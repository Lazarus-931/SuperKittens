// PROFILING — REMOVE BEFORE MERGE (iter7 diagnostic only).
//
// Stage-boundary timing on Apple Silicon. Each "kernel class" runs in its own
// MTLComputeCommandEncoder, opened with a ComputePassDescriptor that has the
// global CounterSampleBuffer attached at start/end sample indices. After
// waitUntilCompleted we resolve N pairs of timestamps; each pair gives the
// GPU time of one encoder. The gaps between consecutive pairs (end of i →
// start of i+1) accumulate into a "_encoder_gap" tag — encoder-setup +
// submit-wait time.
#include "profile.h"
#include "Metal/Metal.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace sk { namespace profile {

Profiler& g() { static Profiler p; return p; }

static bool s_checked_env = false;
static bool s_env_on = false;
static bool env_on() {
    if (!s_checked_env) {
        const char* v = std::getenv("SK_PROFILE");
        s_env_on = (v && v[0] && v[0] != '0');
        s_checked_env = true;
    }
    return s_env_on;
}

bool enabled() { return env_on(); }

static MTL::CounterSet* find_timestamp_set(MTL::Device* dev) {
    auto* arr = dev->counterSets();
    if (!arr) return nullptr;
    const NS::UInteger n = arr->count();
    auto* want = MTL::CommonCounterSetTimestamp;
    for (NS::UInteger i = 0; i < n; ++i) {
        auto* cs = (MTL::CounterSet*)arr->object(i);
        if (!cs) continue;
        auto* nm = cs->name();
        if (nm && want && nm->isEqualToString(want)) return cs;
    }
    return nullptr;
}

void begin_token(MTL::Device* dev, uint32_t capacity) {
    auto& p = g();
    if (!env_on()) { p.active = false; return; }
    p.dev = dev;
    if (!p.cs_ts) {
        p.cs_ts = find_timestamp_set(dev);
        if (!p.cs_ts) {
            std::fprintf(stderr, "[sk_profile] no timestamp counter set; disabling\n");
            p.supported = false; p.active = false;
            return;
        }
        const bool ok_stage =
            dev->supportsCounterSampling(MTL::CounterSamplingPointAtStageBoundary);
        std::fprintf(stderr, "[sk_profile] stage_boundary=%d\n", (int)ok_stage);
        if (!ok_stage) { p.supported = false; p.active = false; return; }
        p.supported = true;
    }
    // Reuse the sample buffer across tokens when its capacity is sufficient.
    if (p.sb && p.capacity >= capacity) {
        // reuse
    } else {
        if (p.sb) { p.sb->release(); p.sb = nullptr; }
        auto* desc = MTL::CounterSampleBufferDescriptor::alloc()->init();
        desc->setCounterSet(p.cs_ts);
        desc->setStorageMode(MTL::StorageModeShared);
        desc->setSampleCount(capacity);
        NS::Error* err = nullptr;
        p.sb = dev->newCounterSampleBuffer(desc, &err);
        desc->release();
        if (!p.sb) {
            std::fprintf(stderr, "[sk_profile] newCounterSampleBuffer(%u) failed\n", capacity);
            if (err) err->release();
            p.active = false;
            return;
        }
        p.capacity = capacity;
    }
    p.next_idx = 0;
    p.enc_tags.clear();
    p.last_deltas_us.clear();
    p.active = true;
}

// Legacy within-encoder sampler — unsupported on Apple Silicon. No-op.
void sample(MTL::ComputeCommandEncoder*, const char*) {}

MTL::ComputeCommandEncoder* open_profiled_encoder(MTL::CommandBuffer* cmd, const char* name) {
    auto& p = g();
    if (!p.active || !p.sb) {
        return cmd->computeCommandEncoder();
    }
    // Each encoder consumes 2 sample slots (start, end).
    if (p.next_idx + 2 > p.capacity) {
        std::fprintf(stderr, "[sk_profile] sample-slot overflow at enc=%zu (cap=%u)\n",
                     p.enc_tags.size(), p.capacity);
        p.active = false;
        return cmd->computeCommandEncoder();
    }
    auto* pass = MTL::ComputePassDescriptor::computePassDescriptor();
    auto* arr = pass->sampleBufferAttachments();
    auto* att = arr->object(0);
    att->setSampleBuffer(p.sb);
    att->setStartOfEncoderSampleIndex(p.next_idx);
    att->setEndOfEncoderSampleIndex(p.next_idx + 1);
    p.enc_tags.push_back(name ? name : "?");
    p.next_idx += 2;
    auto* enc = cmd->computeCommandEncoder(pass);
    return enc;
}

void split_encoder(MTL::CommandBuffer* cmd, MTL::ComputeCommandEncoder** enc, const char* name) {
    if (!enc) return;
    if (*enc) (*enc)->endEncoding();
    *enc = open_profiled_encoder(cmd, name);
}

void end_token() {
    auto& p = g();
    if (!p.active || !p.sb || p.next_idx < 2) {
        p.active = false; return;
    }
    NS::Range r(0, p.next_idx);
    auto* data = p.sb->resolveCounterRange(r);
    if (!data) { p.active = false; return; }
    const auto* ts = reinterpret_cast<const MTL::CounterResultTimestamp*>(data->mutableBytes());
    MTL::Timestamp cpu0 = 0, gpu0 = 0, cpu1 = 0, gpu1 = 0;
    p.dev->sampleTimestamps(&cpu0, &gpu0);
    p.dev->sampleTimestamps(&cpu1, &gpu1);
    double gpu_to_ns;
    if (gpu1 != gpu0) {
        gpu_to_ns = (double)(cpu1 - cpu0) / (double)(gpu1 - gpu0);
    } else {
        gpu_to_ns = 1.0;
    }
    p.last_deltas_us.clear();
    const size_t n_enc = p.enc_tags.size();
    double prev_end_ts = 0.0;
    for (size_t i = 0; i < n_enc; ++i) {
        const uint64_t a = ts[2 * i].timestamp;
        const uint64_t b = ts[2 * i + 1].timestamp;
        if (a == (uint64_t)~0ULL || b == (uint64_t)~0ULL) continue;
        const double a_ns = (double)a * gpu_to_ns;
        const double b_ns = (double)b * gpu_to_ns;
        const double enc_us = (b_ns > a_ns) ? (b_ns - a_ns) / 1000.0 : 0.0;
        p.last_deltas_us.push_back(enc_us);
        p.totals_us[p.enc_tags[i]] += enc_us;
        if (i > 0 && a_ns > prev_end_ts) {
            const double gap_us = (a_ns - prev_end_ts) / 1000.0;
            p.totals_us["_encoder_gap"] += gap_us;
        }
        prev_end_ts = b_ns;
    }
    p.tokens_recorded++;
    p.active = false;
}

void report() {
    auto& p = g();
    if (p.tokens_recorded == 0) {
        std::fprintf(stderr, "[sk_profile] no tokens recorded\n");
        return;
    }
    std::vector<std::pair<std::string,double>> rows;
    rows.reserve(p.totals_us.size());
    double total = 0.0;
    for (auto& kv : p.totals_us) { rows.emplace_back(kv.first, kv.second); total += kv.second; }
    std::sort(rows.begin(), rows.end(),
              [](const auto& a, const auto& b){ return a.second > b.second; });
    std::fprintf(stderr, "\n[sk_profile] per-encoder GPU breakdown — tokens=%u, total_us/tok=%.1f\n",
                 p.tokens_recorded, total / p.tokens_recorded);
    std::fprintf(stderr, "  %-32s  %10s  %8s\n", "tag", "us/tok", "pct");
    std::fprintf(stderr, "  %-32s  %10s  %8s\n",
                 "--------------------------------", "----------", "--------");
    for (auto& r : rows) {
        const double avg = r.second / p.tokens_recorded;
        const double pct = (total > 0) ? (100.0 * r.second / total) : 0.0;
        std::fprintf(stderr, "  %-32s  %10.2f  %7.2f%%\n", r.first.c_str(), avg, pct);
    }
    std::fprintf(stderr, "\n");
}

}}  // namespace sk::profile

extern "C" int  sk_profile_enabled() { return sk::profile::enabled() ? 1 : 0; }
extern "C" void sk_profile_report()  { sk::profile::report(); }
extern "C" void sk_profile_reset() {
    auto& p = sk::profile::g();
    p.totals_us.clear();
    p.tokens_recorded = 0;
}
