// PROFILING — REMOVE BEFORE MERGE (iter7 diagnostic only).
//
// Per-dispatch GPU timestamp profiler using MTLCounterSampleBuffer.
// Threaded as an opt-in global singleton, gated by env SK_PROFILE=1.
// When inactive, SK_PROF_SAMPLE() is a no-op. When active, each call
// inserts a sampleCountersInBuffer at the current encoder and records the
// label tag of the dispatch that just completed (the previous sample).
//
// Lifecycle:
//   sk_profile_begin_token(device)  — alloc CounterSampleBuffer, sample idx=0
//   SK_PROF_SAMPLE(enc, "tag")      — sample timestamp, tag = dispatch JUST ENDED
//   sk_profile_end_token()          — resolve, accumulate µs into per-tag totals
//   sk_profile_report()             — print sorted table to stderr
//
// The first sample (idx 0) is taken right after the FIRST encoder is opened,
// before any dispatch — it is the "baseline" t0. Each subsequent sample tags
// the time between IT and the previous sample.
#ifndef SK_INFERENCE_PROFILE_H
#define SK_INFERENCE_PROFILE_H

#include "Metal/Metal.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sk { namespace profile {

struct Profiler {
    MTL::Device*               dev = nullptr;
    MTL::CounterSampleBuffer*  sb  = nullptr;
    MTL::CounterSet*           cs_ts = nullptr;
    uint32_t                   capacity = 0;
    uint32_t                   next_idx = 0;
    bool                       active   = false;
    bool                       supported = false;
    // Encoder-pair samples: encoder i occupies sample indices 2i and 2i+1
    // (start, end). enc_tags[i] = name. Each split allocates one pair.
    std::vector<std::string>   enc_tags;
    // Aggregate µs per tag across all tokens recorded so far.
    std::unordered_map<std::string, double> totals_us;
    uint32_t                   tokens_recorded = 0;
    // Optional capture of the latest-token raw deltas for sanity inspection.
    std::vector<double>        last_deltas_us;
};

Profiler& g();  // singleton

// Returns true if profiling is enabled (env SK_PROFILE=1) AND the device
// supports timestamp counters at the dispatch boundary. When false, the
// SK_PROF_SAMPLE macro emits no Metal calls.
bool enabled();

// Per-token lifecycle. Capacity = expected max sample count for one token
// (≈ 2 + n_layers * 16 + 8). Allocates a fresh CounterSampleBuffer each token
// (small + simple; this is diagnostic code).
void begin_token(MTL::Device* dev, uint32_t capacity);

// Called after waitUntilCompleted; resolves the sample buffer to GPU
// timestamps, computes per-tag deltas (sample[i] - sample[i-1] → tag[i]),
// converts to µs via the device's CPU↔GPU timestamp ratio, and accumulates.
void end_token();

// Print a sorted table of (tag, total µs, % of token total) to stderr.
// avg_us divides totals by tokens_recorded.
void report();

// Internal: record a sample at idx = next_idx with `tag` naming the dispatch
// that just completed. `barrier=false` is used to avoid extra serialization.
void sample(MTL::ComputeCommandEncoder* enc, const char* tag);

}}  // namespace sk::profile

// One-line call site. Inactive: no-op (a single load + branch).
#define SK_PROF_SAMPLE(enc, tag) \
    do { if (::sk::profile::g().active) ::sk::profile::sample((enc), (tag)); } while (0)

// Stage-boundary path. Apple Silicon supports AtStageBoundary only. We model
// each "kernel class" as its own compute encoder, attaching the global
// CounterSampleBuffer via a ComputePassDescriptor with start/end sample
// indices. After waitUntilCompleted, end-of-encoder minus start-of-encoder
// gives that encoder's GPU time; the gap to the next encoder's start gives
// the encoder-setup / submit idle.
namespace sk { namespace profile {
// Open a new compute encoder that samples its own start+end timestamps into
// the global sample buffer (when active), and tag it with `name`. When the
// profiler is inactive, falls back to a vanilla cmd->computeCommandEncoder().
// Caller must endEncoding() before opening the next.
MTL::ComputeCommandEncoder* open_profiled_encoder(MTL::CommandBuffer* cmd, const char* name);

// End the current encoder, open a new profiled encoder, and overwrite *enc
// with the new pointer. No-op when inactive (just returns *enc unchanged).
void split_encoder(MTL::CommandBuffer* cmd, MTL::ComputeCommandEncoder** enc, const char* name);
}}
#define SK_PROF_SPLIT(cmd, enc_ptr, tag) \
    do { if (::sk::profile::g().active) ::sk::profile::split_encoder((cmd), (enc_ptr), (tag)); } while (0)

#endif
