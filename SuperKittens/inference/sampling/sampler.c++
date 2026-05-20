// Sampler implementation. See sampler.h for rationale.

#include "sampler.h"

#include <cstdio>
#include <cstring>
#include <filesystem>

namespace sk { namespace sampling {

namespace {

// Resolve a sibling libsk.metallib next to the executable's working tree.
// We look at standard locations the rest of SK uses (build/libsk.metallib),
// and fall back to env SK_METALLIB. We avoid a hard build-time path.
MTL::Library* load_default_library(MTL::Device* dev) {
    auto try_path = [&](const char* path) -> MTL::Library* {
        if (!path || !*path) return nullptr;
        if (!std::filesystem::exists(path)) return nullptr;
        auto* url = NS::URL::fileURLWithPath(NS::String::string(path, NS::UTF8StringEncoding));
        NS::Error* err = nullptr;
        return dev->newLibrary(url, &err);
    };
    if (auto* L = try_path(std::getenv("SK_METALLIB"))) return L;
    if (auto* L = try_path("build/libsk.metallib")) return L;
    return nullptr;
}

}  // namespace

Sampler* Sampler::create(MTL::Device* dev, MTL::CommandQueue* q) {
    if (!dev || !q) return nullptr;
    auto* s = new Sampler();
    s->dev_   = dev;
    s->queue_ = q;
    s->lib_   = load_default_library(dev);
    // Allocate the per-row uniform buffer (rows=1, 1 float).
    s->u_buf_ = dev->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
    return s;
}

Sampler Sampler::greedy(MTL::Device* dev, MTL::CommandQueue* q) {
    Sampler s; s.dev_ = dev; s.queue_ = q; s.lib_ = load_default_library(dev);
    s.u_buf_ = dev->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
    s.set_greedy(); return s;
}
Sampler Sampler::top_p(MTL::Device* dev, MTL::CommandQueue* q, float p, float temp) {
    Sampler s = greedy(dev, q); s.set_top_p(p, temp); return s;
}
Sampler Sampler::min_p(MTL::Device* dev, MTL::CommandQueue* q, float p, float temp) {
    Sampler s = greedy(dev, q); s.set_min_p(p, temp); return s;
}
Sampler Sampler::multinomial(MTL::Device* dev, MTL::CommandQueue* q, float temp) {
    Sampler s = greedy(dev, q); s.set_multinomial(temp); return s;
}

Sampler::~Sampler() {
    for (auto& kv : psos_) if (kv.second) kv.second->release();
    psos_.clear();
    if (u_buf_)     u_buf_->release();
    if (probs_buf_) probs_buf_->release();
    if (lib_)       lib_->release();
}

void Sampler::set_greedy()                          { mode_ = Mode::Greedy; }
void Sampler::set_top_p(float p, float temp)        { mode_ = Mode::TopP; p_ = p; temp_ = temp; }
void Sampler::set_min_p(float p, float temp)        { mode_ = Mode::MinP; p_ = p; temp_ = temp; }
void Sampler::set_multinomial(float temp)           { mode_ = Mode::Multinomial; temp_ = temp; }
void Sampler::set_seed(std::uint64_t s)             { rng_.seed(s); }

MTL::ComputePipelineState* Sampler::pso(const char* host_name) {
    auto it = psos_.find(host_name);
    if (it != psos_.end()) return it->second;
    if (!lib_) return nullptr;
    auto* fn = lib_->newFunction(NS::String::string(host_name, NS::UTF8StringEncoding));
    if (!fn) return nullptr;
    NS::Error* err = nullptr;
    auto* p = dev_->newComputePipelineState(fn, &err);
    fn->release();
    if (!p) {
        std::fprintf(stderr, "[sampler] PSO for %s failed\n", host_name);
        return nullptr;
    }
    psos_.emplace(host_name, p);
    return p;
}

void Sampler::ensure_scratch(std::uint32_t vocab_size) {
    // Scratch probs buffer (rows=1, V halfs). Re-allocated only on growth.
    const std::uint32_t need = vocab_size * 2;  // half = 2 bytes
    if (probs_buf_ && probs_cap_ >= need) return;
    if (probs_buf_) probs_buf_->release();
    probs_buf_ = dev_->newBuffer(need, MTL::ResourceStorageModeShared);
    probs_cap_ = need;
}

void Sampler::refresh_uniform() {
    // Re-seedable uniform draw written into the shared u_buf_ host-side.
    // The multinomial kernel reads u[row] verbatim.
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    float u = d(rng_);
    std::memcpy(u_buf_->contents(), &u, sizeof(float));
}

void Sampler::sample(MTL::Buffer* logits_buf, MTL::Buffer* out_buf,
                     std::uint32_t V, MTL::ComputeCommandEncoder* enc) {
    if (!enc || !logits_buf || !out_buf) return;

    const MTL::Size tg   = MTL::Size::Make(1024, 1, 1);
    const MTL::Size grid = MTL::Size::Make(1, 1, 1);  // rows=1

    if (mode_ == Mode::Greedy) {
        auto* p = pso("argmax");
        if (!p) return;
        enc->setComputePipelineState(p);
        enc->setBuffer(logits_buf, 0, 0);
        enc->setBuffer(out_buf,    0, 1);
        enc->setBytes(&V, sizeof(V), 2);
        enc->dispatchThreadgroups(grid, tg);
        return;
    }

    // Stochastic paths share: softmax_temp -> [mask] -> multinomial.
    ensure_scratch(V);
    refresh_uniform();

    auto* sm = pso("softmax_temp");
    if (!sm) return;
    enc->setComputePipelineState(sm);
    enc->setBuffer(logits_buf, 0, 0);
    enc->setBuffer(probs_buf_, 0, 1);
    enc->setBytes(&V, sizeof(V), 2);
    enc->setBytes(&temp_, sizeof(float), 3);
    enc->dispatchThreadgroups(grid, tg);

    if (mode_ == Mode::TopP) {
        auto* mk = pso("top_p_mask");
        if (!mk) return;
        enc->setComputePipelineState(mk);
        enc->setBuffer(probs_buf_, 0, 0);
        enc->setBytes(&V, sizeof(V), 1);
        enc->setBytes(&p_, sizeof(float), 2);
        enc->dispatchThreadgroups(grid, tg);
    } else if (mode_ == Mode::MinP) {
        auto* mk = pso("min_p_mask");
        if (!mk) return;
        enc->setComputePipelineState(mk);
        enc->setBuffer(probs_buf_, 0, 0);
        enc->setBytes(&V, sizeof(V), 1);
        enc->setBytes(&p_, sizeof(float), 2);
        enc->dispatchThreadgroups(grid, tg);
    }
    // mode_ == Multinomial : skip mask.

    auto* mn = pso("multinomial");
    if (!mn) return;
    enc->setComputePipelineState(mn);
    enc->setBuffer(probs_buf_, 0, 0);
    enc->setBuffer(u_buf_,     0, 1);
    enc->setBuffer(out_buf,    0, 2);
    enc->setBytes(&V, sizeof(V), 3);
    enc->dispatchThreadgroups(grid, tg);
}

}}  // namespace sk::sampling
