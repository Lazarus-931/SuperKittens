# Phase 4 — Inference Executor

Consolidates launcher boilerplate (command queue, PSO cache, residency set, scratch pool, ICB lifecycle) into one shared object. Per-model launchers shrink ~50% LOC.

## Pre-reqs
- Phase 2 (token_args + ICB recording) lands first. Executor wraps the ICB lifecycle that Phase 2 introduces.
- Phase 1 (Sampler) optional but ideally landed: executor exposes `executor.attach_sampler(sampler)` to wire the post-logits stage.

## Surface (C++ side)

```c++
namespace sk {

struct Executor {
    Executor(MTL::Device* dev);
    ~Executor();

    // PSO cache. Idempotent; returns nullptr on missing host_name.
    MTL::ComputePipelineState* pso(const char* host_name);

    // Scratch buffer pool. Size is rounded up to next power-of-two.
    // Buffers are reused across tokens; pool clears on reset().
    MTL::Buffer* scratch(size_t nbytes);

    // Residency set. Caller hands in long-lived buffers (weights, KV cache,
    // token_args). Executor keeps them resident for the lifetime of the model.
    void make_resident(MTL::Buffer* buf);

    // Decode-token graph. Caller passes a closure that emits dispatches via
    // executor.dispatch(...). The closure runs exactly once at record time;
    // every subsequent token replays the recorded ICB.
    void record_token_icb(std::function<void(MTL::ComputeCommandEncoder*)> graph_fn);
    void replay_token(MTL::CommandBuffer* cmd);

    // Convenience dispatch. Recorded into the currently-open ICB during
    // record_token_icb; direct-encoded otherwise (for prefill / setup).
    void dispatch(const char* host_name,
                  std::initializer_list<MTL::Buffer*> buffers,
                  MTL::Size grid, MTL::Size tg);

    // Sampler integration (Phase 1 hook).
    void attach_sampler(sk_sampler_t* s);

    // Lifecycle.
    void reset();  // Drop ICB, clear scratch pool. Keeps PSO cache.

private:
    MTL::Device*       _dev;
    MTL::CommandQueue* _queue;
    std::unordered_map<std::string, MTL::ComputePipelineState*> _psos;
    std::vector<MTL::Buffer*> _scratch_pool;
    MTL::ResidencySet* _residency;
    MTL::IndirectCommandBuffer* _icb;
    sk_sampler_t* _sampler;
};

} // namespace sk
```

## Migration target: qwen3

Before:
```c++
// In qwen launcher.c++ (~600 LOC)
void sk_qwen_create(...) {
    // 100 LOC of PSO creation
    // 80 LOC of scratch allocs
    // 50 LOC of residency set setup
    // ...
}
void sk_qwen_forward(...) {
    // Per-token dispatch loop with inline setBytes everywhere.
}
```

After (Phase 2 + Phase 4):
```c++
// In qwen launcher.c++ (~300 LOC)
void sk_qwen_create(handle, cfg) {
    handle->exec = new sk::Executor(device);
    qwen3_load_weights(handle);  // Phase 3 loader
    handle->exec->record_token_icb([&](auto* enc) {
        for (uint l = 0; l < cfg.n_layers; l++) {
            qwen3_emit_layer_dispatches(handle, l, enc);  // pure dispatch sequence
        }
        qwen3_emit_lm_head(handle, enc);
        handle->exec->attach_sampler(handle->sampler)->sample(...);
    });
}
void sk_qwen_forward(handle, token_id, out) {
    // Patch token_args (Phase 2 work).
    handle->args.token_id = token_id;
    handle->args.current_pos++;
    memcpy(handle->token_args_buf->contents(), &handle->args, sizeof(handle->args));
    // Replay.
    auto cmd = handle->exec->_queue->commandBuffer();
    handle->exec->replay_token(cmd);
    cmd->commit(); cmd->waitUntilCompleted();
    // out is written by the sampler at the tail of the ICB.
}
```

## Validation (offline)
- `./build.sh` clean.
- `nm build/libsk.dylib | grep _ZN2sk8Executor` shows the symbols.
- Unit test that constructs/destructs an Executor with a stub device.
- Unit test that PSO cache returns the same handle for repeated `pso("rmsnorm")` calls.
- Static test: qwen3 launcher LOC count is at least 30% lower than pre-migration.

## What this PR does NOT include
- Gemma4 migration (incremental, follow-up).
- DeepSeek migration (out of scope; their launcher is still scaffolding).
- Multi-stream prefetch (would extend Executor in a separate PR).
- Bench-on-mini (real model forward). Numbers come in a follow-up.

## Risk
Phase 2 + Phase 4 together touch every dispatch site in qwen3 and gemma4. Without a bench gate, regression risk is real. The exit criterion for this PR is "static structural checks pass and offline tests pass" — actual perf validation gates a separate follow-up.
