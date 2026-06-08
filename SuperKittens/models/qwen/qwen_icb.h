// Qwen3 decode-graph ICB recorder (T=1 only).
//
// WHY: dispatch_model re-encodes ~10·n_layers+5 compute dispatches on the CPU
// every decoded token (computeCommandEncoder + setComputePipelineState +
// setBuffer×N + setBytes×M + dispatchThreadgroups + endEncoding). On M4 base
// that CPU encode is ~20-25% of the decode budget (decode sits ~71% of
// roofline while the kernels themselves hit 78-95% BW → the gap is encode +
// inter-dispatch). This records the whole decode graph ONCE into an
// MTLIndirectCommandBuffer and replays it with a single executeCommandsInBuffer
// per token, so per-token CPU work collapses to: write 2 cursor uints, point
// input_ids at the new token, re-record the 2 RoPE slots/layer (cos/sin offset
// is per-position), and submit.
//
// Scope: T=1 decode, generic-quant path (every projection routed through
// q4k/q6k/q8_0 _matvec, gate+up+silu_mul, down+add_f16). This is exactly the
// Q4_K_M and Q8_0 decode layer; the Q8_0-only fused swiglu/prenorm/addres
// dispatch_layer shortcuts are NOT recorded here — callers gate ICB on the
// generic path being the one in use. Prefill (T>1) always uses dispatch_model.
//
// ICB constraint handling: indirectComputeCommand has no setBytes. Every kernel
// scalar (K, N, eps, rows, n_heads, …) is model-static at T=1 and lives in a
// per-handle args pool buffer; each recorded slot binds the pool at the right
// byte offset. The only per-token-varying scalars are kv_cache_write.pos and
// mha_causal.kv_len — both sourced from a 2-uint cursor buffer rewritten each
// token (no re-record). RoPE's cos/sin table offset is position-dependent and
// is expressed as a setKernelBuffer offset, so the 2 RoPE slots per layer are
// re-recorded per token (cheap: 2·n_layers record() calls vs the full graph).

#ifndef SUPERKITTENS_QWEN_ICB_H
#define SUPERKITTENS_QWEN_ICB_H

#include <Metal/Metal.hpp>
#include <cstdint>
#include <vector>

#include "qwen_model.h"
#include "../../kernels/runtime_bindings.h"
#include "../../inference/silicon/icb_recorder.h"

namespace meow { namespace qwen {

// Returns true when the decode-layer graph for these weights is the generic
// quant-matvec path (no Q8_0-only fused shortcut), i.e. the path qwen_icb
// records. Q4_K_M satisfies this (gate/up=Q4_K, down=Q4_K/Q6_K → no q8_0 fuse).
inline bool icb_decode_path_is_generic(const ModelWeights& W) {
    auto is_q = [](sk::Dtype d) {
        return d == sk::Dtype::Q4_K || d == sk::Dtype::Q6_K || d == sk::Dtype::Q8_0;
    };
    // QKV / O / gate / up must all be quant matvecs (the per-layer recorder
    // assumes quant_matvec_pso != nullptr for them).
    if (!is_q(W.dt_qkv) || !is_q(W.dt_o) || !is_q(W.dt_gate) || !is_q(W.dt_up))
        return false;
    // gate AND up both Q8_0 would route through q8_0_swiglu_m1 in dispatch_layer
    // — keep ICB on the non-fused path so byte-for-byte parity holds.
    if (W.dt_gate == sk::Dtype::Q8_0 && W.dt_up == sk::Dtype::Q8_0) return false;
    return true;
}

struct QwenDecodeIcb {
    sk::silicon::IcbRecorder* rec = nullptr;
    MTL::Buffer* args = nullptr;          // static scalar pool (uint/float words)
    MTL::Buffer* cursor = nullptr;        // [pos, kv_len] uint32, per-token
    uint32_t     n_slots = 0;
    uint32_t     n_layers = 0;

    // Slot indices of the two RoPE dispatches per layer (need per-token
    // cos/sin offset re-record). Sized 2·n_layers.
    std::vector<uint32_t> rope_slots;
    // Cached binding state for each RoPE slot so re-record is a pure offset
    // bump (no re-derivation of buffers/grid each token).
    struct RopeRec {
        MTL::ComputePipelineState* pso;
        MTL::Buffer* x;       // Q or K buffer
        MTL::Buffer* cos_tbl;
        MTL::Buffer* sin_tbl;
        MTL::Buffer* args;    // args pool (seq, head_dim, n_heads triple)
        NS::UInteger args_off;
        MTL::Size grid, tg;
    };
    std::vector<RopeRec> rope_recs;
};

// Build the per-handle decode ICB. Returns nullptr if any required ICB PSO is
// missing or Metal allocation fails (caller then keeps the per-dispatch path).
QwenDecodeIcb* qwen_icb_build(MTL::Device* dev,
                              const ModelPSOs& P,
                              const ModelWeights& W,
                              ModelBuffers& B,
                              const ModelParams& M);

// Per-token: stamp cursor scalars + RoPE offsets for absolute position `pos`,
// with attention key length `kv_len`. Cheap (a couple memcpys + 2·n_layers
// IcbRecorder::record offset bumps). Caller has already written input_ids[0].
void qwen_icb_prepare(QwenDecodeIcb* icb, uint32_t pos, uint32_t kv_len,
                      uint32_t head_dim);

// Replay the whole decode graph with a single executeCommandsInBuffer.
void qwen_icb_replay(QwenDecodeIcb* icb, MTL::CommandBuffer* cmd);

void qwen_icb_destroy(QwenDecodeIcb* icb);

}}  // namespace meow::qwen

#endif  // SUPERKITTENS_QWEN_ICB_H
