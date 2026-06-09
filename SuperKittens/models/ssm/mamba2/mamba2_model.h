// Mamba 2 (mamba2-130m-hf) dispatch orchestrator scaffold.
//
// Pipeline per layer (HF Mamba2Mixer, transformers/models/mamba2/modeling_mamba2.py):
//   1. RMSNorm(x)                                                  -> x_n
//   2. in_proj: GEMM (T, D) @ (D, IN_OUT) -> packed (T, IN_OUT)
//        IN_OUT = 2*E + 2*G*N + H
//        split: z (T,E) | xBC (T, E+2*G*N) | dt_raw (T, H)
//   3. conv1d_silu on xBC (causal depthwise, K=conv_kernel=4, then SiLU)
//        -> xBC_post (T, E+2*G*N)
//        split: x (T,E) reshape (T,H,P) | B (T,G,N) | C (T,G,N)
//   4. dt = softplus(dt_raw + dt_bias).clamp(time_step_min, time_step_max)
//        A  = -exp(A_log)                                          (H,)
//   5. SSD: mamba2_ssd / mamba2_step
//        prefill: chunked associative scan over L (HF L398-586)
//        decode : per-token recurrence using ssm_state
//        out: y (T, H, P) reshape (T, E)
//        + D[h] * x[t,h,p] skip
//   6. gate_norm: y = norm_gated(y, z, γ=mixer.norm.weight)        // SiLU(z) * RMSNorm(y)
//   7. out_proj: GEMM (T, E) @ (E, D) -> y (T, D)
//   8. residual + y -> hidden
// Final: RMSNorm(hidden) -> logits via tied lm_head (transpose embed).

#ifndef SUPERKITTENS_MAMBA2_MODEL_H
#define SUPERKITTENS_MAMBA2_MODEL_H

#include <Metal/Metal.hpp>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <string>

namespace meow {
namespace mamba2 {

// SK_MAMBA_SKIP=<csv of stage tokens> drops the named decode stages from the
// dispatch (output is garbage — for ablation timing only). Tokens:
//   inproj outproj lmhead ssd conv gate split prenorm add embed argmax fnorm
// Read once. Delta vs baseline us/step = that stage's GPU contribution.
struct SkipFlags {
    bool inproj=false, outproj=false, lmhead=false, ssd=false, conv=false,
         gate=false, split=false, prenorm=false, add=false, embed=false,
         argmax=false, fnorm=false;
};
// out_proj (N=d_model=768) under-occupies the plain gemv and runs at ~56 GB/s
// vs ~101 for in_proj; split-K (KS K-slices → 4*KS more TGs → reduce) lifts it
// to ~79 GB/s and the whole decode step ~+9% (227→248 tok/s on M4, KS=4).
// On by default; SK_MAMBA_SPLITK=<KS> overrides KS, SK_MAMBA_NO_SPLITK disables.
inline uint32_t splitk_ks() {
    static uint32_t v = []{
        if (std::getenv("SK_MAMBA_NO_SPLITK")) return 0u;
        const char* e = std::getenv("SK_MAMBA_SPLITK");
        if (e) { int k = std::atoi(e); if (k > 0) return (uint32_t)k; }
        return 4u;
    }();
    return v;
}
// Chunked SSD prefill (mamba2_ssd_chunked.metal): SK_MAMBA2_SSD_CHUNKED=1
// selects it for T>1; SK_MAMBA2_SSD_CHUNK=<Q> sets the chunk size. Read fresh
// each call (not static) so one process can A/B without reloading the model.
inline bool ssd_chunked_enabled() {
    const char* e = std::getenv("SK_MAMBA2_SSD_CHUNKED");
    return e && e[0] == '1';
}
inline uint32_t ssd_chunk_q() {
    const char* e = std::getenv("SK_MAMBA2_SSD_CHUNK");
    if (e) { int v = std::atoi(e); if (v >= 32) return (uint32_t)v; }
    // Sweep on M4 base (130m, T=128/512/1024): TTFT improves monotonically with
    // Q — the GPU is already saturated at B*H*(P/4) threadgroups, so chunk
    // parallelism buys nothing and pass 3's extra C reads cost; the win comes
    // from the p-blocked scan. Largest measured Q is the empirical optimum.
    return 1024u;
}

inline const SkipFlags& skip_flags() {
    static SkipFlags s = []{
        SkipFlags f; const char* e = std::getenv("SK_MAMBA_SKIP");
        if (e) { std::string v(e);
            auto has=[&](const char* k){ return v.find(k)!=std::string::npos; };
            f.inproj=has("inproj"); f.outproj=has("outproj"); f.lmhead=has("lmhead");
            f.ssd=has("ssd"); f.conv=has("conv"); f.gate=has("gate");
            f.split=has("split"); f.prenorm=has("prenorm"); f.add=has("add");
            f.embed=has("embed"); f.argmax=has("argmax"); f.fnorm=has("fnorm"); }
        return f; }();
    return s;
}

struct LayerParams {
    uint32_t batch        = 1;
    uint32_t seq          = 1;
    uint32_t d_model      = 768;
    uint32_t intermediate = 1536;   // E
    uint32_t n_heads      = 24;     // H
    uint32_t head_dim     = 64;     // P
    uint32_t state_size   = 128;    // N
    uint32_t n_groups     = 1;      // G
    uint32_t conv_kernel  = 4;
    uint32_t chunk_size   = 256;
    float    eps          = 1e-5f;
    // HF time_step_limit (default (0, inf)); time_step_{min,max} only bound dt_bias init.
    float    dt_min       = 0.0f;
    float    dt_max       = INFINITY;

    uint32_t layer_idx    = 0;
    uint32_t is_decode    = 0;      // 0 = prefill (chunked SSD), 1 = decode (single-step)
    uint32_t pos          = 0;
};

struct LayerPSOs {
    MTL::ComputePipelineState* rmsnorm;
    MTL::ComputePipelineState* rmsnorm_t1 = nullptr;  // optional T=1 fast path
    MTL::ComputePipelineState* gemm;          // gemm_fp16 (prefill T>1)
    // M=1 decode matvec: gemm_fp16's BM=32 MMA tile wastes 31/32 rows and runs
    // the in/out_proj projections at 13-54 GB/s. gemv_fp16_m1 (transB=0) is a
    // real column-per-thread matvec at ~90-110 GB/s on M4 base.
    MTL::ComputePipelineState* gemv   = nullptr;   // gemv_fp16_m1 (transB=0)
    // Split-K matvec for small-N projections (out_proj N=768 under-occupies the
    // plain gemv). Gated on SK_MAMBA_SPLITK; nullptr → use gemv.
    MTL::ComputePipelineState* gevm_splitk_p1 = nullptr;
    MTL::ComputePipelineState* gevm_splitk_p2 = nullptr;
    MTL::ComputePipelineState* split_packed;
    MTL::ComputePipelineState* conv1d_silu;
    MTL::ComputePipelineState* conv1d_silu_step = nullptr;   // O(1) decode conv
    MTL::ComputePipelineState* conv_state_capture = nullptr; // prefill conv_state init
    MTL::ComputePipelineState* mamba2_ssd;    // prefill chunked associative scan
    // Flag-gated chunked-scan prefill (SK_MAMBA2_SSD_CHUNKED=1); all three or none.
    MTL::ComputePipelineState* ssd_chunk_scan = nullptr;
    MTL::ComputePipelineState* ssd_chunk_prop = nullptr;
    MTL::ComputePipelineState* ssd_chunk_fix  = nullptr;
    // p-blocked (4 rows/simdgroup) scan/fix: B/C reads shared 4×. Preferred when
    // P % 4 == 0 and Nstate <= 128; SK_MAMBA2_SSD_PB=1 forces the plain pair.
    MTL::ComputePipelineState* ssd_chunk_scan_pb4 = nullptr;
    MTL::ComputePipelineState* ssd_chunk_fix_pb4  = nullptr;
    MTL::ComputePipelineState* mamba2_step;   // decode per-token recurrence
    MTL::ComputePipelineState* gate_norm;     // SiLU(z) * RMSNorm(y) γ
    MTL::ComputePipelineState* add;
};

struct ModelPSOs {
    LayerPSOs layer;
    MTL::ComputePipelineState* embedding_lookup;
    // LM head is transB=1 (tied embed (V,D)); 2dtile gemv hits ~109 GB/s vs
    // gemm_fp16's 61 at this M=1 / N=50288 shape.
    MTL::ComputePipelineState* gemv_t = nullptr;   // gemv_t_fp16_2dtile_m1
    MTL::ComputePipelineState* argmax;
    // Optional 2-pass fp16 argmax PSOs (≈1.8× at V=50288).
    MTL::ComputePipelineState* argmax_partial = nullptr;
    MTL::ComputePipelineState* argmax_reduce  = nullptr;
};

struct ModelWeights {
    MTL::Buffer* w_embed;
    MTL::Buffer* w_final_norm;
    // per-layer concatenated
    MTL::Buffer* w_pre_norm;     // (n_layers, d_model)
    MTL::Buffer* w_in_proj;      // (n_layers, d_model, IN_OUT)
    MTL::Buffer* w_conv;         // (n_layers, conv_kernel, C_in)
    MTL::Buffer* w_conv_b;       // (n_layers, C_in)
    MTL::Buffer* w_dt_bias;      // (n_layers, n_heads)
    MTL::Buffer* w_A_log;        // (n_layers, n_heads)
    MTL::Buffer* w_D;            // (n_layers, n_heads)
    MTL::Buffer* w_norm;         // (n_layers, intermediate)  mixer.norm.weight
    MTL::Buffer* w_out_proj;     // (n_layers, intermediate, d_model)
};

struct LayerState {
    // conv1d sliding window of last (K-1) inputs to xBC pre-conv.
    MTL::Buffer* conv_state;     // (batch, K-1, C_in) fp16
    // SSM hidden state per head / channel / state.
    MTL::Buffer* ssm_state;      // (batch, H, P, N)   fp16
};

struct ModelBuffers {
    MTL::Buffer* tok_ids;
    MTL::Buffer* x;              // (T, D)
    MTL::Buffer* x_norm;
    MTL::Buffer* in_proj_out;    // (T, IN_OUT)
    MTL::Buffer* z;              // (T, E)
    MTL::Buffer* xBC;            // (T, E + 2*G*N)
    MTL::Buffer* dt_raw;         // (T, H)
    MTL::Buffer* xBC_post;       // (T, E + 2*G*N) after conv+silu
    MTL::Buffer* ssd_out;        // (T, E)
    MTL::Buffer* gated;          // (T, E)
    MTL::Buffer* out_proj_out;   // (T, D)
    MTL::Buffer* logits;         // (V,)
    LayerState* layer_states;    // [n_layers]

    // 2-pass argmax scratch (ceil(vocab_size/16384) partials each).
    MTL::Buffer* argmax_val_buf = nullptr;
    MTL::Buffer* argmax_idx_buf = nullptr;

    // split-K out_proj partials (KS, d_model) fp32.
    MTL::Buffer* splitk_partial = nullptr;

    // Chunked-SSD prefill scratch (fp32): (B, nc_max, H, P, N) and (B, seq_max, H).
    MTL::Buffer* ssd_chunk_states = nullptr;
    MTL::Buffer* ssd_cumdecay     = nullptr;
    uint32_t     ssd_nc_max       = 0;
};

// ── Encode helpers ──────────────────────────────────────────────────

inline void encode_gemm_mb(
    MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
    MTL::Buffer* A, size_t off_A,
    MTL::Buffer* B, size_t off_B,
    MTL::Buffer* C, size_t off_C,
    uint32_t M, uint32_t N, uint32_t K, int transB = 0)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    uint32_t ldA = K;
    uint32_t ldB = (transB ? K : N);
    uint32_t ldC = N;
    int transA = 0, has_bias = 0;
    enc->setBuffer(A, off_A, 0);
    enc->setBuffer(B, off_B, 1);
    enc->setBuffer(C, off_C, 2);
    enc->setBytes(&M, 4, 3); enc->setBytes(&N, 4, 4); enc->setBytes(&K, 4, 5);
    enc->setBytes(&ldA, 4, 6); enc->setBytes(&ldB, 4, 7); enc->setBytes(&ldC, 4, 8);
    enc->setBytes(&transA, 4, 9); enc->setBytes(&transB, 4, 10); enc->setBytes(&has_bias, 4, 11);
    enc->setBuffer(C, off_C, 12);
    enc->dispatchThreadgroups(MTL::Size((N + 63) / 64, (M + 31) / 32, 1),
                              MTL::Size(64, 1, 1));
    enc->endEncoding();
}

// M=1 matvec y[1,N] = x[1,K] @ W[K,N] (transB=0). 128 cols/TG, 128 threads.
inline void encode_gemv_mb(
    MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
    MTL::Buffer* x, size_t off_x,
    MTL::Buffer* W, size_t off_W,
    MTL::Buffer* y, size_t off_y,
    uint32_t N, uint32_t K)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(x, off_x, 0);
    enc->setBuffer(W, off_W, 1);
    enc->setBuffer(y, off_y, 2);
    enc->setBytes(&N, 4, 3);
    enc->setBytes(&K, 4, 4);
    enc->dispatchThreadgroups(MTL::Size((N + 127) / 128, 1, 1),
                              MTL::Size(128, 1, 1));
    enc->endEncoding();
}

// Split-K M=1 matvec (raises TG count for small N): KS K-slices → partial(KS,N)
// → reduce → y. partial must be >= KS*N fp32.
inline void encode_gevm_splitk(
    MTL::CommandBuffer* cmd,
    MTL::ComputePipelineState* p1, MTL::ComputePipelineState* p2,
    MTL::Buffer* x, size_t off_x,
    MTL::Buffer* W, size_t off_W,
    MTL::Buffer* y, size_t off_y,
    MTL::Buffer* partial,
    uint32_t N, uint32_t K, uint32_t KS)
{
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p1);
        enc->setBuffer(x, off_x, 0);
        enc->setBuffer(W, off_W, 1);
        enc->setBuffer(partial, 0, 2);
        enc->setBytes(&N, 4, 3);
        enc->setBytes(&K, 4, 4);
        enc->setBytes(&KS, 4, 5);
        enc->dispatchThreadgroups(MTL::Size((N + 127) / 128, KS, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p2);
        enc->setBuffer(partial, 0, 0);
        enc->setBuffer(y, off_y, 1);
        enc->setBytes(&N, 4, 2);
        enc->setBytes(&KS, 4, 3);
        enc->dispatchThreadgroups(MTL::Size((N + 255) / 256, 1, 1),
                                  MTL::Size(256, 1, 1));
        enc->endEncoding();
    }
}

// M=1 matvec y[1,N] = x[1,K] @ W[N,K]^T (transB=1). 2D-tile: 64 rows/TG,
// (32,16) threads. For the tied LM head (W = embed (V,D)).
inline void encode_gemv_t_2dtile_mb(
    MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
    MTL::Buffer* x, size_t off_x,
    MTL::Buffer* W, size_t off_W,
    MTL::Buffer* y, size_t off_y,
    uint32_t N, uint32_t K)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(x, off_x, 0);
    enc->setBuffer(W, off_W, 1);
    enc->setBuffer(y, off_y, 2);
    enc->setBytes(&N, 4, 3);
    enc->setBytes(&K, 4, 4);
    enc->dispatchThreadgroups(MTL::Size((N + 63) / 64, 1, 1),
                              MTL::Size(32, 16, 1));
    enc->endEncoding();
}

inline void encode_rmsnorm_mb(
    MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
    MTL::Buffer* x, MTL::Buffer* gamma, size_t off_gamma,
    MTL::Buffer* out, uint32_t rows, uint32_t n, float eps,
    MTL::ComputePipelineState* pso_t1 = nullptr)
{
    const bool use_t1 = (pso_t1 != nullptr) && (rows == 1u);
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(use_t1 ? pso_t1 : pso);
    enc->setBuffer(x,     0,         0);
    enc->setBuffer(gamma, off_gamma, 1);
    enc->setBuffer(out,   0,         2);
    enc->setBytes(&rows, 4, 3);
    enc->setBytes(&n,    4, 4);
    enc->setBytes(&eps,  4, 5);
    if (use_t1) {
        enc->dispatchThreadgroups(MTL::Size(1, rows, 1),
                                  MTL::Size(256, 1, 1));
    } else {
        enc->dispatchThreadgroups(MTL::Size(1, (rows + 3) / 4, 1),
                                  MTL::Size(128, 1, 1));
    }
    enc->endEncoding();
}

inline void encode_add_f16(MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
                           MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* y, uint32_t n)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(a, 0, 0);
    enc->setBuffer(b, 0, 1);
    enc->setBuffer(y, 0, 2);
    enc->setBytes(&n, 4, 3);
    uint32_t total = (n / 4u) + (n & 3u);
    enc->dispatchThreadgroups(MTL::Size((total + 127) / 128, 1, 1),
                              MTL::Size(128, 1, 1));
    enc->endEncoding();
}

inline void encode_conv1d_silu(
    MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
    MTL::Buffer* x, MTL::Buffer* w, size_t off_w,
    MTL::Buffer* bias, size_t off_b,
    MTL::Buffer* y, uint32_t Bn, uint32_t L, uint32_t C)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(x,    0,     0);
    enc->setBuffer(w,    off_w, 1);
    enc->setBuffer(bias, off_b, 2);
    enc->setBuffer(y,    0,     3);
    enc->setBytes(&Bn, 4, 4);
    enc->setBytes(&L,  4, 5);
    enc->setBytes(&C,  4, 6);
    uint32_t row_blocks = (L + 3) / 4;
    enc->dispatchThreadgroups(MTL::Size(Bn, row_blocks, 1),
                              MTL::Size(128, 1, 1));
    enc->endEncoding();
}

inline void encode_conv1d_silu_step(
    MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
    MTL::Buffer* x_new, MTL::Buffer* w, size_t off_w,
    MTL::Buffer* bias, size_t off_b,
    MTL::Buffer* y, MTL::Buffer* conv_state,
    uint32_t Bn, uint32_t C, uint32_t K, size_t off_cs = 0)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(x_new,      0,      0);
    enc->setBuffer(w,          off_w,  1);
    enc->setBuffer(bias,       off_b,  2);
    enc->setBuffer(y,          0,      3);
    enc->setBuffer(conv_state, off_cs, 4);
    enc->setBytes(&Bn, 4, 5);
    enc->setBytes(&C,  4, 6);
    enc->setBytes(&K,  4, 7);
    enc->dispatchThreads(MTL::Size(C, Bn, 1), MTL::Size(128, 1, 1));
    enc->endEncoding();
}

inline void encode_conv_state_capture(
    MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
    MTL::Buffer* x, MTL::Buffer* conv_state,
    uint32_t Bn, uint32_t L, uint32_t C, uint32_t K, size_t off_cs = 0)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(x,          0,      0);
    enc->setBuffer(conv_state, off_cs, 1);
    enc->setBytes(&Bn, 4, 2);
    enc->setBytes(&L,  4, 3);
    enc->setBytes(&C,  4, 4);
    enc->setBytes(&K,  4, 5);
    enc->dispatchThreads(MTL::Size(C, K - 1, Bn), MTL::Size(128, 1, 1));
    enc->endEncoding();
}

inline void encode_gate_norm(
    MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
    MTL::Buffer* ssm_out, MTL::Buffer* z,
    MTL::Buffer* w_norm, size_t off_w,
    MTL::Buffer* y, uint32_t rows, uint32_t E, float eps)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(ssm_out, 0,     0);
    enc->setBuffer(z,       0,     1);
    enc->setBuffer(w_norm,  off_w, 2);
    enc->setBuffer(y,       0,     3);
    enc->setBytes(&rows, 4, 4);
    enc->setBytes(&E,    4, 5);
    enc->setBytes(&eps,  4, 6);
    uint32_t row_blocks = (rows + 3) / 4;
    enc->dispatchThreadgroups(MTL::Size(1, row_blocks, 1),
                              MTL::Size(128, 1, 1));
    enc->endEncoding();
}

inline void encode_split(MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
                         MTL::Buffer* src, MTL::Buffer* outA, MTL::Buffer* outB,
                         uint32_t T, uint32_t A, uint32_t B)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(src,  0, 0);
    enc->setBuffer(outA, 0, 1);
    enc->setBuffer(outB, 0, 2);
    enc->setBytes(&T, 4, 3);
    enc->setBytes(&A, 4, 4);
    enc->setBytes(&B, 4, 5);
    enc->dispatchThreads(MTL::Size(A + B, T, 1), MTL::Size(128, 1, 1));
    enc->endEncoding();
}

// ── Per-layer dispatch ─────────────────────────────────────────────

struct DispatchBufs {
    MTL::Buffer* x_in;
    MTL::Buffer* x_out;
    MTL::Buffer* x_norm;
    MTL::Buffer* in_proj_out;
    MTL::Buffer* z;
    MTL::Buffer* xBC;
    MTL::Buffer* dt_raw;
    MTL::Buffer* xBC_post;
    MTL::Buffer* ssd_out;
    MTL::Buffer* gated;
    MTL::Buffer* out_proj_out;
    MTL::Buffer* conv_state;
    MTL::Buffer* ssm_state;
    // Per-lane state offset (bytes) into conv_state / ssm_state. 0 = lane 0
    // (single-stream / batched dispatch from lane 0). A per-lane prefill points
    // these at lane i so the b=0 kernel write lands in lane i's state slot.
    size_t       conv_state_off = 0;
    size_t       ssm_state_off  = 0;
    MTL::Buffer* w_pre_norm;
    MTL::Buffer* w_in_proj;
    MTL::Buffer* w_conv;
    MTL::Buffer* w_conv_b;
    MTL::Buffer* w_dt_bias;
    MTL::Buffer* w_A_log;
    MTL::Buffer* w_D;
    MTL::Buffer* w_norm;
    MTL::Buffer* w_out_proj;
    MTL::Buffer* splitk_partial = nullptr;   // (KS, d_model) fp32 scratch
    MTL::Buffer* ssd_chunk_states = nullptr; // (B, nc_max, H, P, N) fp32 scratch
    MTL::Buffer* ssd_cumdecay     = nullptr; // (B, seq_max, H) fp32 scratch
    uint32_t     ssd_nc_max       = 0;
};

inline void dispatch_layer(
    MTL::CommandBuffer* cmd,
    const LayerPSOs&    P,
    const LayerParams&  p,
    DispatchBufs&       b)
{
    const uint32_t T  = p.batch * p.seq;
    const uint32_t D  = p.d_model;
    const uint32_t E  = p.intermediate;
    const uint32_t H  = p.n_heads;
    const uint32_t Pd = p.head_dim;
    const uint32_t Gv = p.n_groups;
    const uint32_t Nv = p.state_size;
    const uint32_t K  = p.conv_kernel;
    const uint32_t IN_OUT = 2*E + 2*Gv*Nv + H;
    const uint32_t C_in   = E + 2*Gv*Nv;
    const uint32_t L  = p.layer_idx;
    const size_t fp16 = 2;

    const size_t off_pre   = (size_t)L * D * fp16;
    const size_t off_in    = (size_t)L * D * IN_OUT * fp16;
    const size_t off_conv  = (size_t)L * K * C_in * fp16;
    const size_t off_convb = (size_t)L * C_in * fp16;
    const size_t off_dtb   = (size_t)L * H * fp16;
    const size_t off_A     = (size_t)L * H * fp16;
    const size_t off_D     = (size_t)L * H * fp16;
    const size_t off_norm  = (size_t)L * E * fp16;
    const size_t off_outp  = (size_t)L * E * D * fp16;
    const SkipFlags& SK = skip_flags();

    // 1. pre-norm
    if (!SK.prenorm)
    encode_rmsnorm_mb(cmd, P.rmsnorm, b.x_in, b.w_pre_norm, off_pre,
                      b.x_norm, T, D, p.eps, P.rmsnorm_t1);

    // 2. in_proj: (T,D) x (D,IN_OUT) -> (T,IN_OUT). M=1 decode → gemv.
    if (!SK.inproj) {
    if (T == 1 && P.gemv)
        encode_gemv_mb(cmd, P.gemv, b.x_norm, 0, b.w_in_proj, off_in,
                       b.in_proj_out, 0, IN_OUT, D);
    else
        encode_gemm_mb(cmd, P.gemm, b.x_norm, 0, b.w_in_proj, off_in,
                       b.in_proj_out, 0, T, IN_OUT, D);
    }

    // 3. Split packed [z(E) | xBC(C_in) | dt(H)]
    //    First split: z vs (xBC+dt). Write xBC+dt into xBC_post temporarily.
    if (!SK.split) {
    encode_split(cmd, P.split_packed, b.in_proj_out, b.z, b.xBC_post,
                 T, E, C_in + H);
    //    Second split: xBC vs dt_raw.
    encode_split(cmd, P.split_packed, b.xBC_post, b.xBC, b.dt_raw,
                 T, C_in, H);
    }

    // 4. conv1d + silu on xBC (C = C_in). Decode (L=1) reads + rolls the carried
    //    (K-1)-token conv_state; prefill convolves the whole seq (state starts
    //    empty) and captures its last K-1 tokens for the first decode step.
    if (!SK.conv) {
    if (p.is_decode && P.conv1d_silu_step && P.conv_state_capture) {
        encode_conv1d_silu_step(cmd, P.conv1d_silu_step, b.xBC, b.w_conv, off_conv,
                                b.w_conv_b, off_convb, b.xBC_post, b.conv_state,
                                p.batch, C_in, K, b.conv_state_off);
    } else {
        encode_conv1d_silu(cmd, P.conv1d_silu, b.xBC, b.w_conv, off_conv,
                           b.w_conv_b, off_convb, b.xBC_post,
                           p.batch, p.seq, C_in);
        if (P.conv_state_capture)
            encode_conv_state_capture(cmd, P.conv_state_capture, b.xBC, b.conv_state,
                                      p.batch, p.seq, C_in, K, b.conv_state_off);
    }
    }

    // 5. SSD reference. xBC_post layout: [x(T,E) | B(T,G*N) | C(T,G*N)] flat.
    const size_t off_x_in = 0;
    const size_t off_B_in = (size_t)E * fp16;
    const size_t off_C_in = (size_t)(E + Gv * Nv) * fp16;
    if (!SK.ssd) {
        // Chunked-scan prefill. NC=1 is valid (pass 3 early-outs on the zero
        // incoming state) and still wins via the p-blocked scan's shared B/C
        // reads — the serial kernel re-reads B/C once per p-row.
        uint32_t Qc = 0, NC = 0;
        const bool chunked = !p.is_decode && p.seq > 1 && ssd_chunked_enabled()
            && P.ssd_chunk_scan && P.ssd_chunk_prop && P.ssd_chunk_fix
            && b.ssd_chunk_states && b.ssd_cumdecay
            && ((Qc = ssd_chunk_q()),
                (NC = (p.seq + Qc - 1) / Qc),
                (NC >= 1 && NC <= b.ssd_nc_max));
        if (chunked) {
            const char* pbe = std::getenv("SK_MAMBA2_SSD_PB");
            const bool pb4 = P.ssd_chunk_scan_pb4 && P.ssd_chunk_fix_pb4
                && (Pd % 4 == 0) && Nv <= 128
                && !(pbe && pbe[0] == '1');
            const uint32_t grid_p = pb4 ? Pd / 4 : Pd;
            {
                auto* enc = cmd->computeCommandEncoder();
                enc->setComputePipelineState(pb4 ? P.ssd_chunk_scan_pb4
                                                 : P.ssd_chunk_scan);
                enc->setBuffer(b.xBC_post,  off_x_in, 0);
                enc->setBuffer(b.dt_raw,    0,        1);
                enc->setBuffer(b.w_A_log,   off_A,    2);
                enc->setBuffer(b.xBC_post,  off_B_in, 3);
                enc->setBuffer(b.xBC_post,  off_C_in, 4);
                enc->setBuffer(b.w_D,       off_D,    5);
                enc->setBuffer(b.w_dt_bias, off_dtb,  6);
                enc->setBuffer(b.ssd_out,          0, 7);
                enc->setBuffer(b.ssd_chunk_states, 0, 8);
                enc->setBuffer(b.ssd_cumdecay,     0, 9);
                enc->setBytes(&p.batch,  4, 10);
                enc->setBytes(&p.seq,    4, 11);
                enc->setBytes(&H,        4, 12);
                enc->setBytes(&Pd,       4, 13);
                enc->setBytes(&Gv,       4, 14);
                enc->setBytes(&Nv,       4, 15);
                enc->setBytes(&p.dt_min, 4, 16);
                enc->setBytes(&p.dt_max, 4, 17);
                enc->setBytes(&C_in,     4, 18);
                enc->setBytes(&Qc,       4, 19);
                enc->setBytes(&NC,       4, 20);
                enc->dispatchThreadgroups(MTL::Size(p.batch * H, grid_p, NC),
                                          MTL::Size(32, 1, 1));
                enc->endEncoding();
            }
            {
                auto* enc = cmd->computeCommandEncoder();
                enc->setComputePipelineState(P.ssd_chunk_prop);
                enc->setBuffer(b.ssd_chunk_states, 0,               0);
                enc->setBuffer(b.ssd_cumdecay,     0,               1);
                enc->setBuffer(b.ssm_state,        b.ssm_state_off, 2);
                enc->setBytes(&p.batch, 4, 3);
                enc->setBytes(&p.seq,   4, 4);
                enc->setBytes(&H,       4, 5);
                enc->setBytes(&Pd,      4, 6);
                enc->setBytes(&Nv,      4, 7);
                enc->setBytes(&Qc,      4, 8);
                enc->setBytes(&NC,      4, 9);
                enc->dispatchThreadgroups(MTL::Size(p.batch * H, Pd, 1),
                                          MTL::Size(32, 1, 1));
                enc->endEncoding();
            }
            {
                auto* enc = cmd->computeCommandEncoder();
                enc->setComputePipelineState(pb4 ? P.ssd_chunk_fix_pb4
                                                 : P.ssd_chunk_fix);
                enc->setBuffer(b.xBC_post,         off_C_in, 0);
                enc->setBuffer(b.ssd_cumdecay,     0,        1);
                enc->setBuffer(b.ssd_chunk_states, 0,        2);
                enc->setBuffer(b.ssd_out,          0,        3);
                enc->setBytes(&p.batch, 4, 4);
                enc->setBytes(&p.seq,   4, 5);
                enc->setBytes(&H,       4, 6);
                enc->setBytes(&Pd,      4, 7);
                enc->setBytes(&Gv,      4, 8);
                enc->setBytes(&Nv,      4, 9);
                enc->setBytes(&C_in,    4, 10);
                enc->setBytes(&Qc,      4, 11);
                enc->setBytes(&NC,      4, 12);
                enc->dispatchThreadgroups(MTL::Size(p.batch * H, grid_p, NC),
                                          MTL::Size(32, 1, 1));
                enc->endEncoding();
            }
        } else {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.mamba2_ssd);
        enc->setBuffer(b.xBC_post,  off_x_in, 0);
        enc->setBuffer(b.dt_raw,    0,        1);
        enc->setBuffer(b.w_A_log,   off_A,    2);
        enc->setBuffer(b.xBC_post,  off_B_in, 3);
        enc->setBuffer(b.xBC_post,  off_C_in, 4);
        enc->setBuffer(b.w_D,       off_D,    5);
        enc->setBuffer(b.w_dt_bias, off_dtb,  6);
        enc->setBuffer(b.ssd_out,   0,               7);
        enc->setBuffer(b.ssm_state, b.ssm_state_off, 8);
        enc->setBytes(&p.batch,  4, 9);
        enc->setBytes(&p.seq,    4, 10);
        enc->setBytes(&H,        4, 11);
        enc->setBytes(&Pd,       4, 12);
        enc->setBytes(&Gv,       4, 13);
        enc->setBytes(&Nv,       4, 14);
        enc->setBytes(&p.dt_min, 4, 15);
        enc->setBytes(&p.dt_max, 4, 16);
        enc->setBytes(&C_in,     4, 17);   // x/B/C share the interleaved C_in token stride
        enc->dispatchThreadgroups(MTL::Size(p.batch * H, Pd, 1),
                                  MTL::Size(Nv, 1, 1));
        enc->endEncoding();
        }
    }

    // 6. gate_norm
    if (!SK.gate)
    encode_gate_norm(cmd, P.gate_norm, b.ssd_out, b.z,
                     b.w_norm, off_norm, b.gated, T, E, p.eps);

    // 7. out_proj: (T,E)x(E,D)->(T,D). M=1 decode → gemv (or split-K gemv).
    if (!SK.outproj) {
    const uint32_t KS = splitk_ks();
    if (T == 1 && KS && P.gevm_splitk_p1 && P.gevm_splitk_p2 && b.splitk_partial)
        encode_gevm_splitk(cmd, P.gevm_splitk_p1, P.gevm_splitk_p2,
                           b.gated, 0, b.w_out_proj, off_outp,
                           b.out_proj_out, 0, b.splitk_partial, D, E, KS);
    else if (T == 1 && P.gemv)
        encode_gemv_mb(cmd, P.gemv, b.gated, 0, b.w_out_proj, off_outp,
                       b.out_proj_out, 0, D, E);
    else
        encode_gemm_mb(cmd, P.gemm, b.gated, 0, b.w_out_proj, off_outp,
                       b.out_proj_out, 0, T, D, E);
    }

    // 8. residual in-place: x_in += out_proj_out  (add allows aliasing)
    if (!SK.add)
    encode_add_f16(cmd, P.add, b.x_in, b.out_proj_out, b.x_out, T * D);
}

// ── Whole-model dispatch ──────────────────────────────────────────

struct ModelParams {
    uint32_t batch        = 1;
    uint32_t seq          = 1;
    uint32_t n_layers     = 24;
    uint32_t d_model      = 768;
    uint32_t intermediate = 1536;
    uint32_t n_heads      = 24;
    uint32_t head_dim     = 64;
    uint32_t state_size   = 128;
    uint32_t n_groups     = 1;
    uint32_t conv_kernel  = 4;
    uint32_t chunk_size   = 256;
    uint32_t vocab_size   = 50288;
    float    eps          = 1e-5f;
    float    dt_min       = 0.0f;
    float    dt_max       = INFINITY;
    // Batched-decode: when seq==1 and batch=N>1, argmax every row so each
    // request (lane) gets its own next-token (output_id[0..N]). Default 0 keeps
    // the single-row (last-position) argmax byte-identical.
    uint32_t decode_all_rows = 0;
    // Per-lane prefill: dispatch batch=1 but write state into lane `prefill_lane`
    // of the (batch=N) state buffers (offset = lane * per-lane bytes). Used to
    // populate each request's conv/ssm state from its own prompt before the
    // lockstep batched decode. -1 (default) = lane 0 / no offset.
    int32_t  prefill_lane = -1;
};

inline void dispatch_model(
    MTL::CommandBuffer* cmd,
    const ModelPSOs&    P,
    const ModelWeights& W,
    ModelBuffers&       B,
    LayerState*         states,
    const ModelParams&  M,
    MTL::Buffer*        output_id)
{
    const uint32_t T = M.batch * M.seq;
    const SkipFlags& SK = skip_flags();

    // A. Embedding lookup
    if (!SK.embed) {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.embedding_lookup);
        enc->setBuffer(W.w_embed,   0, 0);
        enc->setBuffer(B.tok_ids,   0, 1);
        enc->setBuffer(B.x,         0, 2);
        enc->setBytes(&T,            4, 3);
        enc->setBytes(&M.d_model,    4, 4);
        enc->setBytes(&M.vocab_size, 4, 5);
        const uint32_t D4 = M.d_model / 4;
        enc->dispatchThreadgroups(MTL::Size((D4 + 127) / 128, T, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    // B. Layer stack — in-place residual on B.x
    for (uint32_t L = 0; L < M.n_layers; ++L) {
        LayerParams lp;
        lp.batch        = M.batch;
        lp.seq          = M.seq;
        lp.d_model      = M.d_model;
        lp.intermediate = M.intermediate;
        lp.n_heads      = M.n_heads;
        lp.head_dim     = M.head_dim;
        lp.state_size   = M.state_size;
        lp.n_groups     = M.n_groups;
        lp.conv_kernel  = M.conv_kernel;
        lp.chunk_size   = M.chunk_size;
        lp.eps          = M.eps;
        lp.dt_min       = M.dt_min;
        lp.dt_max       = M.dt_max;
        lp.layer_idx    = L;
        lp.is_decode    = (M.seq == 1) ? 1u : 0u;

        DispatchBufs db{};
        db.x_in         = B.x;
        db.x_out        = B.x;
        db.x_norm       = B.x_norm;
        db.in_proj_out  = B.in_proj_out;
        db.z            = B.z;
        db.xBC          = B.xBC;
        db.dt_raw       = B.dt_raw;
        db.xBC_post     = B.xBC_post;
        db.ssd_out      = B.ssd_out;
        db.gated        = B.gated;
        db.out_proj_out = B.out_proj_out;
        db.conv_state   = states[L].conv_state;
        db.ssm_state    = states[L].ssm_state;
        if (M.prefill_lane >= 0) {
            const size_t C_in = M.intermediate + 2 * (size_t)M.n_groups * M.state_size;
            const size_t conv_lane = (size_t)(M.conv_kernel - 1) * C_in * 2;  // fp16
            const size_t ssm_lane  = (size_t)M.n_heads * M.head_dim
                                   * M.state_size * sizeof(float);
            db.conv_state_off = (size_t)M.prefill_lane * conv_lane;
            db.ssm_state_off  = (size_t)M.prefill_lane * ssm_lane;
        }
        db.w_pre_norm   = W.w_pre_norm;
        db.w_in_proj    = W.w_in_proj;
        db.w_conv       = W.w_conv;
        db.w_conv_b     = W.w_conv_b;
        db.w_dt_bias    = W.w_dt_bias;
        db.w_A_log      = W.w_A_log;
        db.w_D          = W.w_D;
        db.w_norm       = W.w_norm;
        db.w_out_proj   = W.w_out_proj;
        db.splitk_partial = B.splitk_partial;
        db.ssd_chunk_states = B.ssd_chunk_states;
        db.ssd_cumdecay     = B.ssd_cumdecay;
        db.ssd_nc_max       = B.ssd_nc_max;

        dispatch_layer(cmd, P.layer, lp, db);
    }

    // C. Final RMSNorm  B.x -> B.x_norm
    if (!SK.fnorm)
    encode_rmsnorm_mb(cmd, P.layer.rmsnorm, B.x, W.w_final_norm, 0,
                      B.x_norm, T, M.d_model, M.eps, P.layer.rmsnorm_t1);

    // D. LM head (tied): (T,D) x (V,D)^T → (T,V) logits. M=1 decode → gemv_t.
    if (!SK.lmhead) {
    if (T == 1 && P.gemv_t) {
        encode_gemv_t_2dtile_mb(cmd, P.gemv_t, B.x_norm, 0, W.w_embed, 0,
                                B.logits, 0, M.vocab_size, M.d_model);
    } else {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.layer.gemm);
        const uint32_t M_v = T, K_v = M.d_model, N_v = M.vocab_size;
        uint32_t ldA = K_v, ldB = K_v, ldC = N_v;
        int transA = 0, transB = 1, has_bias = 0;
        enc->setBuffer(B.x_norm,   0, 0);
        enc->setBuffer(W.w_embed,  0, 1);
        enc->setBuffer(B.logits,   0, 2);
        enc->setBytes(&M_v,      4, 3); enc->setBytes(&N_v,      4, 4);
        enc->setBytes(&K_v,      4, 5); enc->setBytes(&ldA,      4, 6);
        enc->setBytes(&ldB,      4, 7); enc->setBytes(&ldC,      4, 8);
        enc->setBytes(&transA,   4, 9); enc->setBytes(&transB,   4, 10);
        enc->setBytes(&has_bias, 4, 11);
        enc->setBuffer(B.logits, 0, 12);
        enc->dispatchThreadgroups(MTL::Size((N_v + 63) / 64, (M_v + 31) / 32, 1),
                                  MTL::Size(64, 1, 1));
        enc->endEncoding();
    }
    }

    // E. Argmax. Single-stream / prefill: LAST row only -> output_id[0].
    //    Batched-decode (decode_all_rows, seq==1): every row r -> output_id[r],
    //    one next-token per request lane. The 2-pass reduce writes a single int,
    //    so per-row it needs its own partials region; with a shared scratch we
    //    keep the simple single-pass argmax per row (correctness over the ~us
    //    saving — argmax is a sliver of the bandwidth-bound decode step).
    if (!SK.argmax) {
        const bool all_rows = (M.decode_all_rows && M.seq == 1u && T > 1u);
        const bool can_2pass = P.argmax_partial && P.argmax_reduce
                            && B.argmax_val_buf && B.argmax_idx_buf;
        if (all_rows) {
            for (uint32_t r = 0; r < T; ++r) {
                const size_t row_off = (size_t)r * M.vocab_size * 2;
                auto* enc = cmd->computeCommandEncoder();
                enc->setComputePipelineState(P.argmax);
                enc->setBuffer(B.logits,  row_off,             0);
                enc->setBuffer(output_id, (size_t)r * sizeof(int32_t), 1);
                enc->setBytes(&M.vocab_size, 4, 2);
                enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
                enc->endEncoding();
            }
        } else if (can_2pass) {
            const size_t last_off = (size_t)(T - 1) * M.vocab_size * 2;
            constexpr uint32_t ELTS_PER_TG = 16384u;
            const uint32_t n_blocks = (M.vocab_size + ELTS_PER_TG - 1u) / ELTS_PER_TG;
            {
                auto* enc = cmd->computeCommandEncoder();
                enc->setComputePipelineState(P.argmax_partial);
                enc->setBuffer(B.logits,         last_off, 0);
                enc->setBuffer(B.argmax_val_buf, 0,        1);
                enc->setBuffer(B.argmax_idx_buf, 0,        2);
                enc->setBytes(&M.vocab_size,     4, 3);
                enc->dispatchThreadgroups(MTL::Size(n_blocks, 1, 1),
                                          MTL::Size(1024, 1, 1));
                enc->endEncoding();
            }
            {
                auto* enc = cmd->computeCommandEncoder();
                enc->setComputePipelineState(P.argmax_reduce);
                enc->setBuffer(B.argmax_val_buf, 0, 0);
                enc->setBuffer(B.argmax_idx_buf, 0, 1);
                enc->setBuffer(output_id,        0, 2);
                enc->setBytes(&n_blocks,         4, 3);
                enc->dispatchThreadgroups(MTL::Size(1, 1, 1),
                                          MTL::Size(1024, 1, 1));
                enc->endEncoding();
            }
        } else {
            const size_t last_off = (size_t)(T - 1) * M.vocab_size * 2;
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.argmax);
            enc->setBuffer(B.logits,  last_off, 0);
            enc->setBuffer(output_id, 0,        1);
            enc->setBytes(&M.vocab_size, 4, 2);
            enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
            enc->endEncoding();
        }
    }
}

}  // namespace mamba2
}  // namespace meow

#endif
