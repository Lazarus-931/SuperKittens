// qwen_icb.c++ — decode-graph ICB recorder. See qwen_icb.h for rationale.

#include "qwen_icb.h"

#include <cstdio>
#include <cstring>

namespace meow { namespace qwen {

namespace {

// Append-only scalar pool. Words are 4 bytes (uint32 or float, same width).
// Returns the byte offset of the just-appended run. Metal reads a
// `constant T&` from buffer(i) at the recorded offset, so each scalar argument
// of a kernel gets its own word here and the slot binds the pool at that word.
struct ArgsPool {
    std::vector<uint32_t> words;
    NS::UInteger put_u(uint32_t v) { NS::UInteger o = words.size() * 4; words.push_back(v); return o; }
    NS::UInteger put_f(float v)    { uint32_t u; std::memcpy(&u, &v, 4); return put_u(u); }
};

// ICB-compatible PSOs for the decode graph. All resolved via bindings_pso_icb
// (built with setSupportIndirectCommandBuffers=true); any nullptr aborts the
// build and the caller falls back to per-dispatch encode.
struct IcbPSOs {
    MTL::ComputePipelineState* embedding_lookup;
    MTL::ComputePipelineState* rmsnorm;
    MTL::ComputePipelineState* rmsnorm_t1;   // T=1 full-d_model norm (matches dispatch_model)
    MTL::ComputePipelineState* split_packed;
    MTL::ComputePipelineState* rope_qk;
    MTL::ComputePipelineState* kv_cache_write;
    MTL::ComputePipelineState* attn;            // mha_causal
    MTL::ComputePipelineState* add_rmsnorm;
    MTL::ComputePipelineState* silu_mul;
    MTL::ComputePipelineState* add;
    MTL::ComputePipelineState* q4k_matvec;
    MTL::ComputePipelineState* q6k_matvec;
    MTL::ComputePipelineState* q8_0_matvec;
    MTL::ComputePipelineState* gemv_t_2dtile_m1;
    MTL::ComputePipelineState* gemv_t_m1;
    MTL::ComputePipelineState* q6k_head;         // lm-head matvec (Q6_K) — same as q6k_matvec
    MTL::ComputePipelineState* argmax_partial;
    MTL::ComputePipelineState* argmax_reduce;
    bool ok = true;
};

static MTL::ComputePipelineState* req(IcbPSOs& P, const char* name) {
    auto* p = sk::bindings_pso_icb(name);
    if (!p) { P.ok = false; std::fprintf(stderr, "qwen_icb: missing ICB PSO %s\n", name); }
    return p;
}

static MTL::ComputePipelineState* quant_icb_pso(const IcbPSOs& P, sk::Dtype dt) {
    switch (dt) {
        case sk::Dtype::Q8_0: return P.q8_0_matvec;
        case sk::Dtype::Q4_K: return P.q4k_matvec;
        case sk::Dtype::Q6_K: return P.q6k_matvec;
        default:              return nullptr;
    }
}

}  // namespace

QwenDecodeIcb* qwen_icb_build(MTL::Device* dev,
                              const ModelPSOs& P0,
                              const ModelWeights& W,
                              ModelBuffers& B,
                              const ModelParams& M) {
    if (!dev) return nullptr;
    if (M.batch != 1 || M.seq != 1) return nullptr;
    if (M.layers_run != 0 && M.layers_run < M.n_layers) return nullptr;  // debug knob disables ICB
    if (M.capture_layer >= 0) return nullptr;
    if (!icb_decode_path_is_generic(W)) return nullptr;

    IcbPSOs P{};
    P.embedding_lookup = req(P, "embedding_lookup");
    P.rmsnorm          = req(P, "rmsnorm");
    // dispatch_model picks rmsnorm_t1 for the rows==1 (T=1) full-d_model norms
    // (pre-attn, final); the SIMD-tree reduction order differs from rmsnorm, so
    // matching it is required for byte-identical logits. Optional: if absent,
    // fall back to rmsnorm everywhere (parity may then break — guarded below).
    P.rmsnorm_t1       = sk::bindings_pso_icb("rmsnorm_t1");
    P.split_packed     = req(P, "split_packed");
    P.rope_qk          = req(P, "qwen_rope_qk");
    P.kv_cache_write   = req(P, "kv_cache_write");
    P.attn             = req(P, "mha_causal");
    P.add_rmsnorm      = req(P, "add_rmsnorm");
    P.silu_mul         = req(P, "silu_mul_f16");
    P.add              = req(P, "add_f16");
    // Quant matvecs: only require the ones this model actually uses.
    P.q4k_matvec  = sk::bindings_pso_icb("q4k_matvec");
    P.q6k_matvec  = sk::bindings_pso_icb("q6k_matvec");
    P.q8_0_matvec = sk::bindings_pso_icb("q8_0_matvec");
    P.argmax_partial = req(P, "argmax_partial");
    P.argmax_reduce  = req(P, "argmax_reduce");
    if (!P.ok) return nullptr;

    // Parity guard: dispatch_model uses rmsnorm_t1 for the T=1 full-d_model
    // norms iff P0.layer.rmsnorm_t1 is registered. The ICB must use the same
    // kernel or logits diverge in the last fp16 bit. If dispatch_model would
    // use t1 but the ICB-variant PSO is unavailable, bail to per-dispatch.
    const bool model_uses_t1 = (P0.layer.rmsnorm_t1 != nullptr);
    if (model_uses_t1 && !P.rmsnorm_t1) {
        std::fprintf(stderr, "qwen_icb: rmsnorm_t1 ICB PSO unavailable; "
                             "using per-dispatch decode for parity\n");
        return nullptr;
    }
    // PSO for the T=1 full-d_model norm: t1 when the model uses it, else plain.
    MTL::ComputePipelineState* norm_full_pso = model_uses_t1 ? P.rmsnorm_t1 : P.rmsnorm;
    const bool norm_full_is_t1 = model_uses_t1;

    const uint32_t T   = 1;
    const uint32_t hd  = M.head_dim;
    const uint32_t qN  = M.n_heads * hd;
    const uint32_t kvN = M.n_kv_heads * hd;
    const uint32_t qkv_N = qN + 2 * kvN;
    const uint32_t Hg = M.n_heads / M.n_kv_heads;
    const bool v_split = !W.w_v.empty();

    // LM head: Q6_K or Q8_0 native matvec, else fp16 transposed gemv. Match
    // dispatch_model's preference order exactly so logits are byte-identical.
    MTL::Buffer* w_head = W.w_lm_head ? W.w_lm_head : W.w_embed;
    const size_t off_head = (w_head == W.w_lm_head) ? W.off_w_lm_head : 0;
    MTL::ComputePipelineState* head_pso = nullptr;
    bool head_is_quant = false;
    if (W.dt_lm_head == sk::Dtype::Q6_K && W.w_lm_head && P.q6k_matvec) {
        head_pso = P.q6k_matvec; head_is_quant = true;
    } else if (W.dt_lm_head == sk::Dtype::Q8_0 && W.w_lm_head && P.q8_0_matvec) {
        head_pso = P.q8_0_matvec; head_is_quant = true;
    } else {
        P.gemv_t_2dtile_m1 = sk::bindings_pso_icb("gemv_t_fp16_2dtile_m1");
        P.gemv_t_m1        = sk::bindings_pso_icb("gemv_t_fp16_m1");
        head_pso = P.gemv_t_2dtile_m1 ? P.gemv_t_2dtile_m1 : P.gemv_t_m1;
        if (!head_pso) return nullptr;
    }

    // Per-layer slot budget. Worst case (V split): rmsnorm, qkv, v, split,
    // split, qnorm, knorm, rope-q, rope-k, kv_write, attn, o, add_rmsnorm,
    // gate, up, silu, down, add = 18. Plus model: embed(1) + final_norm(1) +
    // lm_head(1) + argmax(2) = 5.
    const uint32_t per_layer_max = 18u;
    const uint32_t max_slots = M.n_layers * per_layer_max + 8u;
    // Highest kernel-buffer index + 1 across the graph. kv_cache_write binds
    // buffer indices 0..9 (new_k,new_v,k_cache,v_cache,B,H_kv,D,seq_in,pos,
    // cache_size) → 10. add_rmsnorm uses 0..7, mha_causal 0..8, rope 0..6.
    const uint32_t max_bind = 10u;

    auto* rec = sk::silicon::IcbRecorder::create(dev, max_slots, max_bind);
    if (!rec) return nullptr;

    auto* out = new QwenDecodeIcb();
    out->rec = rec;
    out->n_layers = M.n_layers;
    out->rope_slots.reserve(2u * M.n_layers);
    out->rope_recs.reserve(2u * M.n_layers);

    // Cursor buffer: [pos, kv_len]. Written per token by qwen_icb_prepare.
    out->cursor = dev->newBuffer(2 * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(out->cursor->contents(), 0, 2 * sizeof(uint32_t));

    // Build the static scalar pool. We append every distinct scalar tuple a
    // kernel needs; offsets are captured inline as we record.
    ArgsPool pool;
    // Pre-stage the args buffer creation after we know the word count: record
    // into a temp list of (slot, build-fn) closures would be cleaner, but the
    // pool only grows and slot recording reads buffer pointers, so we build the
    // args buffer in a second pass. Collect record requests here.
    struct Rec {
        uint32_t slot;
        MTL::ComputePipelineState* pso;
        std::vector<const MTL::Buffer*> bufs;
        std::vector<NS::UInteger> offs;
        // For args-pool bindings we store the binding index into `bufs` whose
        // buffer is the (not-yet-created) args pool, plus its word offset.
        std::vector<int> args_idx;     // indices into bufs that point at args pool
        MTL::Size grid, tg;
        bool barrier;
    };
    std::vector<Rec> recs;
    uint32_t slot = 0;

    auto mark = [&](MTL::Buffer* b) { rec->mark_resource(b); };

    // ── A. Embedding lookup: (table, ids, out, N, D, V) ──
    {
        Rec r; r.slot = slot++; r.pso = P.embedding_lookup; r.barrier = false;
        r.bufs = { W.w_embed, B.input_ids, B.x_a, nullptr, nullptr, nullptr };
        NS::UInteger oN = pool.put_u(T), oD = pool.put_u(M.d_model), oV = pool.put_u(M.vocab_size);
        r.offs = { 0, 0, 0, oN, oD, oV };
        r.args_idx = { 3, 4, 5 };
        const uint32_t D4 = M.d_model / 4;
        r.grid = MTL::Size((D4 + 127) / 128, T, 1); r.tg = MTL::Size(128, 1, 1);
        recs.push_back(std::move(r));
        mark(W.w_embed); mark(B.input_ids); mark(B.x_a);
    }

    // Layer-stack ping-pong x_a ↔ x_b. cur = input to layer L, nxt = output.
    MTL::Buffer* cur = B.x_a;
    MTL::Buffer* nxt = B.x_b;

    auto add_matvec = [&](MTL::ComputePipelineState* pso,
                          MTL::Buffer* A, NS::UInteger offA,
                          MTL::Buffer* Wt, NS::UInteger offW,
                          MTL::Buffer* C, NS::UInteger offC,
                          uint32_t N, uint32_t K) {
        Rec r; r.slot = slot++; r.pso = pso; r.barrier = true;
        r.bufs = { A, Wt, C, nullptr, nullptr };
        NS::UInteger oK = pool.put_u(K), oN = pool.put_u(N);
        r.offs = { offA, offW, offC, oK, oN };
        r.args_idx = { 3, 4 };
        const uint32_t rows_per_tg = 2;
        r.grid = MTL::Size((N + rows_per_tg - 1) / rows_per_tg, 1, 1);
        r.tg   = MTL::Size(128, 1, 1);
        recs.push_back(std::move(r));
        mark(A); mark(Wt); mark(C);
    };

    for (uint32_t L = 0; L < M.n_layers; ++L) {
        const size_t off_norm = (size_t)L * M.d_model * 2;
        const size_t off_qn   = (size_t)L * hd * 2;
        const size_t off_kn   = (size_t)L * hd * 2;

        const sk::Dtype dt_v   = W.dt_v_layer.empty()    ? W.dt_qkv : W.dt_v_layer[L];
        const sk::Dtype dt_down= W.dt_down_layer.empty() ? sk::Dtype::F16 : W.dt_down_layer[L];

        // 1. pre-attn rmsnorm: (x, gamma, y, rows, d, eps). rows==T==1 → t1.
        {
            Rec r; r.slot = slot++; r.pso = norm_full_pso; r.barrier = true;
            r.bufs = { cur, W.w_pre_attn_norm, B.x_norm, nullptr, nullptr, nullptr };
            NS::UInteger oR = pool.put_u(T), oD = pool.put_u(M.d_model), oE = pool.put_f(M.eps);
            r.offs = { 0, off_norm, 0, oR, oD, oE };
            r.args_idx = { 3, 4, 5 };
            if (norm_full_is_t1) { r.grid = MTL::Size(1, T, 1); r.tg = MTL::Size(256, 1, 1); }
            else                 { r.grid = MTL::Size(1, (T + 3) / 4, 1); r.tg = MTL::Size(128, 1, 1); }
            recs.push_back(std::move(r));
            mark(cur); mark(W.w_pre_attn_norm); mark(B.x_norm);
        }

        // 2. QKV matvec(s).
        MTL::ComputePipelineState* pso_qkv = quant_icb_pso(P, W.dt_qkv);
        if (v_split) {
            MTL::ComputePipelineState* pso_v = quant_icb_pso(P, dt_v);
            // QK band [0:qN+kvN], ldC = qkv_N.
            add_matvec(pso_qkv, B.x_norm, 0, W.w_qkv[L], W.w_qkv_off[L],
                       B.qkv_packed, 0, qN + kvN, M.d_model);
            add_matvec(pso_v, B.x_norm, 0, W.w_v[L], W.w_v_off[L],
                       B.qkv_packed, (size_t)(qN + kvN) * 2, kvN, M.d_model);
        } else {
            add_matvec(pso_qkv, B.x_norm, 0, W.w_qkv[L], W.w_qkv_off[L],
                       B.qkv_packed, 0, qkv_N, M.d_model);
        }

        // 3. splits.  split_packed(src, outA, outB, T, A, B)
        auto add_split = [&](MTL::Buffer* src, MTL::Buffer* oA, MTL::Buffer* oB,
                             uint32_t A_, uint32_t B_) {
            Rec r; r.slot = slot++; r.pso = P.split_packed; r.barrier = true;
            r.bufs = { src, oA, oB, nullptr, nullptr, nullptr };
            NS::UInteger oT = pool.put_u(T), oAA = pool.put_u(A_), oBB = pool.put_u(B_);
            r.offs = { 0, 0, 0, oT, oAA, oBB };
            r.args_idx = { 3, 4, 5 };
            const uint32_t tot = A_ + B_;
            r.grid = MTL::Size((tot + 127) / 128, T, 1); r.tg = MTL::Size(128, 1, 1);
            recs.push_back(std::move(r));
            mark(src); mark(oA); mark(oB);
        };
        add_split(B.qkv_packed, B.q, B.kv_pack, qN, 2 * kvN);
        add_split(B.kv_pack, B.k_tmp, B.v_tmp, kvN, kvN);

        // 4. q-norm, k-norm.  rmsnorm(x, gamma, y, rows, d, eps)
        auto add_norm = [&](MTL::Buffer* x, MTL::Buffer* gamma, size_t goff,
                            MTL::Buffer* y, uint32_t rows) {
            Rec r; r.slot = slot++; r.pso = P.rmsnorm; r.barrier = true;
            r.bufs = { x, gamma, y, nullptr, nullptr, nullptr };
            NS::UInteger oR = pool.put_u(rows), oD = pool.put_u(hd), oE = pool.put_f(M.eps);
            r.offs = { 0, goff, 0, oR, oD, oE };
            r.args_idx = { 3, 4, 5 };
            r.grid = MTL::Size(1, (rows + 3) / 4, 1); r.tg = MTL::Size(128, 1, 1);
            recs.push_back(std::move(r));
            mark(x); mark(gamma); mark(y);
        };
        add_norm(B.q, W.w_q_norm, off_qn, B.q, T * M.n_heads);
        add_norm(B.k_tmp, W.w_k_norm, off_kn, B.k_tmp, T * M.n_kv_heads);

        // 5. RoPE on Q and K. cos/sin offset is position-dependent → these two
        // slots are re-recorded per token. Record at pos=0 here; capture the
        // binding state so qwen_icb_prepare can bump the offset cheaply.
        auto add_rope = [&](MTL::Buffer* x, uint32_t n_heads_x) {
            Rec r; r.slot = slot; r.pso = P.rope_qk; r.barrier = true;
            // x,x,cos,sin,seq,head_dim,n_heads — args triple is one pool run.
            NS::UInteger oS = pool.put_u(T), oHD = pool.put_u(hd), oNH = pool.put_u(n_heads_x);
            (void)oHD; (void)oNH;  // contiguous run starting at oS
            r.bufs = { x, x, B.cos_tbl, B.sin_tbl, nullptr, nullptr, nullptr };
            r.offs = { 0, 0, 0, 0, oS, oHD, oNH };
            r.args_idx = { 4, 5, 6 };
            const uint32_t hd4 = (hd / 2) / 4;
            const uint32_t rows_per_tg = (hd4 > 0) ? (1024u / hd4) : 1u;
            const uint32_t row_blocks = (T + rows_per_tg - 1) / rows_per_tg;
            r.grid = MTL::Size(n_heads_x, row_blocks, 1);
            r.tg   = MTL::Size(hd4, rows_per_tg, 1);
            out->rope_slots.push_back(slot);
            QwenDecodeIcb::RopeRec rr;
            rr.pso = P.rope_qk; rr.x = x; rr.cos_tbl = B.cos_tbl; rr.sin_tbl = B.sin_tbl;
            rr.args = nullptr;          // filled with args pool ptr in pass 2
            rr.args_off = oS;
            rr.grid = r.grid; rr.tg = r.tg;
            out->rope_recs.push_back(rr);
            slot++;
            recs.push_back(std::move(r));
            mark(x); mark(B.cos_tbl); mark(B.sin_tbl);
        };
        add_rope(B.q, M.n_heads);
        add_rope(B.k_tmp, M.n_kv_heads);

        // 6. KV cache write. pos comes from cursor[0]; everything else static.
        {
            Rec r; r.slot = slot++; r.pso = P.kv_cache_write; r.barrier = true;
            // new_k,new_v,k_cache,v_cache,B,H_kv,D,seq_in,pos,cache_size
            // B..seq_in + cache_size come from pool; pos from cursor[0].
            r.bufs = { B.k_tmp, B.v_tmp, W.layer_caches[L].k, W.layer_caches[L].v,
                       nullptr, nullptr, nullptr, nullptr, out->cursor, nullptr };
            NS::UInteger oB = pool.put_u(M.batch), oH = pool.put_u(M.n_kv_heads),
                         oD = pool.put_u(hd), oSeq = pool.put_u(T);
            NS::UInteger oCache = pool.put_u(M.cache_max);
            r.offs = { 0, 0, 0, 0, oB, oH, oD, oSeq, /*cursor[0]=pos*/ 0, oCache };
            r.args_idx = { 4, 5, 6, 7, 9 };   // 8 (pos) is cursor, not args
            const uint32_t D4 = hd / 4;
            // dispatchThreads(D4, seq, B*n_kv_heads) tg(32,4,1) → threadgroups:
            r.grid = MTL::Size((D4 + 31) / 32, (T + 3) / 4, M.batch * M.n_kv_heads);
            r.tg   = MTL::Size(32, 4, 1);
            recs.push_back(std::move(r));
            mark(B.k_tmp); mark(B.v_tmp);
            mark(W.layer_caches[L].k); mark(W.layer_caches[L].v); mark(out->cursor);
        }

        // 7. Attention (mha_causal). kv_len from cursor[1].
        {
            Rec r; r.slot = slot++; r.pso = P.attn; r.barrier = true;
            // Q,K,V,O,seq,nheads,n_kv_heads,kv_len,cache_stride
            r.bufs = { B.q, W.layer_caches[L].k, W.layer_caches[L].v, B.attn_out,
                       nullptr, nullptr, nullptr, out->cursor, nullptr };
            NS::UInteger oSeq = pool.put_u(T), oNH = pool.put_u(M.n_heads),
                         oKV = pool.put_u(M.n_kv_heads);
            NS::UInteger oStride = pool.put_u(M.cache_max);
            r.offs = { 0, 0, 0, 0, oSeq, oNH, oKV, /*cursor[1]=kv_len*/ sizeof(uint32_t), oStride };
            r.args_idx = { 4, 5, 6, 8 };   // 7 (kv_len) is cursor
            r.grid = MTL::Size(M.n_kv_heads, (T + 1) / 2, M.batch);
            r.tg   = MTL::Size(Hg * 2 * 32, 1, 1);
            recs.push_back(std::move(r));
            mark(B.q); mark(W.layer_caches[L].k); mark(W.layer_caches[L].v);
            mark(B.attn_out); mark(out->cursor);
        }

        // 8. O-projection.
        MTL::ComputePipelineState* pso_o = quant_icb_pso(P, W.dt_o);
        add_matvec(pso_o, B.attn_out, 0, W.w_o[L], W.w_o_off[L],
                   B.o_proj, 0, M.d_model, M.n_heads * hd);

        // 9. residual + pre-mlp rmsnorm (add_rmsnorm): generic path (non-fused).
        //    (x, delta, gamma, y, y_norm, rows, n, eps)
        {
            Rec r; r.slot = slot++; r.pso = P.add_rmsnorm; r.barrier = true;
            r.bufs = { cur, B.o_proj, W.w_pre_mlp_norm, B.y_attn, B.m_in,
                       nullptr, nullptr, nullptr };
            NS::UInteger oR = pool.put_u(T), oN = pool.put_u(M.d_model), oE = pool.put_f(M.eps);
            r.offs = { 0, 0, off_norm, 0, 0, oR, oN, oE };
            r.args_idx = { 5, 6, 7 };
            r.grid = MTL::Size(1, T, 1); r.tg = MTL::Size(128, 1, 1);
            recs.push_back(std::move(r));
            mark(cur); mark(B.o_proj); mark(W.w_pre_mlp_norm); mark(B.y_attn); mark(B.m_in);
        }

        // 10. gate, up matvecs + silu_mul.
        MTL::ComputePipelineState* pso_gate = quant_icb_pso(P, W.dt_gate);
        MTL::ComputePipelineState* pso_up   = quant_icb_pso(P, W.dt_up);
        add_matvec(pso_gate, B.m_in, 0, W.w_gate[L], W.w_gate_off[L],
                   B.gate_buf, 0, M.n_int, M.d_model);
        add_matvec(pso_up, B.m_in, 0, W.w_up[L], W.w_up_off[L],
                   B.up_buf, 0, M.n_int, M.d_model);
        {
            Rec r; r.slot = slot++; r.pso = P.silu_mul; r.barrier = true;
            r.bufs = { B.gate_buf, B.up_buf, B.up_buf, nullptr };
            NS::UInteger oN = pool.put_u(T * M.n_int);
            r.offs = { 0, 0, 0, oN };
            r.args_idx = { 3 };
            uint32_t N_total = T * M.n_int;
            r.grid = MTL::Size((N_total + 255) / 256, 1, 1); r.tg = MTL::Size(256, 1, 1);
            recs.push_back(std::move(r));
            mark(B.gate_buf); mark(B.up_buf);
        }

        // 11. down matvec + final residual add (generic, non-fused).
        MTL::ComputePipelineState* pso_down = quant_icb_pso(P, dt_down);
        add_matvec(pso_down, B.up_buf, 0, W.w_down[L], W.w_down_off[L],
                   B.mlp_out, 0, M.d_model, M.n_int);
        {
            Rec r; r.slot = slot++; r.pso = P.add; r.barrier = true;
            r.bufs = { B.y_attn, B.mlp_out, nxt, nullptr };
            uint32_t n = T * M.d_model;
            NS::UInteger oN = pool.put_u(n);
            r.offs = { 0, 0, 0, oN };
            r.args_idx = { 3 };
            uint32_t total = (n / 4u) + (n & 3u);
            r.grid = MTL::Size((total + 127) / 128, 1, 1); r.tg = MTL::Size(128, 1, 1);
            recs.push_back(std::move(r));
            mark(B.y_attn); mark(B.mlp_out); mark(nxt);
        }

        MTL::Buffer* tmp = cur; cur = nxt; nxt = tmp;
    }

    // C. final rmsnorm: cur → nxt. rows==T==1 → t1 (matches dispatch_model).
    {
        Rec r; r.slot = slot++; r.pso = norm_full_pso; r.barrier = true;
        r.bufs = { cur, W.w_final_norm, nxt, nullptr, nullptr, nullptr };
        NS::UInteger oR = pool.put_u(T), oD = pool.put_u(M.d_model), oE = pool.put_f(M.eps);
        r.offs = { 0, 0, 0, oR, oD, oE };
        r.args_idx = { 3, 4, 5 };
        if (norm_full_is_t1) { r.grid = MTL::Size(1, T, 1); r.tg = MTL::Size(256, 1, 1); }
        else                 { r.grid = MTL::Size(1, (T + 3) / 4, 1); r.tg = MTL::Size(128, 1, 1); }
        recs.push_back(std::move(r));
        mark(cur); mark(W.w_final_norm); mark(nxt);
    }

    // D. LM head: logits = head(nxt).
    if (head_is_quant) {
        // q6k/q8_0 matvec(B=nxt, A=w_head, C=logits, K=d_model, N=vocab)
        Rec r; r.slot = slot++; r.pso = head_pso; r.barrier = true;
        r.bufs = { nxt, w_head, B.logits, nullptr, nullptr };
        NS::UInteger oK = pool.put_u(M.d_model), oN = pool.put_u(M.vocab_size);
        r.offs = { 0, off_head, 0, oK, oN };
        r.args_idx = { 3, 4 };
        r.grid = MTL::Size((M.vocab_size + 1) / 2, 1, 1); r.tg = MTL::Size(128, 1, 1);
        recs.push_back(std::move(r));
        mark(nxt); mark(w_head); mark(B.logits);
    } else {
        // gemv_t_*: (in, W, out, N, K)
        Rec r; r.slot = slot++; r.pso = head_pso; r.barrier = true;
        r.bufs = { nxt, w_head, B.logits, nullptr, nullptr };
        NS::UInteger oN = pool.put_u(M.vocab_size), oK = pool.put_u(M.d_model);
        r.offs = { 0, off_head, 0, oN, oK };
        r.args_idx = { 3, 4 };
        if (head_pso == P.gemv_t_2dtile_m1) {
            const uint32_t OUT_ROWS_PER_TG = 64;
            r.grid = MTL::Size((M.vocab_size + OUT_ROWS_PER_TG - 1) / OUT_ROWS_PER_TG, 1, 1);
            r.tg   = MTL::Size(32, 16, 1);
        } else {
            const uint32_t BN = 128;
            r.grid = MTL::Size((M.vocab_size + BN - 1) / BN, 1, 1);
            r.tg   = MTL::Size(BN, 1, 1);
        }
        recs.push_back(std::move(r));
        mark(nxt); mark(w_head); mark(B.logits);
    }

    // E. argmax (2-pass): partial → reduce.
    constexpr uint32_t ELTS_PER_TG = 16384u;
    const uint32_t n_blocks = (M.vocab_size + ELTS_PER_TG - 1u) / ELTS_PER_TG;
    if (!B.argmax_val_buf || !B.argmax_idx_buf) {
        if (out->cursor) out->cursor->release();
        delete rec; delete out; return nullptr;
    }
    {
        Rec r; r.slot = slot++; r.pso = P.argmax_partial; r.barrier = true;
        r.bufs = { B.logits, B.argmax_val_buf, B.argmax_idx_buf, nullptr };
        NS::UInteger oV = pool.put_u(M.vocab_size);
        r.offs = { 0, 0, 0, oV };
        r.args_idx = { 3 };
        r.grid = MTL::Size(n_blocks, 1, 1); r.tg = MTL::Size(1024, 1, 1);
        recs.push_back(std::move(r));
        mark(B.logits); mark(B.argmax_val_buf); mark(B.argmax_idx_buf);
    }
    {
        Rec r; r.slot = slot++; r.pso = P.argmax_reduce; r.barrier = true;
        r.bufs = { B.argmax_val_buf, B.argmax_idx_buf, B.output_id, nullptr };
        NS::UInteger oNB = pool.put_u(n_blocks);
        r.offs = { 0, 0, 0, oNB };
        r.args_idx = { 3 };
        r.grid = MTL::Size(1, 1, 1); r.tg = MTL::Size(1024, 1, 1);
        recs.push_back(std::move(r));
        mark(B.argmax_val_buf); mark(B.argmax_idx_buf); mark(B.output_id);
    }

    out->n_slots = slot;

    // Pass 2: materialize the args pool buffer and record every slot. Args-pool
    // bindings (args_idx) had nullptr placeholders; fill with `out->args`.
    out->args = dev->newBuffer(pool.words.size() * 4, MTL::ResourceStorageModeShared);
    std::memcpy(out->args->contents(), pool.words.data(), pool.words.size() * 4);
    rec->mark_resource(out->args);

    for (auto& r : recs) {
        for (int ai : r.args_idx) r.bufs[ai] = out->args;
        rec->record(r.slot, r.pso,
                    r.bufs.data(), r.offs.data(),
                    (uint32_t)r.bufs.size(), r.grid, r.tg, r.barrier);
    }
    // Backfill RoPE rec args pointer for cheap per-token re-record.
    for (auto& rr : out->rope_recs) rr.args = out->args;

    return out;
}

void qwen_icb_prepare(QwenDecodeIcb* icb, uint32_t pos, uint32_t kv_len,
                      uint32_t head_dim) {
    if (!icb) return;
    uint32_t* cur = (uint32_t*)icb->cursor->contents();
    cur[0] = pos;
    cur[1] = kv_len;

    // RoPE cos/sin offset for this absolute position: pos·(hd/2)·2 bytes.
    const NS::UInteger cs_off = (NS::UInteger)pos * (head_dim / 2) * 2;
    for (size_t i = 0; i < icb->rope_recs.size(); ++i) {
        const auto& rr = icb->rope_recs[i];
        const uint32_t slot = icb->rope_slots[i];
        const MTL::Buffer* bufs[7] = { rr.x, rr.x, rr.cos_tbl, rr.sin_tbl,
                                       rr.args, rr.args, rr.args };
        NS::UInteger offs[7] = { 0, 0, cs_off, cs_off,
                                 rr.args_off, rr.args_off + 4, rr.args_off + 8 };
        icb->rec->record(slot, rr.pso, bufs, offs, 7, rr.grid, rr.tg, /*barrier=*/true);
    }
}

void qwen_icb_replay(QwenDecodeIcb* icb, MTL::CommandBuffer* cmd) {
    if (!icb || !cmd) return;
    auto* enc = cmd->computeCommandEncoder();
    icb->rec->execute(enc, 0, icb->n_slots);
    enc->endEncoding();
}

void qwen_icb_destroy(QwenDecodeIcb* icb) {
    if (!icb) return;
    if (icb->rec)    delete icb->rec;
    if (icb->args)   icb->args->release();
    if (icb->cursor) icb->cursor->release();
    delete icb;
}

}}  // namespace meow::qwen
