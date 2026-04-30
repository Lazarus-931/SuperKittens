//
//  mamba3.c++
//  SuperKittens
//
//  Host-side dispatch for Mamba-3 SISO/MIMO forward kernels.
//

#include "../../../meow.h"

namespace meow::mamba::mamba3 {

struct Mamba3Config {
    int batch;
    int n_heads;
    int seq_len;
    int head_dim_qk;
    int head_dim_v;
    int chunk_size;
    int mimo_rank;    // 0 = SISO
};

// Args struct matching the Metal side
struct Mamba3FwdArgs {
    uint32_t batch;
    uint32_t nheads;
    uint32_t seq_len;
    uint32_t n_chunks;
};

inline size_t siso_buffer_bytes(const Mamba3Config& cfg) {
    size_t n = cfg.batch * cfg.n_heads * cfg.seq_len;
    size_t qk = n * cfg.head_dim_qk * sizeof(__fp16);
    size_t v  = n * cfg.head_dim_v  * sizeof(__fp16);
    size_t a  = n * sizeof(float);
    size_t angle = n * (cfg.head_dim_qk / 2) * sizeof(__fp16);
    return 2 * qk + v + 2 * a + angle + v; // Q,K + V + A,B + angle + O
}

inline size_t mimo_buffer_bytes(const Mamba3Config& cfg) {
    size_t n = cfg.batch * cfg.n_heads * cfg.seq_len;
    size_t qk = n * cfg.head_dim_qk * cfg.mimo_rank * sizeof(__fp16);
    size_t v  = n * cfg.head_dim_v  * sizeof(__fp16);
    size_t a  = n * sizeof(float);
    size_t angle = n * (cfg.head_dim_qk / 2) * sizeof(__fp16);
    return 2 * qk + v + 2 * a + angle + v; // Q,K + V + A,B + angle + O
}

// Dispatch SISO forward kernel
inline double dispatch_siso_fwd(
    MTL::Device* device,
    MTL::CommandQueue* queue,
    MTL::ComputePipelineState* pso,
    MTL::Buffer* bufQ,
    MTL::Buffer* bufK,
    MTL::Buffer* bufV,
    MTL::Buffer* bufA,
    MTL::Buffer* bufB,
    MTL::Buffer* bufAngle,
    MTL::Buffer* bufO,
    MTL::Buffer* bufArgs,
    const Mamba3Config& cfg)
{
    uint32_t n_chunks = cfg.seq_len / cfg.chunk_size;
    const uint32_t threads_per_tg = static_cast<uint32_t>(cfg.chunk_size * 4);

    auto* cmd = queue->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);

    enc->setBuffer(bufQ,     0, 0);
    enc->setBuffer(bufK,     0, 1);
    enc->setBuffer(bufV,     0, 2);
    enc->setBuffer(bufA,     0, 3);
    enc->setBuffer(bufB,     0, 4);
    enc->setBuffer(bufAngle, 0, 5);
    enc->setBuffer(bufO,     0, 6);
    enc->setBuffer(bufArgs,  0, 7);

    // Chunks are looped sequentially inside the kernel for state recurrence
    enc->dispatchThreadgroups(
        MTL::Size(cfg.batch, cfg.n_heads, 1),
        MTL::Size(threads_per_tg, 1, 1));

    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    return cmd->GPUEndTime() - cmd->GPUStartTime();
}

// Dispatch MIMO forward kernel
inline double dispatch_mimo_fwd(
    MTL::Device* device,
    MTL::CommandQueue* queue,
    MTL::ComputePipelineState* pso,
    MTL::Buffer* bufQ,
    MTL::Buffer* bufK,
    MTL::Buffer* bufV,
    MTL::Buffer* bufA,
    MTL::Buffer* bufB,
    MTL::Buffer* bufAngle,
    MTL::Buffer* bufO,
    MTL::Buffer* bufArgs,
    const Mamba3Config& cfg)
{
    // Same dispatch shape — the kernel template handles rank internally
    return dispatch_siso_fwd(device, queue, pso,
                             bufQ, bufK, bufV, bufA, bufB, bufAngle, bufO, bufArgs,
                             cfg);
}

// Allocate all buffers for a SISO run
struct Mamba3Buffers {
    MTL::Buffer* Q;
    MTL::Buffer* K;
    MTL::Buffer* V;
    MTL::Buffer* A;
    MTL::Buffer* B;
    MTL::Buffer* angle;
    MTL::Buffer* O;
    MTL::Buffer* args;

    void release() {
        Q->release(); K->release(); V->release();
        A->release(); B->release(); angle->release();
        O->release(); args->release();
    }
};

inline Mamba3Buffers allocate_siso_buffers(MTL::Device* device, const Mamba3Config& cfg) {
    size_t seq_total = (size_t)cfg.batch * cfg.n_heads * cfg.seq_len;
    size_t qk_elems  = seq_total * cfg.head_dim_qk;
    size_t v_elems   = seq_total * cfg.head_dim_v;
    size_t a_elems   = seq_total;
    size_t ang_elems = seq_total * (cfg.head_dim_qk / 2);

    auto mode = MTL::ResourceStorageModeShared;

    Mamba3Buffers bufs;
    bufs.Q     = device->newBuffer(qk_elems  * sizeof(__fp16), mode);
    bufs.K     = device->newBuffer(qk_elems  * sizeof(__fp16), mode);
    bufs.V     = device->newBuffer(v_elems   * sizeof(__fp16), mode);
    bufs.A     = device->newBuffer(a_elems   * sizeof(float),  mode);
    bufs.B     = device->newBuffer(a_elems   * sizeof(float),  mode);
    bufs.angle = device->newBuffer(ang_elems * sizeof(__fp16), mode);
    bufs.O     = device->newBuffer(v_elems   * sizeof(__fp16), mode);

    Mamba3FwdArgs fwd_args;
    fwd_args.batch    = cfg.batch;
    fwd_args.nheads   = cfg.n_heads;
    fwd_args.seq_len  = cfg.seq_len;
    fwd_args.n_chunks = cfg.seq_len / cfg.chunk_size;
    bufs.args = device->newBuffer(&fwd_args, sizeof(fwd_args), mode);

    return bufs;
}

inline Mamba3Buffers allocate_mimo_buffers(MTL::Device* device, const Mamba3Config& cfg) {
    size_t seq_total = (size_t)cfg.batch * cfg.n_heads * cfg.seq_len;
    size_t qk_elems  = seq_total * cfg.head_dim_qk * cfg.mimo_rank;
    size_t v_elems   = seq_total * cfg.head_dim_v;
    size_t a_elems   = seq_total;
    size_t ang_elems = seq_total * (cfg.head_dim_qk / 2);

    auto mode = MTL::ResourceStorageModeShared;

    Mamba3Buffers bufs;
    bufs.Q     = device->newBuffer(qk_elems  * sizeof(__fp16), mode);
    bufs.K     = device->newBuffer(qk_elems  * sizeof(__fp16), mode);
    bufs.V     = device->newBuffer(v_elems   * sizeof(__fp16), mode);
    bufs.A     = device->newBuffer(a_elems   * sizeof(float),  mode);
    bufs.B     = device->newBuffer(a_elems   * sizeof(float),  mode);
    bufs.angle = device->newBuffer(ang_elems * sizeof(__fp16), mode);
    bufs.O     = device->newBuffer(v_elems   * sizeof(__fp16), mode);

    Mamba3FwdArgs fwd_args;
    fwd_args.batch    = cfg.batch;
    fwd_args.nheads   = cfg.n_heads;
    fwd_args.seq_len  = cfg.seq_len;
    fwd_args.n_chunks = cfg.seq_len / cfg.chunk_size;
    bufs.args = device->newBuffer(&fwd_args, sizeof(fwd_args), mode);

    return bufs;
}

// Pipeline state creation helpers
inline MTL::ComputePipelineState* make_siso_pso(
    MTL::Device* device, MTL::Library* lib,
    const char* name = "mamba3_siso_fwd_64_64_64")
{
    NS::Error* err = nullptr;
    auto* fn = lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn) return nullptr;
    auto* pso = device->newComputePipelineState(fn, &err);
    fn->release();
    return pso;
}

inline MTL::ComputePipelineState* make_mimo_pso(
    MTL::Device* device, MTL::Library* lib,
    const char* name = "mamba3_mimo_fwd_32_64_64_r2")
{
    return make_siso_pso(device, lib, name);
}

} // namespace meow::mamba::mamba3
