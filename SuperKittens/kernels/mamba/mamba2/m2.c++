//
//  mamba2.c++ — v1
//  SuperKittens — host dispatch for Mamba-2 SSD forward.
//

#include "../../../meow.h"
#include "../mamba_impl.h"

namespace meow::mamba::mamba2 {

struct Mamba2Config {
    int batch;
    int n_heads;
    int seq_len;
    int head_dim_qk;
    int head_dim_v;
    int chunk_size;
};

struct Mamba2FwdArgsHost {
    uint32_t batch;
    uint32_t nheads;
    uint32_t seq_len;
    uint32_t n_chunks;
};

inline MTL::ComputePipelineState* make_pso(
    MTL::Device* device, MTL::Library* lib,
    const char* name = "mamba2_siso_fwd_32_64_64")
{
    NS::Error* err = nullptr;
    auto* fn = lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn) return nullptr;
    auto* pso = device->newComputePipelineState(fn, &err);
    fn->release();
    return pso;
}

struct Mamba2Buffers {
    MTL::Buffer* Q;
    MTL::Buffer* K;
    MTL::Buffer* V;
    MTL::Buffer* A;
    MTL::Buffer* O;
    MTL::Buffer* params;

    void release() {
        Q->release(); K->release(); V->release();
        A->release(); O->release(); params->release();
    }
};

inline Mamba2Buffers allocate_buffers(MTL::Device* device, const Mamba2Config& cfg) {
    const size_t per_head_seq = (size_t)cfg.batch * cfg.n_heads * cfg.seq_len;
    const size_t qk_elems = per_head_seq * cfg.head_dim_qk;
    const size_t vo_elems = per_head_seq * cfg.head_dim_v;
    const size_t a_elems  = per_head_seq;
    auto mode = MTL::ResourceStorageModeShared;

    Mamba2Buffers bufs;
    bufs.Q = device->newBuffer(qk_elems * sizeof(__fp16), mode);
    bufs.K = device->newBuffer(qk_elems * sizeof(__fp16), mode);
    bufs.V = device->newBuffer(vo_elems * sizeof(__fp16), mode);
    bufs.A = device->newBuffer(a_elems  * sizeof(float),  mode);
    bufs.O = device->newBuffer(vo_elems * sizeof(__fp16), mode);

    Mamba2FwdArgsHost args{
        (uint32_t)cfg.batch,
        (uint32_t)cfg.n_heads,
        (uint32_t)cfg.seq_len,
        (uint32_t)(cfg.seq_len / cfg.chunk_size)
    };
    bufs.params = device->newBuffer(&args, sizeof(args), mode);
    return bufs;
}

inline double dispatch_fwd(
    MTL::CommandQueue* queue,
    MTL::ComputePipelineState* pso,
    const Mamba2Buffers& bufs,
    const Mamba2Config& cfg)
{
    auto* cmd = queue->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);

    enc->setBuffer(bufs.Q,      0, 0);
    enc->setBuffer(bufs.K,      0, 1);
    enc->setBuffer(bufs.V,      0, 2);
    enc->setBuffer(bufs.A,      0, 3);
    enc->setBuffer(bufs.O,      0, 4);
    enc->setBuffer(bufs.params, 0, 5);

    const uint32_t threads = (uint32_t)((cfg.chunk_size / 8) * 32);
    enc->dispatchThreadgroups(
        MTL::Size(cfg.batch, cfg.n_heads, 1),
        MTL::Size(threads, 1, 1));

    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    return cmd->GPUEndTime() - cmd->GPUStartTime();
}

}
