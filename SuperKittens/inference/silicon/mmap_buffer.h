// MmapBuffer: zero-copy view of a file as an MTL::Buffer.
//
// On Apple Silicon the GPU and CPU share physical memory; an mmap'd region can
// be handed to `MTL::Device::newBufferWithBytesNoCopy(...)` so the GPU reads
// pages directly without any host-side memcpy. The intended use is GGUF weights
// — the file's bytes ARE the GPU's input bytes (Q8_0 blocks, fp16, etc.).
//
// Lifetime: the MmapBuffer owns the mmap and the MTL::Buffer; destruction
// releases the buffer (which triggers our deallocator → munmap + close).
#ifndef SK_INFERENCE_SILICON_MMAP_BUFFER_H
#define SK_INFERENCE_SILICON_MMAP_BUFFER_H

#include "Metal/Metal.hpp"

#include <cstddef>
#include <string>

namespace sk { namespace silicon {

class MmapBuffer {
public:
    // mmap `path` read-only and wrap it as a shared-storage MTL::Buffer.
    // Returns nullptr on failure (file missing, mmap fails, MTL alloc fails).
    static MmapBuffer* from_file(MTL::Device* dev, const std::string& path);

    // mmap only [file_offset, file_offset + length) of `path` (rounded out to
    // page boundaries). The resulting `data()` points at the page-aligned
    // base; the byte offset of the requested region within `data()` is
    // returned via `out_inner_off` (call setBuffer:offset:inner_off+sub).
    // Intended for the GGUF case where we only need a single tensor's bytes
    // mapped, not the whole multi-GB file.
    static MmapBuffer* from_file_range(MTL::Device* dev, const std::string& path,
                                       std::size_t file_offset, std::size_t length,
                                       std::size_t* out_inner_off);

    // Same as from_file_range, but reuses a caller-owned fd. The mapping holds
    // no fd of its own (Darwin/POSIX: mmap retains its own reference on the
    // backing vnode, so the caller may close `fd` once all desired mappings
    // are created). Lets loaders open the GGUF once and mmap hundreds of
    // tensor ranges off that single fd instead of `open()`ing per tensor —
    // critical when n_tensors > `ulimit -n`.
    static MmapBuffer* from_fd_range(MTL::Device* dev, int fd,
                                     std::size_t file_offset, std::size_t length,
                                     std::size_t* out_inner_off);

    ~MmapBuffer();

    MTL::Buffer* buffer() const { return buf_; }    // borrowed; lifetime tied to *this
    const void*  data()   const { return base_; }
    std::size_t  size()   const { return size_; }

    MmapBuffer(const MmapBuffer&) = delete;
    MmapBuffer& operator=(const MmapBuffer&) = delete;

private:
    MmapBuffer(MTL::Buffer* buf, void* base, std::size_t size, int fd)
        : buf_(buf), base_(base), size_(size), fd_(fd) {}

    MTL::Buffer* buf_;
    void*        base_;
    std::size_t  size_;
    int          fd_;
};

}}  // namespace sk::silicon

#endif  // SK_INFERENCE_SILICON_MMAP_BUFFER_H
