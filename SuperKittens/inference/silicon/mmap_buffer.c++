#include "mmap_buffer.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sk { namespace silicon {

MmapBuffer* MmapBuffer::from_file(MTL::Device* dev, const std::string& path) {
    if (!dev) return nullptr;

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "MmapBuffer: open('%s') failed: %s\n",
                     path.c_str(), std::strerror(errno));
        return nullptr;
    }

    struct stat st{};
    if (::fstat(fd, &st) < 0 || st.st_size <= 0) {
        std::fprintf(stderr, "MmapBuffer: fstat failed for '%s'\n", path.c_str());
        ::close(fd);
        return nullptr;
    }
    const std::size_t size = (std::size_t)st.st_size;

    // MAP_PRIVATE so any (illegal under PROT_READ, but a defensive choice) write
    // would be COW. MAP_NOCACHE is the Apple Silicon idiom for "stream this
    // through, don't pollute the unified buffer cache" — pages still come from
    // disk lazily on first touch, and the GPU can map them too.
    void* base = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE | MAP_NOCACHE, fd, 0);
    if (base == MAP_FAILED) {
        std::fprintf(stderr, "MmapBuffer: mmap failed for '%s': %s\n",
                     path.c_str(), std::strerror(errno));
        ::close(fd);
        return nullptr;
    }

    // Hint the kernel: we'll read this sequentially-ish, large region.
    ::madvise(base, size, MADV_WILLNEED);

    // Zero-copy hand-off to Metal. nullptr deallocator → buffer never frees the
    // mapping; that's fine for v0 because the MmapBuffer (and the model) lives
    // for process lifetime. When we want a clean shutdown we'll thread through
    // a block-based deallocator that calls munmap + close.
    MTL::Buffer* buf = dev->newBuffer(base, (NS::UInteger)size,
                                      MTL::ResourceStorageModeShared,
                                      nullptr);
    if (!buf) {
        std::fprintf(stderr, "MmapBuffer: newBufferWithBytesNoCopy failed (size=%zu)\n", size);
        ::munmap(base, size);
        ::close(fd);
        return nullptr;
    }

    return new MmapBuffer(buf, base, size, fd);
}

MmapBuffer::~MmapBuffer() {
    if (buf_) buf_->release();
    // NOTE: we deliberately do NOT munmap/close here yet — Metal may still
    // hold references to the pages via outstanding command buffers, and we
    // pass `nullptr` deallocator to newBufferWithBytesNoCopy so there's no
    // safe synchronization point. Process-lifetime ownership is the v0
    // contract; revisit when MmapBuffers are created on the hot path.
    (void)base_;
    (void)size_;
    (void)fd_;
}

}}  // namespace sk::silicon
