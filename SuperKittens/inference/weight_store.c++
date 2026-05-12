#include "weight_store.h"
#include "../kernels/runtime_bindings.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

namespace sk {

WeightStore::WeightStore() = default;

WeightStore::~WeightStore() {
    for (auto& kv : tensors_) {
        if (kv.second.buffer) kv.second.buffer->release();
    }
    for (auto& m : maps_) {
        if (m.base) munmap(m.base, m.size);
        if (m.fd >= 0) close(m.fd);
    }
}

int WeightStore::map_file(const char* path, const void** out_base, size_t* out_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st{};
    if (fstat(fd, &st) < 0) { close(fd); return -2; }
    void* m = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { close(fd); return -3; }
    maps_.push_back({fd, m, (size_t)st.st_size});
    bytes_mapped_ += (size_t)st.st_size;
    if (out_base) *out_base = m;
    if (out_size) *out_size = (size_t)st.st_size;
    return 0;
}

MTL::Buffer* WeightStore::add(const std::string& name,
                              const void* src, size_t nbytes,
                              Dtype dtype, std::vector<int64_t> shape,
                              bool zero_copy)
{
    auto* dev = sk::bindings_device();
    if (!dev) return nullptr;

    MTL::Buffer* buf = nullptr;
    if (zero_copy) {
        buf = dev->newBuffer(const_cast<void*>(src), nbytes,
                             MTL::ResourceStorageModeShared, nullptr);
    } else {
        buf = dev->newBuffer(nbytes, MTL::ResourceStorageModeShared);
        if (buf) std::memcpy(buf->contents(), src, nbytes);
        bytes_resident_ += nbytes;
    }
    if (!buf) return nullptr;

    TensorView v{ src, nbytes, dtype, std::move(shape), buf };
    tensors_[name] = std::move(v);
    return buf;
}

const TensorView* WeightStore::get(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

MTL::Buffer* WeightStore::buffer(const std::string& name) const {
    auto* v = get(name);
    return v ? v->buffer : nullptr;
}

}  // namespace sk
