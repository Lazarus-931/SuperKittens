#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sk::gguf {

namespace {

constexpr uint32_t GGUF_MAGIC = 0x46554747u;

struct TypeInfo {
    const char* name;
    uint32_t    block_elems;
    uint32_t    block_bytes;
};

static const TypeInfo kTypes[] = {
    {"f32",      1,   4}, {"f16",      1,   2}, {"q4_0",    32,  18}, {"q4_1",    32,  20},
    {nullptr, 0, 0},      {nullptr, 0, 0},      {"q5_0",    32,  22}, {"q5_1",    32,  24},
    {"q8_0",    32,  34}, {"q8_1",    32,  40}, {"q2_k",   256,  84}, {"q3_k",   256, 110},
    {"q4_k",   256, 144}, {"q5_k",   256, 176}, {"q6_k",   256, 210}, {"q8_k",   256, 292},
    {"iq2_xxs",256,  66}, {"iq2_xs", 256,  74}, {"iq3_xxs",256,  98}, {"iq1_s",  256, 110},
    {"iq4_nl", 256,  50}, {"iq3_s",  256, 110}, {"iq2_s",  256,  82}, {"iq4_xs", 256, 136},
    {"i8",       1,   1}, {"i16",      1,   2}, {"i32",      1,   4}, {"i64",      1,   8},
    {"f64",      1,   8}, {"iq1_m",  256,  56}, {"bf16",     1,   2},
};
constexpr uint32_t kNumTypes = sizeof(kTypes) / sizeof(kTypes[0]);

const TypeInfo* type_info(uint32_t t) {
    if (t >= kNumTypes || kTypes[t].name == nullptr) return nullptr;
    return &kTypes[t];
}

bool ggml_to_dtype(uint32_t t, Dtype* out) {
    switch (t) {
        case 0:  *out = Dtype::F32;     return true;
        case 1:  *out = Dtype::F16;     return true;
        case 30: *out = Dtype::BF16;    return true;
        case 10: *out = Dtype::Q2_K;    return true;
        case 12: *out = Dtype::Q4_K;    return true;
        case 14: *out = Dtype::Q6_K;    return true;
        case 16: *out = Dtype::IQ2_XXS; return true;
        case 8:  *out = Dtype::Q8_0;    return true;
        case 6:  *out = Dtype::Q5_0;    return true;
        default: return false;
    }
}

struct Cursor {
    const uint8_t* base;
    uint64_t       size;
    uint64_t       pos;
    bool           ok;
};

bool c_read(Cursor& c, void* dst, uint64_t n) {
    if (n > c.size || c.pos > c.size - n) { c.ok = false; return false; }
    std::memcpy(dst, c.base + c.pos, (size_t)n);
    c.pos += n;
    return true;
}
bool c_skip(Cursor& c, uint64_t n) {
    if (n > c.size || c.pos > c.size - n) { c.ok = false; return false; }
    c.pos += n; return true;
}
bool c_u32(Cursor& c, uint32_t* v) { return c_read(c, v, 4); }
bool c_u64(Cursor& c, uint64_t* v) { return c_read(c, v, 8); }
bool c_string(Cursor& c, std::string* s) {
    uint64_t len;
    if (!c_u64(c, &len)) return false;
    if (len > c.size || c.pos > c.size - len) { c.ok = false; return false; }
    s->assign(reinterpret_cast<const char*>(c.base + c.pos), (size_t)len);
    c.pos += len;
    return true;
}

uint64_t scalar_size(uint32_t t) {
    switch (t) {
        case V_U8: case V_I8: case V_BOOL: return 1;
        case V_U16: case V_I16: return 2;
        case V_U32: case V_I32: case V_F32: return 4;
        case V_U64: case V_I64: case V_F64: return 8;
        default: return 0;
    }
}

bool skip_value(Cursor& c, uint32_t type, int depth) {
    if (depth > 8) { c.ok = false; return false; }
    uint64_t s = scalar_size(type);
    if (s) return c_skip(c, s);
    if (type == V_STRING) {
        uint64_t len;
        if (!c_u64(c, &len)) return false;
        return c_skip(c, len);
    }
    if (type == V_ARRAY) {
        uint32_t item_type; uint64_t len;
        if (!c_u32(c, &item_type)) return false;
        if (!c_u64(c, &len)) return false;
        uint64_t is = scalar_size(item_type);
        if (is) return c_skip(c, len * is);
        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, item_type, depth + 1)) return false;
        }
        return true;
    }
    c.ok = false;
    return false;
}

bool capture_value(Cursor& c, MetaValue& v) {
    v.raw_pos = c.pos;
    uint64_t s = scalar_size(v.type);
    if (s) {
        uint64_t buf = 0;
        if (!c_read(c, &buf, s)) return false;
        switch (v.type) {
            case V_U8:  v.scalar.u64 = (uint8_t)buf; break;
            case V_I8:  v.scalar.i64 = (int8_t)buf; break;
            case V_BOOL:v.scalar.u64 = (uint8_t)buf ? 1 : 0; break;
            case V_U16: v.scalar.u64 = (uint16_t)buf; break;
            case V_I16: v.scalar.i64 = (int16_t)buf; break;
            case V_U32: v.scalar.u64 = (uint32_t)buf; break;
            case V_I32: v.scalar.i64 = (int32_t)buf; break;
            case V_F32: { float f; std::memcpy(&f, &buf, 4); v.scalar.f64 = f; break; }
            case V_U64: v.scalar.u64 = buf; break;
            case V_I64: std::memcpy(&v.scalar.i64, &buf, 8); break;
            case V_F64: std::memcpy(&v.scalar.f64, &buf, 8); break;
        }
        return true;
    }
    if (v.type == V_STRING) return c_string(c, &v.str);
    if (v.type == V_ARRAY) {
        uint32_t item_type; uint64_t len;
        if (!c_u32(c, &item_type)) return false;
        if (!c_u64(c, &len)) return false;
        v.arr_type = item_type;
        v.arr_len  = len;
        v.raw_pos  = c.pos;
        uint64_t is = scalar_size(item_type);
        if (is) return c_skip(c, len * is);
        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, item_type, 0)) return false;
        }
        return true;
    }
    c.ok = false;
    return false;
}

uint64_t align_up(uint64_t v, uint64_t a) {
    uint64_t r = v % a; return r ? v + a - r : v;
}

struct Mapping {
    int    fd  = -1;
    void*  ptr = nullptr;
    size_t len = 0;
};

static std::vector<Mapping>& mapping_registry() {
    static std::vector<Mapping> r;
    return r;
}

static int parse_from_mem(const uint8_t* base, size_t size, Model& m) {
    m.map_base = base;
    m.map_size = size;
    Cursor c{ base, size, 0, true };
    uint32_t magic;
    if (!c_u32(c, &magic) || magic != GGUF_MAGIC) return -2;
    if (!c_u32(c, &m.version) || m.version != 3) return -3;
    if (!c_u64(c, &m.n_tensors) || !c_u64(c, &m.n_kv)) return -4;

    m.alignment = 32;
    m.meta.reserve((size_t)m.n_kv);
    for (uint64_t i = 0; i < m.n_kv; i++) {
        std::string key;
        if (!c_string(c, &key)) return -5;
        MetaValue v;
        if (!c_u32(c, &v.type)) return -6;
        if (!capture_value(c, v)) return -7;
        if (key == "general.alignment" && v.type == V_U32 && v.scalar.u64 != 0) {
            m.alignment = v.scalar.u64;
        }
        m.meta.emplace(std::move(key), std::move(v));
    }

    m.tensors.resize((size_t)m.n_tensors);
    for (uint64_t i = 0; i < m.n_tensors; i++) {
        TensorInfo& t = m.tensors[i];
        if (!c_string(c, &t.name)) return -8;
        uint32_t ndim;
        if (!c_u32(c, &ndim) || ndim == 0 || ndim > 4) return -9;
        t.shape.resize(ndim);
        t.elements = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            uint64_t dim;
            if (!c_u64(c, &dim)) return -10;
            t.shape[d] = (int64_t)dim;
            t.elements *= dim;
        }
        if (!c_u32(c, &t.ggml_type)) return -11;
        uint64_t rel;
        if (!c_u64(c, &rel)) return -12;

        const TypeInfo* ti = type_info(t.ggml_type);
        if (ti && ti->block_elems) {
            uint64_t blocks = (t.elements + ti->block_elems - 1) / ti->block_elems;
            t.nbytes = blocks * ti->block_bytes;
        }
        t.abs_offset = rel;
        t.supported = ggml_to_dtype(t.ggml_type, &t.dtype);
    }

    uint64_t data_pos = align_up(c.pos, m.alignment);
    m.tensor_data_pos = data_pos;
    for (auto& t : m.tensors) {
        t.abs_offset += data_pos;
        if (t.nbytes && (t.abs_offset > size || t.nbytes > size - t.abs_offset)) return -13;
    }
    return 0;
}

bool mmap_file(const char* path, Mapping& out) {
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st{};
    if (::fstat(fd, &st) < 0) { ::close(fd); return false; }
    void* p = ::mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { ::close(fd); return false; }
    out.fd = fd; out.ptr = p; out.len = (size_t)st.st_size;
    return true;
}

}  // namespace

int parse_gguf(const char* path, Model& m) {
    Mapping mp;
    if (!mmap_file(path, mp)) return -1;
    int rc = parse_from_mem((const uint8_t*)mp.ptr, mp.len, m);
    if (rc != 0) { ::munmap(mp.ptr, mp.len); ::close(mp.fd); return rc; }
    mapping_registry().push_back(mp);
    return 0;
}

int load_gguf(const char* path, WeightStore& store, Model* out_model) {
    Model local;
    Model& m = out_model ? *out_model : local;

    const void* base = nullptr;
    size_t size = 0;
    int rc = store.map_file(path, &base, &size);
    if (rc) return rc;
    rc = parse_from_mem((const uint8_t*)base, size, m);
    if (rc != 0) return rc;

    for (const auto& t : m.tensors) {
        if (!t.supported || t.nbytes == 0) continue;
        const void* src = m.map_base + t.abs_offset;
        store.add(t.name, src, (size_t)t.nbytes, t.dtype, t.shape, true);
    }
    return 0;
}

const MetaValue* meta_find(const Model& m, const std::string& key) {
    auto it = m.meta.find(key);
    return it == m.meta.end() ? nullptr : &it->second;
}

bool meta_u32(const Model& m, const std::string& key, uint32_t* out) {
    auto* v = meta_find(m, key);
    if (!v || v->type != V_U32) return false;
    *out = (uint32_t)v->scalar.u64;
    return true;
}
bool meta_u64(const Model& m, const std::string& key, uint64_t* out) {
    auto* v = meta_find(m, key);
    if (!v) return false;
    if (v->type == V_U64) { *out = v->scalar.u64; return true; }
    if (v->type == V_U32) { *out = v->scalar.u64; return true; }
    return false;
}
bool meta_f32(const Model& m, const std::string& key, float* out) {
    auto* v = meta_find(m, key);
    if (!v || v->type != V_F32) return false;
    *out = (float)v->scalar.f64;
    return true;
}
bool meta_string(const Model& m, const std::string& key, std::string* out) {
    auto* v = meta_find(m, key);
    if (!v || v->type != V_STRING) return false;
    *out = v->str;
    return true;
}

}  // namespace sk::gguf
