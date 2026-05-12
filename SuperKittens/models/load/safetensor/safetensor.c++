#include "safetensor.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

namespace sk {
namespace {

struct MappedFile {
    void*  base = nullptr;
    size_t size = 0;
    int    fd   = -1;
};

static int map_ro(const char* path, MappedFile& out) {
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st{};
    if (::fstat(fd, &st) < 0) { ::close(fd); return -2; }
    void* m = ::mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { ::close(fd); return -3; }
    out.base = m;
    out.size = (size_t)st.st_size;
    out.fd   = fd;
    return 0;
}

static bool parse_dtype(const std::string& s, Dtype& out) {
    if (s == "F32")  { out = Dtype::F32;  return true; }
    if (s == "F16")  { out = Dtype::F16;  return true; }
    if (s == "BF16") { out = Dtype::BF16; return true; }
    return false;
}

static size_t dtype_size(Dtype d) {
    switch (d) {
        case Dtype::F32:  return 4;
        case Dtype::F16:  return 2;
        case Dtype::BF16: return 2;
        default:          return 0;
    }
}

// Minimal JSON scanner tailored to safetensors metadata.
struct J {
    const char* p;
    const char* e;
    void skip_ws() {
        while (p < e) {
            char c = *p;
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++p;
            else break;
        }
    }
    bool eat(char c) { skip_ws(); if (p < e && *p == c) { ++p; return true; } return false; }
    bool peek(char c) { skip_ws(); return p < e && *p == c; }
    bool parse_string(std::string& out) {
        skip_ws();
        if (p >= e || *p != '"') return false;
        ++p;
        out.clear();
        while (p < e && *p != '"') {
            if (*p == '\\' && p + 1 < e) {
                char n = p[1];
                if (n == 'n') out.push_back('\n');
                else if (n == 't') out.push_back('\t');
                else if (n == 'r') out.push_back('\r');
                else out.push_back(n);
                p += 2;
            } else {
                out.push_back(*p++);
            }
        }
        if (p >= e) return false;
        ++p;
        return true;
    }
    bool parse_int(int64_t& out) {
        skip_ws();
        const char* s = p;
        if (p < e && (*p == '-' || *p == '+')) ++p;
        while (p < e && *p >= '0' && *p <= '9') ++p;
        if (p == s) return false;
        out = std::strtoll(std::string(s, p - s).c_str(), nullptr, 10);
        return true;
    }
    bool parse_int_array(std::vector<int64_t>& out) {
        if (!eat('[')) return false;
        skip_ws();
        if (eat(']')) return true;
        while (p < e) {
            int64_t v;
            if (!parse_int(v)) return false;
            out.push_back(v);
            skip_ws();
            if (eat(',')) continue;
            if (eat(']')) return true;
            return false;
        }
        return false;
    }
    // Skip an arbitrary JSON value (string/number/object/array/bool/null).
    bool skip_value() {
        skip_ws();
        if (p >= e) return false;
        char c = *p;
        if (c == '"') { std::string tmp; return parse_string(tmp); }
        if (c == '{' || c == '[') {
            char open = c, close = (c == '{') ? '}' : ']';
            int depth = 0;
            while (p < e) {
                char q = *p;
                if (q == '"') { std::string tmp; if (!parse_string(tmp)) return false; continue; }
                if (q == open) ++depth;
                else if (q == close) { --depth; ++p; if (depth == 0) return true; continue; }
                ++p;
            }
            return false;
        }
        // number / bool / null — read until delimiter
        while (p < e) {
            char q = *p;
            if (q == ',' || q == '}' || q == ']' || q == ' ' || q == '\t' || q == '\n' || q == '\r') break;
            ++p;
        }
        return true;
    }
};

static int parse_header_and_add(const uint8_t* base, size_t size, WeightStore* store,
                                std::vector<SafetensorEntry>* enum_out)
{
    if (size < 8) return -10;
    uint64_t hlen = 0;
    std::memcpy(&hlen, base, 8);
    if (hlen + 8 > size) return -11;
    const uint8_t* data_start = base + 8 + hlen;

    J j{ (const char*)(base + 8), (const char*)(base + 8 + hlen) };
    if (!j.eat('{')) return -12;
    while (true) {
        j.skip_ws();
        if (j.eat('}')) break;
        std::string key;
        if (!j.parse_string(key)) return -13;
        if (!j.eat(':')) return -14;
        if (key == "__metadata__") {
            if (!j.skip_value()) return -15;
        } else {
            if (!j.eat('{')) return -16;
            std::string dtype_s;
            std::vector<int64_t> shape;
            int64_t off_start = -1, off_end = -1;
            while (true) {
                j.skip_ws();
                if (j.eat('}')) break;
                std::string k2;
                if (!j.parse_string(k2)) return -17;
                if (!j.eat(':')) return -18;
                if (k2 == "dtype") {
                    if (!j.parse_string(dtype_s)) return -19;
                } else if (k2 == "shape") {
                    if (!j.parse_int_array(shape)) return -20;
                } else if (k2 == "data_offsets") {
                    std::vector<int64_t> offs;
                    if (!j.parse_int_array(offs)) return -21;
                    if (offs.size() != 2) return -22;
                    off_start = offs[0]; off_end = offs[1];
                } else {
                    if (!j.skip_value()) return -23;
                }
                j.skip_ws();
                if (j.eat(',')) continue;
            }
            Dtype dt;
            if (!parse_dtype(dtype_s, dt)) {
                // skip unknown dtype (e.g. I64 token ids — but safetensors weight files won't have these typically)
            } else if (off_start >= 0 && off_end >= off_start) {
                size_t nbytes = (size_t)(off_end - off_start);
                const void* ptr = data_start + off_start;
                if ((const uint8_t*)ptr + nbytes > base + size) return -24;
                if (enum_out) {
                    enum_out->push_back(SafetensorEntry{ key, dt, shape, nbytes });
                } else if (store) {
                    store->add(key, ptr, nbytes, dt, shape, true);
                }
            }
        }
        j.skip_ws();
        if (j.eat(',')) continue;
    }
    return 0;
}

}  // namespace

int load_safetensors(const char* path, WeightStore& store) {
    const void* base = nullptr;
    size_t size = 0;
    int rc = store.map_file(path, &base, &size);
    if (rc) return rc;
    return parse_header_and_add((const uint8_t*)base, size, &store, nullptr);
}

int enumerate_safetensors(const char* path, std::vector<SafetensorEntry>& out) {
    MappedFile m;
    int rc = map_ro(path, m);
    if (rc) return rc;
    rc = parse_header_and_add((const uint8_t*)m.base, m.size, nullptr, &out);
    ::munmap(m.base, m.size);
    ::close(m.fd);
    return rc;
}

int load_safetensors_index(const char* index_json_path, WeightStore& store) {
    MappedFile im;
    int rc = map_ro(index_json_path, im);
    if (rc) return rc;

    std::string base_dir(index_json_path);
    {
        auto pos = base_dir.find_last_of('/');
        if (pos == std::string::npos) base_dir = "";
        else                          base_dir = base_dir.substr(0, pos + 1);
    }

    J j{ (const char*)im.base, (const char*)im.base + im.size };
    if (!j.eat('{')) { ::munmap(im.base, im.size); ::close(im.fd); return -30; }

    std::vector<std::string> shards;
    auto add_shard = [&](const std::string& s) {
        for (auto& x : shards) if (x == s) return;
        shards.push_back(s);
    };

    while (true) {
        j.skip_ws();
        if (j.eat('}')) break;
        std::string key;
        if (!j.parse_string(key)) { ::munmap(im.base, im.size); ::close(im.fd); return -31; }
        if (!j.eat(':')) { ::munmap(im.base, im.size); ::close(im.fd); return -32; }
        if (key == "weight_map") {
            if (!j.eat('{')) { ::munmap(im.base, im.size); ::close(im.fd); return -33; }
            while (true) {
                j.skip_ws();
                if (j.eat('}')) break;
                std::string tname, shard;
                if (!j.parse_string(tname)) { ::munmap(im.base, im.size); ::close(im.fd); return -34; }
                if (!j.eat(':')) { ::munmap(im.base, im.size); ::close(im.fd); return -35; }
                if (!j.parse_string(shard)) { ::munmap(im.base, im.size); ::close(im.fd); return -36; }
                add_shard(shard);
                j.skip_ws();
                if (j.eat(',')) continue;
            }
        } else {
            if (!j.skip_value()) { ::munmap(im.base, im.size); ::close(im.fd); return -37; }
        }
        j.skip_ws();
        if (j.eat(',')) continue;
    }
    ::munmap(im.base, im.size);
    ::close(im.fd);

    for (auto& s : shards) {
        std::string full = base_dir + s;
        rc = load_safetensors(full.c_str(), store);
        if (rc) return rc;
    }
    return 0;
}

}  // namespace sk
