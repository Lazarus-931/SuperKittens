// quantize-to-gguf.c
// HF safetensors -> GGUF (Q8_0 or F16). Architecture is detected from
// config.json:architectures[0]. Supported families:
//   * Qwen3   (Qwen3ForCausalLM)               -> gguf arch "qwen3"   [validated]
//   * Gemma   (Gemma3ForCausalLM / Gemma4...)  -> gguf arch "gemma3"  [validated on small fixtures]
//   * Mamba2  (Mamba2ForCausalLM)              -> gguf arch "mamba2"  [validated on small fixtures]
//   * DeepSeek (DeepseekV2/V3ForCausalLM)      -> gguf arch "deepseek2" [code-only, no local weights]
//
// Build:
//   clang -O3 -ffast-math -framework Accelerate -pthread \
//         -o quantize-to-gguf quantize-to-gguf.c
//
// Usage:
//   quantize-to-gguf <safetensors_dir> <out.gguf> --quant q8_0   (default)
//   quantize-to-gguf <safetensors_dir> <out.gguf> --quant f16

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <math.h>
#include <pthread.h>
#include <ctype.h>
#include <Accelerate/Accelerate.h>

// ---------------- GGUF enums (subset) ----------------
enum {
    GGUF_TYPE_UINT8=0, GGUF_TYPE_INT8=1, GGUF_TYPE_UINT16=2, GGUF_TYPE_INT16=3,
    GGUF_TYPE_UINT32=4, GGUF_TYPE_INT32=5, GGUF_TYPE_FLOAT32=6, GGUF_TYPE_BOOL=7,
    GGUF_TYPE_STRING=8, GGUF_TYPE_ARRAY=9, GGUF_TYPE_UINT64=10, GGUF_TYPE_INT64=11,
    GGUF_TYPE_FLOAT64=12
};
enum {
    GGML_TYPE_F32=0, GGML_TYPE_F16=1, GGML_TYPE_Q8_0=8
};

#define GGUF_DEFAULT_ALIGNMENT 32

// ---------------- helpers ----------------
static void die(const char* m){ fprintf(stderr,"fatal: %s\n", m); exit(1); }
static void diep(const char* m){ perror(m); exit(1); }

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

static void* xmalloc(size_t n){ void*p=malloc(n); if(!p) die("oom"); return p; }
static char* xstrdup(const char* s){ char* r=strdup(s); if(!r) die("oom"); return r; }

// Map a file read-only.
typedef struct { const uint8_t* data; size_t size; int fd; } mmap_file;
static mmap_file mmap_ro(const char* path){
    int fd = open(path, O_RDONLY);
    if(fd<0) diep(path);
    struct stat st; if(fstat(fd,&st)) diep("fstat");
    void* p = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if(p==MAP_FAILED) diep("mmap");
    // advise random read; safetensors are read sequentially per tensor but tensors may be picked
    madvise(p, st.st_size, MADV_WILLNEED);
    return (mmap_file){p,(size_t)st.st_size,fd};
}
static void munmap_file(mmap_file* m){ munmap((void*)m->data,m->size); close(m->fd); }

// ---------------- BF16 -> F32 ----------------
static inline float bf16_to_f32(uint16_t b){
    uint32_t x = ((uint32_t)b)<<16; float f; memcpy(&f,&x,4); return f;
}
#include <arm_neon.h>
static void bf16_row_to_f32(const uint16_t* src, float* dst, size_t n){
    size_t i=0;
    // NEON: load 8 bf16, expand to 8 f32 = two 4-lane float vectors.
    for(; i+8<=n; i+=8){
        uint16x8_t v = vld1q_u16(src+i);
        // Widen low/high halves to u32, shift left 16, reinterpret as float.
        uint32x4_t lo = vshll_n_u16(vget_low_u16(v),  16);
        uint32x4_t hi = vshll_n_u16(vget_high_u16(v), 16);
        vst1q_f32(dst+i,   vreinterpretq_f32_u32(lo));
        vst1q_f32(dst+i+4, vreinterpretq_f32_u32(hi));
    }
    for(; i<n; i++) dst[i]=bf16_to_f32(src[i]);
}

// ---------------- F32 -> F16 (IEEE half, round-to-nearest-even) ----------------
static inline uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x,&f,4);
    uint32_t sign = (x>>16)&0x8000;
    uint32_t mant = x&0x7fffff;
    int32_t  exp  = (int32_t)((x>>23)&0xff) - 127 + 15;
    if (((x>>23)&0xff)==0xff) {
        // inf / nan
        if (mant) return (uint16_t)(sign | 0x7e00);
        return (uint16_t)(sign | 0x7c00);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t r = mant >> shift;
        uint32_t rem = mant & ((1u<<shift)-1);
        uint32_t halfway = 1u<<(shift-1);
        if (rem > halfway || (rem==halfway && (r&1))) r += 1;
        return (uint16_t)(sign | r);
    }
    uint32_t r = mant >> 13;
    uint32_t rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem==0x1000 && (r&1))) {
        r += 1;
        if (r==0x400){ r=0; exp+=1; if (exp>=31) return (uint16_t)(sign|0x7c00); }
    }
    return (uint16_t)(sign | ((uint32_t)exp<<10) | r);
}

// ---------------- Q8_0 quantize one block of 32 ----------------
// Match llama.cpp's quantize_row_q8_0_ref exactly so byte-diff hits zero.
#define QK8 32
static inline void q8_0_block(const float* x, uint8_t* out) {
    // amax via scalar (matches llama.cpp's quantize_row_q8_0_ref exactly).
    float amax = 0.f;
    for (int j=0;j<QK8;j++){ float a = fabsf(x[j]); if (a>amax) amax=a; }
    float d  = amax / 127.f;
    float id = (d==0.f) ? 0.f : 1.f/d;
    uint16_t dh = f32_to_f16(d);
    memcpy(out, &dh, 2);
    int8_t* qs = (int8_t*)(out+2);
    for (int j=0;j<QK8;j++) qs[j] = (int8_t)roundf(x[j]*id);
}

// Fast path: BF16 -> Q8_0 in one block of 32, using NEON.
static inline void q8_0_block_bf16(const uint16_t* bf, uint8_t* out) {
    // Load 32 bf16 -> four float32x4 vectors per half; total 8 f32 vectors.
    float32x4_t f0,f1,f2,f3,f4,f5,f6,f7;
    #define LOAD8(off) do { \
        uint16x8_t v = vld1q_u16(bf + (off)); \
        uint32x4_t a = vshll_n_u16(vget_low_u16(v), 16); \
        uint32x4_t b = vshll_n_u16(vget_high_u16(v), 16); \
        f##off##_lo = vreinterpretq_f32_u32(a); f##off##_hi = vreinterpretq_f32_u32(b); \
    } while(0)
    uint16x8_t v0 = vld1q_u16(bf);
    uint16x8_t v1 = vld1q_u16(bf+8);
    uint16x8_t v2 = vld1q_u16(bf+16);
    uint16x8_t v3 = vld1q_u16(bf+24);
    f0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v0), 16));
    f1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(v0), 16));
    f2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v1), 16));
    f3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(v1), 16));
    f4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v2), 16));
    f5 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(v2), 16));
    f6 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v3), 16));
    f7 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(v3), 16));

    // amax via NEON max-of-abs
    float32x4_t af0=vabsq_f32(f0), af1=vabsq_f32(f1), af2=vabsq_f32(f2), af3=vabsq_f32(f3);
    float32x4_t af4=vabsq_f32(f4), af5=vabsq_f32(f5), af6=vabsq_f32(f6), af7=vabsq_f32(f7);
    float32x4_t m01=vmaxq_f32(af0,af1), m23=vmaxq_f32(af2,af3);
    float32x4_t m45=vmaxq_f32(af4,af5), m67=vmaxq_f32(af6,af7);
    float32x4_t m0123=vmaxq_f32(m01,m23), m4567=vmaxq_f32(m45,m67);
    float32x4_t mall=vmaxq_f32(m0123,m4567);
    float amax = vmaxvq_f32(mall);

    float d  = amax / 127.f;
    float id = (d==0.f) ? 0.f : 1.f/d;
    uint16_t dh = f32_to_f16(d);
    memcpy(out, &dh, 2);

    // Multiply by id, round, narrow to int8.
    float32x4_t vid = vdupq_n_f32(id);
    int32x4_t i0=vcvtnq_s32_f32(vmulq_f32(f0,vid));
    int32x4_t i1=vcvtnq_s32_f32(vmulq_f32(f1,vid));
    int32x4_t i2=vcvtnq_s32_f32(vmulq_f32(f2,vid));
    int32x4_t i3=vcvtnq_s32_f32(vmulq_f32(f3,vid));
    int32x4_t i4=vcvtnq_s32_f32(vmulq_f32(f4,vid));
    int32x4_t i5=vcvtnq_s32_f32(vmulq_f32(f5,vid));
    int32x4_t i6=vcvtnq_s32_f32(vmulq_f32(f6,vid));
    int32x4_t i7=vcvtnq_s32_f32(vmulq_f32(f7,vid));
    int16x8_t h0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
    int16x8_t h1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
    int16x8_t h2 = vcombine_s16(vqmovn_s32(i4), vqmovn_s32(i5));
    int16x8_t h3 = vcombine_s16(vqmovn_s32(i6), vqmovn_s32(i7));
    int8x16_t b0 = vcombine_s8(vqmovn_s16(h0), vqmovn_s16(h1));
    int8x16_t b1 = vcombine_s8(vqmovn_s16(h2), vqmovn_s16(h3));
    vst1q_s8((int8_t*)(out+2),    b0);
    vst1q_s8((int8_t*)(out+2+16), b1);
    #undef LOAD8
}

// Quantize a flat F32 buffer of (nrows*ncols) elements row-major to Q8_0.
// row_size = 2 + 32 bytes per 32 elements -> ncols/32 * 34 per row
static size_t quantize_q8_0(const float* src, uint8_t* dst, size_t nrows, size_t ncols) {
    if (ncols % QK8) die("ncols not divisible by 32 for Q8_0");
    size_t nb_per_row = ncols/QK8;
    size_t row_bytes = nb_per_row * 34;
    for(size_t r=0;r<nrows;r++){
        const float* xrow = src + r*ncols;
        uint8_t* drow = dst + r*row_bytes;
        for(size_t b=0;b<nb_per_row;b++){
            q8_0_block(xrow + b*QK8, drow + b*34);
        }
    }
    return nrows * row_bytes;
}

// ---------------- minimal JSON reader for what we need ----------------
// We only parse two JSONs: safetensors header (objects of objects) and config.json.
// Approach: tiny scanner producing string -> string-of-value map by lookup helpers.
// Not a full parser; enough for these well-formed inputs.

static const char* skip_ws(const char* p){ while(*p && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r')) p++; return p; }

// find numeric value for "key" in a JSON object string `s` of length n.
// Returns 1 + sets *out, else 0. Searches whole buffer (sufficient when keys are unique).
static int json_find_int(const char* s, size_t n, const char* key, long long* out){
    char pat[128]; snprintf(pat,sizeof(pat),"\"%s\"",key);
    const char* p = memmem(s,n,pat,strlen(pat));
    if(!p) return 0;
    p += strlen(pat);
    while(p<s+n && *p!=':') p++;
    if(p>=s+n) return 0;
    p++;
    p = skip_ws(p);
    char* end; long long v = strtoll(p,&end,10);
    if(end==p) return 0;
    *out=v; return 1;
}
static int json_find_float(const char* s, size_t n, const char* key, double* out){
    char pat[128]; snprintf(pat,sizeof(pat),"\"%s\"",key);
    const char* p = memmem(s,n,pat,strlen(pat));
    if(!p) return 0;
    p += strlen(pat);
    while(p<s+n && *p!=':') p++;
    if(p>=s+n) return 0;
    p++; p = skip_ws(p);
    char* end; double v = strtod(p,&end);
    if(end==p) return 0;
    *out=v; return 1;
}
static int json_find_string(const char* s, size_t n, const char* key, char* buf, size_t bufsz){
    char pat[128]; snprintf(pat,sizeof(pat),"\"%s\"",key);
    const char* p = memmem(s,n,pat,strlen(pat));
    if(!p) return 0;
    p += strlen(pat);
    while(p<s+n && *p!=':') p++;
    if(p>=s+n) return 0;
    p++; p = skip_ws(p);
    if(*p!='"') return 0;
    p++;
    size_t i=0;
    while(p<s+n && *p!='"' && i+1<bufsz){
        if(*p=='\\' && p+1<s+n){ buf[i++]=*++p; p++; }
        else buf[i++]=*p++;
    }
    buf[i]=0; return 1;
}

// ---------------- Safetensors tensor table ----------------
typedef struct {
    char     name[256];
    char     dtype[8];     // "BF16", "F32", "F16"
    int      ndim;
    int64_t  shape[4];
    uint64_t off_begin;    // absolute file offset into data region
    uint64_t off_end;
} st_tensor;

typedef struct {
    mmap_file f;
    uint64_t  data_base;   // offset in file where tensor data starts (8 + header_len)
    st_tensor* t;
    size_t n;
} st_db;

// Parse the safetensors header (well-formed JSON, but we keep it specialized).
static void st_open(st_db* db, const char* path){
    db->f = mmap_ro(path);
    if(db->f.size < 8) die("safetensors too small");
    uint64_t hlen; memcpy(&hlen, db->f.data, 8);
    if(hlen > db->f.size-8) die("bad safetensors header length");
    db->data_base = 8 + hlen;
    const char* h = (const char*)db->f.data + 8;
    const char* hend = h + hlen;

    // Reserve big array (HF safetensors usually <2k tensors)
    size_t cap = 1024;
    db->t = xmalloc(cap*sizeof(st_tensor));
    db->n = 0;

    // Walk top-level keys: simple scan for '"name"' patterns followed by '{ ... }' values
    const char* p = h;
    // skip leading '{'
    while(p<hend && *p!='{') p++;
    if(p<hend) p++;
    while(p<hend){
        p = skip_ws(p);
        if(*p=='}') break;
        if(*p!='"') { p++; continue; }
        // parse key
        p++;
        const char* ks=p; while(p<hend && *p!='"') p++;
        size_t klen = p-ks;
        char keybuf[256]; if(klen>=sizeof(keybuf)) klen=sizeof(keybuf)-1;
        memcpy(keybuf,ks,klen); keybuf[klen]=0;
        if(p<hend) p++; // closing quote
        p = skip_ws(p);
        if(*p!=':') die("safetensors header: expected ':'");
        p++; p=skip_ws(p);
        if(strcmp(keybuf,"__metadata__")==0){
            // skip object
            int depth=0;
            while(p<hend){
                if(*p=='{') depth++;
                else if(*p=='}'){ depth--; if(depth==0){p++; break;} }
                else if(*p=='"'){ p++; while(p<hend && *p!='"'){ if(*p=='\\') p++; p++; } }
                p++;
            }
        } else {
            // tensor entry: must be object
            if(*p!='{') die("safetensors: expected object");
            const char* objs = p;
            int depth=0;
            while(p<hend){
                if(*p=='{') depth++;
                else if(*p=='}'){ depth--; if(depth==0){p++; break;} }
                else if(*p=='"'){ p++; while(p<hend && *p!='"'){ if(*p=='\\') p++; p++; } }
                p++;
            }
            size_t olen = p-objs;
            if(db->n==cap){ cap*=2; db->t=realloc(db->t, cap*sizeof(st_tensor)); }
            st_tensor* t = &db->t[db->n++];
            memset(t,0,sizeof(*t));
            strncpy(t->name, keybuf, sizeof(t->name)-1);
            json_find_string(objs,olen,"dtype",t->dtype,sizeof(t->dtype));
            // shape
            const char* sh = memmem(objs,olen,"\"shape\"",7);
            if(!sh) die("missing shape");
            sh += 7;
            while(sh<objs+olen && *sh!='[') sh++;
            sh++;
            while(sh<objs+olen && *sh!=']'){
                sh = skip_ws(sh);
                if(*sh==']') break;
                char* end; long long v = strtoll(sh,&end,10);
                if(end==sh) break;
                if(t->ndim>=4) die("too many dims");
                t->shape[t->ndim++] = v;
                sh = end; while(sh<objs+olen && (*sh==' '||*sh==',')) sh++;
            }
            // data_offsets
            const char* dof = memmem(objs,olen,"\"data_offsets\"",14);
            if(!dof) die("missing data_offsets");
            dof += 14;
            while(dof<objs+olen && *dof!='[') dof++;
            dof++;
            char* end;
            t->off_begin = (uint64_t)strtoll(dof,&end,10);
            dof = end; while(dof<objs+olen && (*dof==' '||*dof==',')) dof++;
            t->off_end = (uint64_t)strtoll(dof,&end,10);
        }
        p = skip_ws(p);
        if(*p==',') { p++; continue; }
        if(*p=='}') break;
    }
}
static st_tensor* st_find(st_db* db, const char* name){
    for(size_t i=0;i<db->n;i++) if(strcmp(db->t[i].name,name)==0) return &db->t[i];
    return NULL;
}
static const void* st_data(st_db* db, const st_tensor* t){
    return db->f.data + db->data_base + t->off_begin;
}
static size_t st_nelem(const st_tensor* t){
    size_t n=1; for(int i=0;i<t->ndim;i++) n*=(size_t)t->shape[i]; return n;
}

// ---------------- GGUF writer ----------------
typedef struct {
    FILE* fp;
    size_t bytes;
} gw;
static void gw_write(gw* g, const void* p, size_t n){
    if(fwrite(p,1,n,g->fp)!=n) die("write");
    g->bytes += n;
}
static void gw_u32(gw* g, uint32_t v){ gw_write(g,&v,4); }
static void gw_u64(gw* g, uint64_t v){ gw_write(g,&v,8); }
static void gw_i32(gw* g, int32_t v){ gw_write(g,&v,4); }
static void gw_f32(gw* g, float v){ gw_write(g,&v,4); }
static void gw_str(gw* g, const char* s){
    uint64_t n = strlen(s);
    gw_u64(g, n);
    gw_write(g, s, n);
}

// KV helpers
static void kv_str(gw* g, const char* k, const char* v){
    gw_str(g,k); gw_u32(g,GGUF_TYPE_STRING); gw_str(g,v);
}
static void kv_u32(gw* g, const char* k, uint32_t v){
    gw_str(g,k); gw_u32(g,GGUF_TYPE_UINT32); gw_u32(g,v);
}
static void kv_i32(gw* g, const char* k, int32_t v){
    gw_str(g,k); gw_u32(g,GGUF_TYPE_INT32); gw_i32(g,v);
}
static void kv_u64(gw* g, const char* k, uint64_t v){
    gw_str(g,k); gw_u32(g,GGUF_TYPE_UINT64); gw_u64(g,v);
}
static void kv_f32(gw* g, const char* k, float v){
    gw_str(g,k); gw_u32(g,GGUF_TYPE_FLOAT32); gw_f32(g,v);
}
static void kv_bool(gw* g, const char* k, int v){
    gw_str(g,k); gw_u32(g,GGUF_TYPE_BOOL); uint8_t b = v?1:0; gw_write(g,&b,1);
}

// ---------------- name mapping HF -> GGUF (table-driven, per family) ----------------
//
// Each row maps a HuggingFace tensor name pattern to its GGUF equivalent.
// Patterns may use %d (layer index) and %%d (literal '%d', for double-indexed
// expert tensors). The `n_indices` field describes how many `%d` slots to fill:
//   0 -> global tensor (no indices in pattern)
//   1 -> layer-indexed: model.layers.%d.X  ->  blk.%d.Y
//   2 -> layer + expert: model.layers.%d.mlp.experts.%d.X -> blk.%d.Z.%d.W
//
// Matching is performed by sscanf with the corresponding number of %d
// placeholders. The first matching row wins.

typedef struct {
    const char* hf_pat;
    const char* gguf_pat;
    int n_indices;
} tensor_rename_t;

// ---- Qwen3 ----
static const tensor_rename_t QWEN3_RENAMES[] = {
    {"model.embed_tokens.weight", "token_embd.weight",   0},
    {"model.norm.weight",         "output_norm.weight",  0},
    {"lm_head.weight",            "output.weight",       0},
    {"model.layers.%d.input_layernorm.weight",          "blk.%d.attn_norm.weight",    1},
    {"model.layers.%d.post_attention_layernorm.weight", "blk.%d.ffn_norm.weight",     1},
    {"model.layers.%d.self_attn.q_proj.weight",         "blk.%d.attn_q.weight",       1},
    {"model.layers.%d.self_attn.k_proj.weight",         "blk.%d.attn_k.weight",       1},
    {"model.layers.%d.self_attn.v_proj.weight",         "blk.%d.attn_v.weight",       1},
    {"model.layers.%d.self_attn.o_proj.weight",         "blk.%d.attn_output.weight",  1},
    {"model.layers.%d.self_attn.q_norm.weight",         "blk.%d.attn_q_norm.weight",  1},
    {"model.layers.%d.self_attn.k_norm.weight",         "blk.%d.attn_k_norm.weight",  1},
    {"model.layers.%d.mlp.gate_proj.weight",            "blk.%d.ffn_gate.weight",     1},
    {"model.layers.%d.mlp.up_proj.weight",              "blk.%d.ffn_up.weight",       1},
    {"model.layers.%d.mlp.down_proj.weight",            "blk.%d.ffn_down.weight",     1},
    {NULL, NULL, 0}
};

// ---- Gemma3/Gemma4 ----
// HF Gemma3 conditional-generation tensors are nested under model.language_model.*
// for multimodal checkpoints; text-only checkpoints use model.* directly. We
// accept both prefixes by listing them. Gemma3 attention has q_norm and k_norm.
static const tensor_rename_t GEMMA4_RENAMES[] = {
    {"model.embed_tokens.weight",         "token_embd.weight",  0},
    {"model.norm.weight",                 "output_norm.weight", 0},
    {"lm_head.weight",                    "output.weight",      0},
    // Multimodal-wrapped variants (Gemma3 text tower under language_model)
    {"language_model.model.embed_tokens.weight", "token_embd.weight",  0},
    {"language_model.model.norm.weight",         "output_norm.weight", 0},
    {"language_model.lm_head.weight",            "output.weight",      0},
    // Per-block (text-only path)
    {"model.layers.%d.input_layernorm.weight",            "blk.%d.attn_norm.weight",      1},
    {"model.layers.%d.post_attention_layernorm.weight",   "blk.%d.post_attention_norm.weight", 1},
    {"model.layers.%d.pre_feedforward_layernorm.weight",  "blk.%d.ffn_norm.weight",       1},
    {"model.layers.%d.post_feedforward_layernorm.weight", "blk.%d.post_ffw_norm.weight",  1},
    {"model.layers.%d.self_attn.q_proj.weight",           "blk.%d.attn_q.weight",         1},
    {"model.layers.%d.self_attn.k_proj.weight",           "blk.%d.attn_k.weight",         1},
    {"model.layers.%d.self_attn.v_proj.weight",           "blk.%d.attn_v.weight",         1},
    {"model.layers.%d.self_attn.o_proj.weight",           "blk.%d.attn_output.weight",    1},
    {"model.layers.%d.self_attn.q_norm.weight",           "blk.%d.attn_q_norm.weight",    1},
    {"model.layers.%d.self_attn.k_norm.weight",           "blk.%d.attn_k_norm.weight",    1},
    {"model.layers.%d.mlp.gate_proj.weight",              "blk.%d.ffn_gate.weight",       1},
    {"model.layers.%d.mlp.up_proj.weight",                "blk.%d.ffn_up.weight",         1},
    {"model.layers.%d.mlp.down_proj.weight",              "blk.%d.ffn_down.weight",       1},
    // Multimodal-wrapped per-block
    {"language_model.model.layers.%d.input_layernorm.weight",            "blk.%d.attn_norm.weight",           1},
    {"language_model.model.layers.%d.post_attention_layernorm.weight",   "blk.%d.post_attention_norm.weight", 1},
    {"language_model.model.layers.%d.pre_feedforward_layernorm.weight",  "blk.%d.ffn_norm.weight",            1},
    {"language_model.model.layers.%d.post_feedforward_layernorm.weight", "blk.%d.post_ffw_norm.weight",       1},
    {"language_model.model.layers.%d.self_attn.q_proj.weight",           "blk.%d.attn_q.weight",              1},
    {"language_model.model.layers.%d.self_attn.k_proj.weight",           "blk.%d.attn_k.weight",              1},
    {"language_model.model.layers.%d.self_attn.v_proj.weight",           "blk.%d.attn_v.weight",              1},
    {"language_model.model.layers.%d.self_attn.o_proj.weight",           "blk.%d.attn_output.weight",         1},
    {"language_model.model.layers.%d.self_attn.q_norm.weight",           "blk.%d.attn_q_norm.weight",         1},
    {"language_model.model.layers.%d.self_attn.k_norm.weight",           "blk.%d.attn_k_norm.weight",         1},
    {"language_model.model.layers.%d.mlp.gate_proj.weight",              "blk.%d.ffn_gate.weight",            1},
    {"language_model.model.layers.%d.mlp.up_proj.weight",                "blk.%d.ffn_up.weight",              1},
    {"language_model.model.layers.%d.mlp.down_proj.weight",              "blk.%d.ffn_down.weight",            1},
    {NULL, NULL, 0}
};

// ---- Mamba2 ----
// Mamba2 SSM tensors live under backbone.layers.N.mixer.*. The norm preceding
// the mixer is mixer.norm. Final norm is backbone.norm_f. Embedding lives at
// backbone.embeddings; lm_head is tied in most checkpoints.
static const tensor_rename_t MAMBA2_RENAMES[] = {
    {"backbone.embeddings.weight",  "token_embd.weight",   0},
    {"backbone.embedding.weight",   "token_embd.weight",   0},  // alt spelling
    {"backbone.norm_f.weight",      "output_norm.weight",  0},
    {"lm_head.weight",              "output.weight",       0},
    {"backbone.layers.%d.norm.weight",            "blk.%d.attn_norm.weight",   1},
    {"backbone.layers.%d.mixer.in_proj.weight",   "blk.%d.ssm_in.weight",      1},
    {"backbone.layers.%d.mixer.conv1d.weight",    "blk.%d.ssm_conv1d.weight",  1},
    {"backbone.layers.%d.mixer.conv1d.bias",      "blk.%d.ssm_conv1d.bias",    1},
    {"backbone.layers.%d.mixer.dt_bias",          "blk.%d.ssm_dt.bias",        1},
    {"backbone.layers.%d.mixer.A_log",            "blk.%d.ssm_a",              1},
    {"backbone.layers.%d.mixer.D",                "blk.%d.ssm_d",              1},
    {"backbone.layers.%d.mixer.norm.weight",      "blk.%d.ssm_norm.weight",    1},
    {"backbone.layers.%d.mixer.out_proj.weight",  "blk.%d.ssm_out.weight",     1},
    {NULL, NULL, 0}
};

// ---- DeepSeek V2/V3 ----
// MLA: q is decomposed (q_a_proj + q_a_layernorm + q_b_proj) for some configs,
// or single q_proj for smaller variants. KV is always decomposed
// (kv_a_proj_with_mqa + kv_a_layernorm + kv_b_proj).
// MoE: per-layer router + (optional) shared experts + N routed experts.
// Per-expert tensors are kept as separate GGUF tensors here (one per expert),
// using the `blk.%d.ffn_*_exps.%d.weight` naming. NOTE: llama.cpp's canonical
// loader expects experts pre-stacked into a single `blk.%d.ffn_*_exps.weight`
// of shape [n_experts, d_in, d_out]; that repacking is *not* implemented here.
// This map is sufficient for tooling that consumes per-expert tensors directly;
// downstream consumers needing the stacked layout must restack as a post-pass.
static const tensor_rename_t DEEPSEEK_RENAMES[] = {
    {"model.embed_tokens.weight", "token_embd.weight",   0},
    {"model.norm.weight",         "output_norm.weight",  0},
    {"lm_head.weight",            "output.weight",       0},
    // Norms
    {"model.layers.%d.input_layernorm.weight",          "blk.%d.attn_norm.weight",    1},
    {"model.layers.%d.post_attention_layernorm.weight", "blk.%d.ffn_norm.weight",     1},
    // MLA attention
    {"model.layers.%d.self_attn.q_proj.weight",           "blk.%d.attn_q.weight",          1},
    {"model.layers.%d.self_attn.q_a_proj.weight",         "blk.%d.attn_q_a.weight",        1},
    {"model.layers.%d.self_attn.q_a_layernorm.weight",    "blk.%d.attn_q_a_norm.weight",   1},
    {"model.layers.%d.self_attn.q_b_proj.weight",         "blk.%d.attn_q_b.weight",        1},
    {"model.layers.%d.self_attn.kv_a_proj_with_mqa.weight","blk.%d.attn_kv_a_mqa.weight",  1},
    {"model.layers.%d.self_attn.kv_a_layernorm.weight",   "blk.%d.attn_kv_a_norm.weight",  1},
    {"model.layers.%d.self_attn.kv_b_proj.weight",        "blk.%d.attn_kv_b.weight",       1},
    {"model.layers.%d.self_attn.o_proj.weight",           "blk.%d.attn_output.weight",     1},
    // Dense MLP (first few layers in V2/V3 are dense, not MoE)
    {"model.layers.%d.mlp.gate_proj.weight",            "blk.%d.ffn_gate.weight",     1},
    {"model.layers.%d.mlp.up_proj.weight",              "blk.%d.ffn_up.weight",       1},
    {"model.layers.%d.mlp.down_proj.weight",            "blk.%d.ffn_down.weight",     1},
    // MoE router + optional bias (V3 adds e_score_correction_bias)
    {"model.layers.%d.mlp.gate.weight",                       "blk.%d.ffn_gate_inp.weight",       1},
    {"model.layers.%d.mlp.gate.e_score_correction_bias",      "blk.%d.exp_probs_b.bias",          1},
    // Shared experts (V2/V3)
    {"model.layers.%d.mlp.shared_experts.gate_proj.weight",   "blk.%d.ffn_gate_shexp.weight",     1},
    {"model.layers.%d.mlp.shared_experts.up_proj.weight",     "blk.%d.ffn_up_shexp.weight",       1},
    {"model.layers.%d.mlp.shared_experts.down_proj.weight",   "blk.%d.ffn_down_shexp.weight",     1},
    // Per-expert (two indices: layer + expert)
    {"model.layers.%d.mlp.experts.%d.gate_proj.weight",       "blk.%d.ffn_gate_exps.%d.weight",   2},
    {"model.layers.%d.mlp.experts.%d.up_proj.weight",         "blk.%d.ffn_up_exps.%d.weight",     2},
    {"model.layers.%d.mlp.experts.%d.down_proj.weight",       "blk.%d.ffn_down_exps.%d.weight",   2},
    {NULL, NULL, 0}
};

// ---- arch enum + dispatch ----
typedef enum {
    ARCH_UNKNOWN = 0,
    ARCH_QWEN3,
    ARCH_GEMMA4,
    ARCH_MAMBA2,
    ARCH_DEEPSEEK,
} arch_kind_t;

typedef struct {
    arch_kind_t kind;
    const char* gguf_arch_key;        // e.g. "qwen3"
    const char* pre_tokenizer;        // e.g. "qwen2"
    const tensor_rename_t* renames;
    int file_type_q8;                 // GGUF file_type code for Q8_0 (== 7)
    int file_type_f16;                // for F16 (== 1)
} arch_handler_t;

static const arch_handler_t ARCH_HANDLERS[] = {
    { ARCH_QWEN3,    "qwen3",     "qwen2",   QWEN3_RENAMES,    7, 1 },
    { ARCH_GEMMA4,   "gemma3",    "default", GEMMA4_RENAMES,   7, 1 },
    { ARCH_MAMBA2,   "mamba2",    "default", MAMBA2_RENAMES,   7, 1 },
    { ARCH_DEEPSEEK, "deepseek2", "deepseek-llm", DEEPSEEK_RENAMES, 7, 1 },
};

static const arch_handler_t* arch_from_hf(const char* hf_arch){
    if(!hf_arch) return NULL;
    if(strstr(hf_arch,"Qwen3"))    return &ARCH_HANDLERS[0];
    if(strstr(hf_arch,"Gemma3") || strstr(hf_arch,"Gemma4")) return &ARCH_HANDLERS[1];
    if(strstr(hf_arch,"Mamba2"))   return &ARCH_HANDLERS[2];
    if(strstr(hf_arch,"Deepseek") || strstr(hf_arch,"DeepSeek")) return &ARCH_HANDLERS[3];
    return NULL;
}

// Apply a rename table. Returns malloc'd GGUF name or NULL if no rule matches.
static char* hf_to_gguf_name_for(const tensor_rename_t* tbl, const char* hf){
    char buf[384];
    for(const tensor_rename_t* r = tbl; r->hf_pat; r++){
        if(r->n_indices == 0){
            if(strcmp(hf, r->hf_pat) == 0) return xstrdup(r->gguf_pat);
        } else if(r->n_indices == 1){
            int a;
            // sscanf is "permissive" — guard against partial matches by also
            // requiring the formatted string to round-trip back to `hf`.
            if(sscanf(hf, r->hf_pat, &a) == 1){
                char check[384];
                snprintf(check, sizeof(check), r->hf_pat, a);
                if(strcmp(check, hf) != 0) continue;
                snprintf(buf, sizeof(buf), r->gguf_pat, a);
                return xstrdup(buf);
            }
        } else if(r->n_indices == 2){
            int a, b;
            if(sscanf(hf, r->hf_pat, &a, &b) == 2){
                char check[384];
                snprintf(check, sizeof(check), r->hf_pat, a, b);
                if(strcmp(check, hf) != 0) continue;
                // gguf_pat takes (layer, expert) in same order
                snprintf(buf, sizeof(buf), r->gguf_pat, a, b);
                return xstrdup(buf);
            }
        }
    }
    return NULL;
}

// Choose dst quant type per tensor (q8_0 mode):
//   - 1D tensors (norms)  -> F32
//   - token_embd, output  -> Q8_0
//   - others (2D)         -> Q8_0
// (Convert F16 mode keeps everything F16 except 1D as F32.)
static uint32_t pick_dst_type(const char* gguf_name, int ndim, int mode_q8, arch_kind_t arch) {
    if (ndim == 1) return GGML_TYPE_F32;
    // Mamba2: SSM scalar/conv tensors must stay F32 (precision critical).
    if (arch == ARCH_MAMBA2 && gguf_name){
        if (strstr(gguf_name, ".ssm_a") || strstr(gguf_name, ".ssm_d")
            || strstr(gguf_name, ".ssm_conv1d") || strstr(gguf_name, ".ssm_dt"))
            return GGML_TYPE_F32;
    }
    if (!mode_q8)  return GGML_TYPE_F16;
    return GGML_TYPE_Q8_0;
}

// ---------------- per-tensor work plan ----------------
typedef struct {
    char*    gguf_name;
    st_tensor* src;
    uint32_t dst_type;
    int64_t  ne[4];
    int      n_dims;
    size_t   nbytes;
    uint64_t out_offset; // relative to data section start
    uint8_t* out_buf;    // produced bytes (heap)
} plan_t;

static size_t gguf_type_size(uint32_t t, int64_t n0, size_t nelem){
    switch(t){
        case GGML_TYPE_F32: return nelem*4;
        case GGML_TYPE_F16: return nelem*2;
        case GGML_TYPE_Q8_0: {
            // n0 elements per row; q8_0 row = (n0/32)*34 bytes; nrows = nelem/n0
            size_t nb_per_row = (size_t)n0/32;
            size_t row_bytes = nb_per_row*34;
            size_t nrows = nelem/(size_t)n0;
            return nrows*row_bytes;
        }
        default: die("unsupported dst type"); return 0;
    }
}

static size_t pad_up(size_t v, size_t a){ return (v + a-1) & ~(a-1); }

// ---------------- worker ----------------
typedef struct {
    plan_t*  plans;
    size_t   n;
    size_t   cursor;
} worker_ctx;

static void* worker(void* p){
    worker_ctx* w = (worker_ctx*)p;
    for(;;){
        size_t i = __atomic_fetch_add(&w->cursor, 1, __ATOMIC_RELAXED);
        if (i >= w->n) break;
        plan_t* pl = &w->plans[i];
        st_tensor* t = pl->src;
        size_t nelem = st_nelem(t);
        // Decode source to F32
        float* src_f32 = NULL;
        const void* sd = NULL; // pointer to source bytes if we can use directly
        if(strcmp(t->dtype,"F32")==0){
            sd = (const uint8_t*)t + 0; // placeholder; we'll use the actual mmap pointer below
        }
        // We need the mmap data pointer; the plan stores src tensor which carries offsets.
        // We pass a global db pointer; cleaner: store db in worker_ctx. Refactor:
        die("worker stub: pass db pointer via worker_ctx (see code below)");
        (void)src_f32; (void)nelem; (void)sd;
    }
    return NULL;
}

// We replace the worker_ctx with one that includes the db pointer; redefined for clarity:
typedef struct {
    plan_t*  plans;
    size_t   n;
    st_db*   db;
    size_t   cursor;
} wctx2;

static void* worker2(void* p){
    wctx2* w = (wctx2*)p;
    for(;;){
        size_t i = __atomic_fetch_add(&w->cursor, 1, __ATOMIC_RELAXED);
        if (i >= w->n) break;
        plan_t* pl = &w->plans[i];
        st_tensor* t = pl->src;
        size_t nelem = st_nelem(t);
        const void* sd = st_data(w->db, t);
        pl->out_buf = xmalloc(pl->nbytes);

        if(pl->dst_type == GGML_TYPE_Q8_0 && strcmp(t->dtype,"BF16")==0){
            // Fast path: stream BF16 -> Q8_0 in 32-elem chunks, no big F32 buffer.
            size_t nb = nelem/QK8;
            const uint16_t* bf = (const uint16_t*)sd;
            uint8_t* dst = pl->out_buf;
            for(size_t k=0;k<nb;k++) q8_0_block_bf16(bf + k*QK8, dst + k*34);
        } else if(pl->dst_type == GGML_TYPE_F32 && strcmp(t->dtype,"BF16")==0){
            bf16_row_to_f32((const uint16_t*)sd, (float*)pl->out_buf, nelem);
        } else {
            float* src_f32 = xmalloc(nelem * sizeof(float));
            if(strcmp(t->dtype,"BF16")==0){
                bf16_row_to_f32((const uint16_t*)sd, src_f32, nelem);
            } else if(strcmp(t->dtype,"F32")==0){
                memcpy(src_f32, sd, nelem*sizeof(float));
            } else if(strcmp(t->dtype,"F16")==0){
                vImage_Buffer s = { (void*)sd, 1, nelem, nelem*2 };
                vImage_Buffer d = { src_f32, 1, nelem, nelem*4 };
                if(vImageConvert_Planar16FtoPlanarF(&s,&d,0) != kvImageNoError) die("F16->F32 failed");
            } else die("unsupported source dtype");
            if(pl->dst_type == GGML_TYPE_F32){
                memcpy(pl->out_buf, src_f32, nelem*4);
            } else if(pl->dst_type == GGML_TYPE_F16){
                vImage_Buffer s = { src_f32, 1, nelem, nelem*4 };
                vImage_Buffer d = { pl->out_buf, 1, nelem, nelem*2 };
                if(vImageConvert_PlanarFtoPlanar16F(&s,&d,0) != kvImageNoError) die("F32->F16 failed");
            } else if(pl->dst_type == GGML_TYPE_Q8_0){
                int64_t n0 = t->shape[t->ndim-1];
                size_t nrows = nelem/(size_t)n0;
                quantize_q8_0(src_f32, pl->out_buf, nrows, (size_t)n0);
            }
            free(src_f32);
        }
    }
    return NULL;
}

// ---------------- read tokenizer.json minimally ----------------
// We need: tokenizer model type (BPE), the vocab tokens (id -> str), merges.
// For Qwen3 it's a "BPE" model with vocab as an object {token: id} plus added_tokens list and merges array of strings.
// We'll do a tiny custom parser: extract the vocab object slice & merges array slice from tokenizer.json.

typedef struct {
    char**  tokens;       // index by id
    int32_t* types;
    size_t  ntok;
    char**  merges;
    size_t  nmerges;
} tok_t;

static char* read_file(const char* path, size_t* len){
    FILE* f = fopen(path,"rb"); if(!f) diep(path);
    fseek(f,0,SEEK_END); long L=ftell(f); fseek(f,0,SEEK_SET);
    char* b = xmalloc((size_t)L+1);
    if(fread(b,1,L,f)!=(size_t)L) die("read");
    b[L]=0; fclose(f); *len=(size_t)L; return b;
}

// Locate `"key": <value>` and return pointer to <value> start, value length via brace/bracket match.
static const char* find_kv_block(const char* js, size_t n, const char* key, size_t* outlen, char open, char close){
    char pat[128]; snprintf(pat,sizeof(pat),"\"%s\"",key);
    const char* p = memmem(js,n,pat,strlen(pat));
    if(!p) return NULL;
    p += strlen(pat);
    while(p<js+n && *p!=':') p++;
    if(p>=js+n) return NULL;
    p++; p=skip_ws(p);
    if(*p!=open) return NULL;
    const char* start = p;
    int depth=0;
    while(p<js+n){
        if(*p=='"'){ p++; while(p<js+n && *p!='"'){ if(*p=='\\') p++; p++; } }
        else if(*p==open) depth++;
        else if(*p==close){ depth--; if(depth==0){p++; break;} }
        p++;
    }
    *outlen = p-start;
    return start;
}

// Parse the {token: id} object in vocab. We need id->token. Worst case ~152000 tokens.
static void parse_vocab(const char* vstart, size_t vlen, tok_t* tk){
    // First pass: count and find max id.
    // Tokens may contain JSON-escaped chars. We decode to a raw byte string.
    // BPE vocab uses GPT-2 byte mapping; llama.cpp stores tokens exactly as the JSON-decoded UTF-8 string.
    // Approach: scan key/value pairs.
    size_t cap = 4096;
    char** byid = xmalloc(cap*sizeof(char*));
    memset(byid,0,cap*sizeof(char*));
    size_t maxid = 0;

    const char* p = vstart+1; // skip '{'
    const char* end = vstart+vlen-1; // before '}'
    while(p<end){
        p = skip_ws(p);
        if(*p=='}') break;
        if(*p!='"'){ p++; continue; }
        // parse key (token string)
        p++;
        // decode in place into buffer
        char keybuf[1024]; size_t ki=0;
        while(p<end && *p!='"'){
            if(*p=='\\' && p+1<end){
                p++;
                char c=*p++;
                switch(c){
                    case 'n': keybuf[ki++]='\n'; break;
                    case 't': keybuf[ki++]='\t'; break;
                    case 'r': keybuf[ki++]='\r'; break;
                    case 'b': keybuf[ki++]='\b'; break;
                    case 'f': keybuf[ki++]='\f'; break;
                    case '"': keybuf[ki++]='"'; break;
                    case '\\': keybuf[ki++]='\\'; break;
                    case '/': keybuf[ki++]='/'; break;
                    case 'u': {
                        if(p+4>end) break;
                        char hx[5]={p[0],p[1],p[2],p[3],0}; p+=4;
                        unsigned cp = (unsigned)strtoul(hx,NULL,16);
                        // encode as UTF-8 (assume BMP only; HF tokenizer doesn't use surrogates here)
                        if(cp<0x80) keybuf[ki++]=(char)cp;
                        else if(cp<0x800){ keybuf[ki++]=(char)(0xC0|(cp>>6)); keybuf[ki++]=(char)(0x80|(cp&0x3F)); }
                        else { keybuf[ki++]=(char)(0xE0|(cp>>12)); keybuf[ki++]=(char)(0x80|((cp>>6)&0x3F)); keybuf[ki++]=(char)(0x80|(cp&0x3F)); }
                        break;
                    }
                    default: keybuf[ki++]=c;
                }
            } else {
                keybuf[ki++]=*p++;
            }
            if(ki>=sizeof(keybuf)-4) die("vocab token too long");
        }
        if(p<end) p++; // closing quote
        keybuf[ki]=0;
        p=skip_ws(p);
        if(*p!=':') die("vocab: expected ':'");
        p++; p=skip_ws(p);
        char* eend;
        long long id = strtoll(p,&eend,10);
        p = eend;
        if(id<0) die("negative token id");
        if((size_t)id>=cap){
            size_t ncap=cap;
            while(ncap<=(size_t)id) ncap*=2;
            byid = realloc(byid, ncap*sizeof(char*));
            memset(byid+cap, 0, (ncap-cap)*sizeof(char*));
            cap = ncap;
        }
        char* s = xmalloc(ki+1); memcpy(s,keybuf,ki+1);
        byid[id] = s;
        if((size_t)id>maxid) maxid=(size_t)id;
        p=skip_ws(p);
        if(*p==',') { p++; continue; }
    }
    tk->ntok = maxid+1;
    tk->tokens = byid;
    tk->types = xmalloc(tk->ntok*sizeof(int32_t));
    // Default token type: NORMAL=1
    for(size_t i=0;i<tk->ntok;i++){
        tk->types[i] = 1;
        if(!tk->tokens[i]) tk->tokens[i] = xstrdup(""); // placeholder
    }
}

// Parse "added_tokens" array of objects with "id","content","special": bool. Mark special tokens as CONTROL=3.
// Grows tk arrays if needed.
static void tk_grow(tok_t* tk, size_t new_n){
    if(new_n <= tk->ntok) return;
    tk->tokens = realloc(tk->tokens, new_n*sizeof(char*));
    tk->types  = realloc(tk->types,  new_n*sizeof(int32_t));
    for(size_t i=tk->ntok;i<new_n;i++){ tk->tokens[i]=xstrdup(""); tk->types[i]=1; }
    tk->ntok = new_n;
}
static void parse_added_tokens(const char* s, size_t n, tok_t* tk){
    const char* p = s+1; // skip '['
    const char* end = s+n;
    while(p<end){
        p = skip_ws(p);
        if(*p==']') break;
        if(*p!='{'){ p++; continue; }
        // find this object's slice
        const char* os = p; int depth=0;
        while(p<end){
            if(*p=='"'){ p++; while(p<end && *p!='"'){ if(*p=='\\') p++; p++; } }
            else if(*p=='{') depth++;
            else if(*p=='}'){ depth--; if(depth==0){p++; break;} }
            p++;
        }
        size_t olen = p-os;
        long long id=-1; double spec=0;
        char content[1024]={0};
        json_find_int(os,olen,"id",&id);
        json_find_string(os,olen,"content",content,sizeof(content));
        // "special": true|false
        const char* sp = memmem(os,olen,"\"special\"",9);
        int is_special = 0;
        if(sp){ const char* q = sp+9; while(q<os+olen && *q!=':') q++; q++; q=skip_ws(q);
            if(q<os+olen && (*q=='t'||*q=='T')) is_special=1;
        }
        (void)spec;
        if(id>=0 && content[0]){
            if((size_t)id >= tk->ntok) tk_grow(tk, (size_t)id+1);
            // Overwrite token text (added tokens override vocab id->text)
            free(tk->tokens[id]);
            tk->tokens[id] = xstrdup(content);
            tk->types[id] = is_special ? 3 : 4; // CONTROL or USER_DEFINED
        }
    }
}

// Helper: read a JSON string into buf starting at *pp (which must point at '"'). Advances *pp past closing '"'.
static size_t read_jsonstr(const char** pp, const char* end, char* buf, size_t bufsz){
    const char* p = *pp;
    if(*p!='"') return 0;
    p++;
    size_t bi=0;
    while(p<end && *p!='"'){
        if(*p=='\\' && p+1<end){
            p++;
            char c=*p++;
            switch(c){
                case 'n': if(bi<bufsz-1) buf[bi++]='\n'; break;
                case 't': if(bi<bufsz-1) buf[bi++]='\t'; break;
                case 'r': if(bi<bufsz-1) buf[bi++]='\r'; break;
                case 'b': if(bi<bufsz-1) buf[bi++]='\b'; break;
                case 'f': if(bi<bufsz-1) buf[bi++]='\f'; break;
                case '"': if(bi<bufsz-1) buf[bi++]='"'; break;
                case '\\': if(bi<bufsz-1) buf[bi++]='\\'; break;
                case '/': if(bi<bufsz-1) buf[bi++]='/'; break;
                case 'u': {
                    if(p+4>end) break;
                    char hx[5]={p[0],p[1],p[2],p[3],0}; p+=4;
                    unsigned cp = (unsigned)strtoul(hx,NULL,16);
                    if(cp<0x80){ if(bi<bufsz-1) buf[bi++]=(char)cp; }
                    else if(cp<0x800){ if(bi<bufsz-2){ buf[bi++]=(char)(0xC0|(cp>>6)); buf[bi++]=(char)(0x80|(cp&0x3F)); } }
                    else { if(bi<bufsz-3){ buf[bi++]=(char)(0xE0|(cp>>12)); buf[bi++]=(char)(0x80|((cp>>6)&0x3F)); buf[bi++]=(char)(0x80|(cp&0x3F)); } }
                    break;
                }
                default: if(bi<bufsz-1) buf[bi++]=c;
            }
        } else {
            if(bi<bufsz-1) buf[bi++]=*p;
            p++;
        }
    }
    if(p<end) p++; // closing "
    buf[bi]=0;
    *pp = p;
    return bi;
}

static void parse_merges(const char* s, size_t n, tok_t* tk){
    size_t cap = 4096; char** arr = xmalloc(cap*sizeof(char*)); size_t cnt=0;
    const char* p = s+1; const char* end = s+n;
    // Merges are an array. Each element is either a string ("a b") or an array of two strings (["a","b"]).
    while(p<end){
        p = skip_ws(p);
        if(*p==']') break;
        char left[256]={0}, right[256]={0};
        char out[520]={0};
        if(*p=='['){
            p++; p=skip_ws(p);
            read_jsonstr(&p,end,left,sizeof(left));
            p=skip_ws(p);
            if(*p==',') p++; p=skip_ws(p);
            read_jsonstr(&p,end,right,sizeof(right));
            // find closing ]
            while(p<end && *p!=']') p++;
            if(p<end) p++;
            snprintf(out,sizeof(out),"%s %s",left,right);
        } else if(*p=='"'){
            read_jsonstr(&p,end,out,sizeof(out));
        } else { p++; continue; }
        if(cnt==cap){ cap*=2; arr=realloc(arr,cap*sizeof(char*)); }
        arr[cnt++] = xstrdup(out);
        p=skip_ws(p);
        if(*p==',') { p++; continue; }
    }
    tk->merges = arr; tk->nmerges = cnt;
}

// Write a STRING array KV.
static void kv_arr_str(gw* g, const char* k, char** arr, size_t n){
    gw_str(g,k); gw_u32(g,GGUF_TYPE_ARRAY);
    gw_u32(g, GGUF_TYPE_STRING);
    gw_u64(g, n);
    for(size_t i=0;i<n;i++){
        const char* s = arr[i] ? arr[i] : "";
        uint64_t L = strlen(s);
        gw_u64(g, L);
        gw_write(g, s, L);
    }
}
static void kv_arr_i32(gw* g, const char* k, int32_t* arr, size_t n){
    gw_str(g,k); gw_u32(g,GGUF_TYPE_ARRAY);
    gw_u32(g, GGUF_TYPE_INT32);
    gw_u64(g, n);
    gw_write(g, arr, n*4);
}

// ---------------- main ----------------
int main(int argc, char** argv){
    if(argc<3){
        fprintf(stderr,"usage: %s <safetensors_dir> <output.gguf> [--quant q8_0|f16]\n",argv[0]);
        return 1;
    }
    const char* dir = argv[1];
    const char* out_path = argv[2];
    int mode_q8 = 1;
    for(int i=3;i<argc;i++){
        if(strcmp(argv[i],"--quant")==0 && i+1<argc){
            i++;
            if(strcmp(argv[i],"q8_0")==0) mode_q8=1;
            else if(strcmp(argv[i],"f16")==0) mode_q8=0;
            else die("unknown --quant value");
        }
    }
    double t0 = now_s();

    // 1. Read config.json
    char path[1024];
    snprintf(path,sizeof(path),"%s/config.json",dir);
    size_t cfg_len; char* cfg = read_file(path,&cfg_len);

    // 1a. Detect architecture from architectures[0].
    char hf_arch[128] = {0};
    {
        const char* p = memmem(cfg,cfg_len,"\"architectures\"",15);
        if(p){
            const char* q = p+15;
            while(q < cfg+cfg_len && *q!='[') q++;
            while(q < cfg+cfg_len && *q!='"') q++;
            if(q < cfg+cfg_len){
                q++; size_t i=0;
                while(q < cfg+cfg_len && *q!='"' && i+1 < sizeof(hf_arch)) hf_arch[i++]=*q++;
                hf_arch[i]=0;
            }
        }
    }
    const arch_handler_t* ah = arch_from_hf(hf_arch);
    if(!ah){
        fprintf(stderr,"fatal: unsupported architecture '%s' (config.json:architectures[0]). "
                       "Supported: Qwen3*, Gemma3*/Gemma4*, Mamba2*, Deepseek*\n",
                hf_arch[0]?hf_arch:"(missing)");
        return 2;
    }
    fprintf(stderr,"arch: hf='%s' -> gguf='%s'\n", hf_arch, ah->gguf_arch_key);

    // 1b. Gemma3 nests its hidden/heads/ff under "text_config". Locate that
    // sub-object if present and parse from it; fall back to top-level.
    const char* cfg_scope = cfg;
    size_t cfg_scope_len = cfg_len;
    size_t tc_len = 0;
    const char* tc_start = NULL;
    if(ah->kind == ARCH_GEMMA4){
        tc_start = find_kv_block(cfg, cfg_len, "text_config", &tc_len, '{', '}');
        if(tc_start){ cfg_scope = tc_start; cfg_scope_len = tc_len; }
    }

    long long n_layers=0, hidden=0, n_heads=0, n_kv=0, head_dim=0, ff=0, ctx=0, vocab=0, bos=0, eos=0;
    long long n_experts=0, n_experts_used=0, moe_intermediate=0, n_shared_experts=0, first_k_dense=0;
    long long kv_lora_rank=0, q_lora_rank=0, qk_nope_dim=0, qk_rope_dim=0, v_head_dim=0;
    long long ssm_d_state=0, ssm_d_conv=0, ssm_expand=0, ssm_n_groups=0, ssm_n_heads=0;
    double rope_theta=10000.0, rms_eps=1e-6, routed_scaling=1.0;
    json_find_int(cfg_scope,cfg_scope_len,"num_hidden_layers",&n_layers);
    json_find_int(cfg_scope,cfg_scope_len,"hidden_size",&hidden);
    json_find_int(cfg_scope,cfg_scope_len,"num_attention_heads",&n_heads);
    json_find_int(cfg_scope,cfg_scope_len,"num_key_value_heads",&n_kv);
    json_find_int(cfg_scope,cfg_scope_len,"head_dim",&head_dim);
    json_find_int(cfg_scope,cfg_scope_len,"intermediate_size",&ff);
    json_find_int(cfg_scope,cfg_scope_len,"max_position_embeddings",&ctx);
    json_find_int(cfg_scope,cfg_scope_len,"vocab_size",&vocab);
    // Top-level (always)
    json_find_int(cfg,cfg_len,"bos_token_id",&bos);
    json_find_int(cfg,cfg_len,"eos_token_id",&eos);
    json_find_float(cfg_scope,cfg_scope_len,"rope_theta",&rope_theta);
    json_find_float(cfg_scope,cfg_scope_len,"rms_norm_eps",&rms_eps);
    // DeepSeek MoE / MLA
    json_find_int(cfg_scope,cfg_scope_len,"n_routed_experts",&n_experts);
    json_find_int(cfg_scope,cfg_scope_len,"num_experts_per_tok",&n_experts_used);
    json_find_int(cfg_scope,cfg_scope_len,"moe_intermediate_size",&moe_intermediate);
    json_find_int(cfg_scope,cfg_scope_len,"n_shared_experts",&n_shared_experts);
    json_find_int(cfg_scope,cfg_scope_len,"first_k_dense_replace",&first_k_dense);
    json_find_int(cfg_scope,cfg_scope_len,"kv_lora_rank",&kv_lora_rank);
    json_find_int(cfg_scope,cfg_scope_len,"q_lora_rank",&q_lora_rank);
    json_find_int(cfg_scope,cfg_scope_len,"qk_nope_head_dim",&qk_nope_dim);
    json_find_int(cfg_scope,cfg_scope_len,"qk_rope_head_dim",&qk_rope_dim);
    json_find_int(cfg_scope,cfg_scope_len,"v_head_dim",&v_head_dim);
    json_find_float(cfg_scope,cfg_scope_len,"routed_scaling_factor",&routed_scaling);
    // Mamba2
    json_find_int(cfg_scope,cfg_scope_len,"state_size",&ssm_d_state);
    json_find_int(cfg_scope,cfg_scope_len,"conv_kernel",&ssm_d_conv);
    json_find_int(cfg_scope,cfg_scope_len,"expand",&ssm_expand);
    json_find_int(cfg_scope,cfg_scope_len,"n_groups",&ssm_n_groups);
    json_find_int(cfg_scope,cfg_scope_len,"num_heads",&ssm_n_heads);
    if(!head_dim && n_heads) head_dim = hidden/n_heads;
    // tied embeddings? (top-level)
    int tie = 0;
    const char* tieKey = memmem(cfg,cfg_len,"\"tie_word_embeddings\"",21);
    if(tieKey){
        const char* q = tieKey+21;
        while(q<cfg+cfg_len && *q!=':') q++;
        q++; q=skip_ws(q);
        if(*q=='t'||*q=='T') tie=1;
    }
    // Mamba2 has no lm_head most of the time -> tie defaults true for those.
    if(ah->kind == ARCH_MAMBA2 && !tieKey) tie = 1;

    fprintf(stderr,"config: layers=%lld hidden=%lld n_heads=%lld n_kv=%lld head_dim=%lld ff=%lld vocab=%lld tie=%d\n",
            n_layers,hidden,n_heads,n_kv,head_dim,ff,vocab,tie);
    free(cfg);

    // 2. Open safetensors
    snprintf(path,sizeof(path),"%s/model.safetensors",dir);
    st_db db; st_open(&db, path);
    fprintf(stderr,"safetensors: %zu tensors, data size=%llu MB\n", db.n, (unsigned long long)((db.f.size - db.data_base)>>20));

    // 3. Build plan: skip lm_head if tied.
    plan_t* plans = xmalloc(db.n*sizeof(plan_t));
    size_t np=0;
    for(size_t i=0;i<db.n;i++){
        st_tensor* t = &db.t[i];
        if(tie && strcmp(t->name,"lm_head.weight")==0) continue;
        char* gn = hf_to_gguf_name_for(ah->renames, t->name);
        if(!gn){ fprintf(stderr,"skip (no mapping): %s\n", t->name); continue; }
        plan_t* p = &plans[np++];
        memset(p,0,sizeof(*p));
        p->gguf_name = gn;
        p->src = t;
        // GGUF ne is reversed from row-major: shape [a,b] (PyTorch [out, in]) -> GGUF ne=[in, out]
        p->n_dims = t->ndim;
        for(int d=0; d<t->ndim; d++) p->ne[d] = t->shape[t->ndim-1-d];
        p->dst_type = pick_dst_type(gn, t->ndim, mode_q8, ah->kind);
        size_t nel = st_nelem(t);
        // Q8_0 requires the contiguous (innermost) dim to be a multiple of 32;
        // if it isn't (e.g. mamba2 conv kernels) fall back to F32.
        if (p->dst_type == GGML_TYPE_Q8_0 && (p->ne[0] % QK8)) {
            p->dst_type = GGML_TYPE_F32;
        }
        p->nbytes = gguf_type_size(p->dst_type, p->ne[0], nel);
    }
    fprintf(stderr,"plan: %zu tensors will be written\n", np);

    // 4. Read tokenizer.json
    snprintf(path,sizeof(path),"%s/tokenizer.json",dir);
    double tk_t0 = now_s();
    size_t tk_len; char* tk_buf = read_file(path,&tk_len);
    tok_t tk = {0};
    size_t vlen, alen, mlen;
    const char* vstart = find_kv_block(tk_buf,tk_len,"vocab",&vlen,'{','}');
    if(!vstart) die("tokenizer: vocab not found");
    parse_vocab(vstart, vlen, &tk);
    double tk_t1 = now_s();
    const char* astart = find_kv_block(tk_buf,tk_len,"added_tokens",&alen,'[',']');
    if(astart) parse_added_tokens(astart, alen, &tk);
    const char* mstart = find_kv_block(tk_buf,tk_len,"merges",&mlen,'[',']');
    if(mstart) parse_merges(mstart, mlen, &tk);
    double tk_t2 = now_s();
    fprintf(stderr,"tokenizer: %zu tokens, %zu merges (vocab=%.2fs merges=%.2fs)\n",
            tk.ntok, tk.nmerges, tk_t1-tk_t0, tk_t2-tk_t1);

    // 5. Stage 1 of output: write header + KV + tensor metadata.
    // We need to know tensor offsets before writing data, but offsets are relative to start of data section
    // which is aligned. We must compute tensor data sizes first; then assign offsets.
    size_t cum = 0;
    for(size_t i=0;i<np;i++){
        plans[i].out_offset = cum;
        cum += pad_up(plans[i].nbytes, GGUF_DEFAULT_ALIGNMENT);
    }
    size_t total_data = cum;

    // Open output file. Two-pass: write metadata to memory, count its size, then write file with padding.
    // Simpler: stream into FILE, track bytes, pad at end of header to alignment.
    FILE* fp = fopen(out_path, "wb");
    if(!fp) diep(out_path);
    setvbuf(fp, NULL, _IOFBF, 1<<22);
    gw g = {fp, 0};

    // Header
    gw_write(&g, "GGUF", 4);
    gw_u32(&g, 3);                          // version
    gw_u64(&g, (uint64_t)np);               // tensor_count

    // count KVs we'll emit
    // base architecture/model keys (12) + tokenizer (model, pre, tokens, types, merges, bos, eos, padding, add_bos)
    // We'll count by writing to /dev/null first? Simpler: collect into buffer.

    // Strategy: write KVs to a memory buffer (open_memstream), then write count + buffer.
    char* kvbuf=NULL; size_t kvsz=0;
    FILE* mfp = open_memstream(&kvbuf,&kvsz);
    if(!mfp) die("open_memstream");
    setvbuf(mfp, NULL, _IOFBF, 1<<20);
    gw mg = {mfp, 0};

    // Track KV count manually.
    uint64_t nkv = 0;
    #define KV(fn, ...) do{ fn(&mg, __VA_ARGS__); nkv++; }while(0)

    KV(kv_str, "general.architecture", ah->gguf_arch_key);
    KV(kv_str, "general.name", hf_arch);
    KV(kv_u32, "general.file_type", mode_q8 ? (uint32_t)ah->file_type_q8 : (uint32_t)ah->file_type_f16);
    KV(kv_u32, "general.quantization_version", 2);

    // Per-architecture KVs. We construct keys with the arch prefix so the same
    // emitter works for qwen3/gemma3/mamba2/deepseek2.
    #define AK(suffix) ({ static char _kbuf[128]; snprintf(_kbuf,sizeof(_kbuf),"%s.%s", ah->gguf_arch_key, suffix); _kbuf; })

    KV(kv_u32, AK("block_count"),        (uint32_t)n_layers);
    if(ctx > 0) KV(kv_u32, AK("context_length"),     (uint32_t)ctx);
    KV(kv_u32, AK("embedding_length"),   (uint32_t)hidden);
    if(ff > 0) KV(kv_u32, AK("feed_forward_length"),(uint32_t)ff);

    if(ah->kind == ARCH_QWEN3 || ah->kind == ARCH_GEMMA4){
        KV(kv_u32, AK("attention.head_count"),    (uint32_t)n_heads);
        KV(kv_u32, AK("attention.head_count_kv"), (uint32_t)(n_kv?n_kv:n_heads));
        KV(kv_u32, AK("attention.key_length"),    (uint32_t)head_dim);
        KV(kv_u32, AK("attention.value_length"),  (uint32_t)head_dim);
        KV(kv_f32, AK("attention.layer_norm_rms_epsilon"), (float)rms_eps);
        KV(kv_f32, AK("rope.freq_base"), (float)rope_theta);
        KV(kv_u32, AK("rope.dimension_count"), (uint32_t)head_dim);
    } else if(ah->kind == ARCH_DEEPSEEK){
        KV(kv_u32, AK("attention.head_count"),    (uint32_t)n_heads);
        KV(kv_u32, AK("attention.head_count_kv"), (uint32_t)(n_kv?n_kv:n_heads));
        KV(kv_f32, AK("attention.layer_norm_rms_epsilon"), (float)rms_eps);
        KV(kv_f32, AK("rope.freq_base"), (float)rope_theta);
        // MLA dims
        if(kv_lora_rank > 0)   KV(kv_u32, AK("attention.kv_lora_rank"), (uint32_t)kv_lora_rank);
        if(q_lora_rank  > 0)   KV(kv_u32, AK("attention.q_lora_rank"),  (uint32_t)q_lora_rank);
        if(qk_nope_dim  > 0)   KV(kv_u32, AK("attention.key_length"),   (uint32_t)(qk_nope_dim + qk_rope_dim));
        if(v_head_dim   > 0)   KV(kv_u32, AK("attention.value_length"), (uint32_t)v_head_dim);
        if(qk_rope_dim  > 0)   KV(kv_u32, AK("rope.dimension_count"),   (uint32_t)qk_rope_dim);
        // MoE
        if(n_experts        > 0) KV(kv_u32, AK("expert_count"),        (uint32_t)n_experts);
        if(n_experts_used   > 0) KV(kv_u32, AK("expert_used_count"),   (uint32_t)n_experts_used);
        if(moe_intermediate > 0) KV(kv_u32, AK("expert_feed_forward_length"), (uint32_t)moe_intermediate);
        if(n_shared_experts > 0) KV(kv_u32, AK("expert_shared_count"), (uint32_t)n_shared_experts);
        if(first_k_dense    > 0) KV(kv_u32, AK("leading_dense_block_count"), (uint32_t)first_k_dense);
        KV(kv_f32, AK("expert_weights_scale"), (float)routed_scaling);
    } else if(ah->kind == ARCH_MAMBA2){
        // SSM-specific. No attention KVs.
        if(ssm_d_conv  > 0) KV(kv_u32, AK("ssm.conv_kernel"),    (uint32_t)ssm_d_conv);
        if(ssm_d_state > 0) KV(kv_u32, AK("ssm.state_size"),     (uint32_t)ssm_d_state);
        if(ssm_n_heads > 0) KV(kv_u32, AK("ssm.head_count"),     (uint32_t)ssm_n_heads);
        if(ssm_n_groups> 0) KV(kv_u32, AK("ssm.group_count"),    (uint32_t)ssm_n_groups);
        if(ssm_expand  > 0) KV(kv_u32, AK("ssm.inner_size"),     (uint32_t)(ssm_expand * hidden));
        KV(kv_f32, AK("ssm.layer_norm_rms_epsilon"), (float)rms_eps);
    }

    #undef AK

    // Tokenizer
    KV(kv_str, "tokenizer.ggml.model", "gpt2");
    KV(kv_str, "tokenizer.ggml.pre",   ah->pre_tokenizer);
    // tokens array
    { gw_str(&mg,"tokenizer.ggml.tokens"); gw_u32(&mg,GGUF_TYPE_ARRAY); gw_u32(&mg,GGUF_TYPE_STRING); gw_u64(&mg,tk.ntok);
      for(size_t i=0;i<tk.ntok;i++){
          const char* s = tk.tokens[i]; if(!s) s="";
          uint64_t L = strlen(s); gw_u64(&mg,L); gw_write(&mg,s,L);
      } nkv++; }
    // types
    { gw_str(&mg,"tokenizer.ggml.token_type"); gw_u32(&mg,GGUF_TYPE_ARRAY); gw_u32(&mg,GGUF_TYPE_INT32); gw_u64(&mg,tk.ntok);
      gw_write(&mg, tk.types, tk.ntok*4); nkv++; }
    // merges
    if(tk.nmerges>0){ gw_str(&mg,"tokenizer.ggml.merges"); gw_u32(&mg,GGUF_TYPE_ARRAY); gw_u32(&mg,GGUF_TYPE_STRING); gw_u64(&mg,tk.nmerges);
      for(size_t i=0;i<tk.nmerges;i++){ uint64_t L = strlen(tk.merges[i]); gw_u64(&mg,L); gw_write(&mg,tk.merges[i],L); } nkv++; }
    KV(kv_u32, "tokenizer.ggml.bos_token_id", (uint32_t)bos);
    KV(kv_u32, "tokenizer.ggml.eos_token_id", (uint32_t)eos);
    KV(kv_u32, "tokenizer.ggml.padding_token_id", (uint32_t)eos);
    KV(kv_bool,"tokenizer.ggml.add_bos_token", 0);
    KV(kv_u32, "general.alignment", GGUF_DEFAULT_ALIGNMENT);

    fflush(mfp); fclose(mfp);

    // Now write nkv and the kv blob
    gw_u64(&g, nkv);
    gw_write(&g, kvbuf, kvsz);
    free(kvbuf);

    // 6. Write tensor metadata records
    for(size_t i=0;i<np;i++){
        plan_t* p = &plans[i];
        gw_str(&g, p->gguf_name);
        gw_u32(&g, (uint32_t)p->n_dims);
        for(int d=0; d<p->n_dims; d++) gw_u64(&g, (uint64_t)p->ne[d]);
        gw_u32(&g, p->dst_type);
        gw_u64(&g, p->out_offset);
    }

    // 7. Pad to alignment
    size_t hdr_end = g.bytes;
    size_t data_start = pad_up(hdr_end, GGUF_DEFAULT_ALIGNMENT);
    if (data_start > hdr_end){
        uint8_t z[64] = {0};
        gw_write(&g, z, data_start - hdr_end);
    }
    fflush(fp);
    double t_meta = now_s();

    // 8. Quantize in parallel
    int n_threads = 8;
    {
        const char* e = getenv("Q2G_THREADS");
        if (e) n_threads = atoi(e);
    }
    if (n_threads < 1) n_threads = 1;
    fprintf(stderr,"quantizing with %d threads...\n", n_threads);
    wctx2 ctx2 = { plans, np, &db, 0 };
    pthread_t* th = xmalloc(n_threads*sizeof(pthread_t));
    for(int i=0;i<n_threads;i++) pthread_create(&th[i], NULL, worker2, &ctx2);
    for(int i=0;i<n_threads;i++) pthread_join(th[i], NULL);
    free(th);
    double t_quant = now_s();

    // 9. Write tensor payload (sequential — disk I/O is now the constraint)
    static const uint8_t zeros[GGUF_DEFAULT_ALIGNMENT] = {0};
    for(size_t i=0;i<np;i++){
        plan_t* p = &plans[i];
        gw_write(&g, p->out_buf, p->nbytes);
        size_t padn = pad_up(p->nbytes, GGUF_DEFAULT_ALIGNMENT) - p->nbytes;
        if(padn) gw_write(&g, zeros, padn);
        free(p->out_buf); p->out_buf = NULL;
    }
    fflush(fp); fclose(fp);
    double t_end = now_s();

    fprintf(stderr,"wrote %s (%zu MB)\n", out_path, g.bytes>>20);
    fprintf(stderr,"timing: total=%.2fs (meta=%.2fs, quantize=%.2fs, write=%.2fs)\n",
            t_end-t0, t_meta-t0, t_quant-t_meta, t_end-t_quant);
    fprintf(stderr,"source size=%zu MB -> output=%zu MB (%.2f GB/s in source terms)\n",
            db.f.size>>20, g.bytes>>20, (double)db.f.size/(1024.0*1024.0*1024.0)/(t_end-t0));

    munmap_file(&db.f);
    (void)total_data; (void)worker;
    return 0;
}
