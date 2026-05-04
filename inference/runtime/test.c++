#include "device.h"
#include "buffer.h"
#include "threadgroup.h"
#include <cstdio>
#include <cassert>

int main() {
    // ── device ──
    sk::runtime::init(nullptr, "/Users/alazarmanakelew/SuperKittens/build/libsk.metallib");
    assert(sk::runtime::device());
    assert(sk::runtime::queue());
    assert(sk::runtime::library());
    printf("device: %s\n", sk::runtime::device()->name()->utf8String());

    // ── buffer ──
    float data[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    auto buf = sk::runtime::Buffer::from(data, 4, 4, 1, 1, sk::runtime::DType::f32);
    assert(buf.elements() == 16);
    assert(buf.bytes == 64);
    assert(buf.ndim == 2);

    float back[16];
    buf.read(back);
    for (int i = 0; i < 16; i++) assert(back[i] == data[i]);
    printf("buffer: %zu elements, %zu bytes, ndim=%u ✓\n", buf.elements(), buf.bytes, buf.ndim);
    buf.release();

    // ── constant buffer ──
    uint32_t val = 42;
    auto cb = sk::runtime::constant_buffer(&val, sizeof(val));
    assert(*(uint32_t*)cb.mtl->contents() == 42);
    printf("constant: value=%u ✓\n", *(uint32_t*)cb.mtl->contents());
    cb.release();

    // ── threadgroup compile-time checks ──
    using namespace sk::runtime;
    using TG = Threadgroup<0>;
    static_assert(TG::free == TG_LIMIT);
    using WithQ = TG::add<8192>;   // 8KB Q tile
    using WithK = WithQ::add<16384>; // +16KB K tile = 24KB
    static_assert(WithK::used == 24576);
    static_assert(WithK::free == 8192);
    printf("threadgroup: Q+K tiles = %uKB, %uKB free ✓\n", WithK::used/1024, WithK::free/1024);

    // ── runtime usage ──
    auto usage = TGUsage::from_elements(64 * 128, 2);  // Q tile [64,128] fp16
    printf("Q tile [64,128] fp16: %u bytes (%u%%)\n", usage.bytes, usage.percent);
    assert(usage.bytes == 16384);

    sk::runtime::shutdown();
    printf("PASS\n");
    return 0;
}
