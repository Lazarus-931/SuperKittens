// quantize.h — Q8_0 weight quantization (load-time, vDSP-accelerated).
//
// Q8_0 block layout (matches gguf.quants.Q8_0 bit-exactly):
//     half  d            — per-32-element scale (= amax / 127)
//     int8  qs[32]       — round(x / d)
//   → 34 bytes per 32 weights.
//
// Used by gemma4 to quantize the bf16 LM-head at load time so decode can
// route through the Q8_0 matvec (~3.8x faster than bf16 GEMM at decode shapes).
//
#ifndef SK_INFERENCE_QUANTIZE_H
#define SK_INFERENCE_QUANTIZE_H

#include <cstddef>
#include <cstdint>

namespace sk {

// Source = float[n_elems]. Output = uint8_t[(n_elems / 32) * 34]. n_elems must
// be a multiple of 32. Uses vDSP for the inner amax + scale-multiply, scalar
// for round-to-int8 (vDSP_vfixr8 truncates, doesn't round).
void quantize_q8_0(const float* x, size_t n_elems, uint8_t* out);

// Convenience wrapper: bf16 source → Q8_0 (converts on the fly, block by block).
void quantize_q8_0_bf16(const uint16_t* x_bf16, size_t n_elems, uint8_t* out);

// Q4_K block layout (matches gguf.quants.Q4_K / llama.cpp block_q4_K bit-exactly;
// = what kernels/gemm/q4k_matvec{,_bf16}.metal read):
//     half  d              — super-block scale
//     half  dmin           — super-block min
//     uint8 scales[12]     — 8×(6-bit sub-scale, 6-bit sub-min), get_scale_min_k4 packed
//     uint8 qs[128]        — 256 4-bit quants (low/high nibble pairs)
//   → 144 bytes per 256 weights. n_elems must be a multiple of 256.
//   w = d * sc[j] * q - dmin * m[j]   per 32-element sub-block j.
//
// Used to quantize the gemma4 LM head (bf16 → Q4_K) at load time so decode's
// large-N bandwidth-bound head matvec routes through q4k_matvec_bf16 (~half the
// bytes of Q8_0). Implements llama.cpp's make_qkx2_quants reference algorithm.
//
// Output = uint8_t[(n_elems / 256) * 144].
void quantize_q4_k(const float* x, size_t n_elems, uint8_t* out);

// Convenience wrapper: bf16 source → Q4_K (converts per super-block on the fly).
void quantize_q4_k_bf16(const uint16_t* x_bf16, size_t n_elems, uint8_t* out);

}  // namespace sk

#endif
