// rng_dump.cpp — emit the exact RNG stream the reference EB sampler consumes
// (std::mt19937 + libc++ uniform_int/uniform_real, same construction & call
// order as diffusion_generate_entropy_bound) so the Python mirror can be
// verified bit-for-bit.
//
//   ./rng_dump <seed> <C> <S> <n_vocab> <out.bin>
//
// layout (little-endian):
//   int32 canvas_init[C]
//   per step s in 0..S-1: float32 u[C], int32 renoise[C]
//   (u/renoise interleave per position inside the generator, matching the
//    reference's "pre-draw single-threaded" loop)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 6) { fprintf(stderr, "usage: %s seed C S n_vocab out\n", argv[0]); return 1; }
    const int32_t seed    = atoi(argv[1]);
    const int32_t C       = atoi(argv[2]);
    const int32_t S       = atoi(argv[3]);
    const int32_t n_vocab = atoi(argv[4]);

    std::mt19937                           rng(seed);
    std::uniform_real_distribution<float>  uni01(0.0f, 1.0f);
    std::uniform_int_distribution<int32_t> vocab_dist(0, n_vocab - 1);

    FILE* f = fopen(argv[5], "wb");
    if (!f) { perror("fopen"); return 1; }

    std::vector<int32_t> canvas(C);
    for (int32_t i = 0; i < C; i++) canvas[i] = vocab_dist(rng);
    fwrite(canvas.data(), 4, C, f);

    std::vector<float>   u(C);
    std::vector<int32_t> renoise(C);
    for (int32_t s = 0; s < S; s++) {
        for (int32_t pos = 0; pos < C; pos++) {
            u[pos]       = uni01(rng);
            renoise[pos] = vocab_dist(rng);
        }
        fwrite(u.data(), 4, C, f);
        fwrite(renoise.data(), 4, C, f);
    }
    fclose(f);
    return 0;
}
