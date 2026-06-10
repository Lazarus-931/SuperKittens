// eb_ref_harness.cpp — diffusion_generate_entropy_bound lifted VERBATIM from
// llama.cpp PR #24423 examples/diffusion/diffusion.cpp @ c84e85af, with
// llama_decode replaced by per-step logits files. Drives the same logits into
// the C++ decision path so the Python sampler can be checked token-for-token
// in isolation from any model/forward noise.
//
//   ./eb_ref_harness <logits_dir> <out_dir> <seed> <C> <S> <n_vocab>
//                    [t_min t_max entropy_bound stability confidence]
//
// logits_dir/step_%03d.f32 : [C, n_vocab] f32 (step_idx-indexed; the harness
//                            stops consuming at adaptive stop, like the ref)
// out_dir/step_%03d.dec    : decision record, parsed by compare_eb.py
//   i32 step_idx, i32 cur_step, f32 t,
//   i32 canvas_in[C], f32 u[C], i32 renoise[C], f32 entropy[C],
//   i32 argmax[C], i32 denoiser[C], u8 accepted[C], i32 canvas_next[C],
//   i32 held, u8 finished, f32 entropy_sum
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

typedef int32_t llama_token;

struct eb_params {
    int32_t S;
    float   t_min                = 0.4f;
    float   t_max                = 0.8f;
    float   entropy_bound        = 0.1f;
    int32_t stability_threshold  = 1;
    float   confidence_threshold = 0.005f;
    int32_t seed                 = 0;
};

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: see header\n"); return 1; }
    const std::string ldir = argv[1];
    const std::string odir = argv[2];
    eb_params params;
    params.seed       = atoi(argv[3]);
    const int32_t C   = atoi(argv[4]);
    params.S          = atoi(argv[5]);
    const int32_t n_vocab = atoi(argv[6]);
    if (argc >= 12) {
        params.t_min                = (float) atof(argv[7]);
        params.t_max                = (float) atof(argv[8]);
        params.entropy_bound        = (float) atof(argv[9]);
        params.stability_threshold  = atoi(argv[10]);
        params.confidence_threshold = (float) atof(argv[11]);
    }
    const int32_t S = params.S;

    // ---- verbatim reference body below (file-fed logits) -------------------
    std::mt19937                           rng(params.seed);
    std::uniform_real_distribution<float>  uni01(0.0f, 1.0f);
    std::uniform_int_distribution<int32_t> vocab_dist(0, n_vocab - 1);

    std::vector<llama_token> current_canvas(C);
    for (int32_t i = 0; i < C; i++) {
        current_canvas[i] = vocab_dist(rng);
    }

    std::vector<llama_token> argmax_canvas(C, 0);
    std::vector<llama_token> prev_argmax(C, -1);
    std::vector<float>       entropy(C);
    std::vector<llama_token> denoiser(C);
    std::vector<int32_t>     order(C);
    std::vector<float>       u(C);
    std::vector<llama_token> renoise(C);

    const unsigned hw  = std::thread::hardware_concurrency();
    const unsigned nth = std::max(1u, std::min(hw ? hw : 1u, 32u));

    std::vector<float> logits((size_t) C * n_vocab);

    float prev_temp_inv = 1.0f;
    int   held          = 0;
    bool  finished      = false;
    (void) prev_temp_inv;

    for (int32_t cur_step = S; cur_step >= 1 && !finished; --cur_step) {
        const int32_t step_idx = S - cur_step;
        const float   t        = params.t_min + (params.t_max - params.t_min) * ((float) cur_step / (float) S);
        const float   temp_inv = 1.0f / t;

        // forward stand-in: read this step's logits
        {
            char path[1024];
            snprintf(path, sizeof(path), "%s/step_%03d.f32", ldir.c_str(), step_idx);
            FILE* f = fopen(path, "rb");
            if (!f) { fprintf(stderr, "missing %s\n", path); return 1; }
            if (fread(logits.data(), 4, logits.size(), f) != logits.size()) {
                fprintf(stderr, "short read %s\n", path); return 1;
            }
            fclose(f);
        }
        std::vector<llama_token> canvas_in = current_canvas;

        for (int32_t pos = 0; pos < C; pos++) {
            u[pos]       = uni01(rng);
            renoise[pos] = vocab_dist(rng);
        }

        auto worker = [&](int32_t p0, int32_t p1) {
            for (int32_t pos = p0; pos < p1; pos++) {
                const float* row = logits.data() + (size_t) pos * n_vocab;
                float m = -INFINITY; int32_t amax = 0;
                for (int32_t v = 0; v < n_vocab; v++) {
                    const float z = row[v] * temp_inv;
                    if (z > m) { m = z; amax = v; }
                }
                float Z = 0.0f;
                for (int32_t v = 0; v < n_vocab; v++) {
                    Z += expf(row[v] * temp_inv - m);
                }
                const float target = u[pos] * Z;
                float   cum = 0.0f, H = 0.0f;
                int32_t sampled = n_vocab - 1; bool picked = false;
                for (int32_t v = 0; v < n_vocab; v++) {
                    const float e = expf(row[v] * temp_inv - m);
                    const float p = e / Z;
                    if (p > 0.0f) { H -= p * logf(p); }
                    cum += e;
                    if (!picked && cum >= target) { sampled = v; picked = true; }
                }
                entropy[pos]       = H;
                argmax_canvas[pos] = amax;
                denoiser[pos]      = sampled;
            }
        };
        {
            std::vector<std::thread> pool;
            const int32_t chunk = (C + (int32_t) nth - 1) / (int32_t) nth;
            for (unsigned ti = 0; ti < nth; ti++) {
                const int32_t p0 = (int32_t) ti * chunk;
                const int32_t p1 = std::min(p0 + chunk, C);
                if (p0 < p1) { pool.emplace_back(worker, p0, p1); }
            }
            for (auto& th : pool) { th.join(); }
        }

        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return entropy[a] < entropy[b]; });
        std::vector<char> accepted(C, 0);
        double cumE = 0.0;
        for (int32_t k = 0; k < C; k++) {
            const int32_t pos = order[k];
            cumE += entropy[pos];
            if (cumE - entropy[pos] <= params.entropy_bound) { accepted[pos] = 1; }
        }

        float entropy_sum = 0.0f;
        for (int32_t pos = 0; pos < C; pos++) {
            current_canvas[pos] = accepted[pos] ? denoiser[pos] : renoise[pos];
            entropy_sum += entropy[pos];
        }

        held = (prev_argmax == argmax_canvas) ? held + 1 : 0;
        const bool confident = (entropy_sum / (float) C) < params.confidence_threshold;
        if (held >= params.stability_threshold && confident) { finished = true; }
        prev_argmax   = argmax_canvas;
        prev_temp_inv = temp_inv;

        // ---- decision dump --------------------------------------------------
        {
            char path[1024];
            snprintf(path, sizeof(path), "%s/step_%03d.dec", odir.c_str(), step_idx);
            FILE* f = fopen(path, "wb");
            if (!f) { perror("fopen dec"); return 1; }
            uint8_t fin = finished ? 1 : 0;
            fwrite(&step_idx, 4, 1, f); fwrite(&cur_step, 4, 1, f); fwrite(&t, 4, 1, f);
            fwrite(canvas_in.data(), 4, C, f);
            fwrite(u.data(), 4, C, f);
            fwrite(renoise.data(), 4, C, f);
            fwrite(entropy.data(), 4, C, f);
            fwrite(argmax_canvas.data(), 4, C, f);
            fwrite(denoiser.data(), 4, C, f);
            std::vector<uint8_t> acc8(accepted.begin(), accepted.end());
            fwrite(acc8.data(), 1, C, f);
            fwrite(current_canvas.data(), 4, C, f);
            fwrite(&held, 4, 1, f);
            fwrite(&fin, 1, 1, f);
            fwrite(&entropy_sum, 4, 1, f);
            fclose(f);
        }
    }
    return 0;
}
