"""instrument_diffusion.py — add env-driven per-step dumps to llama.cpp PR
#24423 examples/diffusion/diffusion.cpp (diffusion_generate_entropy_bound).

DG_EB_DUMP=<dir>        : per step write step_%03d.f32 (canvas logits [C, V])
                          and step_%03d.dec (decision record, eb_ref_harness
                          layout) + header.bin (n_input,C,S,seed, eb params,
                          prompt ids) at step 0.
DG_EB_DUMP_THROTTLE=1   : block until the consumer deletes step-2's logits
                          (keeps <=2 x 268 MB in flight on a tight disk).

Idempotent (skips if already instrumented). Usage:
  python3 instrument_diffusion.py <path-to-diffusion.cpp>
"""
import sys
from pathlib import Path

A_INCLUDE = "#include <random>\n"
INS_INCLUDE = "#include <random>\n#include <unistd.h>\n#include <cstdint>\n#include <cstdlib>\n"

A_LOGITS = ("        const float * logits = llama_get_logits(ctx);"
            "             // canvas rows packed: [C or max_length, n_vocab]\n")
INS_LOGITS = A_LOGITS + """
        // DG_EB_DUMP: per-step logits + decisions (SK Stage-2 parity gate)
        const char * dg_dump = getenv("DG_EB_DUMP");
        std::vector<llama_token> dg_canvas_in;
        if (dg_dump) {
            if (step_idx == 0) {
                char hp[1024]; snprintf(hp, sizeof(hp), "%s/header.bin", dg_dump);
                FILE * hf = fopen(hp, "wb");
                if (hf) {
                    int32_t hdr[4] = { n_input, C, S, params.seed };
                    fwrite(hdr, 4, 4, hf);
                    float fl[5] = { params.t_min, params.t_max, params.entropy_bound,
                                    (float) params.stability_threshold, params.confidence_threshold };
                    fwrite(fl, 4, 5, hf);
                    fwrite(input_tokens, 4, n_input, hf);
                    fclose(hf);
                }
            }
            if (getenv("DG_EB_DUMP_THROTTLE") && step_idx >= 2) {
                char prev[1024]; snprintf(prev, sizeof(prev), "%s/step_%03d.f32", dg_dump, step_idx - 2);
                while (access(prev, F_OK) == 0) { usleep(500000); }   // consumer deletes
            }
            char lp[1024]; snprintf(lp, sizeof(lp), "%s/step_%03d.f32", dg_dump, step_idx);
            FILE * lf = fopen(lp, "wb");
            if (lf) {
                fwrite(logits + (size_t) logit_off * n_vocab, sizeof(float), (size_t) C * n_vocab, lf);
                fclose(lf);
            }
            dg_canvas_in = current_canvas;
        }
"""

A_STOP = ("        prev_argmax   = argmax_canvas;\n"
          "        prev_temp_inv = temp_inv;\n")
INS_STOP = A_STOP + """
        if (dg_dump) {
            char dp[1024]; snprintf(dp, sizeof(dp), "%s/step_%03d.dec", dg_dump, step_idx);
            FILE * df = fopen(dp, "wb");
            if (df) {
                uint8_t fin = finished ? 1 : 0;
                fwrite(&step_idx, 4, 1, df); fwrite(&cur_step, 4, 1, df); fwrite(&t, 4, 1, df);
                fwrite(dg_canvas_in.data(), 4, C, df);
                fwrite(u.data(), 4, C, df);
                fwrite(renoise.data(), 4, C, df);
                fwrite(entropy.data(), 4, C, df);
                fwrite(argmax_canvas.data(), 4, C, df);
                fwrite(denoiser.data(), 4, C, df);
                std::vector<uint8_t> acc8(accepted.begin(), accepted.end());
                fwrite(acc8.data(), 1, C, df);
                fwrite(current_canvas.data(), 4, C, df);
                fwrite(&held, 4, 1, df);
                fwrite(&fin, 1, 1, df);
                fwrite(&entropy_sum, 4, 1, df);
                fclose(df);
            }
        }
"""


def main() -> int:
    p = Path(sys.argv[1])
    src = p.read_text()
    if "DG_EB_DUMP" in src:
        print("already instrumented")
        return 0
    for anchor, ins in ((A_INCLUDE, INS_INCLUDE), (A_LOGITS, INS_LOGITS), (A_STOP, INS_STOP)):
        n = src.count(anchor)
        if n != 1:
            print(f"anchor not unique ({n}): {anchor[:60]!r}")
            return 1
        src = src.replace(anchor, ins)
    p.write_text(src)
    print(f"instrumented {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
