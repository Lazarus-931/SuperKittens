"""instrument_sc_dump.py — env-driven named-tensor dumps for the Gate-1 SC
verification, patched into the predecessor's llama.cpp clone:

1. src/models/diffusion-gemma.cpp : cb(sc_sig, "sc_sig", -1) so the SC signal
   is a named graph node ("inp_region" is already named).
2. examples/diffusion-gemma-server/diffusion-gemma-server.cpp :
   DG_DUMP_TENSORS=<n1,n2,...> DG_DUMP_DIR=<dir> installs a sched eval
   callback that writes each named tensor to <dir>/<name>_r%03d.f32 per
   request (f32, ggml layout).

Idempotent. Usage: python3 instrument_sc_dump.py <clone-root>
"""
import sys
from pathlib import Path

A_MODEL = ('            sc_sig = ggml_scale(ctx0, sc_sig, dmodel.sc_use); '
           '// runtime {0,1} gate (0 == first step)\n')
INS_MODEL = A_MODEL + '            cb(sc_sig, "sc_sig", -1);\n'

A_INC = '#include "llama.h"\n'
INS_INC = A_INC + '#include "ggml-backend.h"\n'

A_CB = "int main(int argc, char ** argv) {\n"
INS_CB = """
// DG_DUMP_TENSORS / DG_DUMP_DIR: per-request named-tensor dumps (SK Stage-2 SC gate)
static std::vector<std::string> g_dg_dump_names;
static std::string              g_dg_dump_dir;
static int                      g_dg_req_idx = -1;
static bool dg_dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    (void) user_data;
    bool want = false;
    for (const auto & s : g_dg_dump_names) { if (s == t->name) { want = true; break; } }
    if (ask) { return want; }
    if (want) {
        const size_t n = (size_t) ggml_nelements(t);
        std::vector<float> host(n);
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, host.data(), 0, n * sizeof(float));
        } else {
            std::vector<char> raw(ggml_nbytes(t));
            ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
            ggml_get_type_traits(t->type)->to_float(raw.data(), host.data(), (int64_t) n);
        }
        char p[1024];
        snprintf(p, sizeof(p), "%s/%s_r%03d.f32", g_dg_dump_dir.c_str(), t->name, g_dg_req_idx);
        FILE * f = fopen(p, "wb");
        if (f) { fwrite(host.data(), sizeof(float), n, f); fclose(f); }
    }
    return true;
}

""" + A_CB

A_CTX = "    llama_context * ctx = llama_init_from_model(model, cparams);\n"
INS_CTX = """    if (const char * dt = getenv("DG_DUMP_TENSORS")) {
        g_dg_dump_dir = getenv("DG_DUMP_DIR") ? getenv("DG_DUMP_DIR") : ".";
        std::string s = dt;
        size_t pos = 0;
        while (true) {
            size_t e = s.find(',', pos);
            g_dg_dump_names.push_back(s.substr(pos, e == std::string::npos ? std::string::npos : e - pos));
            if (e == std::string::npos) break;
            pos = e + 1;
        }
        fprintf(stderr, "dg_dump: %zu tensors -> %s\\n", g_dg_dump_names.size(), g_dg_dump_dir.c_str());
        cparams.cb_eval           = dg_dump_cb;
        cparams.cb_eval_user_data = nullptr;
    }
""" + A_CTX

A_REQ = "        std::vector<int32_t> req = read_i32_file(line);\n"
INS_REQ = A_REQ + "        g_dg_req_idx++;\n"


def patch(path: Path, pairs) -> int:
    src = path.read_text()
    if "DG_DUMP_TENSORS" in src or 'cb(sc_sig, "sc_sig", -1)' in src:
        print(f"already instrumented: {path}")
        return 0
    for anchor, ins in pairs:
        n = src.count(anchor)
        if n != 1:
            print(f"anchor not unique ({n}) in {path}: {anchor[:60]!r}")
            return 1
        src = src.replace(anchor, ins)
    path.write_text(src)
    print(f"instrumented {path}")
    return 0


def main() -> int:
    root = Path(sys.argv[1])
    rc = patch(root / "src/models/diffusion-gemma.cpp", [(A_MODEL, INS_MODEL)])
    if rc:
        return rc
    return patch(root / "examples/diffusion-gemma-server/diffusion-gemma-server.cpp",
                 [(A_INC, INS_INC), (A_CB, INS_CB), (A_CTX, INS_CTX), (A_REQ, INS_REQ)])


if __name__ == "__main__":
    sys.exit(main())
