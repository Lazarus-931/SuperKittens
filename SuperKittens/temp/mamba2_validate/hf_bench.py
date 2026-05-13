"""HF fp32 decode tok/s baseline on lexie M4 mini for Mamba2-130m."""
import os, time, torch
import transformers.models.mamba2.modeling_mamba2 as M2
M2.is_fast_path_available = False
M2.mamba_chunk_scan_combined = None
M2.mamba_split_conv1d_scan_combined = None
M2.causal_conv1d_fn = None
M2.causal_conv1d_update = None
M2.selective_state_update = None
from transformers import AutoTokenizer, Mamba2ForCausalLM

W = os.path.expanduser("~/SuperKittens/SuperKittens/model_weights/mamba2-130m-hf")
tok = AutoTokenizer.from_pretrained(W)
m = Mamba2ForCausalLM.from_pretrained(W, torch_dtype=torch.float32).eval()
ids = tok("Hi", return_tensors="pt").input_ids

with torch.no_grad():
    out = m.generate(ids, max_new_tokens=4, do_sample=False)
    print("warm:", tok.decode(out[0]))

N = 32
with torch.no_grad():
    t0 = time.perf_counter()
    out = m.generate(ids, max_new_tokens=N, do_sample=False)
    dt = time.perf_counter() - t0
text = tok.decode(out[0])
print(f"generated: {text!r}")
print(f"new_tokens={N}  elapsed={dt:.3f}s  tok/s={N/dt:.2f}")
