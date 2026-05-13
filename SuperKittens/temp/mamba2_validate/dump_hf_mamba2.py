"""Dump HF Mamba2 reference activations layer-by-layer for SK validation.

Forces the slow torch_forward path so the reference is deterministic and
matches what SK's kernels reproduce step-by-step.

Saves to hf_ref.npz next to this script.
"""
import os, sys, json, numpy as np, torch

# Force-disable the CUDA fast path before importing transformers' mamba2
import transformers.models.mamba2.modeling_mamba2 as M2
M2.is_fast_path_available = False
M2.mamba_chunk_scan_combined = None
M2.mamba_split_conv1d_scan_combined = None
M2.causal_conv1d_fn = None
M2.causal_conv1d_update = None
M2.selective_state_update = None

from transformers import AutoTokenizer, Mamba2ForCausalLM

REPO = "AntonV/mamba2-130m-hf"
WEIGHTS_DIR = os.path.expanduser("~/SuperKittens/SuperKittens/model_weights/mamba2-130m-hf")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_ref.npz")
PROMPT = "Hi"

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
model = Mamba2ForCausalLM.from_pretrained(WEIGHTS_DIR, torch_dtype=torch.float32)
model.eval()

input_ids = tok(PROMPT, return_tensors="pt").input_ids
print(f"prompt={PROMPT!r} ids={input_ids.tolist()}", flush=True)

dump = {}
def save(name, t):
    if isinstance(t, torch.Tensor):
        dump[name] = t.detach().to(torch.float32).cpu().numpy()

# Hook every mixer to capture inputs/outputs and inner activations via monkey-patch of torch_forward
orig_forward = M2.Mamba2Mixer.torch_forward

def patched(self, input_states, cache_params=None, attention_mask=None):
    layer = self.layer_idx
    save(f"L{layer}.mixer_in", input_states)
    # Replicate torch_forward but also capture intermediates
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    projected_states = self.in_proj(input_states)
    save(f"L{layer}.proj", projected_states)
    d_mlp = (projected_states.shape[-1]
             - 2 * self.intermediate_size
             - 2 * self.n_groups * self.ssm_state_size
             - self.num_heads) // 2
    _, _, gate, hidden_states_B_C, dt = projected_states.split(
        [d_mlp, d_mlp, self.intermediate_size,
         self.intermediate_size + 2 * self.n_groups * self.ssm_state_size,
         self.num_heads], dim=-1)
    save(f"L{layer}.gate", gate)
    save(f"L{layer}.xBC_preconv", hidden_states_B_C)
    save(f"L{layer}.dt_pre", dt)
    out = orig_forward(self, input_states, cache_params, attention_mask)
    save(f"L{layer}.mixer_out", out)
    return out

M2.Mamba2Mixer.torch_forward = patched

# Also dump embed + each block residual + final norm + logits
embed = model.backbone.embeddings(input_ids)
save("embed", embed)

with torch.no_grad():
    out = model(input_ids, output_hidden_states=True)

for i, h in enumerate(out.hidden_states):
    save(f"hidden.{i}", h)
save("logits", out.logits)
save("input_ids", input_ids)

# Also save config for SK
cfg = model.config.to_dict()
with open(os.path.join(os.path.dirname(OUT), "config_dump.json"), "w") as f:
    json.dump(cfg, f, indent=2, default=str)

np.savez(OUT, **dump)
print(f"saved {OUT}  keys={len(dump)}  argmax_last={int(out.logits[0,-1].argmax())}", flush=True)
