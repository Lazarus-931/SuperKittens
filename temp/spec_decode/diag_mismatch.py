"""Diagnose the prompt_len=1 K>=4 row-2 mismatch: is it a near-tie (fp noise)
or a structurally wrong logit row?

Compares the FULL logit vector at the disputed position between:
  (a) autoregressive: prefill[10] -> step 35 -> step 5371 -> logits row (current_pos=3)
  (b) verify chunk:   prefill[10] -> forward([35,5371]) -> row 2 (current_pos 1->3)
Both feed identical token context [10,35,5371]; row must be bit-comparable up to
fp-reassociation. Reports top-5 of each and the max abs logit diff.
"""
import _env  # noqa: F401
import numpy as np
import SuperKittens as sk
import SuperKittens.models.qwen.qwen  # noqa

m = sk.load("qwen3-0.6b", cache_max=512)


def top5(r):
    r = r.astype(np.float32)
    idx = np.argsort(r)[-5:][::-1]
    return [(int(i), float(r[i])) for i in idx]


prompt = [10]
# (a) autoregressive to get context tokens
m.reset(); m.forward(np.array(prompt, np.int32))
t0 = int(np.argmax(m.get_logits_rows(1)[-1].astype(np.float32)))   # 35
m.forward(np.array([t0], np.int32))
t1 = int(np.argmax(m.get_logits_rows(1)[-1].astype(np.float32)))   # 5371
m.forward(np.array([t1], np.int32))
auto_row = m.get_logits_rows(1)[-1].astype(np.float32)             # predict after [10,35,5371]
print("context tokens:", prompt, t0, t1)
print("AUTO  top5:", top5(auto_row))

# (b) verify chunk
m.reset(); m.forward(np.array(prompt, np.int32))
m.forward(np.array([t0, t1], np.int32))
vrows = m.get_logits_rows(2).astype(np.float32)
ver_row = vrows[1]   # row 1 predicts after [10,(35),5371] i.e. position of t1
print("VERIFY top5 (row1):", top5(ver_row))
print("max|auto-verify| logit diff:", float(np.max(np.abs(auto_row - ver_row))))
print("argmax auto:", int(np.argmax(auto_row)), " argmax verify:", int(np.argmax(ver_row)))

# Now reproduce the K=4 scenario exactly (chunk = [35,5371,3567,362]-ish) and
# inspect ROW 2 which mismatched.
m.reset(); m.forward(np.array(prompt, np.int32))
# autoregressive 4-step gt
m.reset(); m.forward(np.array(prompt, np.int32))
gt = []
cur = int(np.argmax(m.get_logits_rows(1)[-1].astype(np.float32)))
gt.append(cur)
for _ in range(3):
    m.forward(np.array([cur], np.int32)); cur = int(np.argmax(m.get_logits_rows(1)[-1].astype(np.float32))); gt.append(cur)
print("gt 4-step:", gt)
m.reset(); m.forward(np.array(prompt, np.int32))
m.forward(np.array(gt[:4], np.int32))
vrows = m.get_logits_rows(4).astype(np.float32)
print("VERIFY row2 top5:", top5(vrows[2]))
# ground truth for row2 = token after [10]+gt[:3]
m.reset(); m.forward(np.array(prompt + gt[:3], np.int32))
gtrow2 = m.get_logits_rows(len(prompt)+3)[-1].astype(np.float32)
print("AUTO  row2 top5:", top5(gtrow2))
print("max|diff| row2:", float(np.max(np.abs(vrows[2]-gtrow2))))
m.close()
