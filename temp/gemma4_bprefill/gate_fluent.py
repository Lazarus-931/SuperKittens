"""Gate 1 with FLUENT prompts: 8 unique long passages (no tiling), BOS at pos 0.

Checks (per WIN gate):
  - 8/8 per-lane exact match (first token + 32-tok continuation) B vs A,
    where A = token-by-token lockstep (M=8 per step, never touches the
    BM=32-vs-/64 row-grid sites) and B = sk_gemma4_prefill_batched.
  - identical-prompt invariant, lane-permutation invariant.
  - prints A-side decoded text per lane so fluency is verifiable.
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.expanduser(os.environ.get("SK_TREE", "~/sk-gemma-bprefill-r3")))
import numpy as np
import SuperKittens as sk

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--T", type=int, default=128)
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--cont", type=int, default=32)
ap.add_argument("--json_out", default="")
args = ap.parse_args()

m = sk.load("gemma4-12b-unified", batch=args.batch, seq_max=128, cache_max=512,
            window=512)
print(f"[setup] batch={args.batch} window={m.cfg.window}", flush=True)

TEXTS = [
    "The history of bread baking stretches back over ten thousand years, beginning with flat unleavened cakes cooked on hot stones beside open fires. Ancient Egyptian bakers discovered that dough left to rest would rise on its own, captured wild yeasts turning a dense paste into an airy loaf. Roman bakeries industrialized the craft with large masonry ovens and professional guilds, and by the Middle Ages nearly every European village supported a communal oven where families brought their shaped loaves each morning. The chemistry behind all of this remained mysterious until the nineteenth century, when scientists finally described how yeast ferments sugars into carbon dioxide, stretching gluten networks that bakers had been kneading by intuition for millennia. Modern artisans now combine that scientific understanding with",
    "Metal compute shaders allow a programmer to dispatch thousands of threadgroups across the GPU, each cooperating through fast threadgroup memory while reading and writing device buffers. A well designed kernel keeps its arithmetic intensity high enough to hide memory latency, streaming tiles of data through shared storage so that every byte fetched from DRAM is reused many times before being discarded. On Apple Silicon the unified memory architecture removes explicit copies between processor and accelerator, but the bandwidth ceiling still dominates performance for most inference workloads. Profiling therefore begins with a simple roofline analysis: count the bytes each kernel must move, divide by the measured bandwidth of the chip, and compare that bound against the observed execution time to decide whether",
    "In the deep ocean, hydrothermal vents support entire ecosystems that never see sunlight, powered instead by chemosynthetic bacteria that oxidize hydrogen sulfide gushing from the seafloor. Giant tube worms cluster around the vents in dense thickets, lacking mouths and digestive tracts entirely, nourished by symbiotic microbes housed within their tissues. Crabs, shrimp, and pale octopuses patrol the mineral chimneys, grazing on bacterial mats that coat every surface. When a vent finally goes extinct, the community collapses within months, yet larvae carried on deep currents somehow locate new vents tens of kilometers away and rebuild the entire assemblage. Biologists studying these habitats argue that similar chemical gardens beneath the ice of distant moons could plausibly",
    "A well tuned sourdough starter doubles in volume within four to six hours of feeding, producing a pleasant aroma of ripe fruit and yogurt rather than the sharp smell of acetone that signals neglect. Maintaining that vigor requires a steady rhythm: discard most of the culture, refresh it with equal weights of flour and water, and hold it at a temperature where the yeasts and lactic acid bacteria stay in balance. Bakers who keep their starter in the refrigerator slow this cycle to a weekly feeding, trading some liveliness for convenience. The microbial community inside a mature starter is remarkably stable, resisting invasion by stray organisms because its acidity and competitive ecology leave no niche unfilled, which explains why",
    "The transformer architecture replaced recurrence with attention, letting every token attend directly to every other token in the sequence instead of squeezing history through a fixed size hidden state. This change unlocked massive parallelism during training, since whole sequences could be processed at once on modern accelerators rather than step by step. The cost is quadratic scaling in sequence length, which has driven a decade of research into sparse patterns, sliding windows, linear approximations, and key value caching strategies. During inference the bottleneck shifts: generating each new token requires reading the entire stack of cached keys and values, so memory bandwidth rather than raw arithmetic throughput usually determines how quickly a large language model can",
    "Glaciers carve valleys over millennia, grinding bedrock into fine flour that turns meltwater lakes a striking shade of turquoise blue. As the ice advances it plucks boulders from the valley walls and drags them along its base, scouring deep U shaped troughs that remain long after the climate warms. Moraines of tumbled rock mark each pause in the retreat, and stranded blocks of ice leave kettle ponds dotting the outwash plain. Geologists read these landforms like a written record, reconstructing the extent of ancient ice sheets from ridgelines and erratic boulders perched far from their parent outcrops. Today laser altimetry and satellite gravimetry extend that record forward, measuring yearly losses of ice mass that",
    "Compilers translate high level source code into machine instructions through stages of parsing, optimization, and code generation, each built on decades of formal theory. The front end checks syntax and types, lowering the program into an intermediate representation that captures its meaning while discarding surface details. Optimization passes then rewrite this representation, folding constants, eliminating dead branches, hoisting loop invariant work, and vectorizing inner loops where the hardware allows. Register allocation maps an unbounded supply of virtual values onto a handful of physical registers, spilling the overflow to the stack as cheaply as possible. The final emission stage schedules instructions to keep the processor pipelines full, because even a perfectly optimized sequence of operations will",
    "The annual monsoon arrives on the southwest coast in early June, bringing weeks of heavy rain that replenish rivers and aquifers after the long dry season. Farmers time their sowing to the first reliable downpours, and an early or late onset can shift harvests across an entire subcontinent. The system itself is a vast heat engine: land warms faster than ocean in spring, drawing moist air inland where it rises over the mountains and releases its water. Forecasters track sea surface temperatures thousands of kilometers away because shifts in the Pacific can strengthen or starve the circulation months in advance. Reservoir managers, city planners, and insurance companies all build their yearly calendars around the",
]

rows = []
for t in TEXTS:
    ids = m.tokenizer.encode(t)  # adds BOS
    assert len(ids) >= args.T, f"prompt too short: {len(ids)}"
    rows.append(ids[:args.T])
ids = np.ascontiguousarray(np.array(rows, dtype=np.int32))

def prefill_tbt(p):
    out = None
    for t in range(p.shape[1]):
        out = m.forward_batched(np.ascontiguousarray(p[:, t]))
    return out

def decode_n(first, n):
    toks = [np.array(first, dtype=np.int32).copy()]
    for _ in range(n - 1):
        toks.append(m.forward_batched(toks[-1]).astype(np.int32).copy())
    return np.stack(toks, axis=1)

m.reset()
base_next = prefill_tbt(ids)
base_cont = decode_n(base_next, args.cont)
print("[fluent] A done", flush=True)

m.reset()
new_next = m.prefill_batched(ids, chunk_size=args.chunk)
new_cont = decode_n(new_next, args.cont)
print("[fluent] B done", flush=True)

first_match = [int(base_next[b]) == int(new_next[b]) for b in range(args.batch)]
lane_match = [bool((base_cont[b] == new_cont[b]).all()) for b in range(args.batch)]
print(f"[fluent] first-token match per lane: {first_match}")
print(f"[fluent] {args.cont}-token continuation match per lane: {lane_match}", flush=True)
for b in range(args.batch):
    tag = "OK " if lane_match[b] else "DIV"
    print(f"[fluent] lane {b} {tag} A text: {m.tokenizer.decode(base_cont[b].tolist())!r}")
    if not lane_match[b]:
        a, c = base_cont[b].tolist(), new_cont[b].tolist()
        d = next(i for i in range(len(a)) if a[i] != c[i])
        print(f"[fluent] lane {b} diverges at idx {d}: A={a[max(0,d-2):d+3]} B={c[max(0,d-2):d+3]}")
        print(f"[fluent] lane {b} B text: {m.tokenizer.decode(new_cont[b].tolist())!r}")

# identical-prompt invariant
ids_same = np.tile(ids[0], (args.batch, 1))
m.reset()
same_next = m.prefill_batched(ids_same, chunk_size=args.chunk)
same_ok = bool((same_next == same_next[0]).all())
print(f"[fluent] identical prompts -> identical next tokens: {same_ok}", flush=True)

# lane-permutation invariant
perm = np.array([3, 1, 4, 0, 7, 5, 2, 6])
m.reset()
pnext = m.prefill_batched(np.ascontiguousarray(ids[perm]), chunk_size=args.chunk)
pcont = decode_n(pnext, 8)
m.reset()
qnext = m.prefill_batched(ids, chunk_size=args.chunk)
qcont = decode_n(qnext, 8)
perm_next_ok = bool((np.array(pnext) == np.array(qnext)[perm]).all())
perm_cont_ok = bool((pcont == qcont[perm]).all())
print(f"[perm] next tokens permute exactly: {perm_next_ok}")
print(f"[perm] 8-tok continuations permute exactly: {perm_cont_ok}", flush=True)

res = dict(first_match=first_match, cont_match=lane_match, same_prompt_ok=same_ok,
           perm_next_ok=perm_next_ok, perm_cont_ok=perm_cont_ok,
           A_text=[m.tokenizer.decode(base_cont[b].tolist()) for b in range(args.batch)])
if args.json_out:
    json.dump(res, open(args.json_out, "w"), indent=1)
print("[done]", flush=True)
