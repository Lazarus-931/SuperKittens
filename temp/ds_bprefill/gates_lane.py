"""N=8 lane-isolation gates + serving-TTFT A/B for sk_deepseek_prefill_batched.

ONE batch=8 handle, one process. A = token-by-token lockstep prefill
(forward_batched seq=1 x T); B = prefill_batched (chunked seq>1).
Writes artifacts/gates_lane.json (tokens + timings + pass/fail).
"""
import os, sys, time, json
import numpy as np

ROOT = os.path.expanduser("~/sk-ds-bprefill-k9")
sys.path.insert(0, ROOT)
from SuperKittens.inference.registry import load

SNAP = os.path.expanduser("~/ds_fit/DeepSeek-V2-Lite")
GGUF = os.path.expanduser("~/qwen-gguf/DeepSeek-V2-Lite.Q4_K_M.gguf")
N, T, CONT = 8, 128, 32
SEQ_MAX, CACHE_MAX = 64, 192
REPS, WARMUPS, GAP = 7, 2, 0.3

TEXTS = [
    "The history of the Roman Empire spans more than a thousand years, from the "
    "legendary founding of the city on the Palatine Hill through the long "
    "republican period of consuls and senators, the civil wars that ended the "
    "republic, the rise of Augustus and the principate, the flourishing of trade "
    "and law and engineering across three continents, the crisis of the third "
    "century with its soldier emperors and fractured provinces, the reforms of "
    "Diocletian and Constantine, the founding of a new capital on the Bosporus, "
    "and finally the slow transformation of the western provinces into the "
    "successor kingdoms of the early medieval world while the east endured.",

    "Photosynthesis is the process by which green plants, algae, and certain "
    "bacteria convert light energy into chemical energy stored in glucose. In "
    "the light-dependent reactions, chlorophyll molecules in the thylakoid "
    "membranes absorb photons and use that energy to split water molecules, "
    "releasing oxygen as a byproduct and generating ATP and NADPH. In the "
    "Calvin cycle that follows, carbon dioxide from the atmosphere is fixed "
    "into organic molecules using the energy carriers produced earlier. The "
    "rate of the whole process depends on light intensity, carbon dioxide "
    "concentration, temperature, and the availability of water and nutrients "
    "in the soil, which is why greenhouse growers control every one of them.",

    "Modern container ships are among the largest moving structures ever built "
    "by human beings, with the biggest vessels stretching four hundred meters "
    "from bow to stern and carrying more than twenty four thousand standard "
    "containers stacked both below deck and high above it. Their slow-speed "
    "two-stroke diesel engines stand several stories tall and burn heavy fuel "
    "oil, turning a single enormous propeller at fewer than one hundred "
    "revolutions per minute. Navigation, ballast, engine monitoring, and cargo "
    "planning are all computerized, and a crew of barely twenty people can "
    "operate a ship whose cargo would once have required an entire fleet and "
    "thousands of sailors to move across the oceans of the world.",

    "The theory of plate tectonics explains the large-scale motions of Earth's "
    "lithosphere, which is broken into a dozen major plates and several minor "
    "ones that ride on the slowly convecting mantle beneath them. At mid-ocean "
    "ridges, new crust forms as magma rises and solidifies, pushing the plates "
    "apart at rates of a few centimeters per year, roughly the speed at which "
    "fingernails grow. Where plates converge, one may slide beneath another in "
    "a subduction zone, producing deep ocean trenches, volcanic arcs, and the "
    "most powerful earthquakes ever recorded. Transform boundaries, where "
    "plates grind past one another horizontally, store elastic strain for "
    "centuries before releasing it suddenly in destructive seismic events.",

    "In the early days of personal computing, hobbyists assembled machines from "
    "kits, soldering memory chips onto circuit boards and toggling bootstrap "
    "loaders through front-panel switches before any keyboard or display was "
    "connected. Software arrived on cassette tapes or paper tape, and a few "
    "kilobytes of random access memory were considered generous. Within a "
    "single decade the industry transformed completely: floppy disks replaced "
    "tape, color graphics replaced blinking lights, spreadsheets and word "
    "processors gave ordinary offices a reason to buy machines, and the "
    "operating system became a product in its own right, setting the stage for "
    "the software platforms that would dominate the following thirty years.",

    "A traditional sourdough loaf begins days before baking, when a baker mixes "
    "flour and water and lets wild yeast and lactic acid bacteria from the "
    "environment colonize the paste, feeding it regularly until it becomes an "
    "active starter that doubles within hours. The dough itself is mixed from "
    "strong bread flour, water, salt, and a portion of that starter, then "
    "folded at intervals to develop gluten without intensive kneading. A long, "
    "cool fermentation in the refrigerator deepens the flavor as acids "
    "accumulate. Finally the loaf is shaped, proofed, scored with a sharp "
    "blade, and baked in a covered vessel that traps steam, allowing the crust "
    "to spring dramatically before it browns and crackles as it cools.",

    "The James Webb Space Telescope orbits the second Lagrange point, a "
    "gravitationally balanced location one and a half million kilometers from "
    "Earth where the observatory can keep its tennis-court-sized sunshield "
    "pointed at the Sun, Earth, and Moon simultaneously. Its segmented primary "
    "mirror, six and a half meters across and coated in a microscopically thin "
    "layer of gold, collects infrared light from the most distant galaxies "
    "ever observed, from stellar nurseries hidden behind veils of dust, and "
    "from the atmospheres of planets orbiting other stars. Keeping the "
    "instruments within a few degrees of absolute zero is essential, because "
    "any warmth from the telescope itself would swamp the faint signals.",

    "Common law systems, which trace their lineage to medieval England, rely "
    "heavily on judicial precedent: the reasoned decisions of earlier courts "
    "bind or strongly guide later judges confronting similar facts. Civil law "
    "systems, descended from Roman law through the Napoleonic codes, instead "
    "place comprehensive written statutes at the center of legal reasoning, "
    "with scholarly commentary shaping interpretation. The distinction shapes "
    "everything from how lawyers are trained to how contracts are drafted: a "
    "common law agreement tries to anticipate every contingency in exhaustive "
    "clauses, while a civil law contract can lean on default rules supplied by "
    "the code, trusting courts to fill gaps in predictable, codified ways.",
]


def main():
    t0 = time.perf_counter()
    m = load("deepseek-v2-lite", snapshot=SNAP, gguf=GGUF,
             batch=N, seq_max=SEQ_MAX, cache_max=CACHE_MAX)
    print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

    lanes = []
    for tx in TEXTS:
        # Doubled so every prompt tokenizes past T (one was 126 < 128 raw).
        ids = m.tokenizer.encode(tx + " " + tx, bos=True)
        assert len(ids) >= T, f"prompt too short: {len(ids)}"
        lanes.append(np.asarray(ids[:T], dtype=np.int32))
    ids_mat = np.stack(lanes)  # [N, T] request-major

    def tbt_prefill(mat):
        cur = np.zeros(N, dtype=np.int32)
        nxt = None
        for s in range(mat.shape[1]):
            cur[:] = mat[:, s]
            nxt = m._forward_batched(cur)
        return np.asarray(nxt, dtype=np.int32)

    def cont_steps(first, n=CONT):
        rows = [np.asarray(first, dtype=np.int32).copy()]
        cur = rows[0].copy()
        for _ in range(n):
            cur = np.asarray(m._forward_batched(cur), dtype=np.int32).copy()
            rows.append(cur)
        return np.stack(rows, axis=1)  # [N, n+1]

    res = {"config": {"N": N, "T": T, "CONT": CONT, "seq_max": SEQ_MAX,
                      "cache_max": CACHE_MAX}}

    # ── correctness gates ──
    print("[gate] A: token-by-token lockstep prefill + cont", flush=True)
    m.reset()
    a_next = tbt_prefill(ids_mat)
    ta = time.perf_counter(); a_tok = cont_steps(a_next); a_dec = time.perf_counter() - ta
    res["A_tokens"] = a_tok.tolist()

    print("[gate] B: prefill_batched chunk=64 + cont", flush=True)
    m.reset()
    b_next = np.asarray(m.prefill_batched(ids_mat, chunk_size=64), dtype=np.int32)
    tb = time.perf_counter(); b_tok = cont_steps(b_next); b_dec = time.perf_counter() - tb
    res["B64_tokens"] = b_tok.tolist()

    print("[gate] B: prefill_batched chunk=32 (next only)", flush=True)
    m.reset()
    b32_next = np.asarray(m.prefill_batched(ids_mat, chunk_size=32), dtype=np.int32)
    res["B32_next"] = b32_next.tolist()

    print("[gate] chunk-vs-single at T=64", flush=True)
    m.reset()
    s_one = np.asarray(m.prefill_batched(ids_mat[:, :64], chunk_size=0), dtype=np.int32)
    m.reset()
    s_two = np.asarray(m.prefill_batched(ids_mat[:, :64], chunk_size=32), dtype=np.int32)
    res["T64_single_next"] = s_one.tolist()
    res["T64_chunk32_next"] = s_two.tolist()

    print("[gate] identical-prompt lanes", flush=True)
    same = np.tile(ids_mat[0], (N, 1))
    m.reset()
    same_next = np.asarray(m.prefill_batched(same, chunk_size=64), dtype=np.int32)
    res["same_next"] = same_next.tolist()

    gates = {
        "lane_match_33tok": bool((a_tok == b_tok).all()),
        "chunk32_eq_chunk64_next": bool((b32_next == b_tok[:, 0]).all()),
        "T64_single_eq_chunked": bool((s_one == s_two).all()),
        "identical_lanes_identical": bool((same_next == same_next[0]).all()),
        "in_vocab": bool((b_tok >= 0).all() and (b_tok < m.cfg.vocab_size).all()),
    }
    res["gates"] = gates
    print(f"[gates] {gates}", flush=True)
    per_lane = [(bool((a_tok[i] == b_tok[i]).all())) for i in range(N)]
    res["per_lane_match"] = per_lane
    print(f"[per-lane 33-token match] {per_lane}", flush=True)

    # ── TTFT A/B: 2 warmup pairs + 7 measured pairs, interleaved, 0.3s gaps ──
    def time_A():
        m.reset()
        t = time.perf_counter()
        tbt_prefill(ids_mat)
        return (time.perf_counter() - t) * 1e3

    def time_B(cs):
        m.reset()
        t = time.perf_counter()
        m.prefill_batched(ids_mat, chunk_size=cs)
        return (time.perf_counter() - t) * 1e3

    A_ms, B64_ms, B32_ms = [], [], []
    for rep in range(WARMUPS + REPS):
        a = time_A(); time.sleep(GAP)
        b = time_B(64); time.sleep(GAP)
        b32 = time_B(32); time.sleep(GAP)
        tag = "warm" if rep < WARMUPS else "rep"
        print(f"[ttft {tag} {rep}] A={a:.1f}ms B64={b:.1f}ms B32={b32:.1f}ms", flush=True)
        if rep >= WARMUPS:
            A_ms.append(a); B64_ms.append(b); B32_ms.append(b32)
    res["ttft"] = {"A_ms": A_ms, "B64_ms": B64_ms, "B32_ms": B32_ms,
                   "A_med": float(np.median(A_ms)),
                   "B64_med": float(np.median(B64_ms)),
                   "B32_med": float(np.median(B32_ms))}
    sp = res["ttft"]["A_med"] / res["ttft"]["B64_med"]
    print(f"[ttft] A_med={res['ttft']['A_med']:.1f}ms B64_med={res['ttft']['B64_med']:.1f}ms "
          f"speedup={sp:.2f}x improvement={(1-1/sp)*100:.1f}%", flush=True)

    # ── decode aggregate after prefill (32 lockstep steps), 3 reps each ──
    a_decs, b_decs = [a_dec], [b_dec]
    for _ in range(2):
        m.reset(); n0 = tbt_prefill(ids_mat)
        t = time.perf_counter(); cont_steps(n0); a_decs.append(time.perf_counter() - t)
        time.sleep(GAP)
        m.reset(); n1 = m.prefill_batched(ids_mat, chunk_size=64)
        t = time.perf_counter(); cont_steps(n1); b_decs.append(time.perf_counter() - t)
        time.sleep(GAP)
    res["decode32_after_A_s"] = a_decs
    res["decode32_after_B_s"] = b_decs
    am, bm = float(np.median(a_decs)), float(np.median(b_decs))
    print(f"[decode32] after-A med={am:.3f}s after-B med={bm:.3f}s ratio={bm/am:.4f}", flush=True)
    res["decode_ratio_B_over_A"] = bm / am

    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    out = os.path.join(ROOT, "artifacts", "gates_lane.json")
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"GATES_LANE_DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
