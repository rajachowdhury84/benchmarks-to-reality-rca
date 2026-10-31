"""
STEP 02: Top-k tables from the extracted scores (data/re1_scores.json).

Reports, for each of the three localizers:
  * restricted candidate space (5 injected services)  -- kept so the
    reproduction check against the paper's baseline table is possible
  * expanded candidate space (all services in the deployment) -- primary
pooled, per system, and per fault type.
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, RESULTS_DIR as OUT_DIR

LOCALIZERS = ["zscore", "nsigma", "baro"]
LABEL = {"zscore": "Z-score adapter", "nsigma": "N-Sigma (RCAEval)", "baro": "BARO (RCAEval)"}
FAULT_TYPES = ["cpu", "mem", "disk", "delay", "loss"]
SYSTEMS = ["RE1-OB", "RE1-SS", "RE1-TT"]


def rank(rec, loc, space):
    cands = rec["services_expanded"] if space == "expanded" else rec["services_restricted"]
    sc = rec[loc]
    pairs = [(s, sc.get(s, -1e9)) for s in cands]
    pairs.sort(key=lambda kv: (-kv[1], kv[0]))
    return [s for s, _ in pairs]


def topk(recs, loc, space, k):
    if not recs:
        return float("nan")
    hit = sum(1 for r in recs if r["true_service"] in rank(r, loc, space)[:k])
    return 100.0 * hit / len(recs)


def main():
    d = json.load(open(os.path.join(DATA_DIR, "re1_scores.json")))
    recs, meta = d["records"], d["meta"]
    print(f"environment: python {meta['python']}, scikit-learn {meta['sklearn']}, "
          f"numpy {meta['numpy']}, pandas {meta['pandas']}")
    print(f"cases: {meta['n_cases']} (skipped {meta['skipped']})\n")

    n_cand = {}
    for s in SYSTEMS:
        r = next(r for r in recs if r["system"] == s)
        n_cand[s] = (len(r["services_restricted"]), len(r["services_expanded"]))
    print("candidate-set sizes per system (restricted / expanded):")
    for s in SYSTEMS:
        print(f"  {s}: {n_cand[s][0]} / {n_cand[s][1]}")

    out = {"meta": meta, "candidate_sizes": n_cand, "results": {}}
    for space in ("restricted", "expanded"):
        print(f"\n================ candidate space: {space} ================")
        print(f"{'localizer':22s} {'pooled T1':>10s} {'pooled T3':>10s}  per-system Top-1")
        for loc in LOCALIZERS:
            p1, p3 = topk(recs, loc, space, 1), topk(recs, loc, space, 3)
            per_sys = {s: topk([r for r in recs if r["system"] == s], loc, space, 1)
                       for s in SYSTEMS}
            per_sys3 = {s: topk([r for r in recs if r["system"] == s], loc, space, 3)
                        for s in SYSTEMS}
            per_ft = {ft: topk([r for r in recs if r["fault_type"] == ft], loc, space, 1)
                      for ft in FAULT_TYPES}
            per_sys_ft = {s: {ft: topk([r for r in recs if r["system"] == s
                                        and r["fault_type"] == ft], loc, space, 1)
                              for ft in FAULT_TYPES} for s in SYSTEMS}
            ss = "  ".join(f"{s}={per_sys[s]:.1f}" for s in SYSTEMS)
            print(f"{LABEL[loc]:22s} {p1:9.1f}% {p3:9.1f}%  {ss}")
            print(f"{'':22s} per-fault-type Top-1: " +
                  "  ".join(f"{ft}={per_ft[ft]:.1f}" for ft in FAULT_TYPES))
            out["results"][f"{space}/{loc}"] = {
                "pooled_top1": p1, "pooled_top3": p3,
                "per_system_top1": per_sys, "per_system_top3": per_sys3,
                "per_fault_top1": per_ft, "per_system_fault_top1": per_sys_ft,
            }
    with open(os.path.join(OUT_DIR, "topk.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote topk.json")


if __name__ == "__main__":
    main()
