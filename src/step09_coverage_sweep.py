"""
STEP 09: conformal-coverage sweep on the synthetic generator.

Runs the ablation at several conformal targets (coverage 1 - alpha from 95% to 60%) and
records accuracy and escalation for every condition, showing that accuracy stays flat while
Full SDL's escalation rate falls. Writes results/coverage_sweep.json.

Usage:
    python src/step09_coverage_sweep.py                      # 100 seeds, alpha 0.05 ... 0.40
    python src/step09_coverage_sweep.py --alphas 0.05 0.40   # endpoints only
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gbm_ablation as G
from paths import RESULTS_DIR

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30, 0.40])
    ap.add_argument("--out", type=str, default=os.path.join(RESULTS_DIR, "coverage_sweep.json"))
    args = ap.parse_args()

    out = {"seeds": args.seeds, "sweep": []}
    for a in args.alphas:
        s = G.run_all(args.seeds, n_incidents=400, alpha=a, seed_start=1)
        row = {"alpha": a, "coverage": 1 - a}
        for c in G.CONDITIONS:
            row[c] = {"acc": s[c]["acc"]["mean"], "esc": s[c]["esc"]["mean"]}
        out["sweep"].append(row)
        print(f"coverage {100 * (1 - a):4.0f}%  Full SDL acc {100 * row['full_sdl']['acc']:5.1f}%  "
              f"esc {100 * row['full_sdl']['esc']:5.1f}%", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")
