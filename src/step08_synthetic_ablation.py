"""
STEP 08: synthetic ablation table (six SDL conditions, 100 trials of 400 incidents).

Thin wrapper around gbm_ablation.run_all with the paper's settings; writes
results/synthetic_ablation.json (per-condition mean, sd and per-seed values, plus the
baseline ECE).
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gbm_ablation as G
from paths import RESULTS_DIR

if __name__ == "__main__":
    summary = G.run_all(100, n_incidents=400, alpha=G.EPISTEMIC_ALPHA_DEFAULT, seed_start=1)
    out = os.path.join(RESULTS_DIR, "synthetic_ablation.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}")
