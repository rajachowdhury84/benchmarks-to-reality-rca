"""
STEP 12: paired statistics over the seeds produced by step11_memory_validate.py.

Reports the memory-ON vs memory-OFF comparison for Full SDL and the paired condition
comparisons quoted in the synthetic-results text (paired t-test and Wilcoxon signed-rank
across the same seeds). Writes results/memory_merged.json.
"""
import glob, json, os, sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS_DIR

conds = ["baseline", "baseline_ctx", "abstain_only", "ood_only",
         "epistemic_only", "full_sdl", "full_sdl_nomem"]
metrics = ["acc", "known_acc", "novel_acc", "cw", "known_cw", "novel_cw", "esc", "abst"]
merged = {c: {m: [] for m in metrics} for c in conds}
files = sorted(glob.glob(os.path.join(RESULTS_DIR, "memory_shard_*.json")),
               key=lambda f: json.load(open(f))["_meta"]["seed_start"])
if not files:
    sys.exit("no memory_shard_*.json in results/; run step11_memory_validate.py first")
for f in files:
    d = json.load(open(f))
    for c in conds:
        for m in metrics:
            merged[c][m].extend(d[c][m])
n = len(merged["full_sdl"]["acc"])
print(f"=== merged {n} seeds from {len(files)} shards ===\n")


def ms(xs):
    xs = np.array(xs) * 100
    return xs.mean(), xs.std(ddof=1)


print(f"{'Condition':16s} {'Acc':>12s} {'Known':>7s} {'Novel':>7s} {'CW':>7s} {'NovelCW':>8s} {'Esc':>7s}")
for c in conds:
    a, asd = ms(merged[c]["acc"])
    k, _ = ms(merged[c]["known_acc"])
    nv, _ = ms(merged[c]["novel_acc"])
    cw, _ = ms(merged[c]["cw"])
    ncw, _ = ms(merged[c]["novel_cw"])
    e, _ = ms(merged[c]["esc"])
    print(f"{c:16s} {a:6.1f}+/-{asd:4.1f} {k:6.1f} {nv:6.1f} {cw:6.1f} {ncw:7.1f} {e:6.1f}")


def paired(a, b, label):
    a = np.array(a) * 100; b = np.array(b) * 100
    diff = a - b
    t, pt = stats.ttest_rel(a, b)
    try:
        w, pw = stats.wilcoxon(a, b)
    except ValueError:
        w, pw = float('nan'), 1.0
    print(f"  {label}: mean diff {diff.mean():+.2f} pts  (paired t={t:.2f}, p={pt:.2e}; Wilcoxon p={pw:.2e})")


print(f"\n=== INSTITUTIONAL MEMORY: Full SDL memory-ON vs memory-OFF (paired, n={n}) ===")
paired(merged["full_sdl"]["acc"], merged["full_sdl_nomem"]["acc"], "accuracy")
paired(merged["full_sdl"]["known_acc"], merged["full_sdl_nomem"]["known_acc"], "known-cause acc")
paired(merged["full_sdl"]["novel_acc"], merged["full_sdl_nomem"]["novel_acc"], "novel-cause acc")
paired(merged["full_sdl"]["cw"], merged["full_sdl_nomem"]["cw"], "confidently-wrong")
paired(merged["full_sdl"]["known_cw"], merged["full_sdl_nomem"]["known_cw"], "known-cause CW")

print(f"\n=== PAIRED CONDITION COMPARISONS (n={n}) ===")
paired(merged["full_sdl"]["acc"], merged["baseline"]["acc"], "Full SDL vs Baseline")
paired(merged["ood_only"]["acc"], merged["baseline"]["acc"], "OOD-only vs Baseline")
paired(merged["full_sdl"]["acc"], merged["ood_only"]["acc"], "Full SDL vs OOD-only")
paired(merged["full_sdl"]["acc"], merged["baseline_ctx"]["acc"], "Full SDL vs Baseline+Ctx")
paired(merged["ood_only"]["acc"], merged["baseline_ctx"]["acc"], "OOD-only vs Baseline+Ctx")
paired(merged["epistemic_only"]["acc"], merged["baseline"]["acc"], "Epistemic-only vs Baseline")
paired(merged["abstain_only"]["acc"], merged["baseline"]["acc"], "Abstain-only vs Baseline")

out = os.path.join(RESULTS_DIR, "memory_merged.json")
json.dump({c: {m: merged[c][m] for m in metrics} for c in conds}, open(out, "w"))
print(f"\nwrote {out}")
