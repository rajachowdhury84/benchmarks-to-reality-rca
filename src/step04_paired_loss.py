"""
STEP 04: paired and distributional analyses, all on the pre-specified LOSS fold.

(a) Paired C2-D1 bootstrap interval at matched budget (the central comparative claim).
(b) Bootstrap CI on the TPR-FPR gap.
(c) C3 as a distribution: 1000 repeats of the size-matched pseudo-novel draw.

Reuses step03_lofto.py verbatim (same seed, same split, same detector fits).
"""
import json, os, sys
import numpy as np

sys.argv = ["step03_lofto.py", "expanded"]
import step03_lofto as L

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, RESULTS_DIR as OUT
SEED = L.SEED
ITERS = 5000
FT = "loss"
LOC = "baro"
DET = "D1"

d = json.load(open(os.path.join(DATA_DIR, "re1_scores.json")))
recs = d["records"]

folds = [L.run_fold(recs, s, FT, LOC, "expanded", np.random.default_rng(SEED))
         for s in L.SYSTEMS]

conf, correct, y_novel, flags, c2conf = L.pool_fault_type(folds, FT, DET, LOC)
n = len(correct)
q = float(np.mean(flags))
n_esc = min(max(int(round(q * n)), 0), n - 1)
n_keep = n - n_esc

print(f"pooled n={n}  novel={int(y_novel.sum())}  in-dist={int((y_novel==0).sum())}")
print(f"matched budget q={q*100:.1f}%  (n_esc={n_esc}, n_keep={n_keep})")

acc_d1, _ = L.acc_at_rate(conf, correct, q)
acc_c2, _ = L.acc_at_rate(c2conf, correct, q)
print(f"D1 retained acc = {acc_d1*100:.1f}%   C2 retained acc = {acc_c2*100:.1f}%   "
      f"point diff = {(acc_c2-acc_d1)*100:+.1f}")

# ---------------------------------------------------------------- (a) paired C2 - D1
rng = np.random.default_rng(SEED + 41)
deltas = np.empty(ITERS)
for b in range(ITERS):
    idx = rng.integers(0, n, n)
    cc = correct[idx].astype(float)
    o1 = np.argsort(-conf[idx], kind="stable")
    o2 = np.argsort(-c2conf[idx], kind="stable")
    deltas[b] = float(np.mean(cc[o2[:n_keep]])) - float(np.mean(cc[o1[:n_keep]]))
pa = (float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))
print(f"\n(a) paired C2-D1 at matched budget: {pa[0]*100:+.1f} pts "
      f"[{pa[1]*100:+.1f},{pa[2]*100:+.1f}]   "
      f"P(delta>0)={float(np.mean(deltas>0)):.3f}")

# ---------------------------------------------------------------- (b) TPR - FPR gap
tpr = float(np.mean(flags[y_novel == 1]))
fpr = float(np.mean(flags[y_novel == 0]))
rng = np.random.default_rng(SEED + 42)
gaps = np.empty(ITERS)
for b in range(ITERS):
    idx = rng.integers(0, n, n)
    f_, y_ = flags[idx], y_novel[idx]
    if (y_ == 1).sum() == 0 or (y_ == 0).sum() == 0:
        gaps[b] = np.nan
        continue
    gaps[b] = float(np.mean(f_[y_ == 1])) - float(np.mean(f_[y_ == 0]))
gaps = gaps[np.isfinite(gaps)]
gb = (float(np.mean(gaps)), float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5)))
print(f"(b) TPR={tpr*100:.1f}%  FPR={fpr*100:.1f}%  gap={(tpr-fpr)*100:+.1f} pts "
      f"[{gb[1]*100:+.1f},{gb[2]*100:+.1f}]")

# also the enrichment side: precision vs prevalence
prec = float(np.mean(y_novel[flags]))
prev = float(np.mean(y_novel))
rng = np.random.default_rng(SEED + 43)
enr = np.empty(ITERS)
for b in range(ITERS):
    idx = rng.integers(0, n, n)
    f_, y_ = flags[idx], y_novel[idx]
    enr[b] = (float(np.mean(y_[f_])) - float(np.mean(y_))) if f_.any() else np.nan
enr = enr[np.isfinite(enr)]
print(f"    precision={prec*100:.1f}%  prevalence={prev*100:.1f}%  "
      f"enrichment={(prec-prev)*100:+.1f} pts "
      f"[{np.percentile(enr,2.5)*100:+.1f},{np.percentile(enr,97.5)*100:+.1f}]")

# ---------------------------------------------------------------- (c) C3 distribution
# Per system, draw a size-matched pseudo-novel subset of the retained-type TEST cases,
# 1000 times; pool the three draws and record the pooled flag rate.
C3_DRAWS = 1000
rng = np.random.default_rng(SEED + 44)
per_fold = []
for f in folds:
    dd = f["detectors"][DET]
    a = np.asarray(dd["anom_eval"])
    fl = (a >= dd["tau"])
    n_in = f["n_test_in"]
    per_fold.append((fl[:n_in], f["n_novel"]))

rates = np.empty(C3_DRAWS)
for b in range(C3_DRAWS):
    hit = tot = 0
    for fl_in, k in per_fold:
        m = min(k, len(fl_in))
        sel = rng.choice(len(fl_in), m, replace=False)
        hit += int(fl_in[sel].sum())
        tot += m
    rates[b] = hit / tot
true_rate = tpr
pct = float(np.mean(rates <= true_rate)) * 100
print(f"\n(c) C3 pseudo-novel flag rate over {C3_DRAWS} draws: "
      f"mean {rates.mean()*100:.1f}%  sd {rates.std(ddof=1)*100:.1f}  "
      f"95% range [{np.percentile(rates,2.5)*100:.1f},{np.percentile(rates,97.5)*100:.1f}]")
print(f"    true-novel flag rate {true_rate*100:.1f}% sits at the {pct:.0f}th percentile "
      f"of that distribution")
print(f"    single-draw value (for reference): "
      f"{np.mean([np.mean([fl_in[i] for i in range(len(fl_in))]) for fl_in,_ in per_fold])*100:.1f}% "
      f"(= full in-distribution rate)")

json.dump({
    "fold": FT, "localizer": LOC, "detector": DET, "n": int(n),
    "n_novel": int(y_novel.sum()), "n_indist": int((y_novel == 0).sum()),
    "budget_q": q, "acc_D1": acc_d1, "acc_C2": acc_c2,
    "paired_C2_minus_D1": {"mean": pa[0], "ci": [pa[1], pa[2]],
                           "p_gt_0": float(np.mean(deltas > 0))},
    "tpr": tpr, "fpr": fpr,
    "tpr_minus_fpr": {"mean": gb[0], "ci": [gb[1], gb[2]]},
    "precision": prec, "prevalence": prev,
    "enrichment_ci": [float(np.percentile(enr, 2.5)), float(np.percentile(enr, 97.5))],
    "c3_distribution": {"draws": C3_DRAWS, "mean": float(rates.mean()),
                        "sd": float(rates.std(ddof=1)),
                        "ci": [float(np.percentile(rates, 2.5)),
                               float(np.percentile(rates, 97.5))],
                        "true_novel_rate": true_rate, "percentile_of_true": pct},
}, open(os.path.join(OUT, "paired_loss.json"), "w"), indent=2)
print("\nwrote paired_loss.json")
