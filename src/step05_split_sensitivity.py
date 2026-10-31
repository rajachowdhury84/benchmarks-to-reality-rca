"""
STEP 05: split-sensitivity check for the pre-specified LOSS fold.

Re-draws the 50/50 stratified calibration/test split (and the C3 pseudo-novel subset) under
SEEDS different seeds, everything else identical to step03_lofto.py (BARO, D1, expanded space,
IsolationForest random_state fixed). Reports the spread of the LOSS-fold headline quantities.
This quantifies SPLIT variability only; inference over new incidents remains the case-level
bootstrap of steps 03 and 04, so these draws are not pooled into a CI.
"""
import json, os, sys
import numpy as np
sys.argv = [sys.argv[0], "expanded"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step03_lofto as S
from paths import DATA_DIR, RESULTS_DIR

SEEDS = 30
d = json.load(open(os.path.join(DATA_DIR, "re1_scores.json")))
recs = d["records"]
rows = []
for k in range(SEEDS):
    seed = 7000 + k
    folds = [S.run_fold(recs, s, "loss", "baro", "expanded", np.random.default_rng(seed))
             for s in S.SYSTEMS]
    r = S.analyse(folds, "loss", "D1", "baro", np.random.default_rng(seed + 1))
    # confidence-vs-random delta uses C2; typed delta uses D1
    pse = np.mean([f["detectors"]["D1"]["pseudo_novel_flag_rate"] for f in folds])
    rows.append({"seed": seed, "esc": r["escalation_rate"], "prec": r["novel_precision"],
                 "base": r["novel_base_rate"], "recall": r["novel_recall"],
                 "fpr": r["novel_fpr"], "d_typed": r["delta_vs_random"],
                 "d_c2": r["delta_C2_vs_random"], "pseudo_flag": float(pse),
                 "aurc": r["aurc"], "aurc_c2": r["aurc_C2"], "aurc_rand": r["aurc_random"]})

def summ(key, scale=100):
    v = np.array([x[key] for x in rows], float) * scale
    return {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)),
            "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}

out = {"seeds": SEEDS, "rows": rows,
       "summary": {k: summ(k) for k in ["esc", "prec", "recall", "fpr", "d_typed", "d_c2", "pseudo_flag"]},
       "aurc": {k: summ(k, 1) for k in ["aurc", "aurc_c2", "aurc_rand"]},
       "recall_le_fpr_count": int(sum(x["recall"] <= x["fpr"] + 0.0 for x in rows)),
       "recall_minus_fpr": summ("recall") ,
       "c2_beats_typed_count": int(sum(x["d_c2"] > x["d_typed"] for x in rows)),
       "typed_delta_positive_count": int(sum(x["d_typed"] > 0 for x in rows)),
       "aurc_c2_lt_typed_count": int(sum(x["aurc_c2"] < x["aurc"] for x in rows))}
json.dump(out, open(os.path.join(RESULTS_DIR, "split_sensitivity.json"), "w"), indent=2)
for k, v in out["summary"].items(): print(k, {a: round(b, 2) for a, b in v.items()})
for k, v in out["aurc"].items(): print(k, {a: round(b, 3) for a, b in v.items()})
for k in ["recall_le_fpr_count","c2_beats_typed_count","typed_delta_positive_count","aurc_c2_lt_typed_count"]: print(k, out[k], "/", SEEDS)
