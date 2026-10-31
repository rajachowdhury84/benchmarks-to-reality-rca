"""
STEP 03: Leave-one-fault-type-out (LOFTO) distribution-shift protocol on RCAEval RE1.

Design
------
3 systems x 5 fault types = 15 folds. For fold (s, f): hold out fault type f in system s.
Within the four retained types (100 cases) take a single fixed 50/50 stratified
calibration/test split (seed below; NEVER resampled -- split-count is not an inferential
sample size). The novelty detector and tau_ood are fitted on the calibration half only.
The evaluation set is the retained-type test cases (50) plus all 25 held-out-type cases,
which are the novel class.

Detector variants (pre-registered ordering; D1 is primary)
---------------------------------------------------------
D1  density/novelty model over the localizer's per-service score vector
D2  score-shape features: max, top1-top2 gap, top1-top3 gap, entropy, dispersion, ...
D3  density over summary statistics of the metric window itself, independent of the
    localizer (most defensible because it inherits none of the localizer's blind spots,
    and the most likely to be uninformative)
All three use IsolationForest as the estimator, mirroring the synthetic arm.

Controls (all mandatory)
------------------------
C1  matched-coverage RANDOM abstention -- abstain at the same rate q uniformly at random.
    This is the null: retained accuracy equals forced accuracy in expectation.
C2  plain confidence threshold -- Platt-scaled p(top-1 correct) from the localizer's own
    max score, no density model, matched budget.
C3  in-distribution reference -- same protocol with a randomly held-out subset of
    RETAINED-type cases as the pseudo-novel class. If the detector flags these at a
    similar rate it is detecting nothing about novelty.

Calibration map is Platt/logistic, not isotonic: ~50 calibration cases per fold is far
too few for isotonic regression, which would interpolate the calibration set.

Reported
--------
risk-coverage curve + AURC; COUNTED novelty precision/recall against the holdout label;
confident-subset accuracy against the matched-random null with delta and CI; paired
case-level bootstrap CIs (5000 iterations, seed fixed and reported); per-fault-type
cells pooled across systems (75 novel cases) as the primary unit, per system secondary.
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, RESULTS_DIR as OUT_DIR
SEED = 20260908
BOOT_ITERS = 5000
SYSTEMS = ["RE1-OB", "RE1-SS", "RE1-TT"]
FAULT_TYPES = ["cpu", "mem", "disk", "delay", "loss"]
FOLD_ORDER = ["loss"] + [f for f in FAULT_TYPES if f != "loss"]  # LOSS pre-registered first
LOCALIZERS = ["baro", "nsigma", "zscore"]
PRIMARY_LOC = "baro"
SPACE = sys.argv[1] if len(sys.argv) > 1 else "expanded"   # primary candidate space
NEG = -1e9


# ----------------------------------------------------------------- case features
def ranked(rec, loc, space):
    cands = rec["services_expanded"] if space == "expanded" else rec["services_restricted"]
    sc = rec[loc]
    pairs = [(s, sc.get(s, NEG)) for s in cands]
    pairs.sort(key=lambda kv: (-kv[1], kv[0]))
    return pairs


def score_vector(rec, loc, space):
    cands = rec["services_expanded"] if space == "expanded" else rec["services_restricted"]
    v = np.array([rec[loc].get(s, NEG) for s in cands], dtype=float)
    return np.clip(v, -50.0, 1e6)


def shape_features(rec, loc, space):
    v = np.sort(score_vector(rec, loc, space))[::-1]
    v = v[np.isfinite(v)]
    top1 = v[0]
    top2 = v[1] if len(v) > 1 else v[0]
    top3 = v[2] if len(v) > 2 else top2
    pos = np.clip(v, 1e-9, None)
    p = pos / pos.sum()
    ent = float(-(p * np.log(p + 1e-12)).sum())
    med = float(np.median(v))
    return [
        float(np.log1p(max(top1, 0.0))), float(top1 - top2), float(top1 - top3),
        ent, float(np.std(v)), float(np.mean(v)), med,
        float(np.log1p(max(top1 - med, 0.0))),
    ]


def is_correct(rec, loc, space):
    return int(ranked(rec, loc, space)[0][0] == rec["true_service"])


def conf_signal(rec, loc, space):
    """Scalar confidence signal for the plain-threshold control C2."""
    v = np.sort(score_vector(rec, loc, space))[::-1]
    top1 = v[0]
    top2 = v[1] if len(v) > 1 else v[0]
    return [float(np.log1p(max(top1, 0.0))), float(top1 - top2)]


# ----------------------------------------------------------------- metrics
def selective_curve(conf, correct):
    """Risk-coverage curve: sort by confidence descending, risk at every prefix.
    Returns (coverage[], risk[], AURC)."""
    order = np.argsort(-np.asarray(conf, dtype=float), kind="stable")
    c = np.asarray(correct, dtype=float)[order]
    n = len(c)
    cum = np.cumsum(c)
    k = np.arange(1, n + 1)
    risk = 1.0 - cum / k
    cov = k / n
    return cov, risk, float(np.mean(risk))


def acc_at_rate(conf, correct, q):
    """Accuracy on the retained (non-escalated) subset when escalating the q lowest-
    confidence cases; also returns the realised escalation count."""
    n = len(correct)
    n_esc = int(round(q * n))
    n_esc = min(max(n_esc, 0), n - 1)
    order = np.argsort(-np.asarray(conf, dtype=float), kind="stable")
    keep = order[: n - n_esc]
    return float(np.mean(np.asarray(correct, dtype=float)[keep])), n_esc


def boot_delta(conf, correct, q, rng, iters=BOOT_ITERS):
    """Paired case-level bootstrap of (confident-subset accuracy) - (matched-coverage
    random abstention accuracy). Resamples CASES, never splits."""
    correct = np.asarray(correct, dtype=float)
    conf = np.asarray(conf, dtype=float)
    n = len(correct)
    n_esc = min(max(int(round(q * n)), 0), n - 1)
    n_keep = n - n_esc
    deltas = np.empty(iters)
    for b in range(iters):
        idx = rng.integers(0, n, n)
        cc, ff = correct[idx], conf[idx]
        order = np.argsort(-ff, kind="stable")
        sel = float(np.mean(cc[order[:n_keep]]))
        rnd_idx = rng.choice(n, n_keep, replace=False)
        rnd = float(np.mean(cc[rnd_idx]))
        deltas[b] = sel - rnd
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), \
        float(np.percentile(deltas, 97.5))


# ----------------------------------------------------------------- one fold
def run_fold(recs, system, held_out, loc, space, rng):
    sys_recs = [r for r in recs if r["system"] == system]
    retained = [r for r in sys_recs if r["fault_type"] != held_out]
    novel = [r for r in sys_recs if r["fault_type"] == held_out]

    # fixed stratified 50/50 calibration/test split of the retained types
    calib, test_in = [], []
    for ft in FAULT_TYPES:
        if ft == held_out:
            continue
        grp = [r for r in retained if r["fault_type"] == ft]
        idx = rng.permutation(len(grp))
        half = len(grp) // 2
        calib += [grp[i] for i in idx[:half]]
        test_in += [grp[i] for i in idx[half:]]

    # in-distribution reference (C3): a random subset of the retained-type TEST cases,
    # matched in size to the novel set, relabelled as pseudo-novel; the detector is
    # fitted identically and must not flag them at the novel rate.
    pseudo_idx = rng.choice(len(test_in), min(len(novel), len(test_in) - 5), replace=False)
    pseudo_set = set(pseudo_idx.tolist())

    feats = {
        "D1": lambda r: score_vector(r, loc, space),
        "D2": lambda r: np.asarray(shape_features(r, loc, space)),
        "D3": lambda r: np.asarray(r["window_features"], dtype=float),
    }
    evalset = test_in + novel
    y_novel = np.array([0] * len(test_in) + [1] * len(novel))
    correct = np.array([is_correct(r, loc, space) for r in evalset])

    out = {"system": system, "held_out": held_out, "localizer": loc, "space": space,
           "n_calib": len(calib), "n_test_in": len(test_in), "n_novel": len(novel),
           "forced_acc": float(np.mean(correct)),
           "forced_acc_in": float(np.mean(correct[y_novel == 0])),
           "forced_acc_novel": float(np.mean(correct[y_novel == 1])),
           "y_novel": y_novel.tolist(), "correct": correct.tolist(),
           "detectors": {}}

    for name, fn in feats.items():
        Xc = np.vstack([fn(r) for r in calib])
        Xe = np.vstack([fn(r) for r in evalset])
        sc = StandardScaler().fit(Xc)
        iso = IsolationForest(n_estimators=300, random_state=SEED, contamination="auto")
        iso.fit(sc.transform(Xc))
        a_cal = -iso.score_samples(sc.transform(Xc))     # higher = more anomalous
        a_ev = -iso.score_samples(sc.transform(Xe))
        conf = -a_ev                                      # higher = more confident
        cov, risk, aurc = selective_curve(conf, correct)
        # pre-registered operating point: tau_ood at the 90th pct of calibration anomaly
        tau = float(np.percentile(a_cal, 90))
        flag = a_ev >= tau
        q = float(np.mean(flag))
        prec = float(np.mean(y_novel[flag])) if flag.any() else float("nan")
        rec_ = float(np.mean(flag[y_novel == 1])) if (y_novel == 1).any() else float("nan")
        fpr = float(np.mean(flag[y_novel == 0]))
        # C3 in-distribution reference: flag rate on pseudo-novel retained cases
        pseudo_flags = [flag[i] for i in range(len(test_in)) if i in pseudo_set]
        out["detectors"][name] = {
            "anom_eval": a_ev.tolist(), "conf": conf.tolist(),
            "aurc": aurc, "tau": tau, "q": q,
            "novel_precision": prec, "novel_recall": rec_, "novel_fpr": fpr,
            "pseudo_novel_flag_rate": float(np.mean(pseudo_flags)) if pseudo_flags else float("nan"),
            "acc_confident": acc_at_rate(conf, correct, q)[0],
        }

    # C2 plain confidence threshold (Platt-scaled max score, no density model)
    Xc = np.vstack([conf_signal(r, loc, space) for r in calib])
    yc = np.array([is_correct(r, loc, space) for r in calib])
    Xe = np.vstack([conf_signal(r, loc, space) for r in evalset])
    sc = StandardScaler().fit(Xc)
    if len(set(yc.tolist())) < 2:
        p_ev = np.ones(len(evalset)) * float(yc.mean())
    else:
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xc), yc)
        p_ev = lr.predict_proba(sc.transform(Xe))[:, 1]
    cov, risk, aurc_c2 = selective_curve(p_ev, correct)
    out["C2_plain_conf"] = {"conf": p_ev.tolist(), "aurc": aurc_c2}
    return out


# ----------------------------------------------------------------- pooling
def pool_fault_type(folds, ft, det, loc):
    """Primary unit: pool the three systems' folds for held-out type ft (75 novel cases)."""
    sel = [f for f in folds if f["held_out"] == ft and f["localizer"] == loc]
    if not sel:
        return None
    conf, correct, y_novel, flags, c2conf = [], [], [], [], []
    for f in sel:
        d = f["detectors"][det]
        a = np.asarray(d["anom_eval"])
        # per-fold tau (scales are not comparable across systems), pooled outcomes
        flags += (a >= d["tau"]).tolist()
        # rank-normalise confidence within fold so pooling is scale-free
        c = np.asarray(d["conf"])
        r = np.argsort(np.argsort(c)) / max(len(c) - 1, 1)
        conf += r.tolist()
        p = np.asarray(f["C2_plain_conf"]["conf"])
        rp = np.argsort(np.argsort(p)) / max(len(p) - 1, 1)
        c2conf += rp.tolist()
        correct += f["correct"]
        y_novel += f["y_novel"]
    return (np.asarray(conf), np.asarray(correct), np.asarray(y_novel),
            np.asarray(flags), np.asarray(c2conf))


def analyse(folds, ft, det, loc, rng):
    conf, correct, y_novel, flags, c2conf = pool_fault_type(folds, ft, det, loc)
    n = len(correct)
    q = float(np.mean(flags))
    forced = float(np.mean(correct))
    acc_sel, _ = acc_at_rate(conf, correct, q)
    d_mean, d_lo, d_hi = boot_delta(conf, correct, q, rng)
    acc_c2, _ = acc_at_rate(c2conf, correct, q)
    dc_mean, dc_lo, dc_hi = boot_delta(c2conf, correct, q, rng)
    prec = float(np.mean(y_novel[flags])) if flags.any() else float("nan")
    recall = float(np.mean(flags[y_novel == 1]))
    fpr = float(np.mean(flags[y_novel == 0]))
    _, _, aurc = selective_curve(conf, correct)
    _, _, aurc_c2 = selective_curve(c2conf, correct)
    _, _, aurc_rand = selective_curve(rng.random(n), correct)
    return {
        "fault_type": ft, "detector": det, "localizer": loc, "n": n,
        "n_novel": int((y_novel == 1).sum()), "escalation_rate": q,
        "forced_acc": forced, "acc_confident": acc_sel,
        "delta_vs_random": d_mean, "delta_ci": [d_lo, d_hi],
        "acc_confident_C2": acc_c2, "delta_C2_vs_random": dc_mean,
        "delta_C2_ci": [dc_lo, dc_hi],
        "novel_precision": prec, "novel_recall": recall, "novel_fpr": fpr,
        "novel_base_rate": float(np.mean(y_novel)),
        "aurc": aurc, "aurc_C2": aurc_c2, "aurc_random": aurc_rand,
    }


def main():
    d = json.load(open(os.path.join(DATA_DIR, "re1_scores.json")))
    recs = d["records"]
    rng = np.random.default_rng(SEED)

    folds = []
    for loc in LOCALIZERS:
        for ft in FOLD_ORDER:
            for s in SYSTEMS:
                folds.append(run_fold(recs, s, ft, loc, SPACE, np.random.default_rng(SEED)))

    results = {"seed": SEED, "boot_iters": BOOT_ITERS, "space": SPACE,
               "primary_localizer": PRIMARY_LOC, "primary_detector": "D1",
               "pooled": [], "per_system": [], "folds": []}

    print(f"=== LOFTO on RE1, candidate space = {SPACE}, seed = {SEED} ===")
    print("Primary unit: fault type pooled across the three systems (75 novel cases).\n")
    for loc in LOCALIZERS:
        for det in ["D1", "D2", "D3"]:
            for ft in FOLD_ORDER:
                r = analyse(folds, ft, det, loc, np.random.default_rng(SEED + 1))
                results["pooled"].append(r)
                if loc == PRIMARY_LOC:
                    tag = "PRIMARY" if det == "D1" else "secondary"
                    print(f"[{tag:9s}] {loc:6s} {det} holdout={ft:5s} n={r['n']:3d} "
                          f"esc={r['escalation_rate']*100:5.1f}%  "
                          f"forced={r['forced_acc']*100:5.1f}%  "
                          f"conf={r['acc_confident']*100:5.1f}%  "
                          f"delta={r['delta_vs_random']*100:+5.1f} "
                          f"[{r['delta_ci'][0]*100:+5.1f},{r['delta_ci'][1]*100:+5.1f}]  "
                          f"P={r['novel_precision']*100:4.1f}% R={r['novel_recall']*100:5.1f}% "
                          f"FPR={r['novel_fpr']*100:4.1f}%  AURC={r['aurc']:.3f} "
                          f"(rand {r['aurc_random']:.3f}, C2 {r['aurc_C2']:.3f})")
            if loc == PRIMARY_LOC:
                print()

    # per-system secondary cells (underpowered: 25 novel cases each)
    for f in folds:
        for det in ["D1", "D2", "D3"]:
            dd = f["detectors"][det]
            results["per_system"].append({
                "system": f["system"], "held_out": f["held_out"],
                "localizer": f["localizer"], "detector": det,
                "n_novel": f["n_novel"], "escalation_rate": dd["q"],
                "forced_acc": f["forced_acc"], "acc_confident": dd["acc_confident"],
                "novel_precision": dd["novel_precision"],
                "novel_recall": dd["novel_recall"], "novel_fpr": dd["novel_fpr"],
                "pseudo_novel_flag_rate": dd["pseudo_novel_flag_rate"],
                "aurc": dd["aurc"],
            })
    for f in folds:
        f.pop("detectors_raw", None)
    results["folds"] = [{k: v for k, v in f.items() if k != "detectors"} for f in folds]

    with open(os.path.join(OUT_DIR, f"lofto_{SPACE}.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # C3 in-distribution reference summary (primary localizer / primary detector)
    print("=== C3 in-distribution reference (primary localizer/detector) ===")
    for ft in FOLD_ORDER:
        rows = [r for r in results["per_system"] if r["held_out"] == ft
                and r["localizer"] == PRIMARY_LOC and r["detector"] == "D1"]
        nov = np.mean([r["novel_recall"] for r in rows])
        pse = np.mean([r["pseudo_novel_flag_rate"] for r in rows])
        print(f"  holdout={ft:5s}  flag rate on true-novel {nov*100:5.1f}%  "
              f"on pseudo-novel (in-distribution) {pse*100:5.1f}%")
    print(f"\nWrote lofto_{SPACE}.json")


if __name__ == "__main__":
    main()
