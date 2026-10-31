"""
STEP 10: routing instrumentation and matched-budget deferral on the synthetic generator.

4a. INSTRUMENT THE GENERATOR.
    Count tau_ood trips directly against the held-out-cause label, so that
    routing precision and recall are measured rather than derived from aggregate rates. This is a pure detector measurement: it does not
    depend on the decision procedure or on any bookkeeping convention about whether
    oracle failures count as confidently wrong.

4b. MATCHED-BUDGET DEFERRAL COMPARISON.
    All policies emit a scalar defer score; at budget q the top-q scoring incidents
    are deferred. Deferred incidents are resolved by the SAME simulated oracle the SDL
    uses (gbm_ablation.diagnose_precomputed's OOD branch: succeeds with probability
    0.85, contributes the novel hypothesis when the true cause is the held-out one,
    and otherwise returns the classifier's own top class; on failure the tool abstains
    and the case is scored incorrect). Policies:
      SDL-OOD    IsolationForest anomaly score over the feature vector (the SDL's own
                 Knightian signal)
      Chow       -calibrated max posterior (Chow's rule; the paper's break-even c/v
                 operating point is marked on the curve)
      Cortes     learned rejector: a model of P(classifier wrong | x), the plug-in form
                 of the learning-with-rejection family
      MozSontag  consistent surrogate L_CE of Mozannar & Sontag trained with the oracle
                 as the expert over an augmented (K+1)-class output, swept by an offset
                 on the defer logit
      Random     the null
    Reported as accuracy-versus-coverage curves, not single operating points.
"""
import json
import os
import random
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS_DIR as OUT_DIR

import gbm_ablation as G

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
N_DEFER_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
ORACLE_SUCCESS = 0.85
BUDGETS = np.round(np.arange(0.0, 1.001, 0.05), 3)
BOOT_ITERS = 5000
SEED = 20260908


# --------------------------------------------------------------- 4a
def instrument(n_seeds):
    """Direct counts of tau_ood trips against the held-out-cause label."""
    cols = list(G.SYMPTOMS)
    per_seed = []
    for seed in range(1, n_seeds + 1):
        stream = G.build_stream(seed, n_incidents=400, novel_rate=0.15)
        model = G.train_full_pipeline(seed, cols, alpha=G.EPISTEMIC_ALPHA_DEFAULT,
                                      with_context=False)
        X = np.array([G.feats_to_vec(f, cols) for _, f, _ in stream], dtype=float)
        flag = np.asarray(model.is_ood_batch(X), dtype=bool)
        novel = np.array([c == G.NOVEL_CAUSE for c, _, _ in stream])
        tp = int((flag & novel).sum())
        fp = int((flag & ~novel).sum())
        fn = int((~flag & novel).sum())
        tn = int((~flag & ~novel).sum())
        per_seed.append({
            "seed": seed, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "trip_rate": float(flag.mean()), "novel_rate": float(novel.mean()),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        })
    tot = {k: sum(s[k] for s in per_seed) for k in ("tp", "fp", "fn", "tn")}
    counted = {
        "n_seeds": n_seeds, "counts": tot,
        "precision": tot["tp"] / (tot["tp"] + tot["fp"]),
        "recall": tot["tp"] / (tot["tp"] + tot["fn"]),
        "fpr": tot["fp"] / (tot["fp"] + tot["tn"]),
        "trip_rate": (tot["tp"] + tot["fp"]) / sum(tot.values()),
        "precision_seed_mean": float(np.mean([s["precision"] for s in per_seed])),
        "precision_seed_sd": float(np.std([s["precision"] for s in per_seed], ddof=1)),
        "recall_seed_mean": float(np.mean([s["recall"] for s in per_seed])),
        "recall_seed_sd": float(np.std([s["recall"] for s in per_seed], ddof=1)),
        "per_seed": per_seed,
    }
    return counted


# --------------------------------------------------------------- 4b
def ms_surrogate_fit(X, y_idx, expert_correct, n_classes, seed, epochs=400, lr=0.35):
    """Mozannar & Sontag's consistent L_CE surrogate over K+1 outputs.
       loss = -[ log softmax_y(x) + 1{expert correct} * log softmax_defer(x) ]
    Linear model on the raw binary symptom features plus pairwise interactions,
    fitted by full-batch gradient descent (no sklearn estimator implements L_CE)."""
    rng = np.random.default_rng(seed)
    Z = expand(X)
    n, d = Z.shape
    K = n_classes + 1                      # last column is the defer option
    W = rng.normal(0, 0.01, (d, K))
    b = np.zeros(K)
    ec = expert_correct.astype(float)
    for _ in range(epochs):
        logits = Z @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        Gr = p.copy()
        Gr[np.arange(n), y_idx] -= 1.0
        Gd = p.copy()
        Gd[:, K - 1] -= 1.0
        Gtot = Gr + ec[:, None] * Gd
        W -= lr * (Z.T @ Gtot) / n
        b -= lr * Gtot.mean(axis=0)
    return W, b


def expand(X):
    """Raw features plus pairwise products, so the linear surrogate is not weaker
    than the HGB classifier purely through model class."""
    n, d = X.shape
    cols = [X]
    for i in range(d):
        for j in range(i + 1, d):
            cols.append((X[:, i] * X[:, j])[:, None])
    return np.hstack(cols)


def oracle_outcome(true_cause, fallback_pred, rng):
    """Identical semantics to gbm_ablation.diagnose_precomputed's OOD branch."""
    if rng.random() < ORACLE_SUCCESS:
        return (true_cause == G.NOVEL_CAUSE) or (fallback_pred == true_cause)
    return False   # tool abstains; the case is not correctly diagnosed by the tool


def defer_curves(n_seeds):
    cols = list(G.SYMPTOMS)
    curves = {k: [] for k in ("SDL-OOD", "Chow", "Cortes", "MozSontag", "Random")}
    op_points = []
    for seed in range(1, n_seeds + 1):
        rng = np.random.default_rng(SEED + seed)
        stream = G.build_stream(seed, n_incidents=400, novel_rate=0.15)
        model = G.train_full_pipeline(seed, cols, alpha=G.EPISTEMIC_ALPHA_DEFAULT,
                                      with_context=False)
        causes = model.causes
        X = np.array([G.feats_to_vec(f, cols) for _, f, _ in stream], dtype=float)
        truth = [c for c, _, _ in stream]
        proba = model.calibrated_proba(X)
        top_idx = proba.argmax(axis=1)
        pred = [causes[i] for i in top_idx]
        base_correct = np.array([p == t for p, t in zip(pred, truth)], dtype=float)
        anom = -model.iso_forest.score_samples(X)
        maxp = proba.max(axis=1)

        # oracle outcome, drawn once per incident and shared by every policy so the
        # comparison is paired at case level
        orc = np.array([oracle_outcome(t, p, rng) for t, p in zip(truth, pred)], dtype=float)

        # --- training material for the two learned policies: an independent stream
        tr_stream = G.build_stream(seed + 100_000, n_incidents=400, novel_rate=0.15)
        Xtr = np.array([G.feats_to_vec(f, cols) for _, f, _ in tr_stream], dtype=float)
        ttr = [c for c, _, _ in tr_stream]
        ptr = model.calibrated_proba(Xtr)
        predtr = [causes[i] for i in ptr.argmax(axis=1)]
        wrong_tr = np.array([p != t for p, t in zip(predtr, ttr)], dtype=int)
        orc_tr = np.array([oracle_outcome(t, p, rng) for t, p in zip(ttr, predtr)], dtype=int)

        # Cortes-style learned rejector: model P(classifier wrong | x)
        if len(set(wrong_tr.tolist())) > 1:
            rej = HistGradientBoostingClassifier(max_iter=150, random_state=seed)
            rej.fit(Xtr, wrong_tr)
            cortes = rej.predict_proba(X)[:, 1]
        else:
            cortes = np.zeros(len(X))

        # Mozannar-Sontag L_CE with the oracle as the expert
        cause_ix = {c: i for i, c in enumerate(causes)}
        y_tr = np.array([cause_ix.get(t, 0) for t in ttr])
        W, b = ms_surrogate_fit(Xtr, y_tr, orc_tr, len(causes), seed)
        lg = expand(X) @ W + b
        ms_score = lg[:, -1] - lg[:, :-1].max(axis=1)

        scores = {"SDL-OOD": anom, "Chow": -maxp, "Cortes": cortes,
                  "MozSontag": ms_score, "Random": rng.random(len(X))}
        n = len(X)
        for name, s in scores.items():
            order = np.argsort(-s, kind="stable")
            acc = []
            for q in BUDGETS:
                k = int(round(q * n))
                dset = np.zeros(n, dtype=bool)
                dset[order[:k]] = True
                a = float(np.mean(np.where(dset, orc, base_correct)))
                acc.append(a)
            curves[name].append(acc)
        # the SDL's own realised OOD operating point for this seed
        flag = np.asarray(model.is_ood_batch(X), dtype=bool)
        op_points.append({
            "seed": seed, "q": float(flag.mean()),
            "acc": float(np.mean(np.where(flag, orc, base_correct))),
            "base_acc": float(base_correct.mean()),
        })
    out = {"budgets": BUDGETS.tolist(), "n_seeds": n_seeds,
           "oracle_success": ORACLE_SUCCESS, "sdl_ood_operating_point": op_points}
    for name, rows in curves.items():
        arr = np.asarray(rows)
        out[name] = {"mean": arr.mean(axis=0).tolist(),
                     "sd": arr.std(axis=0, ddof=1).tolist()}
    # paired seed-level differences against SDL-OOD at each budget
    sdl = np.asarray(curves["SDL-OOD"])
    out["paired_vs_SDL"] = {}
    for name in ("Chow", "Cortes", "MozSontag", "Random"):
        d = np.asarray(curves[name]) - sdl
        out["paired_vs_SDL"][name] = {
            "mean": d.mean(axis=0).tolist(),
            "ci_lo": np.percentile(d, 2.5, axis=0).tolist(),
            "ci_hi": np.percentile(d, 97.5, axis=0).tolist(),
        }
    return out


def main():
    print(f"=== 4a: direct tau_ood instrumentation, {N_SEEDS} seeds ===")
    counted = instrument(N_SEEDS)
    c = counted["counts"]
    print(f"  trips: TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']} "
          f"(N={sum(c.values())})")
    print(f"  COUNTED precision {counted['precision']*100:.1f}%  "
          f"recall {counted['recall']*100:.1f}%  FPR {counted['fpr']*100:.1f}%  "
          f"trip rate {counted['trip_rate']*100:.1f}%")
    print(f"  per-seed mean+-sd: precision {counted['precision_seed_mean']*100:.1f}"
          f"+-{counted['precision_seed_sd']*100:.1f}, "
          f"recall {counted['recall_seed_mean']*100:.1f}"
          f"+-{counted['recall_seed_sd']*100:.1f}")

    print(f"\n=== 4b: matched-budget deferral comparison, {N_DEFER_SEEDS} seeds ===")
    dc = defer_curves(N_DEFER_SEEDS)
    bs = dc["budgets"]
    hdr = "  budget " + "".join(f"{b*100:7.0f}%" for b in bs[:11])
    print(hdr)
    for name in ("SDL-OOD", "Chow", "Cortes", "MozSontag", "Random"):
        print(f"  {name:10s}" + "".join(f"{v*100:7.1f}" for v in dc[name]["mean"][:11]))
    q = float(np.mean([p["q"] for p in dc["sdl_ood_operating_point"]]))
    a = float(np.mean([p["acc"] for p in dc["sdl_ood_operating_point"]]))
    print(f"  SDL's realised OOD operating point: q={q*100:.1f}%  acc={a*100:.1f}%")

    with open(os.path.join(OUT_DIR, "deferral.json"), "w") as f:
        json.dump({"instrumentation": counted, "deferral": dc}, f, indent=2)
    print("\nWrote deferral.json")


if __name__ == "__main__":
    main()
