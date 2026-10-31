"""
Synthetic generator and Structured Diagnostic Loop (SDL) ablation study.

Pattern-matcher: scikit-learn's HistGradientBoostingClassifier (HGB), a
histogram-based gradient-boosting model.

Knightian / OOD detector: an IsolationForest fitted on the known-cause
training features; tau_ood is a percentile of its anomaly scores on
genuine training data.

Epistemic escalation: split conformal prediction (Sadinle et al. 2019,
"least ambiguous set-valued classifier" / LAC). Nonconformity scores
s_i = 1 - p_hat(y_i | x_i) are computed on a calibration split from
isotonic-calibrated probabilities, and their (1 - alpha)-quantile (with
the standard finite-sample correction) is the set-membership threshold
qhat. A case escalates to the epistemic branch whenever its conformal
prediction set contains more than one candidate cause.

Institutional memory (Full SDL only): periodic batch retraining every
RETRAIN_EVERY resolved incidents.

The world model (causes, symptoms, symptom probabilities and
context-signal generation) is defined below; the six ablation conditions
are listed in CONDITIONS.

Imported by steps 08-12. For the paper's ablation table run
src/step08_synthetic_ablation.py; for a quick check:
    python src/gbm_ablation.py --seeds 5 --n 400
"""
import argparse
import copy
import json
import math
import random
import time
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.isotonic import IsotonicRegression

# ----------------------------------------------------------------
# World model
# ----------------------------------------------------------------
KNOWN_CAUSES = [
    "query_contention",
    "network_degradation",
    "memory_leak",
    "misconfigured_connection_pool",
    "hardware_fault",
]
NOVEL_CAUSE = "novel_cascading_interaction"
ALL_CAUSES = KNOWN_CAUSES + [NOVEL_CAUSE]

SYMPTOMS = [
    "high_latency", "high_error_rate", "cpu_spike", "memory_growth",
    "conn_pool_exhaustion", "packet_loss", "retry_storm",
]

TRUE_SYMPTOM_PROB = {
    "query_contention":              dict(high_latency=.85, high_error_rate=.30, cpu_spike=.55, memory_growth=.10, conn_pool_exhaustion=.35, packet_loss=.05, retry_storm=.20),
    "network_degradation":           dict(high_latency=.80, high_error_rate=.55, cpu_spike=.15, memory_growth=.05, conn_pool_exhaustion=.15, packet_loss=.70, retry_storm=.45),
    "memory_leak":                   dict(high_latency=.75, high_error_rate=.35, cpu_spike=.40, memory_growth=.90, conn_pool_exhaustion=.20, packet_loss=.05, retry_storm=.15),
    "misconfigured_connection_pool": dict(high_latency=.82, high_error_rate=.45, cpu_spike=.30, memory_growth=.10, conn_pool_exhaustion=.88, packet_loss=.05, retry_storm=.30),
    "hardware_fault":                dict(high_latency=.50, high_error_rate=.50, cpu_spike=.50, memory_growth=.50, conn_pool_exhaustion=.10, packet_loss=.50, retry_storm=.10),
    "novel_cascading_interaction":   dict(high_latency=.90, high_error_rate=.85, cpu_spike=.20, memory_growth=.80, conn_pool_exhaustion=.20, packet_loss=.75, retry_storm=.85),
}

EPISTEMIC_ALPHA_DEFAULT = 0.20   # conformal target: 1 - alpha = 80% marginal coverage
CONFIDENT_ESCALATE_ABSTAIN = 0.55
OOD_PERCENTILE_DEFAULT = 0.10
RETRAIN_EVERY = 50


def sample_incident_pure(rng, true_cause):
    feats = {}
    for s in SYMPTOMS:
        p = TRUE_SYMPTOM_PROB[true_cause][s]
        feats[s] = 1 if rng.random() < p else 0
    return feats


def sample_context_pure(rng, true_cause):
    if true_cause == "misconfigured_connection_pool":
        return rng.random() < 0.90
    return rng.random() < 0.08


def feats_to_vec(feats, cols):
    return [feats[c] for c in cols]


def build_stream(seed, n_incidents=400, novel_rate=0.15):
    rng = random.Random(seed)
    stream = []
    for _ in range(n_incidents):
        if rng.random() < novel_rate:
            cause = NOVEL_CAUSE
        else:
            cause = rng.choices(KNOWN_CAUSES, weights=[1] * 5)[0]
        feats = sample_incident_pure(rng, cause)
        ctx = sample_context_pure(rng, cause)
        stream.append((cause, feats, ctx))
    return stream


# ----------------------------------------------------------------
# GBM pattern-matcher + isotonic calibration + IsolationForest OOD +
# split-conformal (LAC) epistemic-set construction
# ----------------------------------------------------------------
class GBMModel:
    def __init__(self, causes, feature_cols):
        self.causes = causes
        self.feature_cols = feature_cols
        self.clf = None
        self.calibrator = None       # isotonic: raw max-prob -> calibrated max-prob
        self.iso_forest = None
        self.ood_threshold = None    # percentile of anomaly score on training data
        self.qhat = None             # conformal quantile (LAC score threshold)

    def _raw_proba(self, X):
        return self.clf.predict_proba(X)

    def fit_classifier(self, X, y):
        self.clf = HistGradientBoostingClassifier(
            max_iter=60, learning_rate=0.15, max_depth=4, random_state=0,
        )
        self.clf.fit(X, y)
        # predict_proba's columns follow clf.classes_ (alphabetical), not
        # necessarily the order self.causes was constructed in -- realign
        # self.causes to match so every downstream index is consistent.
        self.causes = list(self.clf.classes_)

    def fit_calibration(self, X_cal, y_cal):
        """Scalar isotonic calibration: maps the model's raw max-class
        probability to an empirical-accuracy-calibrated probability.
        Simple multiclass 'confidence calibration' (Guo et al. 2017
        style), chosen over per-class Platt scaling for data efficiency
        given the modest per-trial calibration-set size."""
        proba = self._raw_proba(X_cal)
        pred = np.argmax(proba, axis=1)
        raw_conf = proba[np.arange(len(proba)), pred]
        correct = (np.array([self.causes.index(y) for y in y_cal]) == pred).astype(float)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_conf, correct)
        self.calibrator = iso

    def calibrated_proba(self, X):
        """Returns per-class probabilities where the max-class entry has
        been isotonic-calibrated and the remaining mass is redistributed
        proportionally across the other classes (keeps a valid simplex
        for the conformal LAC score below)."""
        proba = self._raw_proba(X).copy()
        pred = np.argmax(proba, axis=1)
        raw_conf = proba[np.arange(len(proba)), pred]
        cal_conf = np.clip(self.calibrator.predict(raw_conf), 1e-6, 1 - 1e-6)
        out = np.zeros_like(proba)
        for i in range(len(proba)):
            rest_mass_raw = 1.0 - proba[i, pred[i]]
            rest_mass_new = 1.0 - cal_conf[i]
            out[i] = proba[i] * (rest_mass_new / rest_mass_raw) if rest_mass_raw > 1e-9 else 0.0
            out[i, pred[i]] = cal_conf[i]
        return out

    def fit_ood(self, X_train, percentile=OOD_PERCENTILE_DEFAULT):
        self.iso_forest = IsolationForest(n_estimators=100, random_state=0, contamination="auto")
        self.iso_forest.fit(X_train)
        scores = self.iso_forest.score_samples(X_train)  # higher = more normal
        scores_sorted = np.sort(scores)
        idx = max(0, int(len(scores_sorted) * percentile) - 1)
        self.ood_threshold = scores_sorted[idx]

    def is_ood(self, x_row):
        s = self.iso_forest.score_samples(x_row.reshape(1, -1))[0]
        return s < self.ood_threshold

    def is_ood_batch(self, X):
        scores = self.iso_forest.score_samples(X)
        return scores < self.ood_threshold

    def fit_conformal(self, X_cal, y_cal, alpha=EPISTEMIC_ALPHA_DEFAULT):
        proba = self.calibrated_proba(X_cal)
        y_idx = np.array([self.causes.index(y) for y in y_cal])
        scores = 1.0 - proba[np.arange(len(proba)), y_idx]
        n = len(scores)
        q_level = math.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)
        self.qhat = float(np.quantile(scores, q_level, method="higher"))

    def conformal_set(self, proba_row):
        """LAC prediction set: {c : p_hat(c|x) >= 1 - qhat}."""
        thresh = 1.0 - self.qhat
        members = [self.causes[i] for i in range(len(self.causes)) if proba_row[i] >= thresh]
        if not members:
            members = [self.causes[int(np.argmax(proba_row))]]
        return members


def train_full_pipeline(seed, feature_cols, n_per_cause=120, n_cal=150,
                         ood_percentile=OOD_PERCENTILE_DEFAULT, alpha=EPISTEMIC_ALPHA_DEFAULT,
                         with_context=False):
    rng = random.Random(seed + 999_999)
    X_train, y_train = [], []
    for c in KNOWN_CAUSES:
        for _ in range(n_per_cause):
            feats = dict(sample_incident_pure(rng, c))
            if with_context:
                feats["ctx_recent_deploy"] = 1 if sample_context_pure(rng, c) else 0
            X_train.append(feats_to_vec(feats, feature_cols))
            y_train.append(c)
    X_train = np.array(X_train, dtype=float)

    model = GBMModel(KNOWN_CAUSES, feature_cols)
    model.fit_classifier(X_train, y_train)
    model.fit_ood(X_train, percentile=ood_percentile)

    X_cal, y_cal = [], []
    for c in KNOWN_CAUSES:
        for _ in range(n_cal // len(KNOWN_CAUSES)):
            feats = dict(sample_incident_pure(rng, c))
            if with_context:
                feats["ctx_recent_deploy"] = 1 if sample_context_pure(rng, c) else 0
            X_cal.append(feats_to_vec(feats, feature_cols))
            y_cal.append(c)
    X_cal = np.array(X_cal, dtype=float)
    model.fit_calibration(X_cal, y_cal)
    model.fit_conformal(X_cal, y_cal, alpha=alpha)
    return model


# ----------------------------------------------------------------
# Decision procedure: OOD (IsolationForest) -> conformal set-size check
# -> optional human-context Bayesian update -> abstain/return.
#
# Precomputed-array form: proba_row/ood_flag are computed in ONE batched
# sklearn call per model segment in run_condition (rather than one
# predict_proba/score_samples call per incident), which is what makes
# the 100-seed x 6-condition run tractable.
# ----------------------------------------------------------------
def diagnose_precomputed(condition, causes, proba_row, ood_flag, qhat, ctx_true, true_cause,
                          dec_rng, log, oracle_noise=0.0, oracle_base_success=0.85):
    if condition in ("baseline", "baseline_ctx"):
        top_idx = int(np.argmax(proba_row))
        return causes[top_idx], float(proba_row[top_idx]), False, False

    use_ood = condition in ("ood_only", "full_sdl", "abstain_only")
    use_epistemic = condition in ("epistemic_only", "full_sdl", "abstain_only")
    use_context = condition in ("epistemic_only", "full_sdl")

    escalated, abstained = False, False

    if use_ood and ood_flag:
        escalated = True
        log["knightian_flagged"] += 1
        success_p = oracle_base_success * (1 - oracle_noise)
        if dec_rng.random() < success_p:
            fallback = causes[int(np.argmax(proba_row))]
            final = NOVEL_CAUSE if true_cause == NOVEL_CAUSE else fallback
            return final, 0.85, escalated, abstained
        else:
            abstained = True
            log["abstained"] += 1
            return None, 0.0, escalated, abstained

    top_idx = int(np.argmax(proba_row))
    top_cause, top_p = causes[top_idx], float(proba_row[top_idx])

    if use_epistemic:
        thresh = 1.0 - qhat
        member_idx = [i for i in range(len(causes)) if proba_row[i] >= thresh]
        if not member_idx:
            member_idx = [top_idx]
    else:
        member_idx = [top_idx]

    if len(member_idx) > 1 and use_epistemic:
        escalated = True
        member_idx.sort(key=lambda i: -proba_row[i])
        i1, i2 = member_idx[0], member_idx[1]
        c1, p1 = causes[i1], float(proba_row[i1])
        c2, p2 = causes[i2], float(proba_row[i2])

        if use_context:
            ctx = ctx_true
            if dec_rng.random() < oracle_noise:
                ctx = not ctx_true

            def ctx_likelihood(cause):
                if cause == "misconfigured_connection_pool":
                    return 0.75 if ctx else 0.30
                return 0.25 if ctx else 0.75

            adj = {c1: p1 * ctx_likelihood(c1), c2: p2 * ctx_likelihood(c2)}
            z = sum(adj.values())
            adj = {k: v / z for k, v in adj.items()}
            new_top = max(adj, key=adj.get)
            conf = adj[new_top]
        else:
            new_top, conf = c1, p1

        if conf < CONFIDENT_ESCALATE_ABSTAIN:
            abstained = True
            log["abstained"] += 1
            return None, conf, escalated, abstained
        return new_top, conf, escalated, abstained

    return top_cause, top_p, escalated, abstained


def run_condition(condition, stream, model, model_ctx, feature_cols, feature_cols_ctx, seed,
                   oracle_noise=0.0, alpha=EPISTEMIC_ALPHA_DEFAULT, oracle_base_success=0.85,
                   memory_update=True, n_per_cause=120):
    dec_rng = random.Random(seed + 555_555)
    log = defaultdict(int)
    rows = []
    active_model = model_ctx if condition == "baseline_ctx" else model
    cols = feature_cols_ctx if condition == "baseline_ctx" else feature_cols

    X_all = np.array([feats_to_vec(f, cols) if condition != "baseline_ctx"
                       else feats_to_vec({**f, "ctx_recent_deploy": 1 if ctx else 0}, cols)
                       for c, f, ctx in stream], dtype=float)

    if condition != "full_sdl":
        proba_all = active_model.calibrated_proba(X_all)
        ood_all = active_model.is_ood_batch(X_all) if condition in ("ood_only", "abstain_only") else np.zeros(len(stream), dtype=bool)
        qhat = active_model.qhat
        causes = active_model.causes

        for i, (true_cause, feats, ctx_true) in enumerate(stream):
            pred, conf, escalated, abstained = diagnose_precomputed(
                condition, causes, proba_all[i], bool(ood_all[i]), qhat, ctx_true, true_cause,
                dec_rng, log, oracle_noise=oracle_noise, oracle_base_success=oracle_base_success,
            )
            correct = (pred == true_cause) if pred is not None else False
            cw = (not correct) and (not abstained) and (pred is not None) and (conf >= 0.55)
            rows.append(dict(correct=correct, is_novel=(true_cause == NOVEL_CAUSE),
                              confidently_wrong=cw, escalated=escalated, abstained=abstained,
                              conf=conf, pred=pred, true=true_cause))
        return rows, log

    # full_sdl: periodic batch retraining (institutional memory). Process
    # the stream in RETRAIN_EVERY-sized chunks, each scored in one batched
    # call against the model trained on data available BEFORE that chunk.
    working_model = copy.deepcopy(model)
    resolved_X, resolved_y = [], []
    n = len(stream)
    for start in range(0, n, RETRAIN_EVERY):
        end = min(start + RETRAIN_EVERY, n)
        chunk = stream[start:end]
        X_chunk = X_all[start:end]
        proba_chunk = working_model.calibrated_proba(X_chunk)
        ood_chunk = working_model.is_ood_batch(X_chunk)
        qhat = working_model.qhat
        causes = working_model.causes
        for j, (true_cause, feats, ctx_true) in enumerate(chunk):
            pred, conf, escalated, abstained = diagnose_precomputed(
                "full_sdl", causes, proba_chunk[j], bool(ood_chunk[j]), qhat, ctx_true, true_cause,
                dec_rng, log, oracle_noise=oracle_noise, oracle_base_success=oracle_base_success,
            )
            correct = (pred == true_cause) if pred is not None else False
            cw = (not correct) and (not abstained) and (pred is not None) and (conf >= 0.55)
            rows.append(dict(correct=correct, is_novel=(true_cause == NOVEL_CAUSE),
                              confidently_wrong=cw, escalated=escalated, abstained=abstained,
                              conf=conf, pred=pred, true=true_cause))
            if memory_update and true_cause != NOVEL_CAUSE:
                resolved_X.append(feats_to_vec(feats, feature_cols))
                resolved_y.append(true_cause)

        if memory_update and end < n:
            rng2 = random.Random(seed + 999_999 + end)
            X_base, y_base = [], []
            for c in KNOWN_CAUSES:
                for _ in range(n_per_cause):
                    X_base.append(feats_to_vec(sample_incident_pure(rng2, c), feature_cols))
                    y_base.append(c)
            X_full = np.array(X_base + resolved_X, dtype=float)
            y_full = y_base + resolved_y
            working_model = copy.deepcopy(model)
            working_model.fit_classifier(X_full, y_full)

    return rows, log


def rate(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0


def ece(confidences, corrects, n_bins=10):
    """Expected calibration error over the whole run's confident-path
    predictions (baseline-style single-shot confidence, matching the
    standard multiclass ECE definition)."""
    confidences = np.array(confidences)
    corrects = np.array(corrects, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(confidences)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc = corrects[mask].mean()
        conf = confidences[mask].mean()
        e += (mask.sum() / total) * abs(acc - conf)
    return e


CONDITIONS = ["baseline", "baseline_ctx", "abstain_only", "ood_only", "epistemic_only", "full_sdl"]


def run_all(n_seeds, n_incidents=400, novel_rate=0.15, alpha=EPISTEMIC_ALPHA_DEFAULT, verbose=False, seed_start=1):
    feature_cols = list(SYMPTOMS)
    feature_cols_ctx = list(SYMPTOMS) + ["ctx_recent_deploy"]

    agg = {c: defaultdict(list) for c in CONDITIONS}
    ece_baseline_conf, ece_baseline_correct = [], []

    t0 = time.time()
    for seed in range(seed_start, seed_start + n_seeds):
        stream = build_stream(seed, n_incidents=n_incidents, novel_rate=novel_rate)
        model = train_full_pipeline(seed, feature_cols, alpha=alpha, with_context=False)
        model_ctx = train_full_pipeline(seed, feature_cols_ctx, alpha=alpha, with_context=True)

        for cond in CONDITIONS:
            rows, log = run_condition(cond, stream, model, model_ctx, feature_cols, feature_cols_ctx, seed, alpha=alpha)
            known = [r for r in rows if not r["is_novel"]]
            novel = [r for r in rows if r["is_novel"]]
            agg[cond]["acc"].append(rate([r["correct"] for r in rows]))
            agg[cond]["known_acc"].append(rate([r["correct"] for r in known]))
            agg[cond]["novel_acc"].append(rate([r["correct"] for r in novel]))
            agg[cond]["cw"].append(rate([r["confidently_wrong"] for r in rows]))
            agg[cond]["known_cw"].append(rate([r["confidently_wrong"] for r in known]))
            agg[cond]["novel_cw"].append(rate([r["confidently_wrong"] for r in novel]))
            agg[cond]["esc"].append(rate([r["escalated"] for r in rows]))
            agg[cond]["abst"].append(rate([r["abstained"] for r in rows]))
            if cond == "baseline":
                ece_baseline_conf.extend([r["conf"] for r in rows])
                ece_baseline_correct.extend([r["correct"] for r in rows])

        if verbose:
            print(f"seed {seed} done ({time.time()-t0:.1f}s elapsed)", flush=True)

    summary = {}
    for c in CONDITIONS:
        a = agg[c]
        summary[c] = {k: {"mean": mean(v), "sd": sd(v), "values": v} for k, v in a.items()}
    summary["baseline_ece"] = ece(ece_baseline_conf, ece_baseline_correct)
    summary["_meta"] = {"n_seeds": n_seeds, "seed_start": seed_start, "n_incidents": n_incidents,
                         "novel_rate": novel_rate, "alpha": alpha, "elapsed_sec": time.time() - t0}
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=EPISTEMIC_ALPHA_DEFAULT)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    summary = run_all(args.seeds, n_incidents=args.n, alpha=args.alpha, verbose=args.verbose, seed_start=args.seed_start)
    if not args.out:
        print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out}, elapsed {summary['_meta']['elapsed_sec']:.1f}s")
