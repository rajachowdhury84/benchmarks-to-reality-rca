"""
STEP 01: Run RCAEval's OFFICIAL baro() and nsigma() -- unmodified, imported
straight from a fresh clone of the RCAEval repository -- inside RCAEval's DOCUMENTED
environment (Python 3.12, stock scikit-learn, numpy<2, pandas>2.0.1), and dump
per-case, per-service scores for every RE1 case.

This script imports `from RCAEval.e2e import baro, nsigma` from the official package,
so the numerics are stock sklearn.

Score recovery WITHOUT modifying RCAEval's source
-------------------------------------------------
baro()/nsigma() return only a ranked column list, not the per-column scores the SDL
needs. We recover them by binding, in the *calling* module namespace, a subclass of the
genuine sklearn scaler that records the max of each transform() output. Subclassing
means every number produced is computed by stock sklearn code; only an observation hook
is added. Each function calls scaler.transform() exactly once per column, in the same
order as its returned node_names, so zipping reconstructs the internal score vector.
The reconstruction is CHECKED on every case: the ranking implied by the recorded scores
must equal the ranking the function itself returned, else the case is aborted.

Candidate spaces
----------------
  restricted : the five fault-injected services per system (the benchmark's own protocol)
  expanded   : every service appearing in that system's telemetry (primary)
Ranking only the five injected services leaks the injection design; an operator does not
know in advance which five components were eligible. Both are emitted.

Outputs: re1_scores.json  (per case: per-service scores for 3 localizers, plus
window summary-statistic features used by the third OOD detector in step 03)
"""
import json
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RCAEVAL_REPO as REPO, RE1_ROOT as BASE, DATA_DIR as OUT_DIR

sys.path.insert(0, REPO)

import sklearn
import sklearn.preprocessing as skp
from RCAEval.e2e import baro, nsigma          # official, unmodified

_REC = []


class _RecRobust(skp.RobustScaler):
    """Stock sklearn RobustScaler; records max(transform(X)) as a side effect."""
    def transform(self, X):
        out = super().transform(X)
        _REC.append(float(np.max(out)))
        return out


class _RecStandard(skp.StandardScaler):
    """Stock sklearn StandardScaler; records max(transform(X)) as a side effect."""
    def transform(self, X):
        out = super().transform(X)
        _REC.append(float(np.max(out)))
        return out


# Rebind the scaler names in each function's own global namespace. The functions
# are left byte-for-byte unmodified; only the class they look up is a recording
# subclass of the genuine sklearn class.
baro.__globals__["RobustScaler"] = _RecRobust
nsigma.__globals__["StandardScaler"] = _RecStandard

SYSTEMS = [
    {"name": "RE1-OB", "display": "Online Boutique", "dir": os.path.join(BASE, "RE1-OB"),
     "injected": ["adservice", "cartservice", "checkoutservice", "currencyservice",
                  "productcatalogservice"],
     "data_file": "data.csv", "dataset_str": "re1-ob"},
    {"name": "RE1-SS", "display": "Sock Shop", "dir": os.path.join(BASE, "RE1-SS"),
     "injected": ["carts", "catalogue", "orders", "payment", "user"],
     "data_file": "simple_data.csv", "dataset_str": "re1-ss"},
    {"name": "RE1-TT", "display": "Train Ticket", "dir": os.path.join(BASE, "RE1-TT"),
     "injected": ["ts-auth-service", "ts-order-service", "ts-route-service",
                  "ts-train-service", "ts-travel-service"],
     "data_file": "simple_data.csv", "dataset_str": "re1-tt"},
]
FAULT_TYPES = ["cpu", "mem", "disk", "delay", "loss"]
POST_WINDOW_SEC = 300          # z-score adapter window
NEG = -1e9                     # score for a service with no surviving column


def list_cases(system):
    out = []
    for d in sorted(os.listdir(system["dir"])):
        full = os.path.join(system["dir"], d)
        if not os.path.isdir(full):
            continue
        parts = d.rsplit("_", 1)
        if len(parts) != 2 or parts[0] not in system["injected"]:
            continue
        svc, ft = parts
        for inst in sorted(os.listdir(full)):
            p = os.path.join(full, inst)
            if os.path.isdir(p):
                out.append((p, svc, ft))
    return out


def col_service(col):
    """Service a metric column belongs to: everything before the final '_'."""
    if col in ("time", "time.1") or "_" not in col:
        return None
    return col.rsplit("_", 1)[0]


def system_services(system, cases):
    """Union of services appearing in the system's telemetry (expanded candidate set)."""
    svcs = set()
    for path, _, _ in cases[:3]:
        df = pd.read_csv(os.path.join(path, system["data_file"]), nrows=1)
        for c in df.columns:
            s = col_service(c)
            if s:
                svcs.add(s)
    return sorted(svcs)


def agg_to_services(col_scores, services):
    """Service score = max over that service's surviving metric columns."""
    out = {s: NEG for s in services}
    for col, val in col_scores.items():
        s = col_service(col)
        if s in out and val > out[s]:
            out[s] = val
    return out


def call_recording(fn, df, inject_time, dataset_str):
    global _REC
    _REC = []
    out = fn(df, inject_time=inject_time, dataset=dataset_str,
             anomalies=None, dk_select_useful=False, sli=None)
    rec = list(_REC)
    names = out.get("node_names", [])
    if len(rec) != len(names):
        return None
    col_scores = dict(zip(names, rec))
    # verification: recorded scores must reproduce the function's own ranking
    reconstructed = [c for c, _ in sorted(col_scores.items(), key=lambda kv: kv[1],
                                          reverse=True)]
    if reconstructed != out["ranks"]:
        # ties can permute equal-score columns; accept only if scores agree elementwise
        a = [col_scores[c] for c in reconstructed]
        b = [col_scores[c] for c in out["ranks"]]
        if not np.allclose(a, b):
            return None
    return col_scores


def zscore_cols(df, inject_time):
    """Z-score adapter: |mean_post - mean_pre| / std_pre per column."""
    pre = df[df["time"] < inject_time]
    post = df[(df["time"] >= inject_time) & (df["time"] <= inject_time + POST_WINDOW_SEC)]
    if len(pre) < 10 or len(post) < 5:
        return None
    out = {}
    for col in df.columns:
        if col_service(col) is None:
            continue
        pv, qv = pre[col].dropna(), post[col].dropna()
        if len(pv) < 5 or len(qv) < 3:
            continue
        sd = pv.std(ddof=1)
        z = abs(qv.mean() - pv.mean()) / sd if sd and sd > 1e-9 else abs(qv.mean() - pv.mean())
        if not math.isnan(z):
            out[col] = float(z)
    return out


def window_features(df, inject_time):
    """Localizer-independent summary statistics of the metric window itself
    (detector variant D3 in step 03). Fixed 12-dim, scale-free by construction."""
    pre = df[df["time"] < inject_time]
    post = df[(df["time"] >= inject_time) & (df["time"] <= inject_time + POST_WINDOW_SEC)]
    if len(pre) < 10 or len(post) < 5:
        return None
    shifts, disp, cv = [], [], []
    for col in df.columns:
        if col_service(col) is None:
            continue
        pv, qv = pre[col].dropna(), post[col].dropna()
        if len(pv) < 5 or len(qv) < 3:
            continue
        sd = float(pv.std(ddof=1))
        if not sd or sd < 1e-9 or math.isnan(sd):
            continue
        shifts.append(abs(float(qv.mean()) - float(pv.mean())) / sd)
        disp.append(float(qv.std(ddof=1)) / sd)
        m = abs(float(pv.mean()))
        cv.append(sd / m if m > 1e-12 else 0.0)
    if len(shifts) < 5:
        return None
    s = np.asarray(shifts)
    d = np.asarray(disp)
    c = np.asarray(cv)
    ls = np.log1p(s)
    return [
        float(np.mean(ls)), float(np.std(ls)),
        float(np.percentile(ls, 50)), float(np.percentile(ls, 90)),
        float(np.percentile(ls, 99)), float(np.max(ls)),
        float(np.mean(s > 3.0)), float(np.mean(s > 6.0)),
        float(np.mean(np.log1p(d))), float(np.std(np.log1p(d))),
        float(np.mean(np.log1p(c))), float(len(s)),
    ]


def main():
    print(f"python {sys.version.split()[0]}  sklearn {sklearn.__version__}  "
          f"numpy {np.__version__}  pandas {pd.__version__}")
    print(f"RCAEval repo: {REPO}")
    records, skipped = [], 0
    for system in SYSTEMS:
        cases = list_cases(system)
        services = system_services(system, cases)
        print(f"\n{system['name']}: {len(cases)} cases, "
              f"{len(services)} services in telemetry "
              f"({len(system['injected'])} injected)")
        for i, (path, true_svc, ft) in enumerate(cases):
            df = pd.read_csv(os.path.join(path, system["data_file"]))
            inject_time = int(open(os.path.join(path, "inject_time.txt")).read().strip())
            # RCAEval main.py's own pre-call hygiene
            df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

            b_cols = call_recording(baro, df.copy(), inject_time, system["dataset_str"])
            n_cols = call_recording(nsigma, df.copy(), inject_time, system["dataset_str"])
            z_cols = zscore_cols(df, inject_time)
            feats = window_features(df, inject_time)
            if b_cols is None or n_cols is None or z_cols is None or feats is None:
                skipped += 1
                continue
            rec = {
                "system": system["name"], "case": os.path.relpath(path, BASE),
                "true_service": true_svc, "fault_type": ft,
                "services_expanded": services,
                "services_restricted": system["injected"],
                "baro": agg_to_services(b_cols, services),
                "nsigma": agg_to_services(n_cols, services),
                "zscore": agg_to_services(z_cols, services),
                "window_features": feats,
            }
            records.append(rec)
            if (i + 1) % 25 == 0:
                print(f"  ... {i+1}/{len(cases)}")
    meta = {
        "python": sys.version.split()[0], "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "n_cases": len(records), "skipped": skipped,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "re1_scores.json"), "w") as f:
        json.dump({"meta": meta, "records": records}, f)
    print(f"\nWrote {len(records)} cases (skipped {skipped}) -> re1_scores.json")


if __name__ == "__main__":
    main()
