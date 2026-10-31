"""
STEP 07: the systems quantities that can be measured from the existing
pipeline without new infrastructure.

(A) Persistent state footprint. The routing layer's per-system state is the fitted
    novelty model (IsolationForest), its feature scaler, the Platt/logistic map used
    by control C2, and the three scalar thresholds. Serialise exactly that and
    measure it.
(B) Offline refit cost. Time to fit the detector and recompute tau_ood on a fold's
    48 calibration cases, which is the periodic asynchronous cost discussed in the paper's deployment section.
(C) Routing working set. Peak Python heap allocated while routing one incident,
    via tracemalloc (allocation high-water mark, not process RSS).

Throughput under concurrent load is NOT measured here: it needs a serving harness
this project does not have, and remains a stated omission.
"""
import json
import os
import pickle
import sys
import time
import tracemalloc

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from paths import DATA_DIR, RESULTS_DIR
sys.argv = ["x", "expanded"]
import step03_lofto as L

REPS = 25

d = json.load(open(os.path.join(DATA_DIR, "re1_scores.json")))
recs = d["records"]

print("%-8s %5s %12s %14s %14s" % ("system", "dim", "state (KiB)", "refit p50 (ms)",
                                   "route peak (KiB)"))
rows = {}
for system in L.SYSTEMS:
    f = L.run_fold(recs, system, "loss", "baro", "expanded",
                   np.random.default_rng(L.SEED))
    sys_recs = [r for r in recs if r["system"] == system]
    cands = sys_recs[0]["services_expanded"]

    # rebuild exactly the persistent state, the way run_fold does
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    retained = [r for r in sys_recs if r["fault_type"] != "loss"]
    rng = np.random.default_rng(L.SEED)
    calib = []
    for ft in L.FAULT_TYPES:
        if ft == "loss":
            continue
        grp = [r for r in retained if r["fault_type"] == ft]
        idx = rng.permutation(len(grp))
        calib += [grp[i] for i in idx[: len(grp) // 2]]

    Xc = np.vstack([L.score_vector(r, "baro", "expanded") for r in calib])
    yc = np.array([L.is_correct(r, "baro", "expanded") for r in calib])

    # (B) offline refit cost
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        sc = StandardScaler().fit(Xc)
        iso = IsolationForest(n_estimators=300, random_state=L.SEED,
                              contamination="auto").fit(sc.transform(Xc))
        tau = float(np.percentile(-iso.score_samples(sc.transform(Xc)), 90))
        ts.append(time.perf_counter() - t0)
    refit_ms = float(np.percentile(np.asarray(ts) * 1000.0, 50))

    # C2's Platt map, part of the same persistent state
    Xc2 = np.vstack([L.conf_signal(r, "baro", "expanded") for r in calib])
    sc2 = StandardScaler().fit(Xc2)
    platt = LogisticRegression(max_iter=2000).fit(sc2.transform(Xc2), yc)

    # (A) footprint of exactly what must persist
    state = {"iso": iso, "scaler": sc, "platt": platt, "platt_scaler": sc2,
             "tau_ood": tau, "tau_m": None, "tau_a": None}
    kib = len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)) / 1024.0

    # (C) routing working set for one incident
    row = L.score_vector(sys_recs[0], "baro", "expanded").reshape(1, -1)
    sc.transform(row)
    iso.score_samples(sc.transform(row))          # warm
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    for _ in range(REPS):
        z = sc.transform(row)
        a = -iso.score_samples(z)[0]
        _ = a >= tau
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    peak_kib = (peak - base) / 1024.0

    rows[system] = (len(cands), kib, refit_ms, peak_kib)
    print("%-8s %5d %12.1f %14.1f %14.1f"
          % (system, len(cands), kib, refit_ms, peak_kib))

json.dump({s: {"n_candidates": v[0], "state_kib": v[1], "refit_p50_ms": v[2],
               "route_peak_kib": v[3]} for s, v in rows.items()},
          open(os.path.join(RESULTS_DIR, "footprint.json"), "w"), indent=2)
print("\nwrote footprint.json")
