"""
STEP 06: two analyses that need the raw per-column scores (needs RCAEval and RE1).

(A) Service-score aggregation sensitivity. The main analysis scores a service by the MAXIMUM over
    its metric columns. With 64 services on Train Ticket that is an obvious extreme-value
    confound for the candidate-space reversal. Recompute Top-1 in both candidate spaces
    under four aggregations:
        max            max_c s_c                        (as reported in the main analysis)
        mean           mean_c s_c
        top3           mean of the three largest s_c
        cnorm          max_c s_c - sqrt(2 ln m_s)       (expected-maximum correction,
                                                         m_s = surviving column count)
(B) Routing-latency microbenchmark. Per-incident BARO localization latency vs per-incident
    D1 routing latency, on the same 375 cases, at 14/15/64 candidates.

Same environment, same unmodified RCAEval functions, same recording-subclass extraction
as step01_extract_scores.py.
"""
import json, math, os, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import RCAEVAL_REPO as REPO, RESULTS_DIR as OUT_DIR
sys.path.insert(0, REPO)

import sklearn
import sklearn.preprocessing as skp
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from RCAEval.e2e import baro, nsigma

import step01_extract_scores as S0

_REC = []


class _RecRobust(skp.RobustScaler):
    def transform(self, X):
        out = super().transform(X)
        _REC.append(float(np.max(out)))
        return out


class _RecStandard(skp.StandardScaler):
    def transform(self, X):
        out = super().transform(X)
        _REC.append(float(np.max(out)))
        return out


baro.__globals__["RobustScaler"] = _RecRobust
nsigma.__globals__["StandardScaler"] = _RecStandard

SEED = 20260908
NEG = -1e9
SYSTEMS = S0.SYSTEMS
col_service = S0.col_service
AGGS = ["max", "mean", "top3", "cnorm"]
LOCS = ["zscore", "nsigma", "baro"]


def call_recording(fn, df, inject_time, dataset_str):
    global _REC
    _REC = []
    t0 = time.perf_counter()
    out = fn(df, inject_time=inject_time, dataset=dataset_str,
             anomalies=None, dk_select_useful=False, sli=None)
    dt = time.perf_counter() - t0
    rec = list(_REC)
    names = out.get("node_names", [])
    if len(rec) != len(names):
        return None, dt
    return dict(zip(names, rec)), dt


def group_cols(col_scores, services):
    g = {s: [] for s in services}
    for col, val in col_scores.items():
        s = col_service(col)
        if s in g:
            g[s].append(float(val))
    return g


def aggregate(g, how):
    out = {}
    for s, vals in g.items():
        if not vals:
            out[s] = NEG
            continue
        v = sorted(vals, reverse=True)
        if how == "max":
            out[s] = v[0]
        elif how == "mean":
            out[s] = float(np.mean(v))
        elif how == "top3":
            out[s] = float(np.mean(v[:3]))
        elif how == "cnorm":
            out[s] = v[0] - math.sqrt(2.0 * math.log(max(len(v), 2)))
    return out


def topk(recs, loc, agg, space, k=1, system=None):
    sel = [r for r in recs if system is None or r["system"] == system]
    hit = 0
    for r in sel:
        cands = r["services_expanded"] if space == "expanded" else r["services_restricted"]
        sc = r[loc][agg]
        order = sorted(cands, key=lambda s: (-sc.get(s, NEG), s))[:k]
        hit += int(r["true_service"] in order)
    return 100.0 * hit / len(sel)


def top1(recs, loc, agg, space, system=None):
    return topk(recs, loc, agg, space, 1, system)


def main():
    print("python %s  sklearn %s  numpy %s  pandas %s"
          % (sys.version.split()[0], sklearn.__version__, np.__version__, pd.__version__))
    records = []
    lat = {"baro": [], "nsigma": [], "zscore": []}
    lat_by_system = {}
    skipped = 0

    for system in SYSTEMS:
        cases = S0.list_cases(system)
        services = S0.system_services(system, cases)
        lat_by_system[system["name"]] = {"n_cand": len(services), "baro": []}
        print("\n%s: %d cases, %d services" % (system["name"], len(cases), len(services)))
        for i, (path, true_svc, ft) in enumerate(cases):
            df = pd.read_csv(os.path.join(path, system["data_file"]))
            inject_time = int(open(os.path.join(path, "inject_time.txt")).read().strip())
            df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

            b_cols, t_b = call_recording(baro, df.copy(), inject_time, system["dataset_str"])
            n_cols, t_n = call_recording(nsigma, df.copy(), inject_time, system["dataset_str"])
            t0 = time.perf_counter()
            z_cols = S0.zscore_cols(df, inject_time)
            t_z = time.perf_counter() - t0
            if b_cols is None or n_cols is None or z_cols is None:
                skipped += 1
                continue
            lat["baro"].append(t_b)
            lat["nsigma"].append(t_n)
            lat["zscore"].append(t_z)
            lat_by_system[system["name"]]["baro"].append(t_b)

            rec = {"system": system["name"], "true_service": true_svc, "fault_type": ft,
                   "services_expanded": services,
                   "services_restricted": system["injected"]}
            for loc, cols in (("baro", b_cols), ("nsigma", n_cols), ("zscore", z_cols)):
                g = group_cols(cols, services)
                rec[loc] = {a: aggregate(g, a) for a in AGGS}
            records.append(rec)
            if (i + 1) % 50 == 0:
                print("  ... %d/%d" % (i + 1, len(cases)))

    print("\n%d cases, skipped %d" % (len(records), skipped))

    # ------------------------------------------------------------------ (A) Top-1
    tbl = {}
    print("\n=== (A) Top-1 by aggregation ===")
    for space in ("expanded", "restricted"):
        print("\n-- %s candidate space --" % space)
        print("%-10s %-6s %7s %6s %6s %6s %7s" % ("localizer", "agg", "pooled", "OB",
                                                  "SS", "TT", "top3acc"))
        for loc in LOCS:
            for agg in AGGS:
                p = top1(records, loc, agg, space)
                ob = top1(records, loc, agg, space, "RE1-OB")
                ss = top1(records, loc, agg, space, "RE1-SS")
                tt = top1(records, loc, agg, space, "RE1-TT")
                t3 = topk(records, loc, agg, space, 3)
                tbl["%s|%s|%s" % (space, loc, agg)] = {"pooled": p, "OB": ob, "SS": ss,
                                                       "TT": tt, "top3": t3}
                print("%-10s %-6s %7.1f %6.1f %6.1f %6.1f %7.1f"
                      % (loc, agg, p, ob, ss, tt, t3))

    print("\n-- adapter ordering by aggregation (pooled) --")
    for agg in AGGS:
        eo = sorted(LOCS, key=lambda l: -tbl["expanded|%s|%s" % (l, agg)]["pooled"])
        ev = ["%s %.1f" % (l, tbl["expanded|%s|%s" % (l, agg)]["pooled"]) for l in eo]
        ro = sorted(LOCS, key=lambda l: -tbl["restricted|%s|%s" % (l, agg)]["pooled"])
        rv = ["%s %.1f" % (l, tbl["restricted|%s|%s" % (l, agg)]["pooled"]) for l in ro]
        print("  %-6s expanded: %s   |   restricted: %s   reversal=%s"
              % (agg, " > ".join(ev), " > ".join(rv), "YES" if eo != ro else "no"))

    # ------------------------------------------------------------------ (B) latency
    print("\n=== (B) latency (ms/incident) ===")
    lat_out = {}
    for k, v in lat.items():
        a = np.asarray(v) * 1000.0
        lat_out[k] = {"n": len(a), "p50": float(np.percentile(a, 50)),
                      "p99": float(np.percentile(a, 99)), "mean": float(a.mean())}
        print("  localize/%-7s n=%3d  p50=%8.1f  p99=%8.1f"
              % (k, len(a), lat_out[k]["p50"], lat_out[k]["p99"]))
    for s, v in lat_by_system.items():
        a = np.asarray(v["baro"]) * 1000.0
        lat_out["baro_" + s] = {"n_cand": v["n_cand"], "p50": float(np.percentile(a, 50)),
                                "p99": float(np.percentile(a, 99))}
        print("  localize/baro %s (%2d cand)  p50=%8.1f  p99=%8.1f"
              % (s, v["n_cand"], np.percentile(a, 50), np.percentile(a, 99)))

    rng = np.random.default_rng(SEED)
    print()
    for system in SYSTEMS:
        name = system["name"]
        sel = [r for r in records if r["system"] == name]
        cands = sel[0]["services_expanded"]
        X = np.array([[r["baro"]["max"].get(s, NEG) for s in cands] for r in sel])
        X = np.clip(X, -50.0, 1e6)
        idx = rng.permutation(len(sel))
        cal, ev = X[idx[:len(idx) // 2]], X[idx[len(idx) // 2:]]
        sc = StandardScaler().fit(cal)
        iso = IsolationForest(n_estimators=300, random_state=SEED,
                              contamination="auto").fit(sc.transform(cal))
        a_cal = -iso.score_samples(sc.transform(cal))
        tau = float(np.percentile(a_cal, 90))
        times = []
        for row in ev:
            for _ in range(5):
                t0 = time.perf_counter()
                z = sc.transform(row.reshape(1, -1))
                a = -iso.score_samples(z)[0]
                _ = a >= tau
                times.append(time.perf_counter() - t0)
        a = np.asarray(times) * 1000.0
        lat_out["route_" + name] = {"n_cand": len(cands), "n": len(a),
                                    "p50": float(np.percentile(a, 50)),
                                    "p99": float(np.percentile(a, 99))}
        print("  route/D1     %s (%2d cand)  p50=%8.3f  p99=%8.3f"
              % (name, len(cands), np.percentile(a, 50), np.percentile(a, 99)))

    json.dump({"meta": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
                        "n_cases": len(records), "skipped": skipped},
               "top1": tbl, "latency_ms": lat_out},
              open(os.path.join(OUT_DIR, "aggregation_latency.json"), "w"), indent=2)
    print("\nwrote aggregation_latency.json")


if __name__ == "__main__":
    main()
