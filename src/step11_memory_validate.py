"""
STEP 11: institutional-memory validation on the synthetic generator.

Runs, per seed, the six ablation conditions (memory-ON Full SDL included, reproducing the
ablation table) PLUS a memory-OFF Full SDL variant that is identical except
run_condition(..., memory_update=False). Dumps per-seed metric arrays so that seed ranges can
be run in parallel and combined by step12_memory_analyze.py.

Usage (the paper's 100 seeds, as ten shards of ten):
    for s in 1 11 21 31 41 51 61 71 81 91; do python src/step11_memory_validate.py --seed-start $s --seeds 10; done
"""
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gbm_ablation as g
from paths import RESULTS_DIR


def metrics_from_rows(rows):
    known = [r for r in rows if not r["is_novel"]]
    novel = [r for r in rows if r["is_novel"]]
    return dict(
        acc=g.rate([r["correct"] for r in rows]),
        known_acc=g.rate([r["correct"] for r in known]),
        novel_acc=g.rate([r["correct"] for r in novel]),
        cw=g.rate([r["confidently_wrong"] for r in rows]),
        known_cw=g.rate([r["confidently_wrong"] for r in known]),
        novel_cw=g.rate([r["confidently_wrong"] for r in novel]),
        esc=g.rate([r["escalated"] for r in rows]),
        abst=g.rate([r["abstained"] for r in rows]),
    )


def run_range(seed_start, n_seeds, n=400, novel_rate=0.15, alpha=0.20):
    feature_cols = list(g.SYMPTOMS)
    feature_cols_ctx = feature_cols + ["ctx_recent_deploy"]
    conds = list(g.CONDITIONS) + ["full_sdl_nomem"]
    agg = {c: defaultdict(list) for c in conds}

    for seed in range(seed_start, seed_start + n_seeds):
        stream = g.build_stream(seed, n_incidents=n, novel_rate=novel_rate)
        model = g.train_full_pipeline(seed, feature_cols, alpha=alpha, with_context=False)
        model_ctx = g.train_full_pipeline(seed, feature_cols_ctx, alpha=alpha, with_context=True)

        for cond in g.CONDITIONS:
            rows, _ = g.run_condition(cond, stream, model, model_ctx,
                                      feature_cols, feature_cols_ctx, seed, alpha=alpha)
            for k, v in metrics_from_rows(rows).items():
                agg[cond][k].append(v)

        rows_off, _ = g.run_condition("full_sdl", stream, model, model_ctx,
                                      feature_cols, feature_cols_ctx, seed, alpha=alpha,
                                      memory_update=False)
        for k, v in metrics_from_rows(rows_off).items():
            agg["full_sdl_nomem"][k].append(v)

    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", type=str, default=None,
                    help="default: results/memory_shard_<seed-start>.json")
    args = ap.parse_args()
    out_path = args.out or os.path.join(RESULTS_DIR, f"memory_shard_{args.seed_start:03d}.json")
    agg = run_range(args.seed_start, args.seeds)
    out = {c: {k: list(v) for k, v in d.items()} for c, d in agg.items()}
    out["_meta"] = dict(seed_start=args.seed_start, n_seeds=args.seeds)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"wrote {out_path}: seeds {args.seed_start}..{args.seed_start + args.seeds - 1}")
