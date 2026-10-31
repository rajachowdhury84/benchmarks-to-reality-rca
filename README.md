# From Benchmarks to Reality: Why Novelty-Typed Abstention Fails in Metric-Based Cloud RCA

Code to reproduce the results in the paper. The pipeline evaluates an uncertainty-typed router
for root-cause analysis (RCA) on RCAEval RE1, using leave-one-fault-type-out (LOFTO) shift and
four controls:

- **C1:** matched-coverage random abstention.
- **C2:** a plain confidence threshold at the same budget.
- **C3:** an in-distribution reference.
- **C4:** routing quality counted directly rather than derived from aggregates.

It also includes a synthetic generator for mechanism analysis.

## Layout

```
src/        all scripts, numbered in run order (step01 ... step12)
data/       re1_scores.json: per-case, per-service scores extracted from RCAEval RE1
```

Every step writes its output to `results/`, which is created on the first run.

### About `data/re1_scores.json`

This file is **extracted from the RCAEval RE1 dataset** ([Zenodo](https://zenodo.org/records/14590730)) by step 01
(`src/step01_extract_scores.py`). It is not new data. For each of the 375 RE1 cases it holds:
- the per-service scores from BARO, N-Sigma and the z-score adapter;
- the metric-window summary features used by detector D3.

It is included so you **don't have to download the dataset (about 2.5 GB extracted) or install
RCAEval** to reproduce the results. With it, every step except 01 and 06 runs directly from this
repository. Running step 01 on the RE1 dataset regenerates this file exactly.

To run steps 01 and 06, or to regenerate `re1_scores.json` yourself, see
[Using the RE1 dataset](#using-the-re1-dataset).

## Environment

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run order

Run every command from the repository root, in the order below. Steps marked *raw data* need
the RE1 dataset and RCAEval; you can skip them, because `data/re1_scores.json` is provided.

**Part A: RCAEval RE1 (real telemetry)**

| Step | Command | Output (`results/`) | Paper content |
|---|---|---|---|
| 01 | `python src/step01_extract_scores.py` | `data/re1_scores.json` | *Raw data, optional.* Runs RCAEval's unmodified `baro()` and `nsigma()` plus a z-score adapter on all 375 cases, and checks every case against the function's own ranking. |
| 02 | `python src/step02_topk_table.py` | `topk.json` | Table I (Top-1 and Top-3, both candidate spaces). |
| 03 | `python src/step03_lofto.py expanded`<br>`python src/step03_lofto.py restricted` | `lofto_expanded.json`<br>`lofto_restricted.json` | Table II and the LOFTO results: 15 folds, 3 localizers, 3 detectors, controls C1–C3, AURC and paired bootstrap. |
| 04 | `python src/step04_paired_loss.py` | `paired_loss.json` | LOSS fold: paired C2−D1 contrast, TPR−FPR gap CI, and the 1,000 C3 draws. |
| 05 | `python src/step05_split_sensitivity.py` | `split_sensitivity.json` | Split sensitivity over 30 re-drawn calibration splits. |
| 06 | `python src/step06_aggregation_latency.py` | `aggregation_latency.json` | *Raw data.* Aggregation sensitivity (lower block of Table I) and routing latency. |
| 07 | `python src/step07_footprint.py` | `footprint.json` | Routing-state footprint, refit cost and working set. |

**Part B: synthetic generator**

| Step | Command | Output (`results/`) | Paper content |
|---|---|---|---|
| 08 | `python src/step08_synthetic_ablation.py` | `synthetic_ablation.json` | Table III (six conditions, 100 trials of 400 incidents) and the baseline ECE. |
| 09 | `python src/step09_coverage_sweep.py` | `coverage_sweep.json` | Conformal-coverage sweep (95% to 60%). |
| 10 | `python src/step10_deferral.py 100 30` | `deferral.json` | Counted routing precision and recall, and Table IV (matched-budget deferral). |
| 11 | `for s in 1 11 21 31 41 51 61 71 81 91; do python src/step11_memory_validate.py --seed-start $s --seeds 10; done` | `memory_shard_*.json` | Per-seed runs including memory-off Full SDL (ten shards of ten seeds). |
| 12 | `python src/step12_memory_analyze.py` | `memory_merged.json` (statistics printed) | Paired tests between conditions, and the institutional-memory on/off comparison. Needs step 11. |

`src/gbm_ablation.py` is the synthetic generator itself. Steps 08–12 import it, and you don't
run it directly. `src/paths.py` holds the shared folder locations.

Steps 02–07 and 10 take a few minutes each. Steps 08 and 11 take about 15 minutes each, and
step 09 about 75 minutes (five coverage targets of 100 seeds).

## Using the RE1 dataset

The RE1 data is published by the RCAEval authors on Zenodo:
<https://zenodo.org/records/14590730> (DOI [10.5281/zenodo.14590730](https://doi.org/10.5281/zenodo.14590730)). It comes as three
archives, one per system: `RE1-OB.zip`, `RE1-SS.zip` and `RE1-TT.zip` (about 390 MB to download).
Run all commands from the repository root.

**1. Clone RCAEval** next to the code (the paper used commit `bb48c5a`):

```bash
git clone https://github.com/phamquiluan/RCAEval RCAEval
git -C RCAEval checkout {commit_id}
```

**2. Download RE1** with RCAEval's own utility. This fetches the three archives from Zenodo and
extracts them into `datasets/RE1/`:

```bash
PYTHONPATH=RCAEval python -c "from RCAEval.utility import download_re1_dataset; download_re1_dataset('datasets')"
```

You can also download the three zips by hand from the Zenodo record and extract them into
`datasets/RE1/`. Either way you should end up with:

```
datasets/RE1/RE1-OB/<service>_<fault>/<1..5>/data.csv, inject_time.txt
datasets/RE1/RE1-SS/<service>_<fault>/<1..5>/simple_data.csv, inject_time.txt
datasets/RE1/RE1-TT/<service>_<fault>/<1..5>/simple_data.csv, inject_time.txt
```

> Use the Zenodo CSV release. The Hugging Face Parquet copy of RCAEval uses a different file
> layout, which these scripts do not read.

**3. Run steps 01 and 06:**

```bash
python src/step01_extract_scores.py   # regenerates data/re1_scores.json
python src/step06_aggregation_latency.py
```

Then continue with the run order above.

If RCAEval or the dataset is stored somewhere else, point the scripts at it with environment
variables:

```bash
RCAEVAL_REPO=/path/to/RCAEval RE1_ROOT=/path/to/RE1 python src/step01_extract_scores.py
```

Here `RE1_ROOT` is the folder that contains `RE1-OB`, `RE1-SS` and `RE1-TT`. `DATA_DIR` and
`RESULTS_DIR` can be overridden the same way (see `src/paths.py`).

## Reproducibility notes

- All seeds are fixed in the scripts. The LOFTO split seed is `20260908`.
- The RE1 dataset and the RCAEval source are not redistributed here. Please cite RCAEval
  (Pham et al.) when using them.

## License

- **Code:** MIT (see [LICENSE](LICENSE)).
- **`data/re1_scores.json`:** derived from the RCAEval RE1 dataset, which is licensed under
  [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
  The file contains scores and summary features computed from the raw RE1 telemetry by
  `src/step01_extract_scores.py`; the raw data itself is not included. It is shared under the
  same CC BY 4.0 license. Dataset source: Pham, L. (2024). *RCAEval: A Benchmark for Root Cause
  Analysis of Microservice Systems* [Dataset]. Zenodo.
  [https://doi.org/10.5281/zenodo.14590730](https://doi.org/10.5281/zenodo.14590730)
