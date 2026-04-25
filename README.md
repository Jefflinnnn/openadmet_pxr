
# OpenADMET PXR Challenge (Activity Track) — CheMeleon baseline

This repo contains an end-to-end baseline for the **OpenADMET PXR Blind Challenge** **Activity Track**.

## Setup

Requirements: `uv` installed (you already have it).

```bash
uv venv .venv --python 3.11
source .venv/bin/activate

# install dependencies
uv pip install -U pip
uv pip install "chemprop>=2.2.0" pandas numpy scikit-learn datasets huggingface_hub pyarrow tqdm rich pydantic scipy
```

## 1) Download data

```bash
source .venv/bin/activate
python scripts/download_data.py --out-dir data
```

This creates:
- `data/activity_train.csv`
- `data/activity_test_blinded.csv`

## 2) Train (Chemprop fine-tune from CheMeleon)

```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv data/activity_train.csv \
  --out-dir runs/chemeleon_baseline \
  --epochs 20
```

### Useful training knobs (all still Chemprop CLI under the hood)

**Scaffold-balanced split** (better default than random):

```bash
python scripts/train_chemeleon_chemprop.py \
  --train-csv data/activity_train.csv \
  --out-dir runs/chemeleon_scaffold \
  --split SCAFFOLD_BALANCED \
  --epochs 30
```

**K-fold CV** (example: 5-fold scaffold-balanced):

```bash
python scripts/train_chemeleon_chemprop.py \
  --train-csv data/activity_train.csv \
  --out-dir runs/chemeleon_cv5_scaffold \
  --split SCAFFOLD_BALANCED \
  --num-folds 5 \
  --epochs 30
```

**Analog-mimic validation split** (ECFP4 Tanimoto neighborhoods; closer to how the blinded test was built):

```bash
python scripts/make_analog_splits.py \
  --train-csv data/activity_train.csv \
  --out-path runs/splits_analog_mimic.json \
  --seed-threshold 6.0 \
  --tanimoto 0.4 \
  --val-frac 0.2

python scripts/train_chemeleon_chemprop.py \
  --train-csv data/activity_train.csv \
  --out-dir runs/chemeleon_analog_mimic \
  --splits-file runs/splits_analog_mimic.json \
  --loss mae \
  --patience 10 \
  --epochs 50
```

**Ensembles inside Chemprop**:

```bash
python scripts/train_chemeleon_chemprop.py \
  --train-csv data/activity_train.csv \
  --out-dir runs/chemeleon_ensemble5 \
  --split SCAFFOLD_BALANCED \
  --ensemble-size 5 \
  --epochs 30
```

**Optional extra features** (examples; see `chemprop train --help` for supported values):

```bash
python scripts/train_chemeleon_chemprop.py \
  --train-csv data/activity_train.csv \
  --out-dir runs/chemeleon_feats \
  --split SCAFFOLD_BALANCED \
  --molecule-featurizers rdkit_2d_normalized morgan_count \
  --epochs 30
```

## 3) Predict on blinded test

```bash
source .venv/bin/activate
python scripts/predict_chemprop.py \
  --model-path runs/chemeleon_baseline/model_0/best.pt \
  --test-csv data/activity_test_blinded.csv \
  --preds-csv runs/chemeleon_baseline/test_preds.csv
```

## 4) Build + validate submission

```bash
source .venv/bin/activate
python scripts/make_submission.py \
  --test-csv data/activity_test_blinded.csv \
  --preds-csv runs/chemeleon_baseline/test_preds.csv \
  --out-csv submissions/chemeleon_baseline.csv

python scripts/validate_submission.py submissions/chemeleon_baseline.csv
```

The submission must have **exactly 513 rows** and columns:
`SMILES`, `Molecule Name`, `pEC50`.

## Utilities

### Ensemble multiple prediction CSVs

```bash
python scripts/ensemble_preds.py \
  --preds-csv runs/run1/test_preds.csv runs/run2/test_preds.csv \
  --out-csv runs/ensemble/test_preds.csv
```

### Score a validation split (MAE / RAE / R2 / Spearman / Kendall)

1) Generate (or reuse) a splits file (see analog-mimic example above)
2) Predict on the **training CSV** so row indices align:

```bash
chemprop predict \
  --model-path runs/chemeleon_analog_mimic/model_0/best.pt \
  --test-path data/activity_train.csv \
  --preds-path runs/chemeleon_analog_mimic/train_preds.csv

python scripts/score_preds.py \
  --data-csv data/activity_train.csv \
  --preds-csv runs/chemeleon_analog_mimic/train_preds.csv \
  --splits-file runs/splits_analog_mimic.json
```

## Internal benchmark harness (repeated scaffold CV + practical method comparison)

This repo also includes a small benchmarking harness to compare multiple
modeling approaches on **the same** repeated cross-validation splits.

Key idea (JCIM “practically significant method comparison” guidance):
don’t compare methods on a single split. Instead, sample a **distribution**
of performance values (here: repeated scaffold CV), then do paired statistics.

### 1) Generate repeated scaffold CV splits (default: 5 folds × 5 repeats = 25)

```bash
python scripts/make_repeated_scaffold_cv_splits.py \
  --data-csv data/activity_train.csv \
  --out-path runs/splits_scaffold_cv5x5.json \
  --k-folds 5 \
  --repeats 5 \
  --data-seed 0
```

### 2) Run a method across all splits

Chemprop `--splits-file` consumes a JSON list of split objects; we train one
model per split entry.

Example (very small smoke run):

```bash
for split_idx in $(seq 0 24); do \
  python scripts/train_chemeleon_chemprop.py \
    --train-csv data/activity_train.csv \
    --out-dir runs/bench/chemeleon_mae/split_${split_idx} \
    --splits-file runs/splits_scaffold_cv5x5.json \
    --epochs 3 \
    --loss mae \
    --chemprop-args --split-idx ${split_idx}; \
  \
  chemprop predict \
    --model-path runs/bench/chemeleon_mae/split_${split_idx}/model_0/best.pt \
    --test-path data/activity_train.csv \
    --preds-path runs/bench/chemeleon_mae/split_${split_idx}/train_preds.csv; \
done
```

Notes:
- We predict on `activity_train.csv` so row indices align to the splits file.
- `--chemprop-args` is a passthrough for extra Chemprop CLI flags.

### 3) Score across all splits

```bash
python scripts/score_preds_all.py \
  --data-csv data/activity_train.csv \
  --splits-file runs/splits_scaffold_cv5x5.json \
  --preds-csv-template "runs/bench/chemeleon_mae/split_{split_idx}/train_preds.csv" \
  --k-folds 5 \
  --method chemeleon_mae \
  --out-per-split-csv runs/bench/chemeleon_mae/metrics_per_split.csv \
  --out-summary-json runs/bench/chemeleon_mae/metrics_summary.json
```

Primary metric is **RAE** (train-mean baseline for the denominator), with MAE/R2/Spearman/Kendall also reported.

### 4) Compare methods statistically

Once you have multiple `metrics_per_split.csv` files (one per method), compare
them with paired statistics:

```bash
python scripts/benchmark_stats.py \
  --results-csv runs/bench/*/metrics_per_split.csv \
  --metric rae \
  --out-dir runs/bench/_stats
```

Outputs:
- `summary_rae.csv`: mean/std per method
- `pairwise_paired_t_rae.csv`: paired t-tests + Holm multiple-comparison control + Cohen’s dz
- `anova_rm_rae.csv`: repeated-measures ANOVA table (only if `statsmodels` is installed)

To enable RM-ANOVA:

```bash
uv pip install statsmodels
```

