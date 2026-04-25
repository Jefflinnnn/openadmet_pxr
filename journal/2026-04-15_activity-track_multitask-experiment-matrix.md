# 2026-04-15 — Activity Track multitask experiment matrix (PXR + CYP3A4 + nuclear receptors)

## Goal
Improve **OpenADMET PXR Activity Track** leaderboard performance by using **multitask training** with adjacent endpoints, while ensuring the **final submission is PXR pEC50 only**.

Key constraint: multitask CSV has **very low overlap** between PXR-labeled rows and other tasks (e.g., CYP3A4 overlaps PXR on ~131 rows), so **task weights matter**.

---

## Datasets / splits

**Multitask training CSV (union of tasks):**

- `data/multitask/pxr_plus_adme3521_plus_nr_cyp3a4.csv`
- columns:
  - `pxr_pEC50`
  - `chembl_cyp3a4_pchembl`
  - `chembl_ahr_pchembl`
  - `chembl_car_pchembl`
  - ADME: `adme_log_hlm_clint`, `adme_log_rlm_clint`, `adme_log_mdr1_mdck_er`, `adme_log_solubility_ph68`, `adme_log_ppb_human_unbound`, `adme_log_ppb_rat_unbound`

**Lifted splits (indices correspond to the multitask CSV rows):**

- repeated scaffold CV (10 split entries = 5 folds × 2 repeats)
  - `runs/splits_scaffold_cv5x2_pxr_on_multitask.json`
- analog mimic (5 split entries)
  - `runs/splits_analog_mimic_pxr5_on_multitask.json`

---

## Evaluation protocol (what “wins”)

### Target metric / selection
- Primary: **mean MAE on `pxr_pEC50`** across all split entries.
- Secondary: Spearman/Kendall on `pxr_pEC50`.

### How we compute mean across splits
For a run directory `RUN_DIR` trained with `--splits-file`:

1) Generate predictions for each split entry (Chemprop stores checkpoints under `replicate_{i}/model_0/best.pt`):

```bash
source .venv/bin/activate

RUN_DIR=runs/EXAMPLE
DATA_MT=data/multitask/pxr_plus_adme3521_plus_nr_cyp3a4.csv

for d in ${RUN_DIR}/replicate_*; do
  i=$(basename "$d" | sed 's/replicate_//')
  python scripts/predict_chemprop.py \
    --model-path ${d}/model_0/best.pt \
    --test-csv ${DATA_MT} \
    --preds-csv ${RUN_DIR}/split_${i}_train_preds.csv \
    --accelerator mps --devices 1
done
```

2) Score across all splits:

```bash
source .venv/bin/activate

SPLITS=runs/splits_scaffold_cv5x2_pxr_on_multitask.json

python scripts/score_preds_all.py \
  --data-csv ${DATA_MT} \
  --preds-csv-template "${RUN_DIR}/split_{split_idx}_train_preds.csv" \
  --splits-file ${SPLITS} \
  --target-col pxr_pEC50 \
  --k-folds 5 \
  --method "$(basename ${RUN_DIR})" \
  --out-per-split-csv "${RUN_DIR}/pxr_metrics_per_split.csv" \
  --out-summary-json "${RUN_DIR}/pxr_metrics_summary.json"
```

Repeat with `SPLITS=runs/splits_analog_mimic_pxr5_on_multitask.json` for analog-mimic.

---

## Run conventions

Recommended common flags (feel free to edit):

```bash
ACCEL=mps
DEV=1
EPOCHS=60
WARMUP=3
BS=128
PATIENCE=10

DATA_MT=data/multitask/pxr_plus_adme3521_plus_nr_cyp3a4.csv
SPLITS_CV=runs/splits_scaffold_cv5x2_pxr_on_multitask.json
SPLITS_ANALOG=runs/splits_analog_mimic_pxr5_on_multitask.json
```

I treat each “run” below as training once on a chosen splits file.

### Tooling notes (reproducibility + multitask safety)

- **Lifted splits**: all multitask runs use split indices that refer to rows in the multitask CSV.
  - Artifact: `runs/splits_scaffold_cv5x2_pxr_on_multitask.json`
  - Created by lifting the PXR-only scaffold CV splits onto the multitask CSV by SMILES (see `scripts/lift_splits_by_smiles.py`).
- **Training wrapper**: `scripts/train_chemeleon_chemprop.py`
  - Single-task: `--target-col pxr_pEC50`
  - Multitask: `--target-cols pxr_pEC50 chembl_cyp3a4_pchembl ...`
  - Optional: `--task-weights 1.0 0.2 ...` (must match `--target-cols` length)
- **Prediction wrapper**: `scripts/predict_chemprop.py`
  - Uses `--model-path <.../best.pt> --test-csv <csv> --preds-csv <out.csv>`
- **NaN-safe scoring**: `scripts/score_preds_all.py`
  - When the dataset is a multitask union CSV, many rows have NaN labels for the target task; scoring masks out NaNs in y/yhat per split.
- **Submission**: `scripts/make_submission.py`
  - For multitask prediction files, force the competition column with `--pred-col pxr_pEC50`.

---

## Completed runs (paired comparisons on identical lifted splits)

### Lifted scaffold CV 5×2 (10 split entries)

Splits file:

- `runs/splits_scaffold_cv5x2_pxr_on_multitask.json` (10 split entries)
- example sizes (split 0): train=3311, val=828

Dataset:

- `data/multitask/pxr_plus_adme3521_plus_nr_cyp3a4.csv`

#### R00 — PXR-only baseline (scaffold CV 5×2)

Train:

```bash
source .venv/bin/activate && python scripts/train_chemeleon_chemprop.py \
  --train-csv data/multitask/pxr_plus_adme3521_plus_nr_cyp3a4.csv \
  --out-dir runs/R00_pxr_only_cv \
  --smiles-col SMILES \
  --target-col pxr_pEC50 \
  --epochs 60 \
  --warmup-epochs 3 \
  --batch-size 128 \
  --splits-file runs/splits_scaffold_cv5x2_pxr_on_multitask.json \
  --accelerator mps --devices 1 \
  --patience 10
```

Predict (per split entry):

```bash
source .venv/bin/activate && for d in runs/R00_pxr_only_cv/replicate_*; do 
  python scripts/predict_chemprop.py \
    --model-path "$d/model_0/best.pt" \
    --test-csv data/multitask/pxr_plus_adme3521_plus_nr_cyp3a4.csv \
    --preds-csv "$d/train_preds.csv" \
    --smiles-col SMILES \
    --accelerator mps --devices 1;
 done
```

Score (PXR only, NaN-safe masking via `score_preds_all.py`):

```bash
source .venv/bin/activate && python scripts/score_preds_all.py \
  --data-csv data/multitask/pxr_plus_adme3521_plus_nr_cyp3a4.csv \
  --preds-csv-template 'runs/R00_pxr_only_cv/replicate_{split_idx}/train_preds.csv' \
  --splits-file runs/splits_scaffold_cv5x2_pxr_on_multitask.json \
  --target-col pxr_pEC50 \
  --k-folds 5 \
  --method R00_pxr_only_cv \
  --out-per-split-csv runs/R00_pxr_only_cv/metrics_per_split_pxr_pEC50.csv \
  --out-summary-json runs/R00_pxr_only_cv/metrics_summary_pxr_pEC50.json
```

Results (mean ± std over 10 split entries):

- MAE: 0.4962613 ± 0.0196352
- RAE: 0.5452129 ± 0.0232790
- R2: 0.6231733 ± 0.0372209
- Spearman rho: 0.7560648 ± 0.0257404
- Kendall tau: 0.5647426 ± 0.0235194

Artifacts:

- `runs/R00_pxr_only_cv/metrics_summary_pxr_pEC50.json`
- `runs/R00_pxr_only_cv/metrics_per_split_pxr_pEC50.csv`

#### MT-v1 — multitask PXR + CYP3A4 (w_cyp=0.2) on same scaffold CV 5×2 lifted splits

Train:

```bash
source .venv/bin/activate && python scripts/train_chemeleon_chemprop.py \
  --train-csv data/multitask/pxr_plus_adme3521_plus_nr_cyp3a4.csv \
  --out-dir runs/chemeleon_mt_v1_pxr_cyp3a4_w02_e50 \
  --smiles-col SMILES \
  --target-cols pxr_pEC50 chembl_cyp3a4_pchembl \
  --task-weights 1.0 0.2 \
  --epochs 50 \
  --warmup-epochs 5 \
  --batch-size 128 \
  --splits-file runs/splits_scaffold_cv5x2_pxr_on_multitask.json \
  --accelerator mps --devices 1 \
  --patience 10
```

Predict:

```bash
source .venv/bin/activate && for d in runs/chemeleon_mt_v1_pxr_cyp3a4_w02_e50/replicate_*; do 
  python scripts/predict_chemprop.py \
    --model-path "$d/model_0/best.pt" \
    --test-csv data/multitask/pxr_plus_adme3521_plus_nr_cyp3a4.csv \
    --preds-csv "$d/train_preds.csv" \
    --smiles-col SMILES \
    --accelerator mps --devices 1;
 done
```

Score (PXR only):

```bash
source .venv/bin/activate && python scripts/score_preds_all.py \
  --data-csv data/multitask/pxr_plus_adme3521_plus_nr_cyp3a4.csv \
  --preds-csv-template 'runs/chemeleon_mt_v1_pxr_cyp3a4_w02_e50/replicate_{split_idx}/train_preds.csv' \
  --splits-file runs/splits_scaffold_cv5x2_pxr_on_multitask.json \
  --target-col pxr_pEC50 \
  --k-folds 5 \
  --method chemeleon_mt_v1_pxr_cyp3a4_w02_e50 \
  --out-per-split-csv runs/chemeleon_mt_v1_pxr_cyp3a4_w02_e50/metrics_per_split_pxr_pEC50.csv \
  --out-summary-json runs/chemeleon_mt_v1_pxr_cyp3a4_w02_e50/metrics_summary_pxr_pEC50.json
```

Results (mean ± std over 10 split entries):

- MAE: 0.4983856 ± 0.0188143
- RAE: 0.5475288 ± 0.0218691
- R2: 0.6223425 ± 0.0380762
- Spearman rho: 0.7525962 ± 0.0257725
- Kendall tau: 0.5615292 ± 0.0221417

Artifacts:

- `runs/chemeleon_mt_v1_pxr_cyp3a4_w02_e50/metrics_summary_pxr_pEC50.json`
- `runs/chemeleon_mt_v1_pxr_cyp3a4_w02_e50/metrics_per_split_pxr_pEC50.csv`

#### Paired comparison summary (same splits)

| Method | MAE ↓ | R2 ↑ | Spearman ↑ |
|---|---:|---:|---:|
| R00_pxr_only_cv | 0.4963 | 0.6232 | 0.7561 |
| MT PXR+CYP (w=0.2) | 0.4984 | 0.6223 | 0.7526 |

Observation: on these lifted scaffold CV splits, the PXR-only baseline is extremely close / slightly better than the multitask PXR+CYP (w=0.2) setting. Next step is a task-weight sweep and adding NR tasks (with small weights) to see if multitask can help.

---

## Experiment matrix (20 core runs + optional extras)

### Phase 0 — sanity baselines (2 runs)

**R00 — PXR-only baseline (scaffold CV 5×2)**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} \
  --out-dir runs/R00_pxr_only_cv \
  --target-col pxr_pEC50 \
  --splits-file ${SPLITS_CV} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

**R01 — PXR-only baseline (analog-mimic)**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} \
  --out-dir runs/R01_pxr_only_analog \
  --target-col pxr_pEC50 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

### Phase 1 — PXR + CYP3A4 weight sweep (6 runs)

Targets: `pxr_pEC50`, `chembl_cyp3a4_pchembl`

**R02 — MT PXR+CYP (w_cyp=0.05)**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R02_mt_pxr_cyp_w005 \
  --target-cols pxr_pEC50 chembl_cyp3a4_pchembl \
  --task-weights 1.0 0.05 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

**R03 — MT PXR+CYP (w_cyp=0.1)**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R03_mt_pxr_cyp_w01 \
  --target-cols pxr_pEC50 chembl_cyp3a4_pchembl \
  --task-weights 1.0 0.1 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

**R04 — MT PXR+CYP (w_cyp=0.2)**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R04_mt_pxr_cyp_w02 \
  --target-cols pxr_pEC50 chembl_cyp3a4_pchembl \
  --task-weights 1.0 0.2 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

**R05 — MT PXR+CYP (w_cyp=0.5)**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R05_mt_pxr_cyp_w05 \
  --target-cols pxr_pEC50 chembl_cyp3a4_pchembl \
  --task-weights 1.0 0.5 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

**R06 — MT PXR+CYP (w_cyp=1.0)**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R06_mt_pxr_cyp_w10 \
  --target-cols pxr_pEC50 chembl_cyp3a4_pchembl \
  --task-weights 1.0 1.0 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

**R07 — MT PXR+CYP (w_cyp=0.2) but evaluate on scaffold CV 5×2**

Rationale: pick a mid weight and test robustness.
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R07_mt_pxr_cyp_w02_cv \
  --target-cols pxr_pEC50 chembl_cyp3a4_pchembl \
  --task-weights 1.0 0.2 \
  --splits-file ${SPLITS_CV} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

### Phase 2 — add nuclear receptors (4 runs)

Targets: `pxr_pEC50`, `chembl_cyp3a4_pchembl`, `chembl_ahr_pchembl`, `chembl_car_pchembl`

**R08 — MT + NR (downweight NR hard)**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R08_mt_pxr_cyp_ahr_car_smallnr \
  --target-cols pxr_pEC50 chembl_cyp3a4_pchembl chembl_ahr_pchembl chembl_car_pchembl \
  --task-weights 1.0 0.2 0.05 0.05 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

**R09 — MT + NR (moderate NR)**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R09_mt_pxr_cyp_ahr_car_mednr \
  --target-cols pxr_pEC50 chembl_cyp3a4_pchembl chembl_ahr_pchembl chembl_car_pchembl \
  --task-weights 1.0 0.2 0.1 0.1 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

**R10 — MT + NR (CYP small, NR small)**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R10_mt_allnr_very_small \
  --target-cols pxr_pEC50 chembl_cyp3a4_pchembl chembl_ahr_pchembl chembl_car_pchembl \
  --task-weights 1.0 0.05 0.05 0.05 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

**R11 — MT + NR (CYP heavier, NR tiny)**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R11_mt_cyp_heavy_nr_tiny \
  --target-cols pxr_pEC50 chembl_cyp3a4_pchembl chembl_ahr_pchembl chembl_car_pchembl \
  --task-weights 1.0 0.5 0.02 0.02 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

### Phase 3 — add ADME endpoints (4 runs)

Targets: PXR + CYP + all 6 ADME endpoints (total 8 tasks)

**R12 — MT PXR+CYP+ADME (all ADME tiny)**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R12_mt_pxr_cyp_adme_tiny \
  --target-cols \
    pxr_pEC50 chembl_cyp3a4_pchembl \
    adme_log_hlm_clint adme_log_rlm_clint adme_log_mdr1_mdck_er \
    adme_log_solubility_ph68 adme_log_ppb_human_unbound adme_log_ppb_rat_unbound \
  --task-weights 1.0 0.2 0.05 0.05 0.05 0.05 0.05 0.05 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

**R13 — MT PXR+CYP+ADME (ADME ultra-tiny)**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R13_mt_pxr_cyp_adme_ultratiny \
  --target-cols \
    pxr_pEC50 chembl_cyp3a4_pchembl \
    adme_log_hlm_clint adme_log_rlm_clint adme_log_mdr1_mdck_er \
    adme_log_solubility_ph68 adme_log_ppb_human_unbound adme_log_ppb_rat_unbound \
  --task-weights 1.0 0.2 0.02 0.02 0.02 0.02 0.02 0.02 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

**R14 — MT PXR+ADME only (no CYP)**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R14_mt_pxr_adme_only \
  --target-cols \
    pxr_pEC50 \
    adme_log_hlm_clint adme_log_rlm_clint adme_log_mdr1_mdck_er \
    adme_log_solubility_ph68 adme_log_ppb_human_unbound adme_log_ppb_rat_unbound \
  --task-weights 1.0 0.05 0.05 0.05 0.05 0.05 0.05 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

**R15 — MT PXR+CYP+ADME+NR (everything, tiny auxiliaries)**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R15_mt_everything_tinyaux \
  --target-cols \
    pxr_pEC50 chembl_cyp3a4_pchembl chembl_ahr_pchembl chembl_car_pchembl \
    adme_log_hlm_clint adme_log_rlm_clint adme_log_mdr1_mdck_er \
    adme_log_solubility_ph68 adme_log_ppb_human_unbound adme_log_ppb_rat_unbound \
  --task-weights 1.0 0.2 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

### Phase 4 — architecture/feature knobs on the best MT setting (4 runs)

Pick your current best multitask target/weight setting (I default to PXR+CYP w=0.2 here).

**R16 — MT PXR+CYP w=0.2 + extra RDKit2D + MorganCount features**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R16_mt_pxr_cyp_w02_feats \
  --target-cols pxr_pEC50 chembl_cyp3a4_pchembl \
  --task-weights 1.0 0.2 \
  --molecule-featurizers rdkit_2d_normalized morgan_count \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

**R17 — MT PXR+CYP w=0.2 + deeper net**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R17_mt_pxr_cyp_w02_depth4 \
  --target-cols pxr_pEC50 chembl_cyp3a4_pchembl \
  --task-weights 1.0 0.2 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV} \
  --chemprop-args --depth 4
```

**R18 — MT PXR+CYP w=0.2 + wider hidden dims**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R18_mt_pxr_cyp_w02_h600 \
  --target-cols pxr_pEC50 chembl_cyp3a4_pchembl \
  --task-weights 1.0 0.2 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV} \
  --chemprop-args --message-hidden-dim 600 --ffn-hidden-dim 600
```

**R19 — MT PXR+CYP w=0.2 + dropout sweep**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R19_mt_pxr_cyp_w02_do02 \
  --target-cols pxr_pEC50 chembl_cyp3a4_pchembl \
  --task-weights 1.0 0.2 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV} \
  --chemprop-args --dropout 0.2
```

### Phase 5 — pretrain → finetune (optional extras; not part of the 20 core runs)

Chemprop supports loading a prior checkpoint/model with `--checkpoint`. This is useful even if multitask overlap is low.

**R20 — pretrain on CYP3A4-only (analog-mimic splits)**
```bash
source .venv/bin/activate
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R20_pretrain_cyp_only \
  --target-col chembl_cyp3a4_pchembl \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV}
```

**R21 — finetune PXR-only initialized from a CYP checkpoint**

Pick a single checkpoint to use (e.g. split 0):
`CKPT=runs/R20_pretrain_cyp_only/replicate_0/model_0/best.pt`

```bash
source .venv/bin/activate
CKPT=runs/R20_pretrain_cyp_only/replicate_0/model_0/best.pt
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R21_finetune_pxr_from_cyp \
  --target-col pxr_pEC50 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV} \
  --chemprop-args --checkpoint ${CKPT}
```

**R22 — finetune PXR-only from CYP, with frozen encoder**
```bash
source .venv/bin/activate
CKPT=runs/R20_pretrain_cyp_only/replicate_0/model_0/best.pt
python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} --out-dir runs/R22_finetune_pxr_from_cyp_freeze \
  --target-col pxr_pEC50 \
  --splits-file ${SPLITS_ANALOG} \
  --epochs ${EPOCHS} --warmup-epochs ${WARMUP} --batch-size ${BS} --patience ${PATIENCE} \
  --accelerator ${ACCEL} --devices ${DEV} \
  --chemprop-args --checkpoint ${CKPT} --freeze-encoder
```

---

## Ensembling (recommended for final submission)

Once you have a few strong *single* models on analog-mimic splits (e.g., R04/R08/R16), you can ensemble their **blinded test** predictions.

Example (uniform average):

```bash
source .venv/bin/activate

python scripts/ensemble_preds.py \
  --preds-csv \
    runs/R04_mt_pxr_cyp_w02_final/test_preds.csv \
    runs/R08_mt_pxr_cyp_ahr_car_smallnr_final/test_preds.csv \
    runs/R16_mt_pxr_cyp_w02_feats_final/test_preds.csv \
  --out-csv runs/ENS_final/test_preds.csv \
  --target-col pxr_pEC50
```

---

## Final-fit → predict blinded test → submission

After selecting the best config, do a **final-fit** training run *without* a fixed splits file (use a small internal val split for early stopping). For a multitask model, it’s fine that the blinded test CSV has no target columns.

### Example final-fit (multitask)
```bash
source .venv/bin/activate

python scripts/train_chemeleon_chemprop.py \
  --train-csv ${DATA_MT} \
  --out-dir runs/R04_mt_pxr_cyp_w02_final \
  --target-cols pxr_pEC50 chembl_cyp3a4_pchembl \
  --task-weights 1.0 0.2 \
  --epochs 120 --warmup-epochs 5 --batch-size ${BS} --patience 15 \
  --accelerator ${ACCEL} --devices ${DEV} \
  --split SCAFFOLD_BALANCED
```

### Predict on blinded test
```bash
source .venv/bin/activate
python scripts/predict_chemprop.py \
  --model-path runs/R04_mt_pxr_cyp_w02_final/model_0/best.pt \
  --test-csv data/activity_test_blinded.csv \
  --preds-csv runs/R04_mt_pxr_cyp_w02_final/test_preds.csv \
  --accelerator ${ACCEL} --devices ${DEV}
```

### Make + validate submission (force correct pred col)
```bash
source .venv/bin/activate
mkdir -p submissions

python scripts/make_submission.py \
  --test-csv data/activity_test_blinded.csv \
  --preds-csv runs/R04_mt_pxr_cyp_w02_final/test_preds.csv \
  --pred-col pxr_pEC50 \
  --out-csv submissions/R04_mt_pxr_cyp_w02_final.csv

python scripts/validate_submission.py submissions/R04_mt_pxr_cyp_w02_final.csv
```

---

## Notes / expected outcomes

- Because overlap is low, the most plausible multitask gains come from **representation regularization** rather than shared labeled examples.
- If any auxiliary task hurts, reduce its weight aggressively (e.g., 0.01–0.05) or drop it.
- Once you have 5–10 submissions, it’s often better to **ensemble** than to keep tuning tiny hyperparameters.
