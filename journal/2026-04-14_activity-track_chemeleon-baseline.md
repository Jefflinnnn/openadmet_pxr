# 2026-04-14 — Activity Track CheMeleon (Chemprop) baseline

## Goal
Produce a **valid Activity Track submission** for the OpenADMET PXR Blind Challenge using a simple baseline:

- Model: **Chemprop v2** fine-tuned **from the CheMeleon foundation** (`--from-foundation CheMeleon`)
- Task: regression predicting **pEC50**
- Hardware: Apple Silicon (use **MPS** via PyTorch)

## What I did (high-level)
1. Set up a uv-managed Python 3.11 environment (Apple Silicon compatible).
2. Downloaded competition data from Hugging Face dataset `openadmet/pxr-challenge-train-test`.
3. Implemented small CLI wrapper scripts to:
   - train a Chemprop model from CheMeleon
   - run inference on the blinded test set
   - assemble a submission with the required schema
   - validate the submission locally (rows/cols/finite values)
4. Ran a quick **smoke training** (3 epochs) to confirm the pipeline works.
5. Generated predictions and built a **validated submission**.

## Reasoning / decisions
- **Focus only on Activity Track** first to de-risk submission format and end-to-end tooling.
- Use **Chemprop CLI** rather than re-implementing model code to reduce surface area for bugs.
- Use **uv** for environment management because system Python was older and Apple Silicon wheels can be finicky.

## Commands that produced the working submission

### Predict on blinded test
```bash
source .venv/bin/activate

python scripts/predict_chemprop.py \
  --model-path runs/chemeleon_smoke/model_0/best.pt \
  --test-csv data/activity_test_blinded.csv \
  --preds-csv runs/chemeleon_smoke/test_preds.csv
```

### Build + validate submission
```bash
source .venv/bin/activate

mkdir -p submissions
python scripts/make_submission.py \
  --test-csv data/activity_test_blinded.csv \
  --preds-csv runs/chemeleon_smoke/test_preds.csv \
  --out-csv submissions/chemeleon_smoke.csv

python scripts/validate_submission.py submissions/chemeleon_smoke.csv
```

Validation output observed:
- rows=513
- columns exactly: `SMILES`, `Molecule Name`, `pEC50`
- pEC50 finite, with range approx `(2.656, 6.137)`

## Artifacts
- Model checkpoint used:
  - `runs/chemeleon_smoke/model_0/best.pt`
- Predictions:
  - `runs/chemeleon_smoke/test_preds.csv`
- Submission:
  - `submissions/chemeleon_smoke.csv`

## Next steps
- Run a longer training job (more epochs / tuning) to improve performance.
- Once satisfied, upload `submissions/*.csv` to the HF Space for scoring.
