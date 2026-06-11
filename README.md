# OpenADMET PXR — CheMeleon + Suiren Ensemble

Reproduces `submissions/ens_cm_lr3e-04_3seed_sur_w0.3.csv` (unblinded MAE 0.437),
the best submission for the OpenADMET PXR activity prediction challenge.

The ensemble blends two models:
- **Suiren iw2** — a graph neural network fine-tuned from the Suiren pretrain checkpoint with upweighted inactive and tail compounds, averaged over 3 seeds × 7 epoch checkpoints (epochs 17–23).
- **CheMeleon** — Chemprop v2's foundation MPNN fine-tuned on a 3-task target (pEC50 + two log2FC auxiliary heads), averaged over 3 seeds × 7 epoch checkpoints.

Final blend: `pEC50 = 0.3 × CheMeleon + 0.7 × Suiren`, clipped to [1.0, 8.5].

## Environment setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This creates `.venv/` and installs all pinned dependencies from `uv.lock`, including
PyTorch, PyTorch Geometric, Chemprop 2.2, Lightning, RDKit, and scipy.

> **Note:** `models/suiren/weights/model.pt` (the Suiren pretrain checkpoint) is
> committed to this branch and will be present after cloning. CheMeleon weights
> (~60 MB) are downloaded automatically from Zenodo on first run and cached at
> `~/.chemprop/chemeleon_mp.pt`.

## Reproducing the submission

Run the three steps below in order. Each step is independent and can be skipped
if its output file already exists.

### Step 1 — Train Suiren iw2 (3 seeds)

Trains Suiren with `active_weight=2.0`, `tail_weight=2.0` on all challenge
training compounds. Saves epoch checkpoints 17–23 per seed, then averages all
21 checkpoint predictions into `submissions/iw2_3seed_ep17-23.csv`.

```bash
.venv/bin/python3 scripts/train_inactive_weight.py
```

Optional flags:
```
--seeds 0 1 2        seeds to train (default: 0 1 2)
--skip-training      skip training and run inference + averaging only
```

Checkpoints are written to `runs/inactive_weight/seed_{n}/`.

### Step 2 — Train CheMeleon (3 seeds)

Fine-tunes the CheMeleon MPNN on pEC50 + log2fc auxiliary tasks. Run once per
seed; each invocation trains for 25 epochs and writes a per-seed submission to
`submissions/chemeleon_mt_lr3e-04_s{seed}.csv`.

```bash
.venv/bin/python3 scripts/train_chemeleon.py --seed 0 --max-lr 3e-4
.venv/bin/python3 scripts/train_chemeleon.py --seed 1 --max-lr 3e-4
.venv/bin/python3 scripts/train_chemeleon.py --seed 2 --max-lr 3e-4
```

Checkpoints are written to `runs/chemeleon_mt_lr3e-04/seed{n}/checkpoints/`.

### Step 3 — Assemble the ensemble

Averages the 3 CheMeleon seeds, then blends with the Suiren component at w=0.3.

```bash
.venv/bin/python3 scripts/build_ensemble.py --evaluate
```

`--evaluate` scores the result against `data/phase1_unblinded.csv` if present.

Output: `submissions/ens_cm_lr3e-04_3seed_sur_w0.3.csv`

## Data files

| File | Description |
|------|-------------|
| `data/pxr-challenge_TRAIN.csv` | Challenge training labels (used by Suiren) |
| `data/multitask_train.csv` | Training labels + log2fc auxiliary targets (used by CheMeleon) |
| `data/pxr-challenge_TEST_BLINDED.csv` | Test SMILES for submission generation |
| `data/phase1_unblinded.csv` | Unblinded test labels for local evaluation |
| `submissions/iw2_3seed_ep17-23.csv` | Pre-built Suiren component (output of Step 1) |
