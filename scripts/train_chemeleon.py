"""
CheMeleon (Chemprop v2 foundation model) multi-task training.

Tasks: pEC50 (primary) + log2fc_8um + log2fc_33um (auxiliary, NaN-masked).
Epoch averaging over epochs 17-23 (same protocol as Suiren best).

Usage:
    .venv/bin/python3 scripts/train_chemeleon.py [--seed 0] [--epochs 25]
        [--active-weight 1.0] [--tail-weight 1.0] [--active-threshold 6.0]
        [--tail-threshold 3.0]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MULTITASK_CSV = os.path.join(PROJECT_ROOT, "data/multitask_train.csv")
TEST_CSV      = os.path.join(PROJECT_ROOT, "data/pxr-challenge_TEST_BLINDED.csv")
UNBLINDED_CSV = os.path.join(PROJECT_ROOT, "data/phase1_unblinded.csv")
SUIREN_BEST   = os.path.join(PROJECT_ROOT, "submissions/iw2_3seed_ep17-23.csv")

SAVE_EPOCHS = list(range(17, 24))
PRED_MIN, PRED_MAX = 1.0, 8.5


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_datasets(active_weight=1.0, tail_weight=1.0,
                   active_threshold=6.0, tail_threshold=3.0):
    from chemprop import data as cdata

    df = pd.read_csv(MULTITASK_CSV)

    # Val: 50 inactives held out (same seed/protocol as all other scripts)
    inactive_mask = df["pEC50"] < 6.0
    inactives = df[inactive_mask]
    rng = np.random.default_rng(42)
    val_iloc = rng.choice(len(inactives), size=50, replace=False)
    val_index = inactives.iloc[val_iloc].index
    val_df   = df.loc[val_index]
    train_df = df.drop(index=val_index)

    def sample_weight(pec50):
        if pec50 >= active_threshold:
            return active_weight
        if pec50 < tail_threshold:
            return tail_weight
        return 1.0

    def make_points(frame, weighted=False):
        pts = []
        for _, row in frame.iterrows():
            y = np.array([
                row["pEC50"],
                row["log2fc_8um"]  if pd.notna(row["log2fc_8um"])  else np.nan,
                row["log2fc_33um"] if pd.notna(row["log2fc_33um"]) else np.nan,
            ], dtype=np.float64)
            w = sample_weight(row["pEC50"]) if weighted else 1.0
            pts.append(cdata.MoleculeDatapoint.from_smi(row["SMILES"], y, weight=w))
        return pts

    train_ds = cdata.MoleculeDataset(make_points(train_df, weighted=True))
    val_ds   = cdata.MoleculeDataset(make_points(val_df, weighted=False))

    n_active = (train_df["pEC50"] >= active_threshold).sum()
    n_tail   = (train_df["pEC50"] <  tail_threshold).sum()
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"  Train actives (>={active_threshold}): {n_active}  "
          f"tail (<{tail_threshold}): {n_tail}  "
          f"aw={active_weight}  tw={tail_weight}")
    return train_ds, val_ds


def build_test_dataset():
    from chemprop import data as cdata
    df = pd.read_csv(TEST_CSV)
    pts = [cdata.MoleculeDatapoint.from_smi(smi, np.zeros(3))
           for smi in df["SMILES"]]
    return cdata.MoleculeDataset(pts), df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(train_ds, ffn_layers=1, ffn_hidden=300, ffn_dropout=0.0, max_lr=1e-3):
    from chemprop import nn as cnn, models, data as cdata

    # Normalise all 3 targets in-place; returns sklearn StandardScaler
    scaler = train_ds.normalize_targets()

    # Load CheMeleon weights via the CLI's internal path
    from pathlib import Path
    import urllib.request
    ckpt_dir = Path.home() / ".chemprop"
    ckpt_dir.mkdir(exist_ok=True)
    model_path = ckpt_dir / "chemeleon_mp.pt"
    if not model_path.exists():
        print("  Downloading CheMeleon weights from Zenodo...")
        urllib.request.urlretrieve(
            "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
            model_path,
        )
        print("  Download complete.")

    chemeleon_mp = torch.load(model_path, weights_only=True)
    mp = cnn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
    mp.load_state_dict(chemeleon_mp["state_dict"])

    agg = cnn.NormAggregation()
    ffn = cnn.RegressionFFN(
        input_dim=mp.output_dim,
        n_tasks=3,
        hidden_dim=ffn_hidden,
        n_layers=ffn_layers,
        dropout=ffn_dropout,
    )

    # ScaleTransform wraps the scaler so the model un-normalises predictions
    output_transform = cnn.transforms.UnscaleTransform.from_standard_scaler(scaler)
    ffn.output_transform = output_transform

    model = models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        warmup_epochs=2,
        init_lr=1e-4,
        max_lr=max_lr,
        final_lr=1e-5,
    )
    return model, scaler


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_seed(seed, n_epochs, run_dir, active_weight=1.0, tail_weight=1.0,
               active_threshold=6.0, tail_threshold=3.0,
               ffn_layers=1, ffn_hidden=300, ffn_dropout=0.0, max_lr=1e-3):
    from chemprop import data as cdata

    L.seed_everything(seed)

    train_ds, val_ds = build_datasets(
        active_weight=active_weight, tail_weight=tail_weight,
        active_threshold=active_threshold, tail_threshold=tail_threshold,
    )
    model, scaler = build_model(
        train_ds,
        ffn_layers=ffn_layers, ffn_hidden=ffn_hidden,
        ffn_dropout=ffn_dropout, max_lr=max_lr,
    )

    train_loader = cdata.build_dataloader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
    val_loader   = cdata.build_dataloader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

    ckpt_dir = os.path.join(run_dir, f"seed{seed}", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save every epoch so we can average epochs 17-23
    every_epoch_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:02d}",
        save_top_k=-1,
        every_n_epochs=1,
        save_last=False,
    )

    trainer = L.Trainer(
        max_epochs=n_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[every_epoch_cb],
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=True,
    )
    trainer.fit(model, train_loader, val_loader)

    return model, scaler, ckpt_dir


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(ckpt_path):
    """Reconstruct MPNN with scaler embedded in checkpoint, then load state dict."""
    from chemprop import nn as cnn, models
    from sklearn.preprocessing import StandardScaler
    from pathlib import Path

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    # Scaler values are stored inside the output_transform
    mean_t  = sd["predictor.output_transform.mean"].numpy().flatten()
    scale_t = sd["predictor.output_transform.scale"].numpy().flatten()

    chemeleon_mp = torch.load(
        Path.home() / ".chemprop/chemeleon_mp.pt", weights_only=True
    )
    mp  = cnn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
    agg = cnn.NormAggregation()

    # Infer FFN shape from state dict to support non-default configs
    ffn_keys = [k for k in sd if k.startswith("predictor.ffn.") and k.endswith(".weight")]
    # RegressionFFN keys look like predictor.ffn.N.M.weight; interior layers have hidden->hidden shape
    hidden_dim = 300
    n_layers = 1
    if ffn_keys:
        # Last weight before output has shape (n_tasks, hidden) or (hidden, hidden)
        # Count hidden->hidden layers to infer n_layers
        interior = [k for k in ffn_keys
                    if sd[k].shape[0] == sd[k].shape[1]]  # square = hidden->hidden
        n_layers = len(interior) + 1
        if interior:
            hidden_dim = sd[interior[0]].shape[0]
        else:
            # Single layer: shape is (n_tasks, input_dim) — fall back to default
            hidden_dim = 300
    ffn = cnn.RegressionFFN(input_dim=mp.output_dim, n_tasks=3,
                            hidden_dim=hidden_dim, n_layers=n_layers)

    sc = StandardScaler()
    sc.mean_ = mean_t
    sc.scale_ = scale_t
    sc.var_   = scale_t ** 2
    sc.n_features_in_ = len(mean_t)
    ffn.output_transform = cnn.transforms.UnscaleTransform.from_standard_scaler(sc)

    model = models.MPNN(
        message_passing=mp, agg=agg, predictor=ffn,
        warmup_epochs=2, init_lr=1e-4, max_lr=1e-3, final_lr=1e-5,
    )
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def predict_checkpoint(ckpt_path, test_ds):
    from chemprop import data as cdata

    model = load_model_from_checkpoint(ckpt_path)
    loader = cdata.build_dataloader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    trainer = L.Trainer(
        accelerator="auto", devices=1,
        enable_progress_bar=False, logger=False,
        enable_checkpointing=False,
    )
    batches = trainer.predict(model, loader)
    preds = torch.cat(batches, dim=0).numpy()  # (n, 3)
    return np.clip(preds[:, 0].astype(np.float32), PRED_MIN, PRED_MAX)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_mae(true, pred, label=""):
    ranges = [
        ("tail  <3.0",    true < 3.0),
        ("low  3-4.5",    (true >= 3.0) & (true < 4.5)),
        ("mid  4.5-5.5",  (true >= 4.5) & (true < 5.5)),
        ("hi   5.5-6.0",  (true >= 5.5) & (true < 6.0)),
        ("act  >=6.0",    true >= 6.0),
        ("ALL",           np.ones(len(true), dtype=bool)),
    ]
    print(f"\n  {label}")
    print(f"  {'Region':<18} {'n':>4}  {'MAE':>6}  {'bias':>7}")
    print("  " + "-" * 42)
    for name, mask in ranges:
        if mask.sum() == 0:
            continue
        mae  = float(np.abs(true[mask] - pred[mask]).mean())
        bias = float((pred[mask] - true[mask]).mean())
        print(f"  {name:<18} {mask.sum():>4}  {mae:>6.3f}  {bias:>+7.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",             type=int,   default=0)
    parser.add_argument("--epochs",           type=int,   default=25)
    parser.add_argument("--active-weight",    type=float, default=1.0,
                        help="Loss weight for actives (pEC50 >= active-threshold)")
    parser.add_argument("--tail-weight",      type=float, default=1.0,
                        help="Loss weight for tail compounds (pEC50 < tail-threshold)")
    parser.add_argument("--active-threshold", type=float, default=6.0)
    parser.add_argument("--tail-threshold",   type=float, default=3.0)
    parser.add_argument("--ffn-layers",       type=int,   default=1,
                        help="Number of FFN layers in the prediction head")
    parser.add_argument("--ffn-hidden",       type=int,   default=300,
                        help="Hidden dim of FFN layers")
    parser.add_argument("--ffn-dropout",      type=float, default=0.0,
                        help="Dropout in FFN layers")
    parser.add_argument("--max-lr",           type=float, default=1e-3,
                        help="Peak learning rate (default 1e-3)")
    args = parser.parse_args()

    # Build a descriptive run tag
    tag_parts = []
    if args.active_weight != 1.0 or args.tail_weight != 1.0:
        tag_parts.append(f"aw{args.active_weight}_tw{args.tail_weight}")
    if args.ffn_layers != 1 or args.ffn_hidden != 300 or args.ffn_dropout != 0.0:
        tag_parts.append(f"ffn{args.ffn_layers}x{args.ffn_hidden}_do{args.ffn_dropout}")
    if args.max_lr != 1e-3:
        tag_parts.append(f"lr{args.max_lr:.0e}")
    weight_tag = ("_" + "_".join(tag_parts)) if tag_parts else ""

    run_dir = os.path.join(PROJECT_ROOT, f"runs/chemeleon_mt{weight_tag}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "submissions"), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CheMeleon MT  seed={args.seed}  epochs={args.epochs}")
    print(f"  ffn_layers={args.ffn_layers}  ffn_hidden={args.ffn_hidden}  "
          f"ffn_dropout={args.ffn_dropout}  max_lr={args.max_lr:.0e}")
    print(f"  aw={args.active_weight}  tw={args.tail_weight}")
    print(f"{'='*60}")

    model, scaler, ckpt_dir = train_seed(
        args.seed, args.epochs, run_dir,
        active_weight=args.active_weight, tail_weight=args.tail_weight,
        active_threshold=args.active_threshold, tail_threshold=args.tail_threshold,
        ffn_layers=args.ffn_layers, ffn_hidden=args.ffn_hidden,
        ffn_dropout=args.ffn_dropout, max_lr=args.max_lr,
    )

    test_ds, test_df = build_test_dataset()

    # Collect epoch predictions.
    # Lightning uses 0-indexed epochs: our ep=17 (1-indexed) = epoch=16 on disk.
    import glob
    all_preds = []
    for ep in SAVE_EPOCHS:
        lightning_ep = ep - 1  # convert 1-indexed → 0-indexed
        matches = glob.glob(os.path.join(ckpt_dir, f"epoch={lightning_ep:02d}.ckpt"))
        if not matches:
            print(f"  WARNING: checkpoint for epoch {ep} (epoch={lightning_ep:02d}.ckpt) not found, skipping")
            continue
        ckpt_path = sorted(matches)[-1]
        preds = predict_checkpoint(ckpt_path, test_ds)
        all_preds.append(preds)
        print(f"  Epoch {ep}: mean={preds.mean():.3f}  std={preds.std():.3f}")

    if not all_preds:
        print("ERROR: no predictions generated.")
        sys.exit(1)

    avg_pred = np.clip(np.nanmean(all_preds, axis=0), PRED_MIN, PRED_MAX)
    print(f"\n  Averaged {len(all_preds)} checkpoints: mean={avg_pred.mean():.3f}  std={avg_pred.std():.3f}")

    sub = test_df[["SMILES", "Molecule Name"]].copy()
    sub["pEC50"] = avg_pred
    sub_path = os.path.join(
        PROJECT_ROOT, f"submissions/chemeleon_mt{weight_tag}_s{args.seed}.csv"
    )
    sub.to_csv(sub_path, index=False)
    print(f"  Saved: {sub_path}")

    # Unblinded evaluation
    if not os.path.exists(UNBLINDED_CSV):
        print("  Unblinded labels not found — skipping evaluation.")
        return

    ub = pd.read_csv(UNBLINDED_CSV)[["Molecule Name", "pEC50"]].rename(columns={"pEC50": "true"})
    ev = test_df[["Molecule Name"]].copy()
    ev["pred"] = avg_pred
    ev = ev.merge(ub, on="Molecule Name")
    true_v = ev["true"].values
    pred_v = ev["pred"].values
    print(f"\n  Evaluated on {len(ev)} unblinded compounds.")
    eval_mae(true_v, pred_v,
             label=f"CheMeleon MT{weight_tag} (seed={args.seed}, ep17-23 avg)")

    if os.path.exists(SUIREN_BEST):
        sur = pd.read_csv(SUIREN_BEST)[["Molecule Name", "pEC50"]].rename(columns={"pEC50": "suiren"})
        ev2 = ev.merge(sur, on="Molecule Name")
        true_v = ev2["true"].values
        pred_v = ev2["pred"].values
        sur_v  = ev2["suiren"].values
        eval_mae(true_v, sur_v, label="Suiren best (iw2_3seed) — reference")
        print("\n  Ensemble sweep (CheMeleon + Suiren):")
        for w in [0.2, 0.3, 0.4, 0.5]:
            ens = np.clip(w * pred_v + (1 - w) * sur_v, PRED_MIN, PRED_MAX)
            mae = float(np.abs(true_v - ens).mean())
            delta = mae - float(np.abs(true_v - sur_v).mean())
            print(f"    w={w:.1f}: MAE={mae:.3f} ({delta:+.3f} vs Suiren)")


if __name__ == "__main__":
    main()
