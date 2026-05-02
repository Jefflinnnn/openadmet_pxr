# Suiren-ConfAvg Modeling Plan

## Goal

Establish a baseline with Suiren-ConfAvg fine-tuned on PXR pEC50, then iteratively
augment the training pipeline to reduce RAE on the blind test set.

Each stage builds on the previous. Run analog-mimic 5-fold CV before submitting.

**Primary diagnostic:** always report overall RAE and active-subgroup RAE separately.
Overall RAE is dominated by inactives and can look fine while the active region stays
broken. Active-subgroup RAE (pEC50 ≥ 6 in the validation fold) is the true signal.

---

## Stage 0 — Baseline (simplest possible)

Fine-tune Suiren-ConfAvg end-to-end on training data with no modifications.

- Input: SMILES only
- Target: pEC50
- Loss: L1 (MAE) — more robust to the long tail of inactives than MSE
- Sample weights: uniform
- Data: 4,139 training compounds
- Split: analog-mimic 5-fold CV (`runs/analog_mimic_5fold.json`)
- Config: default Suiren args (`--mode regression --loss l1 --epochs 100 --lr 2e-4`)

**Purpose:** establish a CV RAE floor and confirm Suiren representations are useful
at all before adding complexity. Also reveals whether active underprediction bias is
present (if yes, active-subgroup RAE >> overall RAE).

---

## Stage 1 — Active upweighting sweep

Add per-sample weights that upweight actives (pEC50 ≥ 6) during training. Requires
a custom training loop since Suiren's default loop does not support sample weights.

**Sweep weights:** 2×, 3×, 5×, 10× — do not commit to 5× without evidence.

**Risk:** too-high weight overfits the 67 actives (great active-subgroup CV, worse
inactive region). With ~13 actives per fold, CV variance on the active subset is high
— confirm improvement is consistent across folds, not driven by one lucky fold.

**Also check:** if active-subgroup RAE doesn't improve from Stage 0, upweighting
alone won't fix it — the representations may not distinguish the active region, in
which case Stage 4 (more active-like structures) matters more than loss weighting.

**Gate:** proceed only if active-subgroup RAE improves at ≥ 3 of 5 folds.

---

## Stage 2 — Consolidated sample weighting

Replace the Stage 1 active weight with a single combined weight that folds in
measurement uncertainty. Stages 2 and 3 from the original plan are merged here
because they target the same ~67 compounds and stacking them multiplicatively
risks over-correcting on a small active set.

Combined weight:
```
active_weight = W if pEC50 ≥ 6 else 1.0   # W from Stage 1 sweep
se_weight     = exp(−pEC50_std.error)
final_weight  = active_weight × se_weight
```

**Selectivity weight is deliberately excluded.** Counter-assay data only exists for
compounds that reached EC50 ≤ 1 µM — almost exclusively actives. Adding a third
multiplicative term on the same 67 compounds risks making some genuine actives
near-zero weight. Selectivity is better used as a data quality filter: inspect the
187 compounds with sel_weight < 0.4 and decide whether to exclude them entirely
rather than down-weighting them to near-zero.

**Anti-correlation check:** se_weight and active_weight are partially anti-correlated
(noisy dose-response curves are common at the edge of measurability, i.e., actives).
Verify the mean combined weight on actives is still materially higher than on inactives
after multiplication — if not, se_weight is cancelling the active upweighting.

---

## Stage 3 — Selective pseudo-labeling (active-only)

Expand the training set using pseudo-labels, but only for likely-active unlabeled
compounds. Do not pseudo-label the full 8,126 unlabeled set.

**Why not full pseudo-labeling:** the log2FC → pEC50 mapping is fit on compounds
promoted to DRC — a biased sample skewed toward actives. Applying it to 8,126
mostly-inactive unlabeled compounds generates systematically wrong pseudo-labels for
the inactive majority. Even at 0.3× weight, ~7,500 wrong pseudo-labels can hurt.

**Procedure:**
1. Filter unlabeled single-conc compounds to log2FC > 0.5 at 8.25 µM (~577 compounds,
   7.1% of unlabeled). These are likely true actives that didn't reach the DRC cut.
2. Fit log2FC → pEC50 on labeled compounds that have both values (use only the
   active/near-active region: pEC50 ≥ 5.5 or log2FC > 0.3, so the mapping is
   calibrated where it will be applied).
3. Add the ~577 pseudo-labeled compounds at 0.3× weight with active upweighting
   applied (same W from Stage 1).

**Expected gain:** adds ~577 structurally informative actives — nearly 9× the
current active count. Even at 0.3× weight this is a meaningful signal expansion.

**Risk:** pseudo-labels have unknown error. Monitor whether active-subgroup RAE
improves or degrades — if it degrades, the pseudo-labels are confusing rather than
helping.

---

## Stage 4 — Seed ensemble (same config)

Train 5–7 models on full data with identical config, varying only random seed.
Average predictions.

**Why seed-only, not LR/dropout variants:** ensembling models of different quality
(a slightly overfit high-LR model averaged with an underfit low-LR model) can hurt.
Seed variation captures genuine stochasticity in the fine-tuning without introducing
quality differences. If seed ensemble helps, then try a small LR range (1e-4, 2e-4
only — not 4e-4 which risks overfitting).

**Expected diversity:** pre-trained backbones on small fine-tuning sets tend to
converge to similar solutions across seeds. Measure prediction variance across seeds
on the validation set before committing — if std across seeds < 0.05 pEC50 units,
the ensemble will provide negligible benefit.

**Expected gain:** −0.02 to −0.05 RAE if seed diversity is adequate.

---

## Stage 5 — GP residual correction (conditional)

Post-hoc correction applied only if a systematic active underprediction bias
remains after Stage 3.

**When to use:** if active-subgroup RAE is still substantially worse than overall
RAE, and training residuals show a pattern (actives consistently negative residuals).

**When NOT to use:**
- If earlier stages already fixed the active bias (training residuals near zero
  for actives — the GP will correct nothing)
- If activity cliffs are sharp (Tanimoto 0.4–0.8 neighbours of actives are mostly
  inactive here) — an RBF-kernel GP will smooth across the cliff and may pull
  inactive predictions upward near actives, creating false positives

**Safer alternative:** instead of GP on fingerprints, fit a simple linear correction
`pred_corrected = pred + β × (active_indicator)` where β is estimated from training
residuals. Less flexible but won't hallucinate smooth gradients across cliffs.

---

## Evaluation protocol

- Primary: analog-mimic 5-fold CV RAE (`runs/analog_mimic_5fold.json`)
- Report both **overall RAE** and **active-subgroup RAE** (pEC50 ≥ 6 in val fold)
- Gate each stage: only proceed if active-subgroup RAE improves at ≥ 3 of 5 folds
- Submit to leaderboard only after a stage clears the CV gate
- Do not tune on leaderboard — CV is the development signal

## Key file references

- CV splits: `runs/analog_mimic_5fold.json`
- Training data: `pxr-challenge_TRAIN.csv`
- Single-conc data: `pxr-challenge_single_concentration_TRAIN.csv`
- Counter-assay data: `pxr-challenge_counter-assay_TRAIN.csv`
- Suiren repo: https://github.com/golab-ai/Suiren-Property-Prediction
- Suiren weights: https://huggingface.co/ajy112/Suiren-ConfAvg
