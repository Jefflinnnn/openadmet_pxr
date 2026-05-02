# Internal Testing Protocol

## Purpose

Catch model failures before they reach the leaderboard. The prior runs showed a
large CV-to-leaderboard gap (C22 CV RAE 0.523 → C13 leaderboard RAE 0.6395).
This protocol is designed to reproduce that gap internally so it can be diagnosed
and closed, not discovered on submission.

---

## Layer 1 — Analog-mimic 5-fold CV (existing)

**Splits:** `runs/analog_mimic_5fold.json`

**Metrics to report at every stage:**
- Overall RAE
- Active-subgroup RAE (pEC50 ≥ 6 compounds in the validation fold only)
- Mean and std across 5 folds for both — high std signals the model is unstable
  on the active region, not that it has learned the SAR

**Gate:** a stage passes only if active-subgroup RAE improves at ≥ 3 of 5 folds.
Overall RAE improvement alone is not sufficient — a mean-predictor can improve
overall RAE while getting worse on actives.

---

## Layer 2 — Analog-set cross-validation

Evaluates whether the model captures SAR within analogue series, not just absolute
pEC50 accuracy. A model that predicts everything near the mean can pass Layer 1
but will fail here.

### Cluster construction

Group training compounds into analogue series using Tanimoto similarity on ECFP4:
- Seed each cluster from an active (pEC50 ≥ 6)
- Add any compound with Tanimoto ≥ 0.4 to the nearest active seed
- Compounds with no active neighbour at ≥ 0.4 are excluded from this analysis
  (they're not part of any analogue series and don't contribute SAR signal)

`notebooks/03_cheminformatics_eda.py` has the cliff pair enumeration infrastructure
and Tanimoto computation needed — extend it to produce cluster assignments.

### CV procedure

For each analogue cluster with ≥ 4 members:
1. Hold out one compound at a time (leave-one-out within the cluster)
2. Train on all other compounds (full training set minus the held-out compound)
3. Predict the held-out compound

### Metrics

**Primary: within-series Spearman ρ**
For each cluster, compute Spearman correlation between predicted and true pEC50
rankings. Average across clusters weighted by cluster size.
- ρ ≥ 0.7: model is learning the SAR
- ρ 0.4–0.7: partial SAR capture — some series learned, some not
- ρ < 0.4: model is not learning relative potency within series

**Secondary: mean |Δpred − Δtrue| across analogue pairs**
For all pairs within a cluster, measure how well the model captures the magnitude
of potency differences, not just the direction.
- Complements Spearman ρ — a model can rank correctly but compress all Δ values
  toward zero (underpredicts cliff magnitude)

**Tertiary: per-cluster ρ variance across seeds**
If within-series ρ swings by > 0.3 across seeds for the same cluster, the model
has not learned that series — it's producing different random orderings each run.
This variance should decrease as the model improves.

### Interpretation

A model with good Layer 1 RAE but poor Layer 2 Spearman ρ is a mean-predictor
in disguise. It will score acceptably on the leaderboard for inactives but fail
on the analogue-expansion test compounds where fine-grained SAR is required.

---

## Layer 3 — Residual diagnostics

Run after any full-data training (not CV) to catch systematic bias before submission.

### Plots to generate

1. **Predicted vs. actual pEC50** — colored by active (pEC50 ≥ 6) vs. inactive.
   Active underprediction bias is visible as actives clustering below the diagonal.

2. **Residual vs. predicted pEC50** — should be flat and centered at zero across
   the full range. A systematic negative residual at high pEC50 confirms active
   underprediction.

3. **Residual vs. molecular weight and LogP** — actives cluster in high-MW,
   high-LogP space. If residuals are systematically negative in that region,
   the model is failing on the pharmacophore-relevant chemical space.

4. **Per-fold active-subgroup RAE bar chart** — visualize fold variance. One
   bad fold can mask three good folds in the average.

### Sanity checks on test predictions

Before any submission:
- No NaN or inf values
- All 513 predictions within [1.0, 8.5] — outside this range is extrapolation
  beyond the training distribution (train range: 1.6–7.5)
- Mean test prediction within 0.3 of training mean pEC50 (4.65) — large deviation
  suggests a systematic shift introduced by a training change
- Fraction of test predictions ≥ 6.0 should be plausible given test set is an
  analogue expansion of actives — expect 10–30%, not 0% or 80%

---

## Layer 4 — Seed stability check

Before submitting an ensemble, verify the ensemble is actually adding diversity.

For each pair of seeds:
- Compute Pearson r between their test predictions
- If mean pairwise r > 0.97, the models are nearly identical — ensembling provides
  negligible benefit and the submission slot is better used on a qualitatively
  different approach

Report: mean pairwise r, and the seed pair with the lowest r (most diverse pair).

---

## Summary: decision flowchart per stage

```
Train model(s)
    │
    ├─ Layer 1: 5-fold CV
    │      Active-subgroup RAE improves ≥ 3/5 folds?
    │      No → diagnose (representation? weighting? data?) before proceeding
    │      Yes ↓
    │
    ├─ Layer 2: Analog-set CV
    │      Within-series Spearman ρ ≥ 0.5 on average?
    │      No → model is not capturing SAR; do not submit
    │      Yes ↓
    │
    ├─ Layer 3: Residual diagnostics
    │      Active underprediction bias still present?
    │      Yes → apply linear correction or investigate before submitting
    │      Sanity checks pass?
    │      No → do not submit
    │      Yes ↓
    │
    └─ Submit to leaderboard
```
