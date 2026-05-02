# PXR Challenge — Modeling Strategy

## Problem framing

Predict pEC50 (regression) for 513 blind test compounds against hPXR. Metric: RAE (lower is better).

The test set is an analogue expansion of 63 training actives (ECFP4 Tanimoto > 0.4). This is a lead optimization / SAR scenario, not a scaffold-hop generalization task. Fine-grained potency differences within analogue series are what separate top submissions.

**Core failure mode:** Active underprediction bias. 58/67 training actives are systematically underpredicted by >0.5 log units.

---

## Dataset facts relevant to strategy

- 4,139 training compounds; only 67 actives (pEC50 ≥ 6) — 1.6% of training data
- 64/67 actives have unique Murcko scaffolds — scaffold-based generalisation fails
- Activity cliffs are abrupt: Tanimoto 0.4–0.8 neighbours of actives are almost all inactive
- Counter-assay missingness is **not random** — only compounds with EC50 ≤ 1 µM were counter-screened; missing = weak binder (confirmed inactive), not missing at random
- Single-concentration log2FC at 8.25 µM: r = 0.72 with pEC50 — strongest non-structural signal
- 8,126 unlabeled compounds in the single-conc file — ~3× additional chemical space
- Fold ensemble (25 models) hurt vs. single full-data model — active sparsity per fold is the cause

---

## Strategy catalogue

Strategies are grouped by theme and ranked within each group by expected impact.

---

### A. Loss function & sample weighting

**A1. Active upweighting / focal loss** *(not implemented)*
Apply 3–5× weight to actives (pEC50 ≥ 6) during training, or use a focal-style loss
`weight_i = |pred_i - target_i|^γ` that automatically upweights hard examples.
Directly targets the underprediction bias — the model currently under-invests in the
rare active region because MSE/MAE is dominated by the 98.4% inactive majority.
Expected: −0.05 to −0.08 RAE. Start with 3× fixed weight; tune γ if using focal form.

**A2. Heteroscedastic NLL loss** *(partially implemented as scalar weight)*
Current: `se_weight = exp(−pEC50_std.error)` scalar multiplier.
Upgrade: train model to output μ and σ per compound; optimize
`loss = (pred − target)² / (2σ²) + log σ`. The model learns which compounds are
inherently uncertain rather than using fixed experimental SE. Requires architecture
that exposes a variance head.

**A3. Combined sample weight** *(implemented in C22/C23)*
`sample_weight = exp(−pEC50_std.error) × sigmoid(primary_pEC50 − counter_pEC50)`
Down-weights noisy dose-response fits and likely false positives simultaneously.
**Correction needed:** compounds with missing counter-assay data currently default to
`selectivity_weight = 1.0`, but missing = inactive (not counter-screened because
EC50 > 1 µM). These should default to `selectivity_weight = 0.5` or lower, not 1.0.

**A4. CI-weighted loss** *(not implemented)*
Use the per-compound pEC50 confidence interval width as a Gaussian likelihood weight:
`loss_i = (pred_i − target_i)² / CI_width_i²`. Distinct from A2 in that it uses
experimentally derived uncertainty rather than a learned variance. Note: CI columns
(r ≈ 0.99 with pEC50) must NOT be used as input features — data leakage — but are
valid as loss weights since they only affect training dynamics, not predictions.

---

### B. Auxiliary signals & features

**B1. Single-concentration log2FC at 8.25 µM as input feature** *(identified in EDA, use status unclear)*
r = 0.72 with pEC50 — the strongest non-structural signal in the dataset. Join on
`OCNT_ID`. Only available for ~69% of training compounds (those promoted to DRC);
use a missing indicator for the rest. For test compounds this feature is unavailable,
so the model must learn to rely on it during training while being robust to its absence
at test time (mask randomly during training, à la dropout on features).

**B2. Emax as auxiliary target** *(implemented in multitask runs, marginal gain)*
pEC50 vs Emax correlation: r = −0.12. Orthogonal biological signal. Multitask with
Emax showed ~+0.01 RAE degradation in CV. Use "safe multitask" protocol if retrying:
delayed activation (epoch 10+), small auxiliary weight (0.1×), gradient clipping.

**B3. Counter-assay selectivity as input feature** *(implemented as weight, not feature)*
selectivity = primary_pEC50 − counter_pEC50; r = 0.56 with primary pEC50.
Currently used only as a sample weight. Consider also passing it as an explicit input
feature for models that accept molecular descriptors alongside SMILES.
Imputation for missing values: use 0.0 (neutral selectivity) + a binary missing
indicator, not 1.0 (which implies confirmed selective).

**B4. Physicochemical features** *(identified in EDA, not yet used as model inputs)*
Actives vs inactives show clear separation in:
- Molecular weight (actives higher)
- LogP (actives more lipophilic — consistent with PXR's large hydrophobic LBP)
- H-bond donors (actives fewer)
- Ring count / aromatic ring count (actives higher)
- TPSA (actives lower)
These are well-motivated as explicit global features alongside SMILES-derived representations.

**B5. ChEMBL PXR pre-training** *(attempted as simultaneous multitask in C24, marginal)*
True pre-training (train on ChEMBL PXR assays first, then fine-tune on challenge data)
is different from simultaneous multitask. May help if ChEMBL data distribution is
compatible. Risk: ChEMBL PXR assay noise and heterogeneous protocols may hurt.
Validate ChEMBL data quality (distribution, assay types) before committing.

---

### C. Pairwise / relational learning

**C1. MMP delta head** *(not implemented)*
Enumerate matched molecular pairs (MMPs) from training data — pairs of compounds
differing by a single well-defined structural transformation. Train an auxiliary head
to predict Δ pEC50 = pEC50_A − pEC50_B directly. Forces the model to explicitly
learn what structural changes increase or decrease potency. The cliff enumeration
infrastructure already exists in `notebooks/03_cheminformatics_eda.py`.
Key papers: Hussain & Rea (2010) for MMP enumeration; recent MMPA-Net for GNN
implementation. Requires architecture that exposes per-molecule embeddings.
Works best on congeneric series — favorable here given the analogue test design.

**C2. Contrastive / metric learning on actives** *(not implemented)*
Push active embeddings together and away from inactive embeddings in representation
space. Complementary to A1 (loss weighting) — attacks underprediction from the
representation side rather than the loss side.

---

### D. Semi-supervised & data augmentation

**D1. Pseudo-labeling on 8,126 unlabeled single-conc compounds** *(not implemented)*
8,126 compounds in the single-conc file have no pEC50 label but do have log2FC at
8.25 µM (r = 0.72 with pEC50). Options:
- Train a log2FC → pEC50 mapping on labeled compounds; apply to unlabeled
- Use as pseudo-labels with low confidence weights
- Use for self-supervised pre-training (masked SMILES prediction on the larger set)
7.1% of unlabeled compounds are active at 8.25 µM (log2FC > 0.5) — this pool may
contain actives that didn't make the DRC cut but are structurally informative.

**D2. SMILES augmentation** *(not implemented)*
Enumerate multiple valid SMILES for each molecule (RDKit `MolToSmiles` with
`doRandom=True`) and treat each as a separate training example. Standard technique
for SMILES-based transformers; improves robustness of learned representations.

---

### E. Ensemble & calibration

**E1. Parameter perturbation ensemble** *(not implemented — fold ensemble was tried and hurt)*
Train 7–10 models on full data varying seed, dropout, and learning rate. Average
predictions. Distinct from fold ensemble: avoids the per-fold active sparsity problem
(fold ensemble hurt because each fold had very few actives). Expected: −0.03 to −0.07 RAE.

**E2. GP residual correction** *(not implemented)*
Train base model, collect residuals on training data, fit a Gaussian Process on
(fingerprint features → residual). Apply GP correction to test predictions.
Directly corrects systematic bias without retraining. Most useful if the
underprediction bias is localized in chemical space (which the EDA suggests it is —
actives cluster in high-LogP, high-MW, low-HBD region).
Reference: AUGUR framework (arXiv 2025).

**E3. Curriculum learning** *(not implemented)*
Train on high-confidence samples first (low pEC50_std.error), progressively add
uncertain ones. Prevents noisy labels from corrupting early representations.
Alternative curriculum: easy-to-hard by pEC50 magnitude (inactives first, then
moderate, then actives last for fine-grained tuning).

---

### F. Architecture

**F1. CheMeleon hyperparameter sweep** *(partially done — C13 best: d=4, h=600, dropout=0.2, ffn=2)*
Continue sweep: d=8, d=16; larger ffn depth; different attention heads.
Current best is d=4 which may be underfitting the active SAR.

**F2. Chemprop / D-MPNN as alternative architecture** *(not implemented)*
Message-passing GNN operates on molecular graphs rather than SMILES strings.
Exposes per-atom and per-molecule embeddings needed for MMP delta head (C1) and
contrastive learning (C2). ChemProp v2 supports multitask, uncertainty heads,
and custom loss functions out of the box. Useful baseline: XGBoost on ECFP6 —
if it matches CheMeleon CV within 5%, the GNN may not be learning useful representations.

---

### G. Feature-based discrimination (EDA findings)

**G1. SMARTS pharmacophore features** *(not implemented)*
Encode known PXR pharmacophore motifs as explicit binary features: hydrophobic
patches, H-bond acceptor placement, aromatic halides, sulfonamide groups.
PXR actives are enriched in specific substructures that ECFP4 partially captures
but can't explicitly represent.

**G2. FCFP4 (functional-class fingerprints)** *(not implemented)*
FCFP encodes functional class (HBD, HBA, hydrophobic, etc.) rather than atom
identity. May better capture the HBA/hydrophobic balance that drives PXR agonism,
especially across structurally diverse scaffolds.

---

## What has been ruled out

| Approach | Reason |
|----------|--------|
| 3D shape features (NPR/PMI) | No discrimination between actives and inactives — PXR pocket accepts diverse shapes |
| pEC50 CI columns as features | Data leakage (r ≈ 0.99 with pEC50) |
| Raw counter-assay pEC50 as feature | r = 0.11 alone; only useful via selectivity |
| Fold ensemble (25 models) | Hurt leaderboard performance — active sparsity per fold |
| Higher single-conc concentrations (33, 99 µM) | Cytotoxicity confounds signal |
| Docking scores | PXR pocket flexibility makes docking unreliable (r ~0.3–0.4 literature); poor ROI |

---

## Priority order for next experiments

1. **A1 — Active upweighting (3×)** — highest expected impact, lowest implementation cost
2. **E1 — Parameter perturbation ensemble** — directly replaces the failed fold ensemble
3. **D1 — Pseudo-labeling / semi-supervised on 8,126 unlabeled** — large potential upside
4. **E2 — GP residual correction** — model-agnostic, can be applied to any existing submission
5. **C1 — MMP delta head** — requires architecture change; high ceiling if it works
6. **A3 correction — fix selectivity weight imputation** — low effort, currently wrong
7. **B4 — Physicochemical features as explicit inputs** — well-motivated by EDA, quick to add
