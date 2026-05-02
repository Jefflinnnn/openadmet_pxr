# PXR Challenge — EDA & Feature Investigation Findings

## Dataset overview

| File | Rows | Notes |
|------|------|-------|
| `pxr-challenge_TRAIN.csv` | 4,139 | Full dose-response curves; primary modeling target |
| `pxr-challenge_TEST_BLINDED.csv` | 513 | SMILES only; no labels |
| `pxr-challenge_counter-assay_TRAIN.csv` | 2,859 | PXR-null assay; 69.1% overlap with train |
| `pxr-challenge_single_concentration_TRAIN.csv` | 21,003 rows / 10,870 compounds | Fixed-dose screening; 25.2% overlap with train |
| `pxr-challenge_structure_TEST_BLINDED.csv` | 78 | Fragment structures with X-ray data; separate task |

Test SMILES do not appear in any training file — no leakage risk from the raw data.

---

## Primary task

Predict **pEC50** (−log₁₀ EC50 in molarity) from SMILES. Higher = more potent.
The competition metric is evaluated on `pxr-challenge_TEST_BLINDED.csv`.

---

## Assay biology

### Primary assay
A chimeric cell-based reporter: PXR's ligand-binding domain is fused to a heterologous
DNA-binding domain driving a luciferase gene. Compound binding causes a conformational
change that recruits transcriptional activators, producing measurable luminescence.

### Counter-assay
Identical reporter system but the chimeric PXR gene contains **early nonsense mutations
that abolish functional protein expression**. Any signal reflects non-specific effects
(cytotoxicity, membrane disruption, luciferase interference), not true PXR agonism.

### Single-concentration screening
High-throughput triage run before full dose-response. Compounds are tested at up to four
fixed concentrations (~1 µM, ~8.25 µM, ~33 µM, ~99 µM) using the same PXR reporter.
Readout is **log2 fold-change (log2FC)** vs. vehicle baseline — same scale as Emax but at
a fixed rather than saturating dose.

---

## Training data column reference

### Identifiers
| Column | Meaning |
|--------|---------|
| `Molecule Name` | OpenADMET compound ID — unique per row |
| `SMILES` | 2D molecular structure |
| `OCNT Batch` | Full batch/plate identifier (1:1 with compound in this dataset) |
| `OCNT_ID` | Compound-level ID; join key to counter-assay and single-conc files |
| `source` | All `train_df` — no signal |
| `Split` | All `Train` — no signal |

### Primary targets
| Column | Meaning |
|--------|---------|
| `pEC50` | Potency: −log₁₀(EC50). **Primary competition metric.** Range 1.6–7.5, median 4.65 |
| `Emax_estimate (log2FC vs. baseline)` | Max efficacy: log2FC of reporter vs. vehicle at saturation |
| `Emax.vs.pos.ctrl_estimate (dimensionless)` | Emax normalised to positive control (~1.0 = same as pos ctrl) |

### Uncertainty columns
| Column | Verdict |
|--------|---------|
| `pEC50_std.error` | Use as **sample weight** — reflects dose-response curve reliability |
| `pEC50_ci.lower / ci.upper` | **Drop — data leakage** (r ≈ 0.99 with pEC50; derived from it) |
| `Emax_std.error`, `Emax.vs.pos.ctrl_std.error` | SE on Emax fits; can weight Emax auxiliary target |
| Emax CI columns | Same leakage risk if used as features for Emax prediction |

---

## Key findings

### 1. Emax is an independent auxiliary signal
- pEC50 vs Emax correlation: **r = −0.12**
- Potency and efficacy are nearly orthogonal — a highly potent compound is not necessarily
  a full agonist, and vice versa
- Emax is useful as an **auxiliary target** (multi-task learning) or as an input feature;
  it encodes biological information the model would otherwise have to infer from structure alone

### 2. Measurement uncertainty → sample weights
- 15% of compounds have `pEC50_std.error > 0.3`, indicating poorly-constrained dose-response curves
- These should be **down-weighted** during training, not excluded
- The CI columns (r ≈ 0.99 with pEC50) must not be used as features — direct data leakage

### 3. Counter-assay selectivity
- Counter-assay pEC50 alone: **r = 0.11** with primary pEC50 (nearly useless raw)
- Derived feature **selectivity = primary pEC50 − counter pEC50**: **r = 0.56** with primary pEC50
- Compounds with negative selectivity are likely false positives in the primary screen
- 31% of training compounds have no counter-assay data → selectivity must be imputed or handled with a missing indicator

### 4. Single-concentration log2FC
- log2FC at **8.25 µM** correlates with pEC50 at **r = 0.72** (Pearson and Spearman) — the
  strongest non-structural signal in the dataset
- Signal degrades sharply at higher concentrations (r = 0.50 at 33 µM, r = 0.09 at 99 µM),
  likely due to cytotoxicity confounding the reporter
- **8,126 compounds** in the single-conc file have no pEC50 label — candidates for
  semi-supervised pre-training or pseudo-labeling (~3× additional chemical space)
- 7.1% of unlabeled compounds are active at 8.25 µM (log2FC > 0.5)

---

## Combined sample weighting scheme

Two independent noise sources are combined into a single per-compound training weight:

```
selectivity_weight = sigmoid(primary_pEC50 − counter_pEC50)
se_weight          = exp(−pEC50_std.error)
sample_weight      = se_weight × selectivity_weight
```

- **Sigmoid** provides a smooth gradient rather than a hard cutoff — numerically better
  for gradient-based optimisers
- Compounds with no counter-assay data default to `selectivity_weight = 1.0`
- 187 compounds receive `sel_weight < 0.4` (likely false positives)
- Worst case: selectivity = −4.37, combined weight = 0.007 (effectively excluded)
- Mean combined weight across all 4,139 training compounds: **0.71**

---

## Modeling recommendations

| Source | Use | Expected impact |
|--------|-----|-----------------|
| `Emax_estimate` | Auxiliary target or input feature | Moderate |
| `Emax.vs.pos.ctrl_estimate` | Auxiliary target or input feature | Moderate |
| `sample_weight` (combined) | Per-sample training weight | Moderate |
| Counter-assay selectivity | Engineered feature (join on `OCNT_ID`) | Moderate — r = 0.56 |
| Single-conc `log2_fc` at 8.25 µM | Input feature (join on `OCNT_ID`) | High — r = 0.72 |
| Single-conc 8,126 unlabeled | Semi-supervised pre-training / pseudo-labels | Potentially high |
| `pEC50_ci.lower / ci.upper` | **Drop — data leakage** | — |
| Counter-assay pEC50 raw | **Drop** | r = 0.11; only useful via selectivity |
| `OCNT Batch`, `Split`, `source` | **Drop** | No signal |

---

## Notebooks

| Notebook | Contents |
|----------|----------|
| `notebooks/01_eda_pec50.py` | Dataset sizes, column overview, missing values, pEC50/Emax distributions, batch breakdown, RDKit descriptors, correlation matrix |
| `notebooks/02_feature_investigation.py` | Emax as auxiliary signal, uncertainty weighting, counter-assay selectivity, single-concentration analysis, combined sample weight scheme |
