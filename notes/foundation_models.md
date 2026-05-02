# Foundation Models for PXR pEC50 Prediction

## Candidates

### Suiren-ConfAvg (recommended)
- **Paper:** arXiv:2603.21942 (Suiren-1.0 family)
- **Repo:** https://github.com/golab-ai/Suiren-Property-Prediction
- **Weights:** https://huggingface.co/ajy112/Suiren-ConfAvg (~40 MB)
- **Input:** SMILES or molecular graphs — no 3D conformers needed at inference
- **Architecture:** 12-layer GNN, 256-dim embeddings; distilled from Suiren-Base (1.8B, SE(3)-equivariant) via Conformation Compression Distillation (CCD)
- **Pre-training:** 70M DFT samples → 3D geometry baked into 2D representations
- **Claimed benchmark:** SOTA across molecular property tasks; outperforms Uni-Mol2 (per paper — full table not yet verified)
- **Fine-tuning API:** Regression mode via `main.py --mode regression`; supports custom CSV (SMILES + target)
- **Embeddings:** Extractable for use as frozen features
- **Install:** `pip install torch torch_geometric timm==0.4.12` + `git clone`
- **Gotcha:** Must call `Chem.AddHs(mol)` before inference
- **Use case fit:** High — 3D geometry pre-training aligns with PXR being a shape/hydrophobicity-driven 3D target; SMILES input avoids conformer generation pipeline

#### Architecture (relevant to pEC50 fine-tuning)

**Preprocessing**
- SMILES → RDKit mol with `Chem.AddHs()` (mandatory) + kekulization
- Allowed elements: H, C, N, O, F, P, S, Cl, Br, I — verify all PXR SMILES qualify
- Atom features (5 used in forward pass): atomic_num, chirality, formal_charge, aromaticity, ring membership
- Bond features (3): bond_type, stereo, is_conjugated
- Two edge sets per molecule: sparse (actual bonds) + full-connect (all atom pairs, for attention)

**Model stages**
1. Pre-trained GNN (12-layer GATConv, 256-dim) — frozen backbone; returns all layer outputs
2. Fine-tuned GNN (16-layer GATConv, 256-dim) — conditioned on pre-trained layer outputs
3. Node projection: 3-layer MLP (256→256→256, SiLU)
4. Pooling: scatter mean normalized by hardcoded `avg_atom=35.2160`
5. Prediction head: MLP → scalar (regression) or softmax (classification)

**Embedding extraction**
- Set `output_type="last"` on the GNN + `global_add_pool` on node embeddings → 256-dim molecule vector, bypassing the prediction head

#### Fine-tuning config (key args for pEC50)

| Arg | Default | Notes |
|-----|---------|-------|
| `--mode` | `regression` | Use for pEC50 |
| `--loss` | `l1` | L1 (MAE) or `l2` (MSE); L1 recommended given outlier actives |
| `--epochs` | 100 | |
| `--batch-size` | 8 | |
| `--lr` | 2e-4 | AdamW |
| `--weight-decay` | 0.01 | |
| `--sched` | `cosine` | Cosine annealing |
| `--warmup-epochs` | 0 | |
| `--clip-grad` | None | Enable for stability |
| `--model-ema` | off | EMA of weights; helps on small datasets |
| `--model-ema-decay` | 0.9999 | |
| `--amp` | off | bfloat16 mixed precision |
| `--data-mode` | `smiles_random` | Use `smiles_defined` for our analog-mimic CV splits |
| `--seed` | 0 | |

**Data format:** CSV with exactly two columns: `SMILES`, `value`.

#### Gaps for our task
- **No sample weight support** — training loop uses plain L1Loss/MSELoss; active upweighting requires a custom training loop
- **avg_atom normalization hardcoded to 35.2** — derived from pre-training data; may be slightly mismatched for PXR compounds but unlikely to matter significantly
- **`smiles_defined` split mode** needed to use our analog-mimic CV splits (separate train/val CSVs)

### Uni-Mol (v1)
- **Repo:** https://github.com/dptech-corp/Uni-Mol
- **Install:** `pip install unimol-tools` (v0.1.5) — installs **v1 only**, not Uni-Mol2
- **Input:** 3D conformers (RDKit ETKDG + MMFF sufficient); 209M conformations pre-training
- **Architecture:** Transformer on 3D atom positions (ICLR 2023)
- **Benchmarks:** 14/15 MoleculeNet tasks; well-validated on regression
- **Fine-tuning API:** Mature, documented
- **Embeddings:** Extractable
- **Use case fit:** Good — solid baseline, proven on bioactivity; conformer generation required

### Uni-Mol2
- **Repo:** https://github.com/deepmodeling/Uni-Mol/tree/main/unimol2
- **Paper:** arXiv:2406.14969 (2024)
- **Install:** `pip install unimol-tools` — wraps Uni-Mol2 as of Sep 2024 update
- **Weights:** https://huggingface.co/dptech/Uni-Mol2/ — 5 sizes: 84M, 164M, 310M, 570M, 1.1B params
- **Input:** 3D conformers; two-track transformer integrating atomic, graph, and geometry features
- **Pre-training:** 800M conformations (vs 209M for v1)
- **Fine-tuning:** `torchrun` + `unicore-train`; `unimol_tools` package supports embedding extraction
- **Dependencies:** Uni-Core + PyTorch >2.0, rdkit==2022.09.5
- **Benchmarks:** Surpasses Uni-Mol v1 on molecular property prediction (exact numbers not in README)
- **Use case fit:** Good — publicly available, `unimol-tools` accessible, stronger than v1; conformer generation required

### Uni-QSAR
- **Paper:** arXiv:2304.12239
- **HuggingFace:** https://huggingface.co/Cuiyaning/UniQSAR (sparse docs)
- **Input:** 1D SMILES + 2D graph + 3D conformers (combined)
- **Architecture:** Auto-ML stacking of multiple representations
- **Benchmarks:** 21/22 TDC tasks (6.09% avg improvement over prior SOTA)
- **Fine-tuning API:** Auto-ML black box — limited control over loss function
- **Embeddings:** Not clearly extractable
- **Install:** No pip package; no dedicated GitHub repo found
- **Use case fit:** Poor for our needs — Auto-ML design conflicts with custom active upweighting; public release too sparse to rely on

---

## Recommended approach

Use **Suiren-ConfAvg** embeddings (frozen, 256-dim) as features, combined with:
- log2FC at 8.25 µM (r = 0.72 with pEC50)
- Counter-assay selectivity (r = 0.56)
- RDKit physicochemical descriptors (MW, LogP, HBD, ring count, TPSA)

Train a lightweight head (XGBoost or MLP) with **active upweighting (3–5×)** on the combined feature matrix.

**Fallback:** Uni-Mol embeddings if Suiren-ConfAvg frozen embeddings don't transfer well to bioactivity.

**Not recommended:** Uni-QSAR — public release too sparse and Auto-ML design prevents loss function control.

---

## Open questions
- Verify Suiren-ConfAvg vs. Uni-Mol2 benchmark margins (need full paper PDF tables)
- Test whether frozen vs. fine-tuned Suiren-ConfAvg embeddings differ meaningfully on our CV splits
