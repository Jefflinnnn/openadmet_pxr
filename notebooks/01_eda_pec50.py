import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # PXR Challenge — pEC50 EDA

    ## Column reference

    ### Identifiers
    | Column | Meaning |
    |--------|---------|
    | `Molecule Name` | Internal OpenADMET compound ID (e.g. `OADMET-0006089`) — unique per row |
    | `SMILES` | 2D molecular structure as a SMILES string |
    | `OCNT Batch` | Full batch/plate identifier (e.g. `OCNT-2318682-AA-004`) — encodes compound + plate + well format |
    | `OCNT_ID` | Compound-level ID without batch suffix (e.g. `OCNT-2318682`) |
    | `source` | Data provenance — all rows are `train_df` in the training file |
    | `Split` | Train/test label — all rows are `Train` here; the blinded test is a separate file |

    ### Primary targets
    | Column | Meaning |
    |--------|---------|
    | `pEC50` | Potency: −log₁₀(EC50 in molarity). Higher = more potent. **Primary competition metric.** |
    | `Emax_estimate (log2FC vs. baseline)` | Efficacy: maximum PXR target gene induction in log2 fold-change vs. vehicle baseline |
    | `Emax.vs.pos.ctrl_estimate (dimensionless)` | Same Emax normalized to a positive control reference (~1.0 = same efficacy as positive control) |

    ### Uncertainty / fit quality (from dose-response curve fitting)
    | Column | Meaning |
    |--------|---------|
    | `pEC50_std.error` | Standard error on the pEC50 fit — useful as **sample weight** (tighter = more reliable) |
    | `Emax_std.error` | Standard error on the Emax fit (log2FC units) |
    | `Emax.vs.pos.ctrl_std.error` | Standard error on the normalized Emax |
    | `pEC50_ci.lower / ci.upper` | 95% confidence interval bounds on pEC50 |
    | `Emax_ci.lower / ci.upper` | 95% CI bounds on Emax (log2FC) |
    | `Emax.vs.pos.ctrl_ci.lower / ci.upper` | 95% CI bounds on normalized Emax |
    """)
    return


@app.cell
def _():
    import subprocess
    subprocess.run(
        ["uv", "pip", "install", "matplotlib", "seaborn", "plotly"],
        capture_output=True,
    )
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted")
    return pd, plt, sns


@app.cell
def _(pd):
    import pathlib
    DATA = pathlib.Path(__file__).parent.parent / "data"
    train = pd.read_csv(DATA / "pxr-challenge_TRAIN.csv")
    test  = pd.read_csv(DATA / "pxr-challenge_TEST_BLINDED.csv")
    return test, train


@app.cell
def _(mo, test, train):
    mo.md(f"""
    ## Dataset sizes
    | Split | Rows | Columns |
    |-------|------|---------|
    | Train | {len(train):,} | {train.shape[1]} |
    | Test (blinded) | {len(test):,} | {test.shape[1]} |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Train column overview
    """)
    return


@app.cell
def _(train):
    train.dtypes.reset_index().rename(columns={"index": "column", 0: "dtype"})
    return


@app.cell
def _(mo):
    mo.md("""
    ## Missing values
    """)
    return


@app.cell
def _(train):
    missing = train.isnull().sum().rename("missing").reset_index().rename(columns={"index": "column"})
    missing["pct"] = (missing["missing"] / len(train) * 100).round(2)
    missing[missing["missing"] > 0]
    return


@app.cell
def _(mo):
    mo.md("""
    ## pEC50 distribution
    """)
    return


@app.cell
def _(plt, sns, train):
    fig_pec50, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(train["pEC50"].dropna(), bins=40, kde=True, ax=ax, color="steelblue")
    ax.set_xlabel("pEC50  (−log₁₀ molarity)")
    ax.set_ylabel("Count")
    ax.set_title("pEC50 distribution (train)")
    fig_pec50
    return


@app.cell
def _(mo, train):
    mo.md(f"""
    **Summary stats — pEC50**

    | Stat | Value |
    |------|-------|
    | count | {train['pEC50'].count():,} |
    | mean  | {train['pEC50'].mean():.3f} |
    | std   | {train['pEC50'].std():.3f} |
    | min   | {train['pEC50'].min():.3f} |
    | 25%   | {train['pEC50'].quantile(0.25):.3f} |
    | 50%   | {train['pEC50'].median():.3f} |
    | 75%   | {train['pEC50'].quantile(0.75):.3f} |
    | max   | {train['pEC50'].max():.3f} |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Emax distribution
    """)
    return


@app.cell
def _(plt, sns, train):
    fig_emax, ax2 = plt.subplots(figsize=(8, 4))
    sns.histplot(train["Emax_estimate (log2FC vs. baseline)"].dropna(), bins=40, kde=True, ax=ax2, color="darkorange")
    ax2.set_xlabel("Emax (log2FC vs. baseline)")
    ax2.set_ylabel("Count")
    ax2.set_title("Emax distribution (train)")
    fig_emax
    return


@app.cell
def _(mo):
    mo.md("""
    ## pEC50 vs Emax scatter
    """)
    return


@app.cell
def _(plt, sns, train):
    fig_scatter, ax3 = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=train,
        x="pEC50",
        y="Emax_estimate (log2FC vs. baseline)",
        alpha=0.4,
        s=20,
        ax=ax3,
    )
    ax3.set_xlabel("pEC50")
    ax3.set_ylabel("Emax (log2FC vs. baseline)")
    ax3.set_title("pEC50 vs Emax (train)")
    fig_scatter
    return


@app.cell
def _(mo):
    mo.md("""
    ## Train/test split in training file
    """)
    return


@app.cell
def _(mo, train):
    split_counts = train["Split"].value_counts().reset_index()
    split_counts.columns = ["split", "count"]
    mo.ui.table(split_counts)
    return


@app.cell
def _(plt, sns, train):
    fig_split, ax4 = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=train,
        x="pEC50",
        hue="Split",
        bins=40,
        kde=True,
        ax=ax4,
        alpha=0.5,
    )
    ax4.set_xlabel("pEC50")
    ax4.set_title("pEC50 distribution by Split")
    fig_split
    return


@app.cell
def _(mo):
    mo.md("""
    ## OCNT Batch distribution
    """)
    return


@app.cell
def _(mo, train):
    batch_counts = train["OCNT Batch"].value_counts().reset_index()
    batch_counts.columns = ["batch", "count"]
    mo.ui.table(batch_counts)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Molecular properties (RDKit)
    """)
    return


@app.cell
def _(train):
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    def calc_props(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None, None, None, None, None
        return (
            mol.GetNumHeavyAtoms(),
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
        )

    props = train["SMILES"].apply(calc_props)
    train_props = train.copy()
    train_props[["heavy_atoms", "mol_wt", "logP", "hbd", "hba"]] = list(props)
    return Chem, Descriptors, train_props


@app.cell
def _(mo, train_props):
    mo.md(f"""
    **RDKit descriptor summary**

    | Property | Mean | Std | Min | Max |
    |----------|------|-----|-----|-----|
    | Heavy atom count | {train_props['heavy_atoms'].mean():.1f} | {train_props['heavy_atoms'].std():.1f} | {train_props['heavy_atoms'].min():.0f} | {train_props['heavy_atoms'].max():.0f} |
    | Mol weight (Da) | {train_props['mol_wt'].mean():.1f} | {train_props['mol_wt'].std():.1f} | {train_props['mol_wt'].min():.1f} | {train_props['mol_wt'].max():.1f} |
    | LogP | {train_props['logP'].mean():.2f} | {train_props['logP'].std():.2f} | {train_props['logP'].min():.2f} | {train_props['logP'].max():.2f} |
    | H-bond donors | {train_props['hbd'].mean():.2f} | {train_props['hbd'].std():.2f} | {train_props['hbd'].min():.0f} | {train_props['hbd'].max():.0f} |
    | H-bond acceptors | {train_props['hba'].mean():.2f} | {train_props['hba'].std():.2f} | {train_props['hba'].min():.0f} | {train_props['hba'].max():.0f} |
    """)
    return


@app.cell
def _(plt, sns, train_props):
    fig_props, axes = plt.subplots(1, 3, figsize=(14, 4))

    sns.histplot(train_props["mol_wt"], bins=40, ax=axes[0], color="steelblue", kde=True)
    axes[0].set_xlabel("Molecular weight (Da)")
    axes[0].set_title("Mol weight")

    sns.histplot(train_props["logP"], bins=40, ax=axes[1], color="darkorange", kde=True)
    axes[1].set_xlabel("LogP")
    axes[1].set_title("Lipophilicity (LogP)")

    sns.histplot(train_props["heavy_atoms"], bins=30, ax=axes[2], color="seagreen", kde=True)
    axes[2].set_xlabel("Heavy atom count")
    axes[2].set_title("Molecular size")

    fig_props.tight_layout()
    fig_props
    return axes, fig_props


@app.cell
def _(mo):
    mo.md("""
    ### Molecular properties vs pEC50
    """)
    return


@app.cell
def _(plt, sns, train_props):
    fig_prop_pec50, axes2 = plt.subplots(1, 3, figsize=(14, 4))

    for _ax, _col, _label in zip(
        axes2,
        ["mol_wt", "logP", "heavy_atoms"],
        ["Molecular weight (Da)", "LogP", "Heavy atom count"],
    ):
        sns.scatterplot(data=train_props, x=_col, y="pEC50", alpha=0.3, s=15, ax=_ax)
        _ax.set_xlabel(_label)
        _ax.set_ylabel("pEC50")

    fig_prop_pec50.tight_layout()
    fig_prop_pec50
    return axes2, fig_prop_pec50


@app.cell
def _(mo):
    mo.md("""
    ## Correlation matrix — numeric columns
    """)
    return


@app.cell
def _(plt, sns, train):
    num_cols = [
        "pEC50",
        "Emax_estimate (log2FC vs. baseline)",
        "Emax.vs.pos.ctrl_estimate (dimensionless)",
        "pEC50_std.error (-log10(molarity))",
        "Emax_std.error (log2FC vs. baseline)",
    ]
    corr = train[num_cols].corr()

    fig_corr, ax6 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax6)
    ax6.set_title("Correlation matrix")
    fig_corr
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
