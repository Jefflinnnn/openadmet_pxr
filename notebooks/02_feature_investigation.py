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
    # PXR Challenge — Feature Investigation

    This notebook investigates which columns beyond SMILES are useful for predicting pEC50,
    and flags what to avoid (leakage). Four themes:

    1. **Emax as an auxiliary signal** — potency and efficacy are nearly independent; Emax can inform multi-task learning
    2. **Uncertainty columns as sample weights** — high SE compounds have poorly-constrained dose-response curves
    3. **Counter-assay selectivity** — a derived feature separating true PXR agonists from non-specific hits
    4. **Single-concentration screening** — a cheap activity proxy with semi-supervised potential
    """)
    return


@app.cell
def _():
    import pathlib
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted")

    DATA = pathlib.Path(__file__).parent.parent / "data"
    train   = pd.read_csv(DATA / "pxr-challenge_TRAIN.csv")
    counter = pd.read_csv(DATA / "pxr-challenge_counter-assay_TRAIN.csv")
    single  = pd.read_csv(DATA / "pxr-challenge_single_concentration_TRAIN.csv")
    return counter, np, pd, plt, single, sns, train


@app.cell
def _(mo):
    mo.md("""
    ## 1. Emax as an auxiliary signal
    """)
    return


@app.cell
def _(mo, train):
    r_emax     = train["pEC50"].corr(train["Emax_estimate (log2FC vs. baseline)"])
    r_emax_pos = train["pEC50"].corr(train["Emax.vs.pos.ctrl_estimate (dimensionless)"])
    mo.md(f"""
    pEC50 vs Emax correlation: **r = {r_emax:.3f}**
    pEC50 vs Emax (vs pos ctrl) correlation: **r = {r_emax_pos:.3f}**

    Potency and efficacy are nearly independent — Emax carries orthogonal biological information.
    Including it as an auxiliary target (multi-task learning) or as an input feature could help
    the model learn the underlying PXR biology rather than just fitting pEC50 directly.
    """)
    return


@app.cell
def _(plt, sns, train):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(
        data=train, x="pEC50", y="Emax_estimate (log2FC vs. baseline)",
        alpha=0.3, s=15, ax=axes[0],
    )
    axes[0].set_title("pEC50 vs Emax (log2FC vs baseline)")
    axes[0].set_xlabel("pEC50")
    axes[0].set_ylabel("Emax (log2FC)")

    sns.scatterplot(
        data=train, x="pEC50", y="Emax.vs.pos.ctrl_estimate (dimensionless)",
        alpha=0.3, s=15, ax=axes[1], color="darkorange",
    )
    axes[1].set_title("pEC50 vs Emax (vs positive control)")
    axes[1].set_xlabel("pEC50")
    axes[1].set_ylabel("Emax (vs pos ctrl)")

    fig.suptitle(
        "Potency (pEC50) and efficacy (Emax) are nearly uncorrelated — "
        "Emax carries independent biological signal useful as an auxiliary target.",
        fontsize=9, y=1.02,
    )
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Uncertainty columns — sample weights, not features
    """)
    return


@app.cell
def _(mo, train):
    r_se_pec50    = train["pEC50"].corr(train["pEC50_std.error (-log10(molarity))"])
    r_ci_lo       = train["pEC50"].corr(train["pEC50_ci.lower (-log10(molarity))"])
    r_ci_hi       = train["pEC50"].corr(train["pEC50_ci.upper (-log10(molarity))"])
    high_se       = (train["pEC50_std.error (-log10(molarity))"] > 0.3).sum()
    high_se_pct   = 100 * high_se / len(train)
    mo.md(f"""
    | Column | Correlation with pEC50 | Verdict |
    |--------|------------------------|---------|
    | `pEC50_std.error` | r = {r_se_pec50:.3f} | Use as **sample weight** — do not use as feature |
    | `pEC50_ci.lower` | r = {r_ci_lo:.3f} | **Leakage** — derived directly from pEC50 |
    | `pEC50_ci.upper` | r = {r_ci_hi:.3f} | **Leakage** — derived directly from pEC50 |

    CI bounds are mathematically derived from pEC50 (r ≈ 0.99) — including them as features
    would be direct data leakage. The std error is also correlated (r = {r_se_pec50:.3f}),
    but this reflects measurement reliability, not the target value itself.

    **{high_se:,} compounds ({high_se_pct:.1f}%) have SE > 0.3** — their dose-response curves
    are poorly constrained and should be down-weighted during training.
    """)
    return


@app.cell
def _(pd, plt, sns, train):
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(
        data=train,
        x="pEC50_std.error (-log10(molarity))",
        y="pEC50",
        alpha=0.3, s=15, ax=axes2[0],
    )
    axes2[0].set_title("pEC50 vs measurement std error")
    axes2[0].set_xlabel("pEC50 std error")
    axes2[0].set_ylabel("pEC50")

    se_col = "pEC50_std.error (-log10(molarity))"
    train_copy = train.copy()
    train_copy["reliability"] = pd.cut(
        train_copy[se_col],
        bins=[0, 0.1, 0.2, 0.3, 999],
        labels=["SE ≤ 0.1", "0.1–0.2", "0.2–0.3", "SE > 0.3"],
    )
    sns.histplot(
        data=train_copy, x="pEC50", hue="reliability",
        bins=35, kde=False, ax=axes2[1], alpha=0.6,
    )
    axes2[1].set_title("pEC50 distribution by measurement reliability")
    axes2[1].set_xlabel("pEC50")

    fig2.suptitle(
        "High std error reflects a poorly-fitted dose-response curve, not a noisier compound — "
        "use as a sample weight, not a feature. 15% of compounds have SE > 0.3.",
        fontsize=9, y=1.02,
    )
    fig2.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Counter-assay selectivity

    ### What the counter-assay is

    The primary assay uses a **chimeric cell-based reporter**: PXR's ligand-binding domain is
    fused to a heterologous DNA-binding domain driving a luciferase gene. When a compound binds
    PXR, it induces a conformational change that recruits transcriptional activators, producing
    measurable luminescence.

    The counter-assay uses an **identical reporter system except the chimeric gene contains
    early nonsense mutations that abolish functional PXR protein expression**. Any luminescence
    signal in the counter-assay therefore reflects non-specific effects — cytotoxicity, membrane
    disruption, direct luciferase interference — rather than true PXR agonism.

    A compound potent in *both* assays is a false positive in the primary screen. A compound
    potent only in the primary assay is a genuine PXR agonist.

    ### Column reference
    | Column | Meaning |
    |--------|---------|
    | `pEC50` | Potency in the counter (PXR-null) assay — measures non-specific activity |
    | `Emax_estimate` | Maximum efficacy in the counter assay |
    | All SE / CI columns | Same meaning as in the primary assay, for the counter-assay fit |
    """)
    return


@app.cell
def _(counter, mo, train):
    n_overlap = train["OCNT_ID"].isin(counter["OCNT_ID"]).sum()
    mo.md(f"""
    **{n_overlap:,} / {len(train):,} training compounds ({100*n_overlap/len(train):.1f}%) have counter-assay data.**

    The derived feature **selectivity = primary pEC50 − counter pEC50** separates true PXR
    agonists (high selectivity) from non-specific hits (low or negative selectivity).
    """)
    return


@app.cell
def _(counter, mo, train):
    merged = train.merge(
        counter[["OCNT_ID", "pEC50", "Emax_estimate (log2FC vs. baseline)"]],
        on="OCNT_ID",
        suffixes=("_primary", "_counter"),
    )
    merged["selectivity"] = merged["pEC50_primary"] - merged["pEC50_counter"]

    r_counter = merged["pEC50_primary"].corr(merged["pEC50_counter"])
    r_sel      = merged["pEC50_primary"].corr(merged["selectivity"])

    mo.md(f"""
    | Comparison | Correlation with primary pEC50 |
    |------------|-------------------------------|
    | Counter-assay pEC50 | r = {r_counter:.3f} |
    | Selectivity (primary − counter) | r = {r_sel:.3f} |

    Counter-assay pEC50 is nearly uncorrelated (r = {r_counter:.3f}) with primary pEC50 —
    confirming it measures a different mechanism. The selectivity ratio has a moderate
    positive correlation (r = {r_sel:.3f}): more potent primary hits tend to also be
    more selective, but there's substantial spread worth exploiting.
    """)
    return (merged,)


@app.cell
def _(merged, plt, sns):
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

    sns.scatterplot(
        data=merged, x="pEC50_counter", y="pEC50_primary",
        alpha=0.3, s=15, ax=axes3[0],
    )
    axes3[0].set_title("Primary vs counter-assay pEC50")
    axes3[0].set_xlabel("Counter-assay pEC50")
    axes3[0].set_ylabel("Primary pEC50")

    sns.histplot(merged["selectivity"], bins=40, kde=True, ax=axes3[1], color="steelblue")
    axes3[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes3[1].set_title("Selectivity distribution\n(primary − counter pEC50)")
    axes3[1].set_xlabel("Selectivity")

    sns.scatterplot(
        data=merged, x="selectivity", y="pEC50_primary",
        alpha=0.3, s=15, ax=axes3[2], color="seagreen",
    )
    axes3[2].set_title("Selectivity vs primary pEC50")
    axes3[2].set_xlabel("Selectivity (primary − counter)")
    axes3[2].set_ylabel("Primary pEC50")

    fig3.suptitle(
        "Counter-assay pEC50 is nearly uncorrelated with primary pEC50 (r = 0.11), confirming it measures "
        "a different mechanism. Selectivity (primary − counter) is the useful derived signal (r = 0.56).",
        fontsize=9, y=1.02,
    )
    fig3.tight_layout()
    fig3
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Single-concentration screening

    ### What the single-concentration data is

    Before running a full dose-response experiment (which yields pEC50 and Emax), compounds are
    first screened at **one or two fixed concentrations** using the same PXR reporter assay.
    This is a high-throughput triage step: it's cheaper and faster than a full curve, but only
    gives a binary-ish activity signal rather than a precise potency estimate.

    Compounds are tested at up to four concentrations: ~1 µM, ~8.25 µM, ~33 µM, and ~99 µM.
    Most compounds have measurements at one or two concentrations. The readout is
    **log2 fold-change (log2FC)** of the reporter signal vs. vehicle baseline — the same scale
    as Emax in the primary assay, but at a fixed dose rather than the saturating dose.

    ### Column reference
    | Column | Meaning |
    |--------|---------|
    | `log2_fc_estimate` | Log2 fold-change of PXR reporter signal vs. baseline at this concentration |
    | `log2_fc_stderr` | Standard error of the log2FC estimate |
    | `t_statistic` | t-statistic for the fold-change being different from zero |
    | `p_value` / `fdr_bh` | Raw p-value and Benjamini-Hochberg FDR-corrected p-value |
    | `neg_log10_fdr` | −log₁₀(FDR) — higher means more statistically significant activity |
    | `median_log2_fc` | Median log2FC across replicates |
    | `n_replicates` | Number of replicate wells |
    | `cohens_d` | Effect size (Cohen's d) of the fold-change |
    | `concentration_M` | Tested concentration in molarity |
    | `plate_id` | Experimental plate identifier |
    | `experiment_name` | Screening batch/experiment label |
    | `compound_class` | `Library` (test compound) or `Positive Control` |
    """)
    return


@app.cell
def _(mo, single, train):
    single_lib = single[single["compound_class"] == "Library"]
    n_single_total   = single_lib["OCNT_ID"].nunique()
    n_single_overlap = single_lib["OCNT_ID"].isin(train["OCNT_ID"]).sum()
    n_single_extra   = n_single_total - single_lib[single_lib["OCNT_ID"].isin(train["OCNT_ID"])]["OCNT_ID"].nunique()
    mo.md(f"""
    **{n_single_total:,} unique compounds** in the single-concentration file:
    - **{single_lib[single_lib['OCNT_ID'].isin(train['OCNT_ID'])]['OCNT_ID'].nunique():,}** overlap with the labeled train set — log2FC can be used as a feature
    - **{n_single_extra:,}** have *no* pEC50 label — candidates for semi-supervised pre-training or pseudo-labels
    """)
    return (single_lib,)


@app.cell
def _(mo, single_lib, train):
    conc_rows = []
    for _conc in sorted(single_lib["concentration_M"].unique()):
        _sub = single_lib[single_lib["concentration_M"] == _conc]
        _merged = train.merge(_sub[["OCNT_ID", "log2_fc_estimate"]], on="OCNT_ID", how="inner")
        _r = _merged["log2_fc_estimate"].corr(_merged["pEC50"])
        conc_rows.append((_conc, len(_merged), round(_r, 3)))

    table_rows = "\n".join(
        f"    | {c:.2e} M | {n:,} | **{r}** |" if r == max(x[2] for x in conc_rows)
        else f"    | {c:.2e} M | {n:,} | {r} |"
        for c, n, r in conc_rows
    )
    mo.md(f"""
    ### log2FC correlation with pEC50 by concentration

    Signal quality drops sharply at higher concentrations — likely due to cytotoxicity
    confounding the reporter readout at 99 µM. The **8.25 µM measurement is the most
    informative** (r = 0.72), making it the best concentration to use as a feature.

    | Concentration | n (overlap with train) | r with pEC50 |
    |---------------|------------------------|--------------|
    {table_rows}
    """)
    return


@app.cell
def _(plt, single_lib, sns, train):
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))

    _conc_8um = single_lib[single_lib["concentration_M"].between(8e-6, 9e-6)]
    _merged = train.merge(_conc_8um[["OCNT_ID", "log2_fc_estimate"]], on="OCNT_ID", how="inner")

    sns.scatterplot(
        data=_merged, x="log2_fc_estimate", y="pEC50",
        alpha=0.3, s=15, ax=axes4[0], color="steelblue",
    )
    axes4[0].set_title("log2FC at 8.25 µM vs pEC50")
    axes4[0].set_xlabel("log2FC at 8.25 µM")
    axes4[0].set_ylabel("pEC50")

    _extra = single_lib[~single_lib["OCNT_ID"].isin(train["OCNT_ID"])]
    _extra_8um = _extra[_extra["concentration_M"].between(8e-6, 9e-6)]
    _max_fc = _extra_8um.groupby("OCNT_ID")["log2_fc_estimate"].max()

    sns.histplot(_max_fc, bins=40, kde=True, ax=axes4[1], color="darkorange")
    axes4[1].axvline(0.5, color="red", linestyle="--", linewidth=1, label="active threshold (log2FC=0.5)")
    axes4[1].set_title("log2FC distribution — unlabeled compounds\n(8.25 µM, no pEC50 label)")
    axes4[1].set_xlabel("max log2FC")
    axes4[1].legend(fontsize=8)

    fig4.suptitle(
        "Single-concentration log2FC at 8.25 µM is the strongest non-structural feature (r = 0.72 with pEC50). "
        "8,126 unlabeled compounds (right) are candidates for semi-supervised pre-training.",
        fontsize=9, y=1.02,
    )
    fig4.tight_layout()
    fig4
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Combined sample weights

    Two independent sources of unreliability can be folded into a single per-compound weight
    used during model training:

    1. **Measurement uncertainty** (`pEC50_std.error`) — noisy dose-response curve fit
    2. **Counter-assay selectivity** — low selectivity means the primary pEC50 may partly
       reflect non-specific activity rather than true PXR binding

    The weighting scheme:

    ```
    selectivity_weight = sigmoid(selectivity)   # (0,1); <0.5 for false-positive candidates
    se_weight          = exp(−pEC50_std.error)   # (0,1); penalises noisy fits
    sample_weight      = se_weight × selectivity_weight
    ```

    For the ~31% of compounds with no counter-assay data, `selectivity_weight` defaults to 1.0
    (no penalty), so the combined weight degrades gracefully to SE-only weighting.

    **Sigmoid** is chosen over a hard threshold because it provides a smooth gradient:
    compounds with moderate non-specificity are partially down-weighted rather than
    abruptly excluded, which is more numerically stable for gradient-based optimizers.
    """)
    return


@app.cell
def _(counter, np, pd, plt, sns, train):
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    w_df = train.merge(
        counter[["OCNT_ID", "pEC50"]].rename(columns={"pEC50": "pEC50_counter"}),
        on="OCNT_ID", how="left",
    )
    w_df["selectivity"]     = w_df["pEC50"] - w_df["pEC50_counter"]
    w_df["sel_weight"]      = w_df["selectivity"].apply(lambda x: _sigmoid(x) if pd.notna(x) else 1.0)
    w_df["se_weight"]       = np.exp(-w_df["pEC50_std.error (-log10(molarity))"])
    w_df["sample_weight"]   = w_df["sel_weight"] * w_df["se_weight"]

    n_fp = (w_df["sel_weight"] < 0.4).sum()
    n_no_counter = w_df["pEC50_counter"].isna().sum()

    fig5, axes5 = plt.subplots(1, 3, figsize=(15, 5))

    sns.histplot(w_df["sel_weight"], bins=40, ax=axes5[0], color="steelblue", kde=True)
    axes5[0].axvline(0.5, color="red", linestyle="--", linewidth=1, label="sigmoid(0) = 0.5")
    axes5[0].set_title("Selectivity weight\nsigmoid(primary − counter pEC50)")
    axes5[0].set_xlabel("selectivity_weight")
    axes5[0].legend(fontsize=8)

    sns.histplot(w_df["se_weight"], bins=40, ax=axes5[1], color="darkorange", kde=True)
    axes5[1].set_title("SE weight\nexp(−pEC50_std.error)")
    axes5[1].set_xlabel("se_weight")

    sns.scatterplot(
        data=w_df, x="pEC50", y="sample_weight",
        alpha=0.3, s=15, ax=axes5[2], color="seagreen",
    )
    axes5[2].set_title("Combined sample weight vs pEC50")
    axes5[2].set_xlabel("pEC50")
    axes5[2].set_ylabel("sample_weight")

    fig5.suptitle(
        "Combined weight = sigmoid(selectivity) × exp(−SE). Low-selectivity false positives and "
        "noisy fits are independently penalised; compounds without counter-assay data fall back to SE-only weighting.",
        fontsize=9, y=1.02,
    )
    fig5.tight_layout()
    fig5
    return n_fp, n_no_counter, w_df


@app.cell
def _(mo, n_fp, n_no_counter, w_df):
    mo.md(f"""
    | Statistic | Value |
    |-----------|-------|
    | Compounds with `sel_weight` < 0.4 (likely false positives) | {n_fp:,} |
    | Compounds with no counter-assay data (sel_weight = 1.0) | {n_no_counter:,} |
    | Mean combined weight | {w_df['sample_weight'].mean():.3f} |
    | Min combined weight | {w_df['sample_weight'].min():.3f} |

    The worst-weighted compound has selectivity = −4.37 (more potent in the PXR-null assay
    than the primary — a clear non-specific hit) combined with moderate SE, giving a
    combined weight of ~0.007. It contributes almost nothing to the loss.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    | Column / Source | How to use | Expected impact |
    |-----------------|-----------|-----------------|
    | `Emax_estimate (log2FC vs. baseline)` | Auxiliary target or input feature | Moderate — r = −0.12, orthogonal biological signal |
    | `Emax.vs.pos.ctrl_estimate` | Auxiliary target or input feature | Moderate — normalized efficacy, same rationale |
    | `pEC50_std.error` + counter selectivity | **Combined sample weight** `exp(−SE) × sigmoid(selectivity)` | Moderate — down-weights noisy fits and false positives together |
    | Counter-assay: selectivity = primary − counter pEC50 | Engineered feature (join on `OCNT_ID`) | Moderate — r = 0.56 with pEC50 |
    | Single-conc `log2_fc_estimate` at 8.25 µM | Feature for labeled compounds (join on `OCNT_ID`) | High — r = 0.72 with pEC50 |
    | Single-conc 8,126 unlabeled compounds | Semi-supervised pre-training or pseudo-labels | Potentially high — ~3× more chemical space |
    | `pEC50_ci.lower / ci.upper` | **Drop — data leakage** | r ≈ 0.99; mathematically derived from pEC50 |
    | Counter-assay pEC50 raw | **Drop** | r = 0.11 alone; only useful via selectivity |
    | `OCNT Batch`, `Split`, `source` | **Drop** | No exploitable signal (1:1 compound mapping) |
    """)
    return


if __name__ == "__main__":
    app.run()
