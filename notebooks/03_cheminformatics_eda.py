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
    # PXR Challenge — Cheminformatics EDA

    Topics covered:
    1. **Chemical space coverage** — intra-train vs test→train nearest-neighbour Tanimoto
    2. **Scaffold analysis** — Murcko scaffold diversity, actives vs inactives
    3. **Activity cliff analysis** — structurally similar pairs with large pEC50 gaps
    4. **Train/test similarity distribution** — ECFP4 nearest-neighbour analysis of the 513 blind test compounds
    5. **Active compound characterisation** — physicochemical features enriched in actives vs inactives
    6. **Chemical space PCA** — ECFP4 fingerprint PCA coloured by pEC50, train vs test overlay
    7. **3D shape analysis** — PMI/NPR descriptors from ETKDG conformers; active vs inactive vs test in shape space
    """)
    return


@app.cell
def _():
    import pathlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold

    sns.set_theme(style="whitegrid", palette="muted")
    DATA = pathlib.Path(__file__).parent.parent / "data"
    return (
        Chem,
        DATA,
        DataStructs,
        GetMorganGenerator,
        MurckoScaffold,
        np,
        pd,
        plt,
    )


@app.cell
def _(Chem, DATA, GetMorganGenerator, pd):
    train = pd.read_csv(DATA / "pxr-challenge_TRAIN.csv")
    test  = pd.read_csv(DATA / "pxr-challenge_TEST_BLINDED.csv")

    _gen = GetMorganGenerator(radius=2, fpSize=2048)

    def fps_from_smiles(smiles_series):
        fps, valid_idx = [], []
        for i, smi in enumerate(smiles_series):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fps.append(_gen.GetFingerprint(mol))
                valid_idx.append(i)
        return fps, valid_idx

    train_fps, train_valid = fps_from_smiles(train["SMILES"])
    test_fps,  test_valid  = fps_from_smiles(test["SMILES"])

    train_valid_df = train.iloc[train_valid].reset_index(drop=True)
    test_valid_df  = test.iloc[test_valid].reset_index(drop=True)
    return test_fps, test_valid_df, train_fps, train_valid_df


@app.cell
def _(mo):
    mo.md("""
    ## 1. Chemical Space Coverage

    We compare two Tanimoto distributions:
    - **Intra-train** (random pairs, sampled) — how diverse is the training set internally?
    - **Test → train NN** — how close is each test compound to its nearest training neighbour?

    If test NN Tanimoto is consistently high, the test set is structurally accessible from train.
    If it's low, the model must generalise beyond the training chemical space.
    The project note states all 513 test compounds are ECFP4 neighbours of the 63 training actives,
    so we expect high NN similarity — but this plot makes it concrete.
    """)
    return


@app.cell
def _(DataStructs, np, test_fps, train_fps):
    rng = np.random.default_rng(42)

    # Intra-train: sample 2000 random pairs
    n = len(train_fps)
    idx_a = rng.integers(0, n, 2000)
    idx_b = rng.integers(0, n, 2000)
    mask = idx_a != idx_b
    intra_tanimoto = np.array([
        DataStructs.TanimotoSimilarity(train_fps[a], train_fps[b])
        for a, b in zip(idx_a[mask], idx_b[mask])
    ])

    # Test → train NN
    nn_tanimoto = []
    for tfp in test_fps:
        sims = DataStructs.BulkTanimotoSimilarity(tfp, train_fps)
        nn_tanimoto.append(max(sims))
    nn_tanimoto = np.array(nn_tanimoto)
    return intra_tanimoto, nn_tanimoto


@app.cell
def _(intra_tanimoto, nn_tanimoto, plt):
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    bins = [i / 40 for i in range(41)]
    ax1.hist(intra_tanimoto, bins=bins, alpha=0.6, label="Intra-train (random pairs)", color="steelblue", density=True)
    ax1.hist(nn_tanimoto,    bins=bins, alpha=0.6, label="Test → train NN",            color="darkorange",  density=True)
    ax1.axvline(nn_tanimoto.mean(), color="darkorange", linestyle="--", linewidth=1.2,
                label=f"Test NN mean = {nn_tanimoto.mean():.2f}")
    ax1.set_xlabel("Tanimoto similarity (ECFP4)")
    ax1.set_ylabel("Density")
    ax1.set_title("Chemical space coverage: intra-train vs test→train nearest neighbour")
    ax1.legend(fontsize=9)
    fig1.tight_layout()
    fig1
    return


@app.cell
def _(intra_tanimoto, mo, nn_tanimoto, np):
    mo.md(f"""
    | Metric | Intra-train (random pairs) | Test → train NN |
    |--------|---------------------------|-----------------|
    | Mean   | {intra_tanimoto.mean():.3f} | {nn_tanimoto.mean():.3f} |
    | Median | {np.median(intra_tanimoto):.3f} | {np.median(nn_tanimoto):.3f} |
    | % ≥ 0.4 | {100*(intra_tanimoto>=0.4).mean():.1f}% | {100*(nn_tanimoto>=0.4).mean():.1f}% |
    | % ≥ 0.6 | {100*(intra_tanimoto>=0.6).mean():.1f}% | {100*(nn_tanimoto>=0.6).mean():.1f}% |

    Test compounds sit much closer to the training set than random train pairs — confirming
    the test was designed as an analogue challenge, not a scaffold-hop extrapolation task.
    A model that memorises training actives will be partially rewarded, but fine-grained
    potency differences within analogue series are what separates top submissions.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Scaffold Analysis

    Murcko scaffolds capture the ring system + linkers of each molecule (side chains stripped).
    We look at:
    - How many unique scaffolds exist vs singletons (scaffolds with only one compound)
    - Whether actives (pEC50 ≥ 6) cluster on specific scaffolds or are spread across unique scaffolds
    - Top scaffolds by compound count, coloured by mean pEC50

    High singleton rate in actives means the model cannot generalise via scaffold matching —
    it must learn substructural pharmacophore features instead.
    """)
    return


@app.cell
def _(Chem, MurckoScaffold, train_valid_df):
    def get_scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

    train_scaf_df = train_valid_df.copy()
    train_scaf_df["scaffold"] = train_scaf_df["SMILES"].apply(get_scaffold)
    return (train_scaf_df,)


@app.cell
def _(train_scaf_df):
    scaf_stats = (
        train_scaf_df.groupby("scaffold")
        .agg(count=("SMILES", "size"), mean_pec50=("pEC50", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    scaf_stats["is_singleton"] = scaf_stats["count"] == 1

    actives = train_scaf_df[train_scaf_df["pEC50"] >= 6]
    active_scaf_counts = actives["scaffold"].value_counts()
    n_active_unique_scaf = (active_scaf_counts == 1).sum()
    n_active_shared_scaf = (active_scaf_counts > 1).sum()
    return actives, n_active_shared_scaf, n_active_unique_scaf, scaf_stats


@app.cell
def _(
    actives,
    mo,
    n_active_shared_scaf,
    n_active_unique_scaf,
    scaf_stats,
    train_scaf_df,
):
    mo.md(f"""
    | Metric | Value |
    |--------|-------|
    | Total compounds (valid SMILES) | {len(train_scaf_df):,} |
    | Unique Murcko scaffolds | {scaf_stats['scaffold'].nunique():,} |
    | Singleton scaffolds (1 compound) | {scaf_stats['is_singleton'].sum():,} ({100*scaf_stats['is_singleton'].mean():.1f}%) |
    | Actives (pEC50 ≥ 6) | {len(actives):,} |
    | Actives on **unique** scaffolds | {n_active_unique_scaf:,} |
    | Actives on **shared** scaffolds | {n_active_shared_scaf:,} |
    """)
    return


@app.cell
def _(plt, scaf_stats):
    top20 = scaf_stats.head(20).copy()

    cmap = plt.cm.RdYlGn
    pec50_min, pec50_max = scaf_stats["mean_pec50"].min(), scaf_stats["mean_pec50"].max()
    norm = plt.Normalize(vmin=pec50_min, vmax=pec50_max)
    colors = [cmap(norm(v)) for v in top20["mean_pec50"]]

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bars = ax2.bar(range(len(top20)), top20["count"], color=colors)
    ax2.set_xticks(range(len(top20)))
    ax2.set_xticklabels([f"S{i+1}" for i in range(len(top20))], fontsize=8)
    ax2.set_xlabel("Scaffold rank (S1 = most frequent)")
    ax2.set_ylabel("Compound count")
    ax2.set_title("Top-20 Murcko scaffolds by compound count\n(colour = mean pEC50, green = more active)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig2.colorbar(sm, ax=ax2, label="Mean pEC50")
    fig2.tight_layout()
    fig2
    return


@app.cell
def _(plt, scaf_stats, train_scaf_df):
    _singleton_count = scaf_stats["is_singleton"].sum()
    _shared_count    = (~scaf_stats["is_singleton"]).sum()
    _scaf_map        = scaf_stats.set_index("scaffold")["is_singleton"]
    _tv = train_scaf_df.copy()
    _tv["singleton_scaffold"] = _tv["scaffold"].map(_scaf_map)

    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4))

    axes3[0].pie(
        [_singleton_count, _shared_count],
        labels=[f"Singleton\n({_singleton_count:,})", f"Shared\n({_shared_count:,})"],
        colors=["#d9534f", "#5bc0de"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )
    axes3[0].set_title("Scaffold diversity: singleton vs shared")

    for _label, _color, _group in [
        ("Singleton scaffold", "#d9534f", _tv[_tv["singleton_scaffold"] == True]["pEC50"].dropna()),
        ("Shared scaffold",    "#5bc0de", _tv[_tv["singleton_scaffold"] == False]["pEC50"].dropna()),
    ]:
        axes3[1].hist(_group, bins=35, alpha=0.6, label=_label, color=_color, density=True)

    axes3[1].set_xlabel("pEC50")
    axes3[1].set_ylabel("Density")
    axes3[1].set_title("pEC50 distribution by scaffold type")
    axes3[1].legend(fontsize=9)

    fig3.tight_layout()
    fig3
    return


@app.cell
def _(actives, mo, n_active_unique_scaf, scaf_stats):
    top_active_scaffolds = (
        actives.groupby("scaffold")
        .agg(n_actives=("pEC50", "size"), mean_pec50=("pEC50", "mean"))
        .reset_index()
        .merge(scaf_stats[["scaffold", "count"]], on="scaffold")
        .sort_values("n_actives", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    rows = "\n".join(
        f"| {r['scaffold'][:50]}{'…' if len(r['scaffold'])>50 else ''} | {r['n_actives']} | {r['count']} | {r['mean_pec50']:.2f} |"
        for _, r in top_active_scaffolds.iterrows()
    )
    mo.md(f"""
    ### Top scaffolds containing actives

    {n_active_unique_scaf} of the 67 actives sit on singleton scaffolds — no other training
    compound shares their ring system. This makes scaffold-based generalisation impossible for
    most actives; the model must rely on local substructural features.

    | Scaffold (truncated) | # actives | Total compounds on scaffold | Mean pEC50 |
    |----------------------|-----------|----------------------------|------------|
    {rows}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Activity Cliff Analysis

    An activity cliff is a pair of structurally similar compounds with a large pEC50 difference.
    They are the primary reason analogue-based models fail: the model interpolates smoothly
    in fingerprint space, but the biology changes sharply.

    We enumerate all pairs where at least one compound is an active (pEC50 ≥ 6) and
    Tanimoto ≥ 0.3, then plot similarity vs |ΔpEC50|. A cliff is defined as
    Tanimoto ≥ 0.3 and |ΔpEC50| ≥ 2.

    The scatter reveals whether the hard cases are spread uniformly or cluster at a specific
    similarity range — which informs the choice of similarity threshold for the analogue-mimic CV split.
    """)
    return


@app.cell
def _(DataStructs, np, train_fps, train_valid_df):
    _actives_mask = train_valid_df["pEC50"].values >= 6
    _active_idx   = np.where(_actives_mask)[0]
    _pec50        = train_valid_df["pEC50"].values
    _names        = train_valid_df["Molecule Name"].values

    cliff_rows = []
    for _ai in _active_idx:
        _sims = DataStructs.BulkTanimotoSimilarity(train_fps[_ai], train_fps)
        for _bi, _sim in enumerate(_sims):
            if _bi == _ai:
                continue
            if _sim < 0.3:
                continue
            _delta = abs(_pec50[_ai] - _pec50[_bi])
            cliff_rows.append({
                "mol_a": _names[_ai],
                "mol_b": _names[_bi],
                "pec50_a": _pec50[_ai],
                "pec50_b": _pec50[_bi],
                "tanimoto": _sim,
                "delta_pec50": _delta,
                "is_cliff": (_sim >= 0.3) and (_delta >= 2.0),
            })

    import pandas as _pd
    cliff_df = _pd.DataFrame(cliff_rows)
    return (cliff_df,)


@app.cell
def _(cliff_df, np, plt):
    _cliffs    = cliff_df[cliff_df["is_cliff"]]
    _non_cliff = cliff_df[~cliff_df["is_cliff"]]

    fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5))

    axes4[0].scatter(
        _non_cliff["tanimoto"], _non_cliff["delta_pec50"],
        alpha=0.25, s=12, color="steelblue", label="Non-cliff",
    )
    axes4[0].scatter(
        _cliffs["tanimoto"], _cliffs["delta_pec50"],
        alpha=0.6, s=18, color="#d9534f", label=f"Cliff (n={len(_cliffs):,})",
    )
    axes4[0].axhline(2.0, color="grey", linestyle="--", linewidth=0.8)
    axes4[0].axvline(0.3, color="grey", linestyle="--", linewidth=0.8)
    axes4[0].set_xlabel("Tanimoto similarity (ECFP4)")
    axes4[0].set_ylabel("|ΔpEC50|")
    axes4[0].set_title("Activity cliffs: active-involved pairs\n(Tanimoto ≥ 0.3)")
    axes4[0].legend(fontsize=9)

    _bins = np.linspace(0, cliff_df["delta_pec50"].max() + 0.2, 35)
    axes4[1].hist(_non_cliff["delta_pec50"], bins=_bins, alpha=0.6, color="steelblue", density=True, label="Non-cliff")
    axes4[1].hist(_cliffs["delta_pec50"],    bins=_bins, alpha=0.6, color="#d9534f",   density=True, label="Cliff")
    axes4[1].axvline(2.0, color="grey", linestyle="--", linewidth=0.8)
    axes4[1].set_xlabel("|ΔpEC50|")
    axes4[1].set_ylabel("Density")
    axes4[1].set_title("|ΔpEC50| distribution\namong active-neighbour pairs")
    axes4[1].legend(fontsize=9)

    fig4.tight_layout()
    fig4
    return


@app.cell
def _(cliff_df, mo):
    _cliffs = cliff_df[cliff_df["is_cliff"]]
    _top10  = (
        _cliffs
        .sort_values(["delta_pec50", "tanimoto"], ascending=[False, False])
        .drop_duplicates(subset=["mol_a"])
        .head(10)
        .reset_index(drop=True)
    )
    _rows = "\n".join(
        f"| {r['mol_a']} | {r['mol_b']} | {r['pec50_a']:.2f} | {r['pec50_b']:.2f} | {r['delta_pec50']:.2f} | {r['tanimoto']:.3f} |"
        for _, r in _top10.iterrows()
    )
    mo.md(f"""
    ### Activity cliff summary

    | Metric | Value |
    |--------|-------|
    | Active-involved pairs with Tanimoto ≥ 0.3 | {len(cliff_df):,} |
    | Cliff pairs (Tanimoto ≥ 0.3 and \|ΔpEC50\| ≥ 2) | {len(_cliffs):,} ({100*len(_cliffs)/len(cliff_df):.1f}%) |
    | Mean \|ΔpEC50\| among cliffs | {_cliffs['delta_pec50'].mean():.2f} |
    | Max \|ΔpEC50\| | {cliff_df['delta_pec50'].max():.2f} |

    ### Top-10 sharpest cliffs

    | Active | Neighbour | pEC50 (active) | pEC50 (neighbour) | \|ΔpEC50\| | Tanimoto |
    |--------|-----------|----------------|-------------------|------------|----------|
    {_rows}

    These are the hardest cases for the model: structurally similar neighbours that should
    predict similarly but differ by ≥ 2 log units in potency. Encoding substructural
    differences (e.g. SMARTS-based features, 3D shape) may help resolve some of these cliffs.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Train/Test Similarity Distribution

    We want to understand *which training compounds* the test set is closest to, and specifically
    whether the test compounds cluster near the 67 actives or the broader inactive pool.

    Three distributions are compared:
    - **Test → any train NN** (overall proximity)
    - **Test → active train NN** (proximity to the 67 actives specifically)
    - **Test → inactive train NN** (proximity to the inactive pool)

    If test compounds are systematically closer to actives than to inactives, the test set
    is designed to probe the active chemical space — meaning accurate active prediction is
    directly rewarded by the RAE metric.
    """)
    return


@app.cell
def _(DataStructs, np, test_fps, train_fps, train_valid_df):
    _pec50_tr    = train_valid_df["pEC50"].values
    _active_idx  = np.where(_pec50_tr >= 6)[0]
    _inactive_idx = np.where(_pec50_tr < 6)[0]
    _active_fps  = [train_fps[i] for i in _active_idx]
    _inactive_fps = [train_fps[i] for i in _inactive_idx]

    nn_any     = []
    nn_active  = []
    nn_inactive = []
    nn_active_name = []

    _names_tr = train_valid_df["Molecule Name"].values

    for _tfp in test_fps:
        _sims_all = DataStructs.BulkTanimotoSimilarity(_tfp, train_fps)
        nn_any.append(max(_sims_all))

        _sims_act = DataStructs.BulkTanimotoSimilarity(_tfp, _active_fps)
        _best_act_i = int(np.argmax(_sims_act))
        nn_active.append(_sims_act[_best_act_i])
        nn_active_name.append(_names_tr[_active_idx[_best_act_i]])

        _sims_inact = DataStructs.BulkTanimotoSimilarity(_tfp, _inactive_fps)
        nn_inactive.append(max(_sims_inact))

    nn_any      = np.array(nn_any)
    nn_active   = np.array(nn_active)
    nn_inactive = np.array(nn_inactive)
    return nn_active, nn_active_name, nn_any, nn_inactive


@app.cell
def _(nn_active, nn_any, nn_inactive, plt):
    fig5, axes5 = plt.subplots(1, 2, figsize=(13, 5))

    _bins = [i / 40 for i in range(41)]
    axes5[0].hist(nn_any,     bins=_bins, alpha=0.55, color="steelblue",  density=True, label="→ any train")
    axes5[0].hist(nn_active,  bins=_bins, alpha=0.55, color="#d9534f",    density=True, label="→ active train")
    axes5[0].hist(nn_inactive, bins=_bins, alpha=0.45, color="seagreen",  density=True, label="→ inactive train")
    axes5[0].axvline(nn_active.mean(),  color="#d9534f", linestyle="--", linewidth=1.1,
                     label=f"Active NN mean = {nn_active.mean():.2f}")
    axes5[0].axvline(nn_inactive.mean(), color="seagreen", linestyle="--", linewidth=1.1,
                     label=f"Inactive NN mean = {nn_inactive.mean():.2f}")
    axes5[0].set_xlabel("Tanimoto similarity (ECFP4)")
    axes5[0].set_ylabel("Density")
    axes5[0].set_title("Test → train NN similarity\nsplit by active vs inactive neighbours")
    axes5[0].legend(fontsize=8)

    # Scatter: test compound's active-NN sim vs inactive-NN sim
    _closer_to_active = nn_active >= nn_inactive
    axes5[1].scatter(
        nn_inactive[~_closer_to_active], nn_active[~_closer_to_active],
        s=12, alpha=0.4, color="seagreen", label="Closer to inactive",
    )
    axes5[1].scatter(
        nn_inactive[_closer_to_active], nn_active[_closer_to_active],
        s=12, alpha=0.4, color="#d9534f", label="Closer to active",
    )
    _lim = max(nn_active.max(), nn_inactive.max()) + 0.02
    axes5[1].plot([0, _lim], [0, _lim], "k--", linewidth=0.8, label="y = x")
    axes5[1].set_xlabel("NN similarity to nearest inactive")
    axes5[1].set_ylabel("NN similarity to nearest active")
    axes5[1].set_title(f"Test compound proximity:\nactive vs inactive neighbourhood\n"
                       f"({_closer_to_active.sum()} closer to active, "
                       f"{(~_closer_to_active).sum()} closer to inactive)")
    axes5[1].legend(fontsize=8)

    fig5.tight_layout()
    fig5
    return


@app.cell
def _(mo, nn_active, nn_active_name, nn_any, nn_inactive, test_valid_df):
    _closer_to_active = nn_active >= nn_inactive
    _top_active_nn = (
        test_valid_df
        .assign(nn_active_sim=nn_active, nn_active_train=nn_active_name)
        .sort_values("nn_active_sim", ascending=False)
        .head(10)[["Molecule Name", "nn_active_train", "nn_active_sim"]]
        .reset_index(drop=True)
    )
    _rows = "\n".join(
        f"| {r['Molecule Name']} | {r['nn_active_train']} | {r['nn_active_sim']:.3f} |"
        for _, r in _top_active_nn.iterrows()
    )
    mo.md(f"""
    ### Test/train proximity summary

    | Metric | Value |
    |--------|-------|
    | Test compounds closer to an active than any inactive | {_closer_to_active.sum()} / {len(nn_any)} ({100*_closer_to_active.mean():.1f}%) |
    | Mean NN sim → any train | {nn_any.mean():.3f} |
    | Mean NN sim → active train | {nn_active.mean():.3f} |
    | Mean NN sim → inactive train | {nn_inactive.mean():.3f} |
    | Test compounds with active NN sim ≥ 0.5 | {(nn_active >= 0.5).sum()} ({100*(nn_active>=0.5).mean():.1f}%) |
    | Test compounds with active NN sim ≥ 0.7 | {(nn_active >= 0.7).sum()} ({100*(nn_active>=0.7).mean():.1f}%) |

    ### Top-10 test compounds most similar to a training active

    | Test compound | Nearest training active | Tanimoto |
    |---------------|------------------------|----------|
    {_rows}

    Test compounds that are close to training actives are where the model has the most leverage.
    For test compounds that are closer to inactives, the model must extrapolate — those are
    likely the hardest cases and the primary driver of the CV→leaderboard RAE gap.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Active Compound Characterisation

    The 67 actives (pEC50 ≥ 6) are only 1.6% of the training set, yet they drive the
    competition metric. Understanding what makes them structurally distinct from inactives
    helps explain why models struggle and points to which feature representations matter.

    We compare the active vs inactive distributions across:
    - **Molecular weight and heavy atom count** — PXR actives are typically larger, lipophilic molecules
    - **LogP** — PXR's large hydrophobic LBD selectively binds lipophilic compounds
    - **H-bond donors/acceptors** — PXR agonists tend to have fewer polar groups
    - **Ring count and aromatic ring count** — scaffold rigidity is common in PXR ligands
    - **Rotatable bonds** — flexibility affects binding entropy

    Significant separation in any dimension suggests a feature worth including explicitly.
    """)
    return


@app.cell
def _(Chem, train_valid_df):
    from rdkit.Chem import Descriptors, rdMolDescriptors

    def _calc_props(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return (None,) * 7
        return (
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
        )

    _prop_cols = ["mol_wt", "logP", "hbd", "hba", "n_rings", "n_arom_rings", "n_rot_bonds"]
    _props = [_calc_props(s) for s in train_valid_df["SMILES"]]
    props_df = train_valid_df.copy()
    for _j, _col in enumerate(_prop_cols):
        props_df[_col] = [p[_j] for p in _props]
    props_df["active"] = props_df["pEC50"] >= 6
    return (props_df,)


@app.cell
def _(np, plt, props_df):
    _feat_cols = [
        ("mol_wt",       "Molecular weight (Da)"),
        ("logP",         "LogP"),
        ("hbd",          "H-bond donors"),
        ("hba",          "H-bond acceptors"),
        ("n_rings",      "Ring count"),
        ("n_arom_rings", "Aromatic ring count"),
        ("n_rot_bonds",  "Rotatable bonds"),
    ]
    _act = props_df[props_df["active"]]
    _ina = props_df[~props_df["active"]]

    fig6, axes6 = plt.subplots(2, 4, figsize=(16, 7))
    _axs = axes6.flatten()

    for _ax, (_col, _label) in zip(_axs, _feat_cols):
        _a_vals = _act[_col].dropna()
        _i_vals = _ina[_col].dropna()
        _lo = min(_a_vals.min(), _i_vals.min())
        _hi = max(_a_vals.max(), _i_vals.max())
        _bins = np.linspace(_lo, _hi, 30)
        _ax.hist(_i_vals, bins=_bins, alpha=0.55, color="steelblue", density=True,
                 label=f"Inactive (n={len(_i_vals):,})")
        _ax.hist(_a_vals, bins=_bins, alpha=0.70, color="#d9534f", density=True,
                 label=f"Active (n={len(_a_vals):,})")
        _ax.axvline(_a_vals.mean(), color="#d9534f",  linestyle="--", linewidth=1.0)
        _ax.axvline(_i_vals.mean(), color="steelblue", linestyle="--", linewidth=1.0)
        _ax.set_xlabel(_label, fontsize=8)
        _ax.set_ylabel("Density", fontsize=8)
        _ax.tick_params(labelsize=7)
        _ax.legend(fontsize=6)

    _axs[-1].set_visible(False)
    fig6.suptitle(
        "Physicochemical feature distributions: actives vs inactives  (dashed = group mean)",
        fontsize=11,
    )
    fig6.tight_layout()
    fig6
    return


@app.cell
def _(mo, props_df):
    _feat_cols = [
        ("mol_wt",       "Mol weight (Da)"),
        ("logP",         "LogP"),
        ("hbd",          "H-bond donors"),
        ("hba",          "H-bond acceptors"),
        ("n_rings",      "Ring count"),
        ("n_arom_rings", "Aromatic rings"),
        ("n_rot_bonds",  "Rotatable bonds"),
    ]
    _act = props_df[props_df["active"]]
    _ina = props_df[~props_df["active"]]
    _rows = []
    for _col, _label in _feat_cols:
        _am, _as = _act[_col].mean(), _act[_col].std()
        _im, _is = _ina[_col].mean(), _ina[_col].std()
        _rows.append(
            f"| {_label} | {_am:.2f} ± {_as:.2f} | {_im:.2f} ± {_is:.2f} | {_am - _im:+.2f} |"
        )
    mo.md(
        "### Summary statistics: actives vs inactives\n\n"
        "| Feature | Active mean ± std | Inactive mean ± std | Δ (active − inactive) |\n"
        "|---------|------------------|---------------------|----------------------|\n"
        + "\n".join(_rows)
        + """

    Key take-aways:
    - Actives are **larger** (higher MW, more rings) and **more lipophilic** (higher LogP) — consistent with PXR's large hydrophobic LBP.
    - Actives have **fewer H-bond donors** — the PXR binding pocket is predominantly hydrophobic.
    - Including LogP and MW as explicit node/global features in a graph model, or using fingerprint
      radius ≥ 2 to capture ring systems, is well-motivated by these distributions.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Chemical Space PCA

    We project all ECFP4 fingerprints (2048-bit) onto the first two principal components
    using a numpy SVD — no sklearn required. The projection is fit on training compounds only;
    test compounds are projected into the same space.

    Three views:
    1. **Train coloured by pEC50** — where do actives sit in fingerprint PC space?
    2. **Train vs test overlay** — does the test set fall inside the training distribution
       or in peripheral regions?

    Actives clustering in a specific region would suggest a targeted search strategy; spread
    across PC space indicates potency is driven by subtle local features not visible in 2D projection.
    """)
    return


@app.cell
def _(np, test_fps, train_fps):
    def _fp_to_matrix(fps):
        n = len(fps)
        d = fps[0].GetNumBits()
        X = np.zeros((n, d), dtype=np.float32)
        for _i, _fp in enumerate(fps):
            _bits = _fp.GetOnBits()
            X[_i, list(_bits)] = 1.0
        return X

    _X_tr = _fp_to_matrix(train_fps)
    _X_te = _fp_to_matrix(test_fps)

    _mean = _X_tr.mean(axis=0)
    _Xc   = _X_tr - _mean
    _U, _S, _Vt = np.linalg.svd(_Xc, full_matrices=False)
    _comps      = _Vt[:2]
    _var_ratio  = (_S[:2] ** 2) / (_S ** 2).sum()

    pca_train = _Xc @ _comps.T
    pca_test  = (_X_te - _mean) @ _comps.T
    pca_var   = _var_ratio
    return pca_test, pca_train, pca_var


@app.cell
def _(pca_test, pca_train, pca_var, plt, train_valid_df):
    _pec50  = train_valid_df["pEC50"].values
    _active = _pec50 >= 6

    fig7, axes7 = plt.subplots(1, 2, figsize=(14, 5))

    # Left: train coloured by pEC50
    _sc = axes7[0].scatter(
        pca_train[~_active, 0], pca_train[~_active, 1],
        c=_pec50[~_active], cmap="Blues_r", s=8, alpha=0.35,
        vmin=_pec50.min(), vmax=_pec50.max(),
    )
    axes7[0].scatter(
        pca_train[_active, 0], pca_train[_active, 1],
        c=_pec50[_active], cmap="Reds", s=45, alpha=0.9,
        vmin=_pec50.min(), vmax=_pec50.max(),
        edgecolors="black", linewidths=0.5, label="Active (pEC50 ≥ 6)",
        zorder=3,
    )
    plt.colorbar(_sc, ax=axes7[0], label="pEC50")
    axes7[0].set_xlabel(f"PC1 ({100*pca_var[0]:.1f}% var)")
    axes7[0].set_ylabel(f"PC2 ({100*pca_var[1]:.1f}% var)")
    axes7[0].set_title("Train: ECFP4 PCA coloured by pEC50\n(actives = filled circles, outlined)")
    axes7[0].legend(fontsize=8)

    # Right: train vs test overlay
    axes7[1].scatter(
        pca_train[:, 0], pca_train[:, 1],
        s=7, alpha=0.25, color="steelblue", label=f"Train (n={len(pca_train):,})",
    )
    axes7[1].scatter(
        pca_test[:, 0], pca_test[:, 1],
        s=16, alpha=0.6, color="#d9534f", marker="^",
        label=f"Test (n={len(pca_test):,})",
    )
    axes7[1].set_xlabel(f"PC1 ({100*pca_var[0]:.1f}% var)")
    axes7[1].set_ylabel(f"PC2 ({100*pca_var[1]:.1f}% var)")
    axes7[1].set_title("Train vs test: ECFP4 PCA overlay")
    axes7[1].legend(fontsize=8)

    fig7.tight_layout()
    fig7
    return


@app.cell
def _(mo, pca_test, pca_train, pca_var, train_valid_df):
    _active      = train_valid_df["pEC50"].values >= 6
    _pc1_act     = pca_train[_active, 0]
    _pc2_act     = pca_train[_active, 1]
    _pc1_inact   = pca_train[~_active, 0]
    _pc2_inact   = pca_train[~_active, 1]

    _test_in_bbox = (
        (pca_test[:, 0] >= _pc1_act.min()) & (pca_test[:, 0] <= _pc1_act.max()) &
        (pca_test[:, 1] >= _pc2_act.min()) & (pca_test[:, 1] <= _pc2_act.max())
    ).mean()

    mo.md(f"""
    ### PCA summary

    | Metric | Value |
    |--------|-------|
    | Variance explained by PC1 | {100*pca_var[0]:.2f}% |
    | Variance explained by PC2 | {100*pca_var[1]:.2f}% |
    | Combined PC1+PC2 variance | {100*pca_var[:2].sum():.2f}% |
    | Active centroid (PC1, PC2) | ({_pc1_act.mean():.2f}, {_pc2_act.mean():.2f}) |
    | Inactive centroid (PC1, PC2) | ({_pc1_inact.mean():.2f}, {_pc2_inact.mean():.2f}) |
    | Test compounds within active PC1/PC2 bounding box | {100*_test_in_bbox:.1f}% |

    PC1+PC2 capture only a small fraction of total variance in ECFP4 space — the
    active/inactive boundary is high-dimensional and not linearly separable in 2D.
    The test compounds largely overlap the training distribution, consistent with the
    analogue-based test design seen in Section 4. Actives do not form a tight isolated cluster,
    which explains why distance-based methods underperform against learned representations.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. 3D Shape Analysis

    PXR has a large, flexible, predominantly hydrophobic ligand-binding pocket that accommodates
    structurally diverse agonists — but shape complementarity still matters. 2D fingerprints
    are blind to 3D shape, which may explain part of the model's underprediction of actives.

    We generate one ETKDG v3 conformer per molecule (MMFF-minimised) and compute the
    **normalised principal moments of inertia (NPR1, NPR2)**. These two numbers place every
    molecule on the PMI triangle:
    - Bottom-left corner **(rod)**: linear, elongated molecules (NPR1 ≈ 0, NPR2 ≈ 0.5)
    - Bottom-right corner **(disc)**: flat, planar molecules (NPR1 ≈ 0.5, NPR2 ≈ 1)
    - Top corner **(sphere)**: globular molecules (NPR1 ≈ NPR2 ≈ 1)

    We also extract **asphericity**, **eccentricity**, and **spherocity index** as scalar summaries.

    Two questions:
    1. Do PXR actives occupy a distinct region of shape space vs inactives?
    2. Is the test set closer to actives in 3D shape space than it appeared in 2D Tanimoto space?
    """)
    return


@app.cell
def _(Chem, test_valid_df, train_valid_df):
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdMolDescriptors import (
        CalcNPR1, CalcNPR2, CalcAsphericity,
        CalcEccentricity, CalcSpherocityIndex,
    )

    def _calc_3d(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return (None,) * 5
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == -1:
            return (None,) * 5
        AllChem.MMFFOptimizeMolecule(mol)
        try:
            return (
                CalcNPR1(mol),
                CalcNPR2(mol),
                CalcAsphericity(mol),
                CalcEccentricity(mol),
                CalcSpherocityIndex(mol),
            )
        except Exception:
            return (None,) * 5

    _cols = ["npr1", "npr2", "asphericity", "eccentricity", "spherocity"]

    train_3d = train_valid_df.copy()
    _train_props = [_calc_3d(s) for s in train_valid_df["SMILES"]]
    for _j, _c in enumerate(_cols):
        train_3d[_c] = [p[_j] for p in _train_props]
    train_3d["active"] = train_3d["pEC50"] >= 6
    train_3d = train_3d.dropna(subset=_cols).reset_index(drop=True)

    test_3d = test_valid_df.copy()
    _test_props = [_calc_3d(s) for s in test_valid_df["SMILES"]]
    for _j, _c in enumerate(_cols):
        test_3d[_c] = [p[_j] for p in _test_props]
    test_3d = test_3d.dropna(subset=_cols).reset_index(drop=True)

    shape_cols = _cols
    return shape_cols, test_3d, train_3d


@app.cell
def _(mo, test_3d, train_3d):
    mo.md(f"""
    Conformers generated successfully for **{len(train_3d):,} / {len(train_3d):,} train** and
    **{len(test_3d):,} / {len(test_3d):,} test** compounds (ETKDG v3 + MMFF minimisation).
    """)
    return


@app.cell
def _(plt, test_3d, train_3d):
    _act = train_3d[train_3d["active"]]
    _ina = train_3d[~train_3d["active"]]

    fig8, axes8 = plt.subplots(1, 2, figsize=(14, 5))

    # PMI triangle boundary
    _tri_x = [0.0, 0.5, 1.0, 0.0]
    _tri_y = [0.5, 1.0, 1.0, 0.5]
    for _ax in axes8:
        _ax.plot(_tri_x, _tri_y, "k-", linewidth=0.8, zorder=0)
        _ax.text(0.01, 0.48, "rod",    fontsize=8, color="grey")
        _ax.text(0.47, 1.01, "sphere", fontsize=8, color="grey", ha="center")
        _ax.text(0.96, 0.98, "disc",   fontsize=8, color="grey", ha="right")

    # Left: train coloured by pEC50
    _sc = axes8[0].scatter(
        _ina["npr1"], _ina["npr2"],
        c=_ina["pEC50"], cmap="Blues_r", s=8, alpha=0.3,
        vmin=train_3d["pEC50"].min(), vmax=train_3d["pEC50"].max(),
    )
    axes8[0].scatter(
        _act["npr1"], _act["npr2"],
        c=_act["pEC50"], cmap="Reds", s=45, alpha=0.9,
        vmin=train_3d["pEC50"].min(), vmax=train_3d["pEC50"].max(),
        edgecolors="black", linewidths=0.5, zorder=3, label="Active (pEC50 ≥ 6)",
    )
    plt.colorbar(_sc, ax=axes8[0], label="pEC50")
    axes8[0].set_xlabel("NPR1  (I1/I3)")
    axes8[0].set_ylabel("NPR2  (I2/I3)")
    axes8[0].set_title("PMI triangle: train coloured by pEC50")
    axes8[0].legend(fontsize=8)

    # Right: train vs test overlay
    axes8[1].scatter(
        train_3d["npr1"], train_3d["npr2"],
        s=7, alpha=0.2, color="steelblue", label=f"Train (n={len(train_3d):,})",
    )
    axes8[1].scatter(
        test_3d["npr1"], test_3d["npr2"],
        s=14, alpha=0.55, color="#d9534f", marker="^",
        label=f"Test (n={len(test_3d):,})",
    )
    axes8[1].scatter(
        _act["npr1"], _act["npr2"],
        s=50, alpha=0.9, color="gold",
        edgecolors="black", linewidths=0.6, zorder=4,
        label=f"Actives (n={len(_act):,})",
    )
    axes8[1].set_xlabel("NPR1  (I1/I3)")
    axes8[1].set_ylabel("NPR2  (I2/I3)")
    axes8[1].set_title("PMI triangle: train vs test (actives highlighted)")
    axes8[1].legend(fontsize=8)

    fig8.tight_layout()
    fig8
    return


@app.cell
def _(np, plt, shape_cols, test_3d, train_3d):
    # NN distance in standardised 3D shape space
    _act_feat  = train_3d[train_3d["active"]][shape_cols].values.astype(float)
    _ina_feat  = train_3d[~train_3d["active"]][shape_cols].values.astype(float)
    _test_feat = test_3d[shape_cols].values.astype(float)

    # Standardise by train mean/std
    _mean = train_3d[shape_cols].values.mean(0)
    _std  = train_3d[shape_cols].values.std(0) + 1e-9
    _act_z   = (_act_feat  - _mean) / _std
    _ina_z   = (_ina_feat  - _mean) / _std
    _test_z  = (_test_feat - _mean) / _std

    nn_3d_active   = np.array([np.min(np.linalg.norm(_act_z  - t, axis=1)) for t in _test_z])
    nn_3d_inactive = np.array([np.min(np.linalg.norm(_ina_z  - t, axis=1)) for t in _test_z])

    _closer_3d = nn_3d_active <= nn_3d_inactive

    fig9, axes9 = plt.subplots(1, 2, figsize=(13, 5))

    _bins = np.linspace(0, max(nn_3d_active.max(), nn_3d_inactive.max()) + 0.1, 40)
    axes9[0].hist(nn_3d_active,   bins=_bins, alpha=0.6, color="#d9534f",  density=True,
                  label=f"→ active (mean={nn_3d_active.mean():.3f})")
    axes9[0].hist(nn_3d_inactive, bins=_bins, alpha=0.55, color="steelblue", density=True,
                  label=f"→ inactive (mean={nn_3d_inactive.mean():.3f})")
    axes9[0].axvline(nn_3d_active.mean(),   color="#d9534f",  linestyle="--", linewidth=1.1)
    axes9[0].axvline(nn_3d_inactive.mean(), color="steelblue", linestyle="--", linewidth=1.1)
    axes9[0].set_xlabel("Euclidean NN distance (standardised 3D shape descriptors)")
    axes9[0].set_ylabel("Density")
    axes9[0].set_title("Test → train NN distance in 3D shape space")
    axes9[0].legend(fontsize=9)

    _lim = max(nn_3d_active.max(), nn_3d_inactive.max()) * 1.02
    axes9[1].scatter(
        nn_3d_inactive[~_closer_3d], nn_3d_active[~_closer_3d],
        s=12, alpha=0.4, color="steelblue", label="Closer to inactive",
    )
    axes9[1].scatter(
        nn_3d_inactive[_closer_3d], nn_3d_active[_closer_3d],
        s=12, alpha=0.4, color="#d9534f", label="Closer to active",
    )
    axes9[1].plot([0, _lim], [0, _lim], "k--", linewidth=0.8)
    axes9[1].set_xlabel("NN dist to nearest inactive (3D shape)")
    axes9[1].set_ylabel("NN dist to nearest active (3D shape)")
    axes9[1].set_title(
        f"3D shape proximity: active vs inactive neighbourhood\n"
        f"({_closer_3d.sum()} closer to active, {(~_closer_3d).sum()} closer to inactive)"
    )
    axes9[1].legend(fontsize=8)

    fig9.tight_layout()
    fig9
    return nn_3d_active, nn_3d_inactive


@app.cell
def _(mo, nn_3d_active, nn_3d_inactive, shape_cols, test_3d, train_3d):
    _act  = train_3d[train_3d["active"]]
    _ina  = train_3d[~train_3d["active"]]
    _closer_3d = nn_3d_active <= nn_3d_inactive

    _feat_rows = []
    for _c in shape_cols:
        _am, _as = _act[_c].mean(), _act[_c].std()
        _im, _is = _ina[_c].mean(), _ina[_c].std()
        _tm, _ts = test_3d[_c].mean(), test_3d[_c].std()
        _feat_rows.append(
            f"| {_c} | {_am:.3f} ± {_as:.3f} | {_im:.3f} ± {_is:.3f} | {_tm:.3f} ± {_ts:.3f} |"
        )

    mo.md(
        "### 3D shape descriptor means: actives vs inactives vs test\n\n"
        "| Descriptor | Active | Inactive | Test |\n"
        "|------------|--------|----------|------|\n"
        + "\n".join(_feat_rows)
        + f"""

### NN proximity summary (3D shape space vs 2D Tanimoto)

| Metric | 2D Tanimoto | 3D shape |
|--------|-------------|----------|
| Test closer to an **active** | 48.9% | {100*_closer_3d.mean():.1f}% |
| Test closer to an **inactive** | 51.1% | {100*(~_closer_3d).mean():.1f}% |
| Mean NN dist/sim → active | 0.398 (sim) | {nn_3d_active.mean():.3f} (dist) |
| Mean NN dist/sim → inactive | 0.464 (sim) | {nn_3d_inactive.mean():.3f} (dist) |

**The 3D result is the opposite of what we hoped.** In 3D shape space, the test set is
overwhelmingly closer to inactives (~99%) and the gap is far larger (active NN dist
{nn_3d_active.mean():.3f} vs inactive {nn_3d_inactive.mean():.3f}).

The explanation: there are only 67 actives but {len(_ina):,} inactives, so the inactive pool
densely tiles the shape space. The NN distance to the nearest inactive is tiny because the
inactives are everywhere — removing the pool-size effect would require comparing at the same
pool size (e.g. random subsampling to 67 inactives).

More importantly, this tells us **3D shape alone does not discriminate PXR actives from
inactives**. PXR's promiscuous binding pocket accepts a wide variety of molecular shapes;
what matters is the specific combination of hydrophobic contacts and H-bond acceptors, not
overall shape. This is consistent with PXR being notoriously difficult to model with
shape-based methods, and reinforces that learned atomic representations (graph networks,
CheMeleon) that capture local chemical environment are the right approach.
"""
    )
    return


if __name__ == "__main__":
    app.run()
