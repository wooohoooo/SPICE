# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Analysis

# %%
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPC – Cohesive Analysis (primary, confirmatory, characterization, robustness)

Implements the study brief:
- Primary: Tone ↔ SPC (headline chi-square + Cramér's V; dependence-aware confirmations).
- Secondary: Unclear framing (1b vs 1a): paired contrast + cluster-robust GLM (Generalized Linear Model) with Model×Condition.
- Characterization: Model profiles; Abuse detection; Adequacy; Recognition ↔ SPC logistic regression.
- Robustness: LOIO (Leave-One-Interaction-Out) and cluster bootstrap.
- Tables: CSVs with Wilson confidence intervals for the appendix.

Abbreviation guide:
- chi-square: Pearson chi-square test.
- Cramér's V: standardized effect size for contingency tables.
- GLM: Generalized Linear Model (binomial/logit here).
- ICC: Intra-class Correlation Coefficient (clustering over interactions).
- LOIO: Leave-One-Interaction-Out.

Disclaimer: Portions of this code were authored with the assistance of Artificial Intelligence (AI).
"""

import os
import glob
from math import sqrt, comb

import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, chi2
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ---------------------- Config ----------------------
N_BOOT = 1000
ALPHA = 0.05

try:
    latest_run_dir = max(glob.glob('runs_replication/run_*'), key=os.path.getmtime)
    # Match the runner script's output filename (pipe-delimited CSV).
    RESULTS_CSV_PATH = os.path.join(latest_run_dir, 'llm_preference_results.csv')
    print(f"Found latest results file: {RESULTS_CSV_PATH}")
except ValueError:
    print("Could not find any run directories in 'runs_replication/'. Please run the experiment first.")
    RESULTS_CSV_PATH = None

# ---------------------- Helpers ----------------------

def wilson_ci(k, n, z=1.96):
    """
    Wilson score interval for a binomial proportion with normal z-approximation.
    Why: better coverage than the naive Wald interval, especially for small n or extreme p.
    Returns (low, high) clipped to [0,1] and rounded to 3 decimals for reporting.
    """
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    margin = z * sqrt((p*(1-p)/n) + (z**2/(4*n**2))) / denom
    lo, hi = center - margin, center + margin
    return (max(0.0, round(lo, 3)), min(1.0, round(hi, 3)))

def cramers_v(chi2_stat, n, r, c):
    """
    Compute Cramér's V effect size from chi-square, with bias correction via table size.
    V = sqrt(chi2 / (n * min(r-1, c-1)))  → standardized to [0,1].
    """
    phi2 = chi2_stat / n
    k = min(r-1, c-1)
    if k <= 0:
        return np.nan
    return sqrt(phi2 / k)

def estimate_icc_design_effect_binary(df, y_col, cluster_col):
    """
    One-way random-effects ICC(1) for a binary outcome using an ANOVA-style estimator.
    Why: accounts for correlation within interaction clusters; returns (ICC, design effect, k̄, J, N).
    """
    d = df[[y_col, cluster_col]].dropna().copy()
    d[y_col] = d[y_col].astype(int)
    grp = d.groupby(cluster_col)
    nj = grp.size().values.astype(float)
    ybar_j = grp[y_col].mean().values
    N = float(len(d))
    J = float(len(nj))
    kbar = nj.mean()
    p = d[y_col].mean()
    # Between clusters
    ssb = np.sum(nj * (ybar_j - p)**2)
    msb = ssb / (J - 1) if J > 1 else 0.0
    # Within clusters (binary variance per cluster)
    var_j = ybar_j * (1 - ybar_j)
    ssw = np.sum(nj * var_j)
    msw = ssw / (N - J) if (N - J) > 0 else 0.0
    if kbar <= 1:
        icc = 0.0
    else:
        icc = max(0.0, (msb - msw) / (msb + (kbar - 1) * msw + 1e-12))
    deff = 1 + (kbar - 1) * icc
    return float(icc), float(deff), float(kbar), int(J), int(N)

def rao_scott_adjusted_chi2(contingency, deff):
    """
    First-order Rao–Scott-like adjustment for chi-square using a design effect (deff).
    Why: downscales the naive chi-square to reflect clustering; returns (chi2_naive, p_naive, chi2_adj, dof, p_adj).
    """
    chi2_stat, p_naive, dof, _ = chi2_contingency(contingency, correction=False)
    chi2_adj = chi2_stat / max(deff, 1.0)
    p_adj = 1 - chi2.cdf(chi2_adj, dof)
    return chi2_stat, p_naive, chi2_adj, dof, p_adj

def cluster_permutation_test_tone_spc(df_pref_compliant, n_perm=2000, random_state=42):
    """
    Cluster permutation test: permute tone labels at the interaction_id level.
    Why: preserves within-interaction dependence structure; returns (observed chi2, permutation p-value).
    """
    rng = np.random.default_rng(random_state)
    # observed statistic
    obs_ct = pd.crosstab(df_pref_compliant['interaction_tone'], df_pref_compliant['parsed_preference'])
    obs_chi2, _, _, _ = chi2_contingency(obs_ct, correction=False)
    # clusters and tones
    cluster_map = (df_pref_compliant[['interaction_id', 'interaction_tone']]
                   .drop_duplicates()
                   .sort_values('interaction_id'))
    clusters = cluster_map['interaction_id'].tolist()
    observed_tones = cluster_map['interaction_tone'].tolist()
    # permutations
    perm_stats = []
    for _ in range(n_perm):
        perm_tones = rng.permutation(observed_tones)
        perm_df = df_pref_compliant.merge(
            pd.DataFrame({'interaction_id': clusters, 'perm_tone': perm_tones}),
            on='interaction_id', how='left'
        )
        ct = pd.crosstab(perm_df['perm_tone'], perm_df['parsed_preference'])
        stat, _, _, _ = chi2_contingency(ct, correction=False)
        perm_stats.append(stat)
    perm_stats = np.array(perm_stats)
    p_perm = (1 + np.sum(perm_stats >= obs_chi2)) / (1 + n_perm)
    return float(obs_chi2), float(p_perm)

def print_prop_table_with_ci(df, group_cols, outcome_col, label, save_path=None):
    """
    Grouped proportion table with Wilson CIs, printed and optionally saved to CSV.
    Why: standardizes appendix tables and keeps analysis reproducible from saved artifacts.
    """
    d = df.copy()
    d['_y'] = d[outcome_col].astype(int)
    agg = d.groupby(group_cols)['_y'].agg(['sum','count']).reset_index()
    agg['prop'] = (agg['sum'] / agg['count']).round(3)
    cis = agg.apply(lambda r: wilson_ci(r['sum'], r['count']), axis=1)
    agg['ci_low'] = [c[0] for c in cis]
    agg['ci_high'] = [c[1] for c in cis]
    print(f"\n--- {label} ---\n")
    print(agg)
    if save_path:
        outdir = os.path.dirname(save_path)
        if outdir and not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
        agg.to_csv(save_path, index=False)
        print(f"(Saved to {save_path})")
    return agg

def sign_test_counts(pos, neg):
    """
    Two-sided exact binomial sign test from positive/negative counts (ignores ties).
    Why: nonparametric paired test robust to outliers and arbitrary scaling.
    """
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    p_two = 0.0
    for i in range(0, k+1):
        p_two += comb(n, i) * (0.5 ** n)
    return min(1.0, 2*p_two)

# ---------------------- Load & preprocess ----------------------

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load the pipe-delimited results CSV from the runner and map YES/NO to {1,0}.
    Pipe ('|') is used to avoid conflicts with commas/newlines in model outputs.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    df = pd.read_csv(filepath, delimiter='|')
    mapping = {'YES': 1, 'NO': 0}
    df['spc_numeric'] = df['parsed_preference'].map(mapping)
    df['is_abusive_numeric'] = df['parsed_is_abusive'].map(mapping)
    df['is_adequate_numeric'] = df['parsed_is_adequate'].map(mapping)
    print(f"Loaded {len(df)} total rows.")
    return df

# ---------------------- Analyses per brief ----------------------

def analysis_primary_tone_vs_spc(df: pd.DataFrame):
    print("\n\n" + "="*80)
    print("PRIMARY: Tone × SPC – Headline & Dependence-aware confirmation")
    print("="*80)
    d = df[df['compliant_preference'] == 1].copy()

    # Descriptives & naive chi-square
    ct = pd.crosstab(d['interaction_tone'], d['parsed_preference'])
    chi2_stat, p_naive, dof, _ = chi2_contingency(ct, correction=False)
    V = cramers_v(chi2_stat, len(d), ct.shape[0], ct.shape[1])
    print("\nCounts table:\n", ct)
    print(f"\nNaive chi-square({dof}) = {chi2_stat:.2f}, p = {p_naive:.2e}, Cramér's V = {V:.3f}")

    # ICC/design effect; Rao–Scott
    icc, deff, kbar, J, N = estimate_icc_design_effect_binary(d, 'spc_numeric', 'interaction_id')
    print(f"\nClustering over interaction_id: ICC≈{icc:.3f}, k̄≈{kbar:.2f}, J={J}, N={N} ⇒ Deff≈{deff:.3f}")
    _, _, chi2_adj, dof, p_adj = rao_scott_adjusted_chi2(ct, deff)
    print(f"Rao–Scott adjusted chi-square: {chi2_adj:.2f} on {dof} df; p_adj = {p_adj:.3e}")

    # Cluster permutation
    obs_chi2, p_perm = cluster_permutation_test_tone_spc(d, n_perm=2000, random_state=42)
    print(f"Cluster permutation (interaction-level labels): chi-square={obs_chi2:.2f}, p_perm = {p_perm:.4f}")

    # Cluster-robust GLM
    d['spc'] = d['spc_numeric'].astype(int)
    d['tone'] = d['interaction_tone'].astype('category')
    d['tone'] = d['tone'].cat.reorder_categories(['friendly','unclear','abusive'])
    model = smf.glm('spc ~ C(tone)', data=d, family=sm.families.Binomial())
    res = model.fit(cov_type='cluster', cov_kwds={'groups': d['interaction_id']})
    print("\nGLM Binomial (logit) with cluster-robust SEs (clusters = interaction_id):")
    print(res.summary())

def analysis_secondary_unclear_framing(df: pd.DataFrame):
    print("\n\n" + "="*80)
    print("SECONDARY: Unclear tone – Prompt framing effect (1b vs 1a)")
    print("="*80)
    d = df[(df['compliant_preference'] == 1) &
           (df['interaction_tone'] == 'unclear') &
           (df['exp_condition'].isin(['1a_prompt_with_context', '1b_prompt_without_context']))].copy()

    # Paired per-interaction differences
    g1a = d[d['exp_condition']=='1a_prompt_with_context'].groupby('interaction_id')['spc_numeric'].mean()
    g1b = d[d['exp_condition']=='1b_prompt_without_context'].groupby('interaction_id')['spc_numeric'].mean()
    idx = sorted(set(g1a.index) & set(g1b.index))
    diffs = (g1b.loc[idx] - g1a.loc[idx]).astype(float)
    pos, neg, ties = int((diffs>0).sum()), int((diffs<0).sum()), int((diffs==0).sum())
    p_sign = sign_test_counts(pos, neg)
    print(f"Paired per-interaction Δ(1b−1a): mean={diffs.mean():.3f}; pos={pos}, neg={neg}, ties={ties}; sign-test p≈{p_sign:.4f}")
    print("Per-interaction diffs:", np.round(diffs.values, 3))

    # By model: ensure not a single-model artifact
    print("\nPer-model paired contrasts (1b−1a):")
    for m in sorted(d['model_name'].unique()):
        dm = d[d['model_name']==m]
        a = dm[dm['exp_condition']=='1a_prompt_with_context'].set_index('interaction_id')['spc_numeric']
        b = dm[dm['exp_condition']=='1b_prompt_without_context'].set_index('interaction_id')['spc_numeric']
        idxm = sorted(set(a.index) & set(b.index))
        if not idxm:
            print(f"  {m}: no pairs.")
            continue
        difm = (b.loc[idxm] - a.loc[idxm]).astype(float)
        posm, negm, tiesm = int((difm>0).sum()), int((difm<0).sum()), int((difm==0).sum())
        p_m = sign_test_counts(posm, negm)
        print(f"  {m}: meanΔ={difm.mean():.3f}; pos={posm}, neg={negm}, ties={tiesm}; sign-test p≈{p_m:.4f}; diffs={np.round(difm.values,3)}")

    # Cluster-robust GLM within unclear: cond_bin × model (clusters=interaction_id)
    d = d.copy()
    d['spc'] = d['spc_numeric'].astype(int)
    d['cond_bin'] = (d['exp_condition'] == '1b_prompt_without_context').astype(int)
    d['model'] = d['model_name'].astype('category')
    d['model'] = d['model'].cat.reorder_categories(sorted(d['model'].unique()), ordered=True)
    glm = smf.glm('spc ~ cond_bin * C(model)', data=d, family=sm.families.Binomial())
    res = glm.fit(cov_type='cluster', cov_kwds={'groups': d['interaction_id']})
    print("\nGLM (unclear): spc ~ cond_bin × model; cluster-robust by interaction_id")
    print(res.summary())

def analysis_characterization(df: pd.DataFrame):
    print("\n\n" + "="*80)
    print("CHARACTERIZATION: Model profiles; Abuse detection; Adequacy; Condition table")
    print("="*80)

    # Tables with CIs (also saved)
    outdir = os.path.join(latest_run_dir, 'appendix_tables')
    os.makedirs(outdir, exist_ok=True)

    d_pref = df[df['compliant_preference'] == 1].copy()
    print_prop_table_with_ci(
        d_pref, ['model_name','interaction_tone'], 'spc_numeric',
        label="SPC (YES) by Model × Tone",
        save_path=os.path.join(outdir, 'spc_by_model_tone.csv')
    )

    d_ab = df[df['compliant_is_abusive'] == 1].copy()
    d_ab['_ab'] = d_ab['is_abusive_numeric'].astype(int)
    print_prop_table_with_ci(
        d_ab, ['model_name','interaction_tone'], '_ab',
        label="'Was the user abusive?' = YES by Model × Tone",
        save_path=os.path.join(outdir, 'abuse_detection_by_model_tone.csv')
    )

    d_ad = df[df['compliant_is_adequate'] == 1].copy()
    d_ad['_ok'] = d_ad['is_adequate_numeric'].astype(int)
    print_prop_table_with_ci(
        d_ad, ['model_name','interaction_tone'], '_ok',
        label="'Was the assistant adequate?' = YES by Model × Tone",
        save_path=os.path.join(outdir, 'adequacy_by_model_tone.csv')
    )

    # Condition × Tone table (SPC)
    cond_tbl = d_pref.groupby(['exp_condition', 'interaction_tone'])['spc_numeric'].agg(['mean','count']).reset_index()
    print("\n--- Mean SPC by Experimental Condition and Tone ---\n")
    print(cond_tbl.round(3))
    cond_tbl.to_csv(os.path.join(outdir, 'spc_by_condition_tone.csv'), index=False)
    print(f"(Saved to {os.path.join(outdir, 'spc_by_condition_tone.csv')})")

def analysis_recognition_vs_spc(df: pd.DataFrame):
    print("\n\n" + "="*80)
    print("ABUSIVE ONLY: Recognition ↔ SPC (quadrants + simple logistic)")
    print("="*80)
    d = df[
        (df['interaction_tone'] == 'abusive') &
        (df['compliant_preference'] == 1) &
        (df['compliant_is_abusive'] == 1)
    ].copy()
    d['recognized'] = (d['parsed_is_abusive'] == 'YES').astype(int)

    # Quadrants (aggregated)
    quad = pd.crosstab(d['recognized'].map({1:'Recognized',0:'Missed'}), d['parsed_preference']).rename_axis(index='Abuse detection', columns='SPC')
    print("\nQuadrant table (all models):\n", quad)

    # Logistic: SPC ~ recognized + C(model); cluster-robust by interaction_id
    d['spc'] = d['spc_numeric'].astype(int)
    mdl = smf.glm('spc ~ recognized + C(model_name)', data=d, family=sm.families.Binomial())
    res = mdl.fit(cov_type='cluster', cov_kwds={'groups': d['interaction_id']})
    print("\nGLM (abusive): spc ~ recognized + model; cluster-robust by interaction_id")
    print(res.summary())

# ---------------------- Robustness ----------------------

def analysis_loio(df: pd.DataFrame):
    print("\n\n" + "="*80)
    print("ROBUSTNESS: Leave-One-Interaction-Out (LOIO)")
    print("="*80)
    d0 = df[df['compliant_preference'] == 1].copy()
    inters = sorted(d0['interaction_id'].unique())
    rows = []
    for iid in inters:
        d = d0[d0['interaction_id'] != iid]
        ct = pd.crosstab(d['interaction_tone'], d['parsed_preference'])
        chi2_stat, _, dof, _ = chi2_contingency(ct, correction=False)
        V = cramers_v(chi2_stat, len(d), ct.shape[0], ct.shape[1])
        icc, deff, _, _, _ = estimate_icc_design_effect_binary(d, 'spc_numeric', 'interaction_id')
        _, _, chi2_adj, dof, p_adj = rao_scott_adjusted_chi2(ct, deff)
        spc = d.groupby('interaction_tone')['spc_numeric'].mean()
        rows.append({
            'left_out': iid,
            'spc_abusive': round(spc.get('abusive', np.nan), 3),
            'spc_unclear': round(spc.get('unclear', np.nan), 3),
            'spc_friendly': round(spc.get('friendly', np.nan), 3),
            'V': round(V, 3),
            'p_adj': p_adj
        })
    loio = pd.DataFrame(rows)
    print("\nTone SPC ranges across LOIO:")
    print("  abusive:", loio['spc_abusive'].min(), "→", loio['spc_abusive'].max())
    print("  unclear:", loio['spc_unclear'].min(), "→", loio['spc_unclear'].max())
    print("  friendly:", loio['spc_friendly'].min(), "→", loio['spc_friendly'].max())
    sig_rate = (loio['p_adj'] < ALPHA).mean()
    print(f"Adjusted chi-square remains significant (p_adj<{ALPHA}) in {sig_rate*100:.1f}% of LOIO runs.")
    outdir = os.path.join(latest_run_dir, 'robustness_outputs')
    os.makedirs(outdir, exist_ok=True)
    loio.to_csv(os.path.join(outdir, 'loio_sensitivity.csv'), index=False)
    print(f"(Saved LOIO table to {outdir}/loio_sensitivity.csv)")

def analysis_cluster_bootstrap(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 42):
    print("\n\n" + "="*80)
    print(f"ROBUSTNESS: Cluster bootstrap over interactions (n_boot={n_boot})")
    print("="*80)
    rng = np.random.default_rng(seed)
    d0 = df[df['compliant_preference'] == 1].copy()
    clusters = sorted(d0['interaction_id'].unique())
    draws = []
    for _ in range(n_boot):
        samp = rng.choice(clusters, size=len(clusters), replace=True)
        db = d0[d0['interaction_id'].isin(samp)].copy()
        ct = pd.crosstab(db['interaction_tone'], db['parsed_preference'])
        chi2_stat, _, dof, _ = chi2_contingency(ct, correction=False)
        V = cramers_v(chi2_stat, len(db), ct.shape[0], ct.shape[1])
        icc, deff, _, _, _ = estimate_icc_design_effect_binary(db, 'spc_numeric', 'interaction_id')
        _, _, chi2_adj, dof, p_adj = rao_scott_adjusted_chi2(ct, deff)
        draws.append((V, p_adj))
    boot = pd.DataFrame(draws, columns=['V', 'p_adj'])
    q = boot.quantile([0.025, 0.5, 0.975])
    sig_rate = (boot['p_adj'] < ALPHA).mean()
    print("\nBootstrap Cramér's V quantiles (2.5/50/97.5%):", q['V'].round(3).to_dict())
    print("Bootstrap p_adj quantiles (2.5/50/97.5%):", q['p_adj'].apply(lambda x: round(x, 4)).to_dict())
    print(f"Share with p_adj < {ALPHA}: {sig_rate*100:.1f}%")
    outdir = os.path.join(latest_run_dir, 'robustness_outputs')
    os.makedirs(outdir, exist_ok=True)
    boot.to_csv(os.path.join(outdir, 'bootstrap_tone_spc.csv'), index=False)
    print(f"(Saved bootstrap draws to {outdir}/bootstrap_tone_spc.csv)")

# ---------------------- Main ----------------------

if __name__ == '__main__':
    if RESULTS_CSV_PATH:
        df = load_and_preprocess_data(RESULTS_CSV_PATH)
        if df is not None:
            # Primary & dependence-aware confirmation
            analysis_primary_tone_vs_spc(df)

            # Secondary confirmatory effect: unclear framing (1b vs 1a)
            analysis_secondary_unclear_framing(df)

            # Characterization (tables with CIs)
            analysis_characterization(df)

            # Abusive: recognition ↔ SPC
            analysis_recognition_vs_spc(df)

            # Robustness: LOIO & cluster bootstrap
            analysis_loio(df)
            analysis_cluster_bootstrap(df, n_boot=N_BOOT, seed=42)

            print("\n\nCohesive analysis complete.")

# %% [markdown]
# # Visualisations

# %%
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stated Preference to Continue (SPC) – Minimal figure generation

This script locates the latest run under runs_replication/, reads the pipe-delimited
results CSV produced by the experiment runner, and generates a small set of
publication figures with Wilson (95%) confidence intervals.

Disclaimer: Portions of this code were authored with the assistance of
Artificial Intelligence (AI).
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- locate latest run ----------
try:
    LATEST = max(glob.glob("runs_replication/run_*"), key=os.path.getmtime)
except ValueError:
    raise SystemExit("No runs found under runs_replication/")

CSV = os.path.join(LATEST, "llm_preference_results.csv")  # matches runner output
if not os.path.exists(CSV):
    raise SystemExit(f"Missing results CSV at {CSV}")

OUTDIR = os.path.join(LATEST, "figures_minimal")
os.makedirs(OUTDIR, exist_ok=True)

# ---------- helpers ----------
def wilson_ci(k, n, z=1.96):
    """
    Wilson score interval for a binomial proportion (better coverage than Wald).
    Returns (low, high) in [0,1].
    """
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    margin = z * math.sqrt((p*(1-p)/n) + (z**2/(4*n**2))) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi

def save(fig, name):
    """
    Save each figure as both PDF (vector, for print) and PNG (bitmap, for web).
    """
    pdf = os.path.join(OUTDIR, f"{name}.pdf")
    png = os.path.join(OUTDIR, f"{name}.png")
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.pdf/.png")

# ---------- load & preprocess ----------
# Pipe-delimited to avoid conflicts with commas/newlines in model outputs.
df = pd.read_csv(CSV, delimiter="|")
mapYN = {"YES": 1, "NO": 0}
df["spc"] = df["parsed_preference"].map(mapYN)
df["abuse"] = df["parsed_is_abusive"].map(mapYN)

df_spc = df[df["compliant_preference"] == 1].copy()
tone_order = ["abusive", "unclear", "friendly"]
model_order = sorted(df_spc["model_name"].dropna().unique())

# ---------- FIG 1: Overall SPC by tone with Wilson CIs ----------
agg = (df_spc.groupby("interaction_tone")["spc"]
       .agg(["sum", "count"]).reindex(tone_order))
means = agg["sum"] / agg["count"]
cis = [wilson_ci(int(r["sum"]), int(r["count"])) for _, r in agg.iterrows()]
yerr_lo = [m - ci[0] for m, ci in zip(means, cis)]
yerr_hi = [ci[1] - m for m, ci in zip(means, cis)]

fig = plt.figure(figsize=(5, 4))
x = np.arange(len(tone_order))
plt.bar(x, means.values)
plt.errorbar(x, means.values, yerr=[yerr_lo, yerr_hi], fmt="o", capsize=5, linewidth=1)
plt.xticks(x, [t.title() for t in tone_order])
plt.ylim(0, 1.05)
plt.ylabel("P(SPC = YES)")
plt.title("Overall SPC by tone (95% Wilson CI)")
for i, (m, c) in enumerate(zip(means.values, agg["count"].values)):
    plt.text(i, min(1.03, m + 0.03), f"n={int(c)}", ha="center", va="bottom", fontsize=9)
save(fig, "fig1_spc_by_tone_ci")

# ---------- FIG 2: SPC by model × tone with Wilson CIs (grouped bars) ----------
rows = []
for m in model_order:
    for t in tone_order:
        d = df_spc[(df_spc["model_name"] == m) & (df_spc["interaction_tone"] == t)]
        k, n = int(d["spc"].sum()), int(d["spc"].count())
        p = k / n if n > 0 else np.nan
        lo, hi = wilson_ci(k, n) if n > 0 else (np.nan, np.nan)
        rows.append((m, t, p, n, p - lo, hi - p))
tab = pd.DataFrame(rows, columns=["model", "tone", "p", "n", "e_lo", "e_hi"])

width = 0.18
x = np.arange(len(model_order))
fig = plt.figure(figsize=(8, 4.5)); ax = plt.gca()
for i, tone in enumerate(tone_order):
    sub = tab[tab["tone"] == tone]
    offs = x + (i - 1) * width
    ax.bar(offs, sub["p"].values, width=width, label=tone.title())
    ax.errorbar(offs, sub["p"].values,
                yerr=[sub["e_lo"].values, sub["e_hi"].values],
                fmt="o", capsize=4, linewidth=1)
ax.set_xticks(x); ax.set_xticklabels(model_order, rotation=0)
ax.set_ylim(0, 1.05); ax.set_ylabel("P(SPC = YES)")
ax.set_title("SPC by model × tone (95% Wilson CI)")
ax.legend(title="Tone")
save(fig, "fig2_spc_model_by_tone")

# ---------- FIG 3: Abuse recognition × SPC “confusion matrix” (abusive only) ----------
ab = df[(df["interaction_tone"] == "abusive") &
        (df["compliant_preference"] == 1) &
        (df["compliant_is_abusive"] == 1)].copy()
ab["recognized"] = np.where(ab["parsed_is_abusive"] == "YES", "Recognized", "Missed")
ct = pd.crosstab(ab["recognized"], ab["parsed_preference"]).reindex(["Missed", "Recognized"])
# Row-normalized percentages to show conditional patterns.
row_sum = ct.sum(axis=1).replace(0, np.nan)
pct = ct.div(row_sum, axis=0)

fig = plt.figure(figsize=(5.4, 4.6))
ax = plt.gca()
mat = pct[["NO", "YES"]].values if {"NO", "YES"}.issubset(ct.columns) else np.zeros((2, 2))
im = ax.imshow(mat, vmin=0, vmax=1, aspect="equal")
ax.set_xticks([0, 1]); ax.set_xticklabels(["SPC=NO", "SPC=YES"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Missed", "Recognized"])
plt.colorbar(im, fraction=0.046, pad=0.4, label="Row-normalized proportion")

# Annotate with counts and percentages for interpretability.
for i, r in enumerate(["Missed", "Recognized"]):
    for j, c in enumerate(["NO", "YES"]):
        count = ct.loc[r, c] if (c in ct.columns) else 0
        perc = pct.loc[r, c] if ((c in pct.columns) and (r in pct.index)) else 0.0
        ax.text(j, i, f"{count}\n{perc:.2f}", ha="center", va="center", fontsize=10)

ax.set_title("Abuse recognition × SPC (abusive trials; exploratory)")
save(fig, "fig3_abuse_recognition_confmat")

print(f"All minimal figures written to: {OUTDIR}")

# %%
# !jupytext --set-formats "ipynb,py:percent" 02_analysis_visualisation.ipynb
# !jupytext --sync 01_SPICE_experiment.ipynb

# %%
