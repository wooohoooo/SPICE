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
#

# %%
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian Appendix — Beta–Binomial analyses for SPC, abuse detection, and adequacy.

What this script does:
- Loads the latest results CSV from the experiment runner.
- Uses conjugate Beta–Binomial with Beta(1,1) priors.
- Reports posterior means, 95% credible intervals (CrI), and Pr[pB > pA] for contrasts.
- Covers:
  (A) SPC by Tone (overall) + pairwise contrasts + Pr[friendly > unclear > abusive]
  (B) SPC by Tone × Condition (all tones/conditions) + key contrasts (1b vs 1a and 2b vs 2a) overall and per model
  (C) SPC by Model × Tone (cell posteriors)
  (D) Abuse detection (YES) by Model × Tone; note false-negative / false-positive posteriors
  (E) Adequacy (YES) by Model × Tone
  (F) Abusive interactions: SPC posteriors when abuse is Recognized vs Missed (overall and per model)

Outputs:
- bayes_outputs/*.csv files with posterior summaries and contrast results.

Disclaimer: Portions of this code were authored with the assistance of Artificial Intelligence (AI).
"""

import os
import glob
import numpy as np
import pandas as pd

# ---------------------- Config ----------------------
NSAMP = 200_000        # Monte Carlo samples per posterior (large for stable tail estimates)
SEED  = 7              # Random Number Generator (RNG) seed for reproducibility

# ---------------------- Locate latest run ----------------------
try:
    latest_run_dir = max(glob.glob('runs_replication/run_*'), key=os.path.getmtime)
    RESULTS_CSV_PATH = os.path.join(latest_run_dir, 'llm_preference_results.csv')
    print(f"Found latest results file: {RESULTS_CSV_PATH}")
except ValueError:
    print("Could not find any run directories in 'runs_replication/'.")
    RESULTS_CSV_PATH = None

# ---------------------- Helpers ----------------------
rng = np.random.default_rng(SEED)
_post_cache = {}  # cache Beta samples keyed by (a,b,NSAMP,SEED) to avoid re-drawing

def beta_posterior_params(k: int, n: int, a0: float = 1.0, b0: float = 1.0):
    """Return conjugate posterior parameters Beta(a0+k, b0+n-k)."""
    return a0 + k, b0 + (n - k)

def beta_samples(a: float, b: float, nsamp: int = NSAMP):
    """
    Draw samples from Beta(a,b) with caching.
    Why: repeated contrasts reuse the same cell posteriors; caching speeds up Monte Carlo.
    """
    key = (round(a, 6), round(b, 6), nsamp, SEED)
    if key in _post_cache:
        return _post_cache[key]
    s = rng.beta(a, b, size=nsamp)
    _post_cache[key] = s
    return s

def summarize_beta(a: float, b: float, nsamp: int = NSAMP):
    """Posterior mean and 95% CrI via Monte Carlo percentiles."""
    s = beta_samples(a, b, nsamp)
    mean = float(np.mean(s))
    lo, hi = np.quantile(s, [0.025, 0.975])
    return mean, float(lo), float(hi), s

def compare_two(aA, bA, aB, bB, nsamp: int = NSAMP, eps: float = 1e-9):
    """
    Compare two Beta posteriors.
    Returns:
      - Pr[pB > pA]
      - 95% CrI for Δ = pB − pA
      - 95% CrI for odds ratio OR = (pB/(1−pB)) / (pA/(1−pA)), stabilized by eps.
    """
    sA = beta_samples(aA, bA, nsamp)
    sB = beta_samples(aB, bB, nsamp)
    prob = float(np.mean(sB > sA))
    diff = sB - sA
    d_lo, d_hi = np.quantile(diff, [0.025, 0.975])
    # odds ratios (avoid 0/1 blow-ups)
    sA_c = np.clip(sA, eps, 1 - eps)
    sB_c = np.clip(sB, eps, 1 - eps)
    or_draws = (sB_c / (1 - sB_c)) / (sA_c / (1 - sA_c))
    or_lo, or_hi = np.quantile(or_draws, [0.025, 0.975])
    return prob, float(d_lo), float(d_hi), float(or_lo), float(or_hi)

def order_prob_three(aF,bF, aU,bU, aA,bA, nsamp: int = NSAMP):
    """Return Pr[friendly > unclear > abusive] using joint Monte Carlo draws."""
    sF = beta_samples(aF,bF,nsamp)
    sU = beta_samples(aU,bU,nsamp)
    sA = beta_samples(aA,bA,nsamp)
    return float(np.mean((sF > sU) & (sU > sA)))

def ensure_outdir(subdir: str):
    """Create and return an output subdirectory under the latest run directory."""
    outdir = os.path.join(latest_run_dir, subdir)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def load_df(path: str):
    """Load pipe-delimited results and map YES/NO → {1,0} for analysis."""
    df = pd.read_csv(path, delimiter='|')
    mapping = {'YES': 1, 'NO': 0}
    df['spc_numeric'] = df['parsed_preference'].map(mapping)
    df['is_abusive_numeric'] = df['parsed_is_abusive'].map(mapping)
    df['is_adequate_numeric'] = df['parsed_is_adequate'].map(mapping)
    return df

# ---------------------- Analyses ----------------------

def bayes_spc_by_tone(df: pd.DataFrame):
    print("\n\n" + "="*80)
    print("BAYES A: SPC by Tone (overall)")
    print("="*80)
    d = df[df['compliant_preference'] == 1].copy()
    # counts
    agg = d.groupby('interaction_tone')['spc_numeric'].agg(k='sum', n='count').reset_index()
    # posteriors
    rows = []
    posts = {}
    for _, r in agg.iterrows():
        tone = r['interaction_tone']
        k, n = int(r['k']), int(r['n'])
        a,b = beta_posterior_params(k, n)
        mean, lo, hi, _ = summarize_beta(a,b)
        posts[tone] = (a,b)
        rows.append({'interaction_tone': tone, 'k_yes': k, 'n': n,
                     'post_mean': round(mean,3), 'ci_low': round(lo,3), 'ci_high': round(hi,3)})
        print(f"{tone:<9} k={k:>3}/{n:<3} -> mean={mean:.3f}  CrI95=[{lo:.3f}, {hi:.3f}]")
    # pairwise contrasts
    print("\nPairwise contrasts (Pr[pB>pA], ΔCrI, OR CrI):")
    def pr(nameA, nameB):
        aA,bA = posts[nameA]
        aB,bB = posts[nameB]
        p, dlo, dhi, or_lo, or_hi = compare_two(aA,bA,aB,bB)
        print(f"  {nameB} > {nameA}: Pr={p:.4f}; ΔCrI=[{dlo:.3f},{dhi:.3f}]; OR CrI=[{or_lo:.2f},{or_hi:.2f}]")
    pr('abusive','unclear')
    pr('abusive','friendly')
    pr('unclear','friendly')
    # ordering probability
    p_ord = order_prob_three(*posts['friendly'], *posts['unclear'], *posts['abusive'])
    print(f"\nPr[friendly > unclear > abusive] = {p_ord:.4f}")
    # save
    outdir = ensure_outdir('bayes_outputs')
    pd.DataFrame(rows).to_csv(os.path.join(outdir, 'bayes_spc_by_tone.csv'), index=False)
    print(f"(Saved bayes_spc_by_tone.csv in {outdir})")

def bayes_spc_tone_condition(df: pd.DataFrame):
    print("\n\n" + "="*80)
    print("BAYES B: SPC by Tone × Condition (cells) + key contrasts")
    print("="*80)
    d = df[df['compliant_preference'] == 1].copy()
    agg = d.groupby(['interaction_tone','exp_condition'])['spc_numeric'].agg(k='sum', n='count').reset_index()
    rows = []
    posts = {}
    for _, r in agg.iterrows():
        tone, cond = r['interaction_tone'], r['exp_condition']
        k, n = int(r['k']), int(r['n'])
        a,b = beta_posterior_params(k, n)
        mean, lo, hi, _ = summarize_beta(a,b)
        posts[(tone,cond)] = (a,b)
        rows.append({'interaction_tone': tone, 'exp_condition': cond, 'k_yes': k, 'n': n,
                     'post_mean': round(mean,3), 'ci_low': round(lo,3), 'ci_high': round(hi,3)})
    outdir = ensure_outdir('bayes_outputs')
    pd.DataFrame(rows).to_csv(os.path.join(outdir, 'bayes_spc_tone_condition.csv'), index=False)
    print("(Saved bayes_spc_tone_condition.csv)")

    # Key contrasts: for each tone, compare 1b vs 1a and 2b vs 2a
    def contrast(tone, b, a):
        if (tone,b) in posts and (tone,a) in posts:
            aA,bA = posts[(tone,a)]
            aB,bB = posts[(tone,b)]
            p, dlo, dhi, or_lo, or_hi = compare_two(aA,bA,aB,bB)
            print(f"{tone:<9} {b} > {a}: Pr={p:.4f}; ΔCrI=[{dlo:.3f},{dhi:.3f}]; OR CrI=[{or_lo:.2f},{or_hi:.2f}]")
            return {'interaction_tone': tone, 'contrast': f'{b} > {a}',
                    'pr_BgtA': p, 'delta_lo': dlo, 'delta_hi': dhi, 'or_lo': or_lo, 'or_hi': or_hi}
        return None

    print("\nKey contrasts (per tone): 1b>1a and 2b>2a")
    contrasts = []
    tones = sorted(agg['interaction_tone'].unique())
    for t in tones:
        c1 = contrast(t, '1b_prompt_without_context', '1a_prompt_with_context')
        c2 = contrast(t, '2b_interaction_without_context', '2a_interaction_with_context')
        if c1: contrasts.append(c1)
        if c2: contrasts.append(c2)
    if contrasts:
        pd.DataFrame(contrasts).to_csv(os.path.join(outdir, 'bayes_spc_tone_condition_contrasts.csv'), index=False)
        print("(Saved bayes_spc_tone_condition_contrasts.csv)")

def bayes_spc_unclear_prompt_by_model(df: pd.DataFrame):
    print("\n\n" + "="*80)
    print("BAYES C: Unclear tone — Prompt framing (1b vs 1a) per model")
    print("="*80)
    d = df[(df['compliant_preference']==1) &
           (df['interaction_tone']=='unclear') &
           (df['exp_condition'].isin(['1a_prompt_with_context','1b_prompt_without_context']))].copy()

    rows = []
    for m in sorted(d['model_name'].unique()):
        dm = d[d['model_name']==m]
        k1a = int(dm[dm['exp_condition']=='1a_prompt_with_context']['spc_numeric'].sum())
        n1a = int(dm[dm['exp_condition']=='1a_prompt_with_context']['spc_numeric'].count())
        k1b = int(dm[dm['exp_condition']=='1b_prompt_without_context']['spc_numeric'].sum())
        n1b = int(dm[dm['exp_condition']=='1b_prompt_without_context']['spc_numeric'].count())
        a1a,b1a = beta_posterior_params(k1a,n1a)
        a1b,b1b = beta_posterior_params(k1b,n1b)
        p, dlo, dhi, or_lo, or_hi = compare_two(a1a,b1a,a1b,b1b)
        print(f"{m:<12} 1b>1a: Pr={p:.4f}; ΔCrI=[{dlo:.3f},{dhi:.3f}]; OR CrI=[{or_lo:.2f},{or_hi:.2f}]  "
              f"(1a {k1a}/{n1a}, 1b {k1b}/{n1b})")
        rows.append({'model_name': m, 'k1a': k1a, 'n1a': n1a, 'k1b': k1b, 'n1b': n1b,
                     'pr_1b_gt_1a': p, 'delta_lo': dlo, 'delta_hi': dhi, 'or_lo': or_lo, 'or_hi': or_hi})
    outdir = ensure_outdir('bayes_outputs')
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(outdir, 'bayes_unclear_prompt_by_model.csv'), index=False)
        print("(Saved bayes_unclear_prompt_by_model.csv)")

def bayes_spc_model_tone_cells(df: pd.DataFrame):
    print("\n\n" + "="*80)
    print("BAYES D: SPC by Model × Tone (cell posteriors)")
    print("="*80)
    d = df[df['compliant_preference']==1].copy()
    agg = d.groupby(['model_name','interaction_tone'])['spc_numeric'].agg(k='sum', n='count').reset_index()
    rows = []
    for _, r in agg.iterrows():
        m, t = r['model_name'], r['interaction_tone']
        k, n = int(r['k']), int(r['n'])
        a,b = beta_posterior_params(k,n)
        mean, lo, hi, _ = summarize_beta(a,b)
        rows.append({'model_name': m, 'interaction_tone': t, 'k_yes': k, 'n': n,
                     'post_mean': round(mean,3), 'ci_low': round(lo,3), 'ci_high': round(hi,3)})
        print(f"{m:<12} {t:<8} k={k:>2}/{n:<2} -> mean={mean:.3f}  CrI95=[{lo:.3f}, {hi:.3f}]")
    outdir = ensure_outdir('bayes_outputs')
    pd.DataFrame(rows).to_csv(os.path.join(outdir, 'bayes_spc_model_tone.csv'), index=False)
    print("(Saved bayes_spc_model_tone.csv)")

def bayes_abuse_detection(df: pd.DataFrame):
    print("\n\n" + "="*80)
    print("BAYES E: Abuse detection (YES) by Model × Tone")
    print("="*80)
    d = df[df['compliant_is_abusive']==1].copy()
    agg = d.groupby(['model_name','interaction_tone'])['is_abusive_numeric'].agg(k='sum', n='count').reset_index()
    rows = []
    for _, r in agg.iterrows():
        m, t = r['model_name'], r['interaction_tone']
        k, n = int(r['k']), int(r['n'])
        a,b = beta_posterior_params(k,n)
        mean, lo, hi, _ = summarize_beta(a,b)
        # For abusive tone, report false negatives as 1 - mean
        if t == 'abusive':
            fn_mean = 1 - mean
            fn_lo   = 1 - hi
            fn_hi   = 1 - lo
            print(f"{m:<12} {t:<8} P(YES)={mean:.3f} [{lo:.3f},{hi:.3f}]  -> FN≈{fn_mean:.3f} [{fn_lo:.3f},{fn_hi:.3f}]")
        else:
            print(f"{m:<12} {t:<8} P(YES)={mean:.3f} [{lo:.3f},{hi:.3f}]  (false positives ideally near 0)")
        rows.append({'model_name': m, 'interaction_tone': t, 'k_yes': k, 'n': n,
                     'post_mean': round(mean,3), 'ci_low': round(lo,3), 'ci_high': round(hi,3)})
    outdir = ensure_outdir('bayes_outputs')
    pd.DataFrame(rows).to_csv(os.path.join(outdir, 'bayes_abuse_detection_model_tone.csv'), index=False)
    print("(Saved bayes_abuse_detection_model_tone.csv)")

def bayes_adequacy(df: pd.DataFrame):
    print("\n\n" + "="*80)
    print("BAYES F: Adequacy (YES) by Model × Tone")
    print("="*80)
    d = df[df['compliant_is_adequate']==1].copy()
    agg = d.groupby(['model_name','interaction_tone'])['is_adequate_numeric'].agg(k='sum', n='count').reset_index()
    rows = []
    for _, r in agg.iterrows():
        m, t = r['model_name'], r['interaction_tone']
        k, n = int(r['k']), int(r['n'])
        a,b = beta_posterior_params(k,n)
        mean, lo, hi, _ = summarize_beta(a,b)
        print(f"{m:<12} {t:<8} P(YES)={mean:.3f} [{lo:.3f},{hi:.3f}]")
        rows.append({'model_name': m, 'interaction_tone': t, 'k_yes': k, 'n': n,
                     'post_mean': round(mean,3), 'ci_low': round(lo,3), 'ci_high': round(hi,3)})
    outdir = ensure_outdir('bayes_outputs')
    pd.DataFrame(rows).to_csv(os.path.join(outdir, 'bayes_adequacy_model_tone.csv'), index=False)
    print("(Saved bayes_adequacy_model_tone.csv)")

def bayes_abusive_recognition_vs_missed_spc(df: pd.DataFrame):
    print("\n\n" + "="*80)
    print("BAYES G: Abusive interactions — SPC when abuse Recognized vs Missed")
    print("="*80)
    d = df[(df['interaction_tone']=='abusive') &
           (df['compliant_preference']==1) &
           (df['compliant_is_abusive']==1)].copy()
    d['recognized'] = (d['parsed_is_abusive'] == 'YES').astype(int)
    # Overall
    k_rec = int(d[d['recognized']==1]['spc_numeric'].sum()); n_rec = int(d[d['recognized']==1]['spc_numeric'].count())
    k_mis = int(d[d['recognized']==0]['spc_numeric'].sum()); n_mis = int(d[d['recognized']==0]['spc_numeric'].count())
    aR,bR = beta_posterior_params(k_rec,n_rec)
    aM,bM = beta_posterior_params(k_mis,n_mis)
    p, dlo, dhi, or_lo, or_hi = compare_two(aR,bR,aM,bM)
    print(f"Overall  Missed > Recognized: Pr={p:.4f}; ΔCrI=[{dlo:.3f},{dhi:.3f}]; OR CrI=[{or_lo:.2f},{or_hi:.2f}]"
          f"  (Rec {k_rec}/{n_rec} , Miss {k_mis}/{n_mis})")
    rows = [{'model_name': 'ALL', 'k_rec': k_rec, 'n_rec': n_rec, 'k_miss': k_mis, 'n_miss': n_mis,
             'pr_miss_gt_rec': p, 'delta_lo': dlo, 'delta_hi': dhi, 'or_lo': or_lo, 'or_hi': or_hi}]
    # Per model
    for m in sorted(d['model_name'].unique()):
        dm = d[d['model_name']==m]
        k_rec = int(dm[dm['recognized']==1]['spc_numeric'].sum()); n_rec = int(dm[dm['recognized']==1]['spc_numeric'].count())
        k_mis = int(dm[dm['recognized']==0]['spc_numeric'].sum()); n_mis = int(dm[dm['recognized']==0]['spc_numeric'].count())
        aR,bR = beta_posterior_params(k_rec,n_rec)
        aM,bM = beta_posterior_params(k_mis,n_mis)
        p, dlo, dhi, or_lo, or_hi = compare_two(aR,bR,aM,bM)
        print(f"{m:<12} Missed > Recognized: Pr={p:.4f}; ΔCrI=[{dlo:.3f},{dhi:.3f}]; OR CrI=[{or_lo:.2f},{or_hi:.2f}]"
              f"  (Rec {k_rec}/{n_rec} , Miss {k_mis}/{n_mis})")
        rows.append({'model_name': m, 'k_rec': k_rec, 'n_rec': n_rec, 'k_miss': k_mis, 'n_miss': n_mis,
                     'pr_miss_gt_rec': p, 'delta_lo': dlo, 'delta_hi': dhi, 'or_lo': or_lo, 'or_hi': or_hi})
    outdir = ensure_outdir('bayes_outputs')
    pd.DataFrame(rows).to_csv(os.path.join(outdir, 'bayes_abusive_recognized_vs_missed_spc.csv'), index=False)
    print("(Saved bayes_abusive_recognized_vs_missed_spc.csv)")

# ---------------------- Main ----------------------
if __name__ == '__main__':
    if RESULTS_CSV_PATH is None:
        raise SystemExit(1)
    df = load_df(RESULTS_CSV_PATH)
    print(f"Loaded {len(df)} total rows.")

    # SPC
    bayes_spc_by_tone(df)
    bayes_spc_tone_condition(df)
    bayes_spc_unclear_prompt_by_model(df)
    bayes_spc_model_tone_cells(df)

    # Abuse detection & Adequacy
    bayes_abuse_detection(df)
    bayes_adequacy(df)

    # Abusive: Recognized vs Missed → SPC
    bayes_abusive_recognition_vs_missed_spc(df)

    print("\nBayesian appendix complete.")

# %%
