"""
exponents_analysis.py
---------------------
Dedicated script for accurate extraction of beta, gamma and tau
using three independent methods:

  1. Log-log fit       — standard power law fit with optimised window
  2. Quotient method   — ratios of observables at paired L values,
                         no fitting window needed, finite-size corrections
                         partially cancel
  3. MLE              — maximum likelihood estimator for tau (Clauset
                         et al. 2009), uses raw cluster sizes directly,
                         no binning or window choice needed

Runs its own FSS loop (needed for quotient method) and fine sweep,
then applies all three methods and produces a comparison figure.

Configuration
-------------
Edit the block below. PC_OVERRIDE and NU_OVERRIDE should be your
best values from the previous overnight run.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from joblib import Parallel, delayed

from observables import generate_lattice, spanning_info, percolation_strength, \
                        mean_finite_cluster_size, cluster_size_distribution
from algorithms  import hoshen_kopelman
from analysis    import (
    estimate_pc_single_L, fit_loglog_slope,
    bootstrap_uncertainties, THEORY,
)
from simulation  import estimate_observables_over_p

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

# =============================================================================
# Configuration
# =============================================================================

PC_OVERRIDE = 0.59296       # your FSS pc from the overnight run
NU_OVERRIDE = 1.378         # your FSS nu from the overnight run

L_MAIN  = 256               # primary lattice for log-log and MLE
L_FSS   = [64, 128, 256, 512]  # needed for quotient method

RUNS    = 500               # runs per p-point (balances time vs accuracy)
P_FINE  = np.linspace(0.575, 0.610, 40)

SPAN_MODE  = "LR"
BASE_SEED  = 42
N_BOOT     = 500
FIG_DIR    = "figures"

# Log-log fitting windows (used for method 1 only)
BETA_XMIN,  BETA_XMAX  = 0.01,  0.04
GAMMA_XMIN, GAMMA_XMAX = 0.003, 0.03

# =============================================================================
# Helpers
# =============================================================================

def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


def _run(L, p_values, runs, seed_offset=0, return_runs=False, desc=None):
    return estimate_observables_over_p(
        L=L, p_values=p_values, runs=runs,
        span_mode=SPAN_MODE, n_jobs=-1,
        base_seed=BASE_SEED + seed_offset,
        return_runs=return_runs,
        desc=desc,
    )


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# =============================================================================
# Method 1: Log-log fit
# =============================================================================

def method_loglog(fine, pc):
    """Standard log-log power law fit with optimised window."""
    p_arr = fine["p"]
    Pinf  = fine["Pinf"]
    chi   = fine["chi"]

    # Beta
    mask_b = (p_arr > pc) & (p_arr - pc >= BETA_XMIN) & (p_arr - pc <= BETA_XMAX)
    if mask_b.sum() >= 3:
        m, _, _ = fit_loglog_slope(p_arr[mask_b] - pc, Pinf[mask_b])
        beta_ll = float(m)
    else:
        beta_ll = np.nan

    # Gamma — both sides of pc
    x_g = np.abs(p_arr - pc)
    mask_g = (x_g >= GAMMA_XMIN) & (x_g <= GAMMA_XMAX) & (chi > 0)
    if mask_g.sum() >= 3:
        m, _, _ = fit_loglog_slope(x_g[mask_g], chi[mask_g])
        gamma_ll = float(-m)
    else:
        gamma_ll = np.nan

    # Tau — cluster size distribution at pc
    k_pc = int(np.argmin(np.abs(p_arr - pc)))
    ns   = fine["ns_density"][k_pc]
    if ns:
        s_arr  = np.array(sorted(ns.keys()), dtype=float)
        ns_arr = np.array([ns[int(s)] for s in s_arr], dtype=float)
        mask_t = (s_arr >= 2) & (s_arr <= L_MAIN**2 // 4) & (ns_arr > 0)
        if mask_t.sum() >= 3:
            m, _, _ = fit_loglog_slope(s_arr[mask_t], ns_arr[mask_t])
            tau_ll = float(-m)
        else:
            tau_ll = np.nan
    else:
        tau_ll = np.nan

    return {"beta": beta_ll, "gamma": gamma_ll, "tau": tau_ll}


# =============================================================================
# Method 2: Quotient method
# =============================================================================

def method_quotient(results_by_L, pc):
    """
    Extract exponents from ratios of observables at paired L values.

    At pc:
        P∞(2L) / P∞(L)  ~  2^(beta/nu)   =>  beta  = nu * log2(ratio)
        chi(2L) / chi(L) ~  2^(gamma/nu)  =>  gamma = nu * log2(ratio)
        R slope ratio    ~  L^(1/nu)      =>  nu    from slope

    Finite-size corrections partially cancel in the ratio, making this
    more robust than direct fitting.
    """
    nu = NU_OVERRIDE

    beta_estimates  = []
    gamma_estimates = []
    nu_estimates    = []

    L_vals = sorted(results_by_L.keys())
    pairs  = [(L_vals[i], L_vals[i+1]) for i in range(len(L_vals)-1)
              if L_vals[i+1] == 2 * L_vals[i]]   # only exact doubling pairs

    for L1, L2 in pairs:
        r1 = results_by_L[L1]
        r2 = results_by_L[L2]

        # Find index closest to pc in each
        k1 = int(np.argmin(np.abs(r1["p"] - pc)))
        k2 = int(np.argmin(np.abs(r2["p"] - pc)))

        # Beta from Pinf ratio
        if r1["Pinf"][k1] > 0 and r2["Pinf"][k2] > 0:
            ratio = r2["Pinf"][k2] / r1["Pinf"][k1]
            if ratio > 0:
                beta_nu = np.log(ratio) / np.log(L2 / L1)
                beta_estimates.append(beta_nu * nu)

        # Gamma from chi ratio
        if r1["chi"][k1] > 0 and r2["chi"][k2] > 0:
            ratio = r2["chi"][k2] / r1["chi"][k1]
            if ratio > 0:
                gamma_nu = np.log(ratio) / np.log(L2 / L1)
                gamma_estimates.append(gamma_nu * nu)

        # Nu from R slope (dR/dp scales as L^(1/nu))
        # Estimate slope of R(p) at pc via finite difference
        p1, R1 = r1["p"], r1["R"]
        p2, R2 = r2["p"], r2["R"]
        slope1 = np.gradient(R1, p1)[k1]
        slope2 = np.gradient(R2, p2)[k2]
        if slope1 > 0 and slope2 > 0:
            nu_est = np.log(L2/L1) / np.log(slope2/slope1)
            if 0.8 <= nu_est <= 2.0:
                nu_estimates.append(nu_est)

    results = {}
    results["beta"]  = float(np.mean(beta_estimates))  if beta_estimates  else np.nan
    results["gamma"] = float(np.mean(gamma_estimates)) if gamma_estimates else np.nan
    results["nu"]    = float(np.mean(nu_estimates))    if nu_estimates    else np.nan
    results["beta_pairs"]  = beta_estimates
    results["gamma_pairs"] = gamma_estimates
    results["nu_pairs"]    = nu_estimates
    results["pairs"]       = pairs

    return results


# =============================================================================
# Method 3: MLE for tau
# =============================================================================

def _collect_raw_cluster_sizes(L, p, runs, seed_offset=0):
    """
    Run HK realisations at a single (L, p) and return ALL cluster sizes
    as a flat array — not averaged, not binned.
    """
    def _one(seed):
        rng     = np.random.default_rng(seed)
        lattice = generate_lattice(L, p, rng=rng)
        labels, sizes = hoshen_kopelman(lattice)
        _, _, spanning = spanning_info(labels)
        # Exclude spanning cluster — same convention as chi
        return [s for r, s in sizes.items() if r not in spanning and s >= 2]

    seeds = BASE_SEED + seed_offset + np.arange(runs, dtype=int)
    if _TQDM:
        from tqdm import tqdm
        iter_seeds = tqdm(seeds, desc=f"  MLE raw sizes L={L} p={p:.4f}")
    else:
        iter_seeds = seeds

    all_sizes = []
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_one)(int(s)) for s in seeds
    )
    for r in results:
        all_sizes.extend(r)

    return np.array(all_sizes, dtype=float)


def mle_tau(sizes, smin=2):
    """
    Clauset-Shalizi-Newman MLE for power law exponent.

    tau_hat = 1 + n * [sum_i ln(s_i / smin - 0.5)]^{-1}

    This is the continuous approximation MLE which works well for
    integer-valued distributions when smin is not too small.

    Parameters
    ----------
    sizes : array of cluster sizes (raw, not binned)
    smin  : minimum cluster size to include in fit

    Returns
    -------
    tau  : MLE estimate
    sigma: standard error (from Fisher information)
    n    : number of clusters used
    """
    s = sizes[sizes >= smin]
    n = len(s)
    if n < 10:
        return np.nan, np.nan, 0

    # Continuous MLE (Clauset et al. eq 3.1, adapted for discrete)
    tau   = 1.0 + n / np.sum(np.log(s / (smin - 0.5)))
    sigma = (tau - 1.0) / np.sqrt(n)   # standard error from Fisher info

    return float(tau), float(sigma), int(n)


def method_mle(L, pc, runs, seed_offset=90000):
    """
    Collect raw cluster sizes at pc and apply MLE for tau.
    Also scans smin to find the stable plateau.
    """
    print(f"  Collecting raw cluster sizes at pc={pc:.5f}, L={L}, runs={runs} ...")
    sizes = _collect_raw_cluster_sizes(L, pc, runs, seed_offset=seed_offset)
    print(f"  Total clusters collected: {len(sizes):,}")

    # Scan smin from 2 to L^2/100
    smin_vals = np.unique(np.round(
        np.logspace(np.log10(2), np.log10(max(L**2//100, 10)), 30)
    ).astype(int))

    tau_vals   = []
    sigma_vals = []
    n_vals     = []

    for smin in smin_vals:
        tau, sigma, n = mle_tau(sizes, smin=smin)
        tau_vals.append(tau)
        sigma_vals.append(sigma)
        n_vals.append(n)

    tau_arr   = np.array(tau_vals)
    sigma_arr = np.array(sigma_vals)
    n_arr     = np.array(n_vals, dtype=float)

    # Find stable plateau: region where tau varies less than 2*sigma
    # Use the estimate from the region with most data that is still stable
    best_tau, best_sigma, best_smin = np.nan, np.nan, 2
    for i in range(len(smin_vals)-2):
        window = tau_arr[i:i+3]
        if np.isfinite(window).all():
            spread = window.max() - window.min()
            if spread < 0.1 and n_arr[i] > 50:
                best_tau   = float(tau_arr[i])
                best_sigma = float(sigma_arr[i])
                best_smin  = int(smin_vals[i])
                break

    if np.isnan(best_tau):
        # fallback: use smin=2
        best_tau, best_sigma, _ = mle_tau(sizes, smin=2)
        best_smin = 2

    return {
        "tau":        best_tau,
        "sigma":      best_sigma,
        "best_smin":  best_smin,
        "smin_vals":  smin_vals,
        "tau_vals":   tau_arr,
        "sigma_vals": sigma_arr,
        "n_vals":     n_arr,
        "sizes":      sizes,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_comparison(ll, quot, mle_res):
    """3-panel comparison figure for all three methods."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Exponent extraction — method comparison", fontsize=12)

    theory = {"beta": THEORY["beta"], "gamma": THEORY["gamma"], "tau": THEORY["tau"]}
    methods = ["Log-log fit", "Quotient", "MLE"]

    # ---- Beta ----
    ax = axes[0]
    vals   = [ll["beta"], quot["beta"], np.nan]
    colors = ["C0", "C1", "C2"]
    for i, (v, m, c) in enumerate(zip(vals, methods, colors)):
        if np.isfinite(v):
            ax.bar(i, v, color=c, alpha=0.7, label=m)
            ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=8)
    ax.axhline(theory["beta"], color="red", ls="--", lw=1.5,
               label=f"Theory = {theory['beta']:.4f}")
    ax.set_xticks(range(3)); ax.set_xticklabels(methods, fontsize=8)
    _style(ax, "", "β", "Beta comparison")

    # ---- Gamma ----
    ax = axes[1]
    vals = [ll["gamma"], quot["gamma"], np.nan]
    for i, (v, m, c) in enumerate(zip(vals, methods, colors)):
        if np.isfinite(v):
            ax.bar(i, v, color=c, alpha=0.7, label=m)
            ax.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=8)
    ax.axhline(theory["gamma"], color="red", ls="--", lw=1.5,
               label=f"Theory = {theory['gamma']:.4f}")
    ax.set_xticks(range(3)); ax.set_xticklabels(methods, fontsize=8)
    _style(ax, "", "γ", "Gamma comparison")

    # ---- Tau ----
    ax = axes[2]
    vals = [ll["tau"], np.nan, mle_res["tau"]]
    for i, (v, m, c) in enumerate(zip(vals, methods, colors)):
        if np.isfinite(v):
            ax.bar(i, v, color=c, alpha=0.7, label=m)
            ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=8)
    ax.axhline(theory["tau"], color="red", ls="--", lw=1.5,
               label=f"Theory = {theory['tau']:.4f}")
    ax.set_xticks(range(3)); ax.set_xticklabels(methods, fontsize=8)
    _style(ax, "", "τ", "Tau comparison")

    fig.tight_layout()
    return fig


def plot_mle_smin_scan(mle_res):
    """Show tau vs smin to demonstrate plateau stability."""
    fig, ax = plt.subplots(figsize=(8, 5))
    smin_vals  = mle_res["smin_vals"]
    tau_vals   = mle_res["tau_vals"]
    sigma_vals = mle_res["sigma_vals"]

    valid = np.isfinite(tau_vals)
    ax.errorbar(smin_vals[valid], tau_vals[valid],
                yerr=sigma_vals[valid], fmt="o-", ms=5,
                color="C0", label="MLE τ(s_min)")
    ax.axhline(THEORY["tau"], color="red", ls="--", lw=1.5,
               label=f"Theory τ = {THEORY['tau']:.4f}")
    ax.axvline(mle_res["best_smin"], color="green", ls=":", lw=1.2,
               label=f"Best s_min = {mle_res['best_smin']}")
    ax.axhspan(THEORY["tau"]*0.95, THEORY["tau"]*1.05,
               alpha=0.08, color="red", label="±5% band")
    ax.set_xscale("log")
    _style(ax, "s_min", "τ (MLE)",
           "MLE tau vs s_min cutoff — stable plateau = reliable estimate")
    fig.tight_layout()
    return fig


def plot_quotient_pairs(quot):
    """Show beta and gamma estimates from each L pair."""
    pairs = quot["pairs"]
    if not pairs:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [f"L={L1}→{L2}" for L1, L2 in pairs]

    for ax, key, tv, sym in [
        (axes[0], "beta_pairs",  THEORY["beta"],  "β"),
        (axes[1], "gamma_pairs", THEORY["gamma"], "γ"),
    ]:
        vals = quot[key]
        if vals:
            ax.plot(range(len(vals)), vals, "o-", ms=8, color="C0",
                    label="Quotient estimate")
            ax.axhline(tv, color="red", ls="--", lw=1.5,
                       label=f"Theory = {tv:.4f}")
            ax.axhline(np.mean(vals), color="C1", ls=":", lw=1.2,
                       label=f"Mean = {np.mean(vals):.4f}")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=8)
        _style(ax, "L pair", sym, f"Quotient method — {sym} per L pair")

    fig.tight_layout()
    return fig


# =============================================================================
# Summary table
# =============================================================================

def print_summary(ll, quot, mle_res):
    print("\n" + "="*70)
    print("  Exponent summary — all three methods")
    print("="*70)
    print(f"  {'Exponent':<10} {'Log-log':>12} {'Quotient':>12} {'MLE':>12} {'Theory':>12}")
    print("-"*70)

    for exp, tv, ll_v, q_v, mle_v in [
        ("beta",  THEORY["beta"],  ll["beta"],  quot["beta"],  np.nan),
        ("gamma", THEORY["gamma"], ll["gamma"], quot["gamma"], np.nan),
        ("tau",   THEORY["tau"],   ll["tau"],   np.nan,        mle_res["tau"]),
    ]:
        def fmt(v):
            return f"{v:.4f}" if np.isfinite(v) else "  —   "
        print(f"  {exp:<10} {fmt(ll_v):>12} {fmt(q_v):>12} {fmt(mle_v):>12} {tv:>12.4f}")

    print("="*70)
    print(f"\n  pc = {PC_OVERRIDE:.5f}  nu = {NU_OVERRIDE:.4f}  (from FSS run, unchanged)")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  Exponent analysis — three methods")
    print("=" * 60)
    print(f"  pc = {PC_OVERRIDE:.5f}   nu = {NU_OVERRIDE:.4f}")
    print(f"  L_MAIN = {L_MAIN}   L_FSS = {L_FSS}")
    print(f"  RUNS = {RUNS}   P_FINE points = {len(P_FINE)}")

    pc = PC_OVERRIDE

    # ------------------------------------------------------------------
    # Fine sweep at L_MAIN
    # ------------------------------------------------------------------
    print(f"\n[1/4] Fine sweep  L={L_MAIN}, runs={RUNS} ...")
    fine = _run(L_MAIN, P_FINE, RUNS, seed_offset=10000,
                return_runs=True,
                desc=f"Fine sweep L={L_MAIN}")
    pc_check = estimate_pc_single_L(fine["p"], fine["R"])
    print(f"  pc check = {pc_check:.5f}  (using override {pc:.5f})")

    # ------------------------------------------------------------------
    # FSS loop (needed for quotient method)
    # ------------------------------------------------------------------
    print(f"\n[2/4] FSS loop  L = {L_FSS} ...")
    results_by_L = {}
    for L_val in L_FSS:
        res = _run(L_val, P_FINE, RUNS,
                   seed_offset=20000 + L_val,
                   desc=f"FSS L={L_val:4d}")
        results_by_L[L_val] = res
        pc_L = estimate_pc_single_L(res["p"], res["R"])
        print(f"  L = {L_val:4d}  →  pc(L) = {pc_L:.5f}")

    # ------------------------------------------------------------------
    # Method 1: Log-log fit
    # ------------------------------------------------------------------
    print(f"\n[3/4] Applying all three methods ...")
    print(f"  Method 1: Log-log fit ...")
    ll = method_loglog(fine, pc)
    print(f"    beta  = {ll['beta']:.4f}  (theory {THEORY['beta']:.4f})")
    print(f"    gamma = {ll['gamma']:.4f}  (theory {THEORY['gamma']:.4f})")
    print(f"    tau   = {ll['tau']:.4f}  (theory {THEORY['tau']:.4f})")

    # Method 2: Quotient
    print(f"  Method 2: Quotient method ...")
    quot = method_quotient(results_by_L, pc)
    print(f"    beta  = {quot['beta']:.4f}  (theory {THEORY['beta']:.4f})"
          f"  from {len(quot['beta_pairs'])} pairs")
    print(f"    gamma = {quot['gamma']:.4f}  (theory {THEORY['gamma']:.4f})"
          f"  from {len(quot['gamma_pairs'])} pairs")
    print(f"    nu    = {quot['nu']:.4f}  (theory {THEORY['nu']:.4f})")

    # Method 3: MLE for tau
    print(f"  Method 3: MLE for tau ...")
    mle_res = method_mle(L_MAIN, pc, RUNS)
    print(f"    tau = {mle_res['tau']:.4f} ± {mle_res['sigma']:.4f}"
          f"  (theory {THEORY['tau']:.4f})"
          f"  smin={mle_res['best_smin']}"
          f"  n={mle_res['n_vals'][0]:,.0f} clusters")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_summary(ll, quot, mle_res)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print(f"\n[4/4] Saving figures ...")
    _save(plot_comparison(ll, quot, mle_res),  "exponent_comparison")
    _save(plot_mle_smin_scan(mle_res),         "mle_tau_scan")
    fig_q = plot_quotient_pairs(quot)
    if fig_q:
        _save(fig_q, "quotient_pairs")

    print(f"\nDone. Figures saved to: {FIG_DIR}")
    print(f"Keep all other figures from your overnight run unchanged.")


if __name__ == "__main__":
    main()
