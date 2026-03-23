"""
percolation_figures.py
----------------------
Standalone script producing publication-quality figures matching the
style of Christensen & Moloney "Complexity and Criticality" (2005).

Figures produced
----------------
1. pinf_vs_p.png        — P∞(p;L) vs p, multi-L with inset at pc
2. pinf_scaling.png     — P∞(pc;L) vs L log-log, slope = -β/ν
3. chi_vs_p.png         — χ(p;L) vs p log-linear, multi-L
4. chi_scaling.png      — χ(pc;L) vs L log-log, slope = γ/ν
5. ns_scaled.png        — s^τ n(s,pc;L) vs s, multi-L
6. ns_collapse.png      — s^τ n(s,pc;L) vs s/L^D, data collapse
7. beta_powerlaw.png    — P∞ vs (p-pc) log-log, direct β fit
8. gamma_powerlaw.png   — χ vs |p-pc| log-log, direct γ fit (both sides)
9. quotient_convergence.png — β/ν and γ/ν quotient estimates vs L pair

Checkpointing
-------------
Results are saved to CKPT_DIR after simulation. Delete the .npz
files to force re-simulation, or set FORCE_RESIM = True.

Dependencies
------------
numpy, matplotlib, scipy, joblib, tqdm
observables.py, algorithms.py  (from your project folder)

Usage
-----
    python percolation_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from joblib import Parallel, delayed

from observables import generate_lattice
from algorithms  import hoshen_kopelman

try:
    from tqdm import tqdm as _tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

# =============================================================================
# Configuration
# =============================================================================

PC          = 0.59296       # your FSS pc
NU          = 4/3           # exact theoretical value
BETA        = 5/36          # exact theoretical value
GAMMA       = 43/18         # exact theoretical value
TAU         = 187/91        # exact theoretical value
D_F         = 91/48         # exact fractal dimension

L_VALUES    = [64, 128, 256, 512]

# p sweep for multi-L curves
P_SWEEP     = np.linspace(0.57, 0.62, 40)
RUNS_SWEEP  = 300           # runs per p value per L

# Cluster size distribution + pc scaling — single p at pc per L
L_NS        = [64, 128, 256, 512]   # L values for ns plots
RUNS_NS     = 2000          # runs at pc for cluster sizes (also used for scaling plots)

SPAN_MODE   = "LR"
BASE_SEED   = 42
FIG_DIR     = "figures_book"
CKPT_DIR    = "checkpoints_figures"
FORCE_RESIM = False         # set True to ignore saved checkpoints

# Line styles matching book convention (dotted, dash-dot, dashed, solid)
LINE_STYLES = {
    64:  (0, (1, 1)),       # dotted
    128: (0, (3, 1, 1, 1)), # dash-dot
    256: (0, (5, 2)),       # dashed
    512: "-",               # solid
}
LINE_LABELS = {L: f"$L = {L}$" for L in L_VALUES}

# =============================================================================
# Helpers
# =============================================================================

def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _ckpt_path(name):
    os.makedirs(CKPT_DIR, exist_ok=True)
    return os.path.join(CKPT_DIR, f"{name}.npz")


def _ckpt_exists(name):
    return (not FORCE_RESIM) and os.path.exists(_ckpt_path(name))


# =============================================================================
# Spanning root helper — respects SPAN_MODE exactly
# =============================================================================

def _spanning_roots_by_mode(labels, mode="LR"):
    """
    Return only the cluster roots that span in the requested direction.

    Unlike spanning_info() which returns LR ∪ TB roots, this function
    returns exactly the roots for the requested mode:
      "LR"  — clusters connecting left and right boundaries only
      "TB"  — clusters connecting top and bottom boundaries only
      "ANY" — union of both

    This avoids including TB-only spanning roots when mode="LR", which
    would bias Pinf upward and chi downward.
    """
    left_roots  = set(labels[:,  0][labels[:,  0] > 0] - 1)
    right_roots = set(labels[:, -1][labels[:, -1] > 0] - 1)
    top_roots   = set(labels[ 0, :][labels[ 0, :] > 0] - 1)
    bot_roots   = set(labels[-1, :][labels[-1, :] > 0] - 1)

    lr_roots = left_roots & right_roots
    tb_roots = top_roots  & bot_roots

    if mode == "LR":
        return lr_roots
    elif mode == "TB":
        return tb_roots
    elif mode == "ANY":
        return lr_roots | tb_roots
    else:
        raise ValueError(f"Unknown span_mode: {mode!r}")


# =============================================================================
# Single realisation
# =============================================================================

def _single_run(L, p, seed):
    """One HK realisation — returns Pinf, chi, and raw cluster sizes."""
    rng     = np.random.default_rng(seed)
    lattice = generate_lattice(L, p, rng=rng)
    labels, cluster_sizes = hoshen_kopelman(lattice)

    # Use direct root selection — avoids LR∪TB contamination
    active = _spanning_roots_by_mode(labels, SPAN_MODE)

    N      = L * L
    pinf   = sum(cluster_sizes.get(r, 0) for r in active) / N
    finite = np.array([s for r, s in cluster_sizes.items()
                       if r not in active], dtype=np.float64)
    chi    = float((finite**2).sum() / finite.sum()) if finite.size > 0 else 0.0
    sizes  = [s for r, s in cluster_sizes.items()
              if r not in active and s >= 2]
    return pinf, chi, sizes


def _simulate_sweep(L, p_values, runs, seed_offset=0, desc=None):
    """Simulate runs realisations at each p in p_values."""
    seeds_base = BASE_SEED + seed_offset
    Pinf_arr   = np.zeros(len(p_values))
    chi_arr    = np.zeros(len(p_values))

    iterator = enumerate(p_values)
    if _TQDM and desc:
        iterator = _tqdm(list(iterator), desc=desc,
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                    "[{elapsed}<{remaining}]")

    for j, p in iterator:
        seeds   = seeds_base + j * 10000 + np.arange(runs, dtype=int)
        results = Parallel(n_jobs=-1, prefer="threads",
                           return_as="generator")(
            delayed(_single_run)(L, p, int(s)) for s in seeds
        )
        pinf_list, chi_list = [], []
        for pinf, chi, _ in results:
            pinf_list.append(pinf)
            chi_list.append(chi)
        Pinf_arr[j] = np.mean(pinf_list)
        chi_arr[j]  = np.mean(chi_list)

    return Pinf_arr, chi_arr


def _simulate_at_pc(L, runs, seed_offset=0):
    """Simulate runs realisations at pc, returning Pinf, chi, all sizes."""
    seeds   = BASE_SEED + seed_offset + np.arange(runs, dtype=int)
    results = Parallel(n_jobs=-1, prefer="threads",
                       return_as="generator")(
        delayed(_single_run)(L, PC, int(s)) for s in (
            _tqdm(seeds, desc=f"  pc sim L={L:4d}",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                             "[{elapsed}<{remaining}]")
            if _TQDM else seeds
        )
    )
    pinf_list, chi_list, size_list = [], [], []
    for pinf, chi, sizes in results:
        pinf_list.append(pinf)
        chi_list.append(chi)
        size_list.extend(sizes)
    return float(np.mean(pinf_list)), float(np.mean(chi_list)), \
           np.array(size_list, dtype=float)


# =============================================================================
# Data acquisition
# =============================================================================

def get_sweep_data():
    """Get P∞ and χ vs p for each L. Load from checkpoint or simulate."""
    ckpt = _ckpt_path("sweep")
    if _ckpt_exists("sweep"):
        print("  Loading sweep data from checkpoint ...")
        d = np.load(ckpt, allow_pickle=True)
        Pinf = {L: d[f"Pinf_{L}"] for L in L_VALUES}
        chi  = {L: d[f"chi_{L}"]  for L in L_VALUES}
        return Pinf, chi

    print("  Simulating p-sweep for all L values ...")
    Pinf, chi = {}, {}
    save_dict = {}
    for i, L in enumerate(L_VALUES):
        p_arr, c_arr = _simulate_sweep(
            L, P_SWEEP, RUNS_SWEEP,
            seed_offset=i * 100000,
            desc=f"  Sweep   L={L:4d}",
        )
        Pinf[L] = p_arr
        chi[L]  = c_arr
        save_dict[f"Pinf_{L}"] = p_arr
        save_dict[f"chi_{L}"]  = c_arr
    np.savez(ckpt, **save_dict)
    print(f"  Sweep data saved to {ckpt}")
    return Pinf, chi


def get_pc_data():
    """Get P∞(pc,L), χ(pc,L) and raw sizes for each L."""
    ckpt = _ckpt_path("pc_data")
    if _ckpt_exists("pc_data"):
        print("  Loading pc data from checkpoint ...")
        d    = np.load(ckpt, allow_pickle=True)
        Pinf = {L: float(d[f"Pinf_{L}"]) for L in L_NS}
        chi  = {L: float(d[f"chi_{L}"])  for L in L_NS}
        sizes= {L: d[f"sizes_{L}"]       for L in L_NS}
        return Pinf, chi, sizes

    print("  Simulating at pc for all L values ...")
    Pinf, chi, sizes = {}, {}, {}
    save_dict = {}
    for i, L in enumerate(L_NS):
        p, c, s = _simulate_at_pc(L, RUNS_NS, seed_offset=200000 + i * 100000)
        Pinf[L]  = p
        chi[L]   = c
        sizes[L] = s
        save_dict[f"Pinf_{L}"]  = p
        save_dict[f"chi_{L}"]   = c
        save_dict[f"sizes_{L}"] = s
        print(f"    L={L:4d}  Pinf={p:.5f}  chi={c:.2f}  "
              f"n_clusters={len(s):,}")
    np.savez(ckpt, **save_dict)
    print(f"  pc data saved to {ckpt}")
    return Pinf, chi, sizes


# =============================================================================
# Figure 1: P∞ vs p, multi-L with inset
# =============================================================================

def fig_pinf_vs_p(Pinf_data):
    fig, ax = plt.subplots(figsize=(7, 5))

    for L in L_VALUES:
        ax.plot(P_SWEEP, Pinf_data[L],
                linestyle=LINE_STYLES[L],
                color="black", lw=1.2,
                label=LINE_LABELS[L])

    ax.axvline(PC, color="black", lw=0.8, linestyle=":")
    ax.set_xlabel(r"$p$", fontsize=13)
    ax.set_ylabel(r"$P_\infty(p;\,L)$", fontsize=13)
    ax.set_xlim(P_SWEEP.min() - 0.002, P_SWEEP.max() + 0.002)
    ax.set_ylim(0, None)
    ax.legend(fontsize=9, loc="upper left")
    ax.tick_params(labelsize=10)

    # Inset zoomed around pc
    inset = ax.inset_axes([0.08, 0.45, 0.38, 0.45])
    pc_mask = (P_SWEEP >= PC - 0.008) & (P_SWEEP <= PC + 0.008)
    for L in L_VALUES:
        inset.plot(P_SWEEP[pc_mask], Pinf_data[L][pc_mask],
                   linestyle=LINE_STYLES[L], color="black", lw=1.0)
    inset.axvline(PC, color="black", lw=0.8, linestyle=":")
    inset.set_xlim(PC - 0.008, PC + 0.008)
    inset.set_xlabel(r"$p$", fontsize=8)
    inset.set_ylabel(r"$P_\infty$", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.text(PC + 0.001, inset.get_ylim()[0] * 1.1,
               r"$p_c$", fontsize=8)

    fig.tight_layout()
    return fig


# =============================================================================
# Figure 2: P∞(pc, L) vs L log-log
# =============================================================================

def fig_pinf_scaling(Pinf_pc):
    fig, ax = plt.subplots(figsize=(6, 5))

    L_arr    = np.array(L_NS, dtype=float)
    Pinf_arr = np.array([Pinf_pc[L] for L in L_NS])

    ax.loglog(L_arr, Pinf_arr, "o", color="black", ms=7,
              zorder=5, label="Simulation")

    # Theoretical slope -β/ν = -5/48
    slope = -BETA / NU
    L_fit = np.logspace(np.log10(L_arr.min() * 0.8),
                        np.log10(L_arr.max() * 1.2), 100)
    norm  = Pinf_arr[-1] / (L_arr[-1] ** slope)
    ax.loglog(L_fit, norm * L_fit**slope, "--", color="black",
              lw=1.2, label=rf"slope $= -\beta/\nu = -{BETA/NU:.4f}$")

    ax.set_xlabel(r"$L$", fontsize=13)
    ax.set_ylabel(r"$P_\infty(p_c;\,L)$", fontsize=13)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig


# =============================================================================
# Figure 3: χ vs p, multi-L log-linear
# =============================================================================

def fig_chi_vs_p(chi_data):
    fig, ax = plt.subplots(figsize=(7, 5))

    for L in L_VALUES:
        ax.semilogy(P_SWEEP, chi_data[L],
                    linestyle=LINE_STYLES[L],
                    color="black", lw=1.2,
                    label=LINE_LABELS[L])

    ax.axvline(PC, color="black", lw=0.8, linestyle=":")
    ax.set_xlabel(r"$p$", fontsize=13)
    ax.set_ylabel(r"$\chi(p;\,L)$", fontsize=13)
    ax.set_xlim(P_SWEEP.min() - 0.002, P_SWEEP.max() + 0.002)
    ax.legend(fontsize=9, loc="upper left")
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig


# =============================================================================
# Figure 4: χ(pc, L) vs L log-log
# =============================================================================

def fig_chi_scaling(chi_pc):
    fig, ax = plt.subplots(figsize=(6, 5))

    L_arr   = np.array(L_NS, dtype=float)
    chi_arr = np.array([chi_pc[L] for L in L_NS])

    ax.loglog(L_arr, chi_arr, "o", color="black", ms=7,
              zorder=5, label="Simulation")

    # Theoretical slope γ/ν = 43/24
    slope = GAMMA / NU
    L_fit = np.logspace(np.log10(L_arr.min() * 0.8),
                        np.log10(L_arr.max() * 1.2), 100)
    norm  = chi_arr[-1] / (L_arr[-1] ** slope)
    ax.loglog(L_fit, norm * L_fit**slope, "--", color="black",
              lw=1.2, label=rf"slope $= \gamma/\nu = {slope:.4f}$")

    ax.set_xlabel(r"$L$", fontsize=13)
    ax.set_ylabel(r"$\chi(p_c;\,L)$", fontsize=13)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig


# =============================================================================
# Figure 5: s^τ n(s, pc; L) vs s  — scaled cluster size distribution
# =============================================================================

def _compute_ns(sizes, L, runs, n_bins=50):
    """
    Compute n_s in log bins, correctly normalised per site per unit s.

    n_s = counts / (delta_s * L^2 * runs)

    The runs factor is essential — sizes is pooled from all realisations
    so without it n_s is too large by a factor of runs.
    """
    if len(sizes) == 0:
        return np.array([]), np.array([])
    smax  = float(L**2)
    bins  = np.logspace(np.log10(1), np.log10(smax), n_bins + 1)
    counts, edges = np.histogram(sizes, bins=bins)
    centres = np.sqrt(edges[:-1] * edges[1:])
    widths  = np.diff(edges)
    ns      = counts / (widths * L**2 * runs)
    # Fix 4: conservative mask — exclude noisy tails
    mask    = (counts > 5) & (centres >= 5)
    return centres[mask], ns[mask]


def fig_ns_scaled(sizes_data):
    """s^τ n(s, pc; L) vs s for multiple L — should be flat if τ correct."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for L in L_NS:
        s, ns = _compute_ns(sizes_data[L], L, runs=RUNS_NS)
        if len(s) == 0:
            continue
        scaled = s**TAU * ns
        ax.loglog(s, scaled,
                  linestyle=LINE_STYLES[L],
                  color="black", lw=1.2,
                  label=LINE_LABELS[L])

    ax.set_xlabel(r"$s$", fontsize=13)
    ax.set_ylabel(r"$s^\tau\, n(s,\,p_c;\,L)$", fontsize=13)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)

    # Annotate with tau value used
    ax.text(0.05, 0.05,
            rf"$\tau = {TAU:.4f}$ (exact)",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom")

    fig.tight_layout()
    return fig


# =============================================================================
# Figure 6: s^τ n(s, pc; L) vs s/L^D — data collapse
# =============================================================================

def fig_ns_collapse(sizes_data):
    """Data collapse: s^τ n(s,pc;L) vs s/L^D_f."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for L in L_NS:
        s, ns = _compute_ns(sizes_data[L], L, runs=RUNS_NS)
        if len(s) == 0:
            continue
        scaled_y = s**TAU * ns
        scaled_x = s / (float(L)**D_F)
        ax.loglog(scaled_x, scaled_y,
                  linestyle=LINE_STYLES[L],
                  color="black", lw=1.2,
                  label=LINE_LABELS[L])

    ax.set_xlabel(r"$s / L^{D}$", fontsize=13)
    ax.set_ylabel(r"$s^\tau\, n(s,\,p_c;\,L)$", fontsize=13)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)

    ax.text(0.05, 0.05,
            rf"$D = d_f = {D_F:.4f}$ (exact)",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom")

    fig.tight_layout()
    return fig


# =============================================================================
# Figure 7: P∞ vs (p - pc) log-log — direct β measurement
# =============================================================================

def fig_beta_powerlaw(Pinf_data):
    """
    Log-log plot of P∞ vs (p - pc) for p > pc using the largest L.
    The slope gives β directly.  Theoretical line with β = 5/36 overlaid.
    Uses L=512 (most accurate) and also shows L=256 for comparison.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for L in [256, 512]:
        if L not in Pinf_data:
            continue
        mask = (P_SWEEP > PC) & (Pinf_data[L] > 0)
        x    = P_SWEEP[mask] - PC
        y    = Pinf_data[L][mask]
        if x.sum() == 0:
            continue
        ax.loglog(x, y, linestyle=LINE_STYLES[L], color="black",
                  lw=1.2, label=LINE_LABELS[L])

        # Measured slope from linear regression on log-log
        slope, intercept, *_ = linregress(np.log(x), np.log(y))
        if L == 512:
            beta_meas = slope

    # Theoretical power law P∞ ~ (p-pc)^β
    x_fit = np.logspace(np.log10((P_SWEEP[P_SWEEP > PC] - PC).min()),
                        np.log10((P_SWEEP[P_SWEEP > PC] - PC).max()), 100)
    # Normalise through the midpoint of L=512 data
    L_ref  = max((L for L in [256, 512] if L in Pinf_data), default=max(Pinf_data.keys()))
    mask_r = (P_SWEEP > PC) & (Pinf_data[L_ref] > 0)
    x_ref  = (P_SWEEP[mask_r] - PC)
    y_ref  = Pinf_data[L_ref][mask_r]
    mid    = len(x_ref) // 2
    norm   = y_ref[mid] / (x_ref[mid] ** BETA)
    ax.loglog(x_fit, norm * x_fit**BETA, "--", color="black", lw=1.2,
              label=rf"$\beta = {BETA:.4f}$ (exact)")

    ax.set_xlabel(r"$p - p_c$", fontsize=13)
    ax.set_ylabel(r"$P_\infty$", fontsize=13)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)
    ax.set_title(rf"Order parameter power law: $P_\infty \propto (p-p_c)^\beta$",
                 fontsize=10)
    fig.tight_layout()
    return fig


# =============================================================================
# Figure 8: χ vs |p - pc| log-log — direct γ measurement, both sides
# =============================================================================

def fig_gamma_powerlaw(chi_data):
    """
    Log-log plot of χ vs |p - pc| for both sides of pc using L=512.
    Fitting both sides separately and showing they give consistent γ
    is a direct cross-check of the scaling hypothesis.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    L = max(L_VALUES)
    p_arr = P_SWEEP
    c_arr = chi_data[L]

    # Supercritical side p > pc
    mask_sup = (p_arr > PC) & (p_arr < P_SWEEP.max()) & (c_arr > 0)
    x_sup = p_arr[mask_sup] - PC
    y_sup = c_arr[mask_sup]

    # Subcritical side p < pc
    mask_sub = (p_arr < PC) & (p_arr > P_SWEEP.min()) & (c_arr > 0)
    x_sub = PC - p_arr[mask_sub]
    y_sub = c_arr[mask_sub]

    if x_sup.size > 2:
        ax.loglog(x_sup, y_sup, linestyle="-", color="black", lw=1.2,
                  label=r"$p > p_c$")
        slope_sup, *_ = linregress(np.log(x_sup), np.log(y_sup))
    if x_sub.size > 2:
        ax.loglog(x_sub, y_sub, linestyle="--", color="black", lw=1.2,
                  label=r"$p < p_c$")
        slope_sub, *_ = linregress(np.log(x_sub), np.log(y_sub))

    # Theoretical line χ ~ |p-pc|^{-γ}
    x_all  = np.concatenate([x_sup, x_sub])
    x_fit  = np.logspace(np.log10(x_all.min()), np.log10(x_all.max()), 100)
    # Normalise through a central point on the supercritical side
    if x_sup.size > 2:
        mid  = len(x_sup) // 2
        norm = y_sup[mid] * (x_sup[mid] ** GAMMA)
        ax.loglog(x_fit, norm * x_fit**(-GAMMA), ":", color="black",
                  lw=1.5, label=rf"$\gamma = {GAMMA:.4f}$ (exact)")

    ax.set_xlabel(r"$|p - p_c|$", fontsize=13)
    ax.set_ylabel(r"$\chi$", fontsize=13)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)
    ax.set_title(rf"Susceptibility power law: $\chi \propto |p-p_c|^{{-\gamma}}$"
                 f"\n(L = {L})", fontsize=10)
    fig.tight_layout()
    return fig


# =============================================================================
# Figure 9: Quotient method convergence — β/ν and γ/ν vs L pair
# =============================================================================

def fig_quotient_convergence(Pinf_pc, chi_pc):
    """
    Show β/ν and γ/ν quotient estimates from each doubling pair,
    with the theoretical value as a horizontal dashed line.
    Demonstrates finite-size corrections decreasing with L.

    At pc:
        P∞(2L)/P∞(L) = 2^(-β/ν)  =>  β/ν = -log2(ratio)
        χ(2L)/χ(L)   = 2^(γ/ν)   =>  γ/ν =  log2(ratio)
    """
    pairs = [(L_NS[i], L_NS[i+1]) for i in range(len(L_NS)-1)
             if L_NS[i+1] == 2 * L_NS[i]]

    beta_nu_vals  = []
    gamma_nu_vals = []
    pair_labels   = []

    for L1, L2 in pairs:
        if L1 not in Pinf_pc or L2 not in Pinf_pc:
            continue
        p1, p2 = Pinf_pc[L1], Pinf_pc[L2]
        c1, c2 = chi_pc[L1],  chi_pc[L2]

        if p1 > 0 and p2 > 0:
            beta_nu_vals.append(-np.log2(p2 / p1))
        if c1 > 0 and c2 > 0:
            gamma_nu_vals.append(np.log2(c2 / c1))
        pair_labels.append(f"$L={L1}\\to{L2}$")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Quotient method convergence with system size", fontsize=11)

    for ax, vals, tv, sym, ylabel in [
        (axes[0], beta_nu_vals,  BETA/NU,  "β/ν",
         r"$\beta/\nu$"),
        (axes[1], gamma_nu_vals, GAMMA/NU, "γ/ν",
         r"$\gamma/\nu$"),
    ]:
        if not vals:
            continue
        x = range(len(vals))
        ax.plot(x, vals, "o-", color="black", ms=8, lw=1.2,
                label="Quotient estimate")
        ax.axhline(tv, color="black", ls="--", lw=1.2,
                   label=f"Exact: {sym} = {tv:.4f}")
        ax.set_xticks(list(x))
        ax.set_xticklabels(pair_labels[:len(vals)], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xlabel("$L$ pair", fontsize=11)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.3)

        # Annotate each point with its value
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.4f}", (i, v),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)

    fig.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  Percolation figures — book style")
    print("=" * 60)
    print(f"  pc={PC:.5f}  nu={NU:.4f}  beta={BETA:.4f}  "
          f"gamma={GAMMA:.4f}")
    print(f"  tau={TAU:.4f}  d_f={D_F:.4f}")
    print(f"  L_VALUES={L_VALUES}")
    print(f"  RUNS_SWEEP={RUNS_SWEEP}  RUNS_NS={RUNS_NS}")

    os.makedirs(FIG_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\n[1/3] Getting p-sweep data ...")
    Pinf_sweep, chi_sweep = get_sweep_data()

    print("\n[2/3] Getting pc scaling data ...")
    Pinf_pc, chi_pc, sizes_pc = get_pc_data()

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\n[3/3] Producing figures ...")

    _save(fig_pinf_vs_p(Pinf_sweep),                    "pinf_vs_p")
    _save(fig_pinf_scaling(Pinf_pc),                    "pinf_scaling")
    _save(fig_chi_vs_p(chi_sweep),                      "chi_vs_p")
    _save(fig_chi_scaling(chi_pc),                      "chi_scaling")
    _save(fig_ns_scaled(sizes_pc),                      "ns_scaled")
    _save(fig_ns_collapse(sizes_pc),                    "ns_collapse")
    _save(fig_beta_powerlaw(Pinf_sweep),                "beta_powerlaw")
    _save(fig_gamma_powerlaw(chi_sweep),                "gamma_powerlaw")
    _save(fig_quotient_convergence(Pinf_pc, chi_pc),    "quotient_convergence")

    print(f"\nAll figures saved to: {FIG_DIR}/")
    print("Checkpoints saved to:  checkpoints_figures/")
    print("Re-run to tweak figures — simulation will be skipped.")


if __name__ == "__main__":
    main()
