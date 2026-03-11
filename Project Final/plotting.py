"""
plotting.py
-----------
All matplotlib figures for the percolation project.

  plot_basic_observables()    — R, P∞, chi, nc/N vs p.
  plot_ns_at_pc()             — cluster-size distribution n_s at pc.
  plot_beta_fit()             — P∞ ~ (p-pc)^beta log-log fit.
  plot_gamma_fit()            — chi ~ |p-pc|^{-gamma} log-log fit.
  plot_tau_fit()              — n_s ~ s^{-tau} log-log fit.
  plot_nz_curve()             — Newman-Ziff largest-cluster fraction.
  plot_pc_fss()               — FSS extrapolation of pc(L).
  plot_data_collapse()        — multi-L scaling collapse.
  plot_exponent_convergence() — exponent estimates vs L.
  plot_summary()              — six-panel publication figure.
  plot_all()                  — generate all standard plots at once.
"""

import numpy as np
import matplotlib.pyplot as plt

from analysis import estimate_beta, estimate_gamma, estimate_tau, THEORY

# Consistent colours used across all plots
_CD = "#2166ac"   # data points
_CF = "#d6604d"   # fit lines


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _style(ax, xlabel, ylabel, title, loglog=False):
    """Apply standard axis labels, title, legend, and grid."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8)
    ax.grid(True, which="both" if loglog else "major", alpha=0.3)


def _fitline(ax, x, slope, intercept, color=_CF, label=""):
    """Draw  y = exp(intercept) * x^slope  on a log-log axis."""
    ax.loglog(x, np.exp(intercept + slope * np.log(x)),
              "--", color=color, label=label)


def _unc_label(key, val, uncertainties):
    """Format 'key = val +/- err' or 'key = val' depending on availability."""
    if uncertainties and key in uncertainties:
        return f"{key} = {val:.3f} +/- {uncertainties[key]['std']:.3f}"
    return f"{key} = {val:.3f}"


# ---------------------------------------------------------------------------
# Basic observables
# ---------------------------------------------------------------------------

def plot_basic_observables(results, L, title_prefix=""):
    """Four-panel figure: R(p), P∞(p), chi(p), nc/N(p)."""
    p   = results["p"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"{title_prefix}Basic observables  (L = {L})", fontsize=13)

    ax = axes[0, 0]
    ax.plot(p, results["R"], "o-", ms=4, color=_CD)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="R = 0.5")
    _style(ax, "p", "Spanning probability R(p)", "Spanning probability")

    ax = axes[0, 1]
    ax.plot(p, results["Pinf"], "o-", ms=4, color="C1")
    _style(ax, "p", "Percolation strength P∞(p)", "Percolation strength")

    ax = axes[1, 0]
    ax.semilogy(p, results["chi"], "o-", ms=4, color="C2")
    _style(ax, "p", "Mean finite cluster size chi(p)", "Mean cluster size",
           loglog=True)

    ax = axes[1, 1]
    ax.plot(p, results["nc_per_site"], "o-", ms=4, color="C3")
    _style(ax, "p", "Clusters per site", "Cluster density")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cluster-size distribution at pc
# ---------------------------------------------------------------------------

def plot_ns_at_pc(results, pc, tau=None):
    """Log-log plot of n_s at the p closest to pc."""
    p_arr = results["p"]
    k     = int(np.argmin(np.abs(p_arr - pc)))
    ns    = results["ns_density"][k]
    s     = np.array(sorted(ns.keys()), dtype=float)
    n_s   = np.array([ns[int(si)] for si in s], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(s, n_s, "o", ms=4, color=_CD, label=f"p = {p_arr[k]:.4f}")

    if tau is not None:
        c  = np.log(n_s[n_s > 0][0]) + tau * np.log(s[n_s > 0][0])
        sf = np.logspace(np.log10(s.min()), np.log10(s.max()), 200)
        ax.loglog(sf, np.exp(c - tau * np.log(sf)), "--", color=_CF,
                  label=f"tau = {tau:.3f}  (theory {THEORY['tau']:.3f})")

    _style(ax, "Cluster size s", "n_s  (clusters per site)",
           f"Cluster-size distribution at pc = {pc:.5f}", loglog=True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Beta fit
# ---------------------------------------------------------------------------

def plot_beta_fit(results, pc, xmin=0.01, xmax=0.08):
    """Log-log: P∞ vs (p - pc) with beta power-law fit."""
    p_arr = results["p"]
    Pinf  = results["Pinf"]
    beta, c, mask = estimate_beta(p_arr, Pinf, pc, xmin=xmin, xmax=xmax)
    x = p_arr - pc

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(x[x > 0], Pinf[x > 0], "o", ms=4, color=_CD, label="data")
    _fitline(ax, x[mask], beta, c,
             label=f"beta = {beta:.3f}  (theory {THEORY['beta']:.3f})")
    _style(ax, "p - pc", "P∞(p)", f"beta fit  (pc = {pc:.5f})", loglog=True)
    fig.tight_layout()
    return fig, beta


# ---------------------------------------------------------------------------
# Gamma fit
# ---------------------------------------------------------------------------

def plot_gamma_fit(results, pc, xmin=0.01, xmax=0.08):
    """Log-log: chi vs |p - pc| with gamma power-law fit."""
    p_arr = results["p"]
    chi   = results["chi"]
    gamma, c, mask = estimate_gamma(p_arr, chi, pc, xmin=xmin, xmax=xmax)
    x = np.abs(p_arr - pc)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(x[x > 0], chi[x > 0], "o", ms=4, color=_CD, label="data")
    _fitline(ax, x[mask], -gamma, c,
             label=f"gamma = {gamma:.3f}  (theory {THEORY['gamma']:.3f})")
    _style(ax, "|p - pc|", "chi(p)", f"gamma fit  (pc = {pc:.5f})", loglog=True)
    fig.tight_layout()
    return fig, gamma


# ---------------------------------------------------------------------------
# Tau fit
# ---------------------------------------------------------------------------

def plot_tau_fit(results, pc, smin=2, L=None):
    """Log-log: n_s at criticality with tau (Fisher exponent) fit."""
    p_arr = results["p"]
    k     = int(np.argmin(np.abs(p_arr - pc)))
    smax  = (L * L // 4) if L is not None else None
    tau, c, s, ns = estimate_tau(results["ns_density"][k], smin=smin, smax=smax)
    mask  = (s >= smin) & (ns > 0) & (s <= smax if smax else True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(s, ns, "o", ms=4, color=_CD, label="data")
    _fitline(ax, s[mask], -tau, c,
             label=f"tau = {tau:.3f}  (theory {THEORY['tau']:.3f})")
    _style(ax, "Cluster size s", "n_s",
           f"tau fit  (p ~ {p_arr[k]:.4f})", loglog=True)
    fig.tight_layout()
    return fig, tau


# ---------------------------------------------------------------------------
# Newman-Ziff curve
# ---------------------------------------------------------------------------

def plot_nz_curve(largest_cluster, L, pc=None):
    """Largest-cluster fraction vs p from a Newman-Ziff run."""
    N    = L * L
    p_nz = np.arange(len(largest_cluster)) / N

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(p_nz, largest_cluster / N, lw=1.2, color=_CD,
            label="NZ (single run)")
    if pc is not None:
        ax.axvline(pc, color="red", ls="--", lw=1, label=f"pc = {pc:.4f}")
    _style(ax, "p  (~n / N)", "Largest cluster fraction",
           f"Newman-Ziff  (L = {L})")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# FSS of pc(L)
# ---------------------------------------------------------------------------

def plot_pc_fss(L_arr, pc_L_arr, pc_inf=None, nu=None):
    """pc(L) vs 1/L with optional FSS extrapolation curve."""
    L_arr    = np.asarray(L_arr,    dtype=float)
    pc_L_arr = np.asarray(pc_L_arr, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(1.0 / L_arr, pc_L_arr, "o", ms=6, color=_CD,
            label="pc(L) from R = 0.5")

    if pc_inf is not None and nu is not None:
        L_fine = np.linspace(L_arr.min(), L_arr.max() * 3, 300)
        a = (pc_L_arr[0] - pc_inf) / (L_arr[0] ** (-1.0 / nu))
        ax.plot(1.0 / L_fine, pc_inf + a * L_fine ** (-1.0 / nu), "--",
                color=_CF,
                label=f"Fit: pc(inf) = {pc_inf:.5f},  nu = {nu:.3f}")
        ax.axhline(pc_inf, color=_CF, ls=":", lw=0.8)

    ax.axhline(THEORY["pc"], color="grey", ls="--", lw=0.8,
               label=f"Theory pc = {THEORY['pc']}")
    _style(ax, "1 / L", "pc(L)", "Finite-size scaling of pc")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Data collapse
# ---------------------------------------------------------------------------

def plot_data_collapse(results_by_L, pc, nu, observable="chi",
                       beta_over_nu=None):
    """
    Rescale curves from multiple L values onto a single master curve.

    x-axis: (p - pc) * L^{1/nu}
    y-axis: chi / L^{gamma/nu}  |  P∞ * L^{beta/nu}  |  R (no scaling)

    A good collapse confirms your nu estimate is correct.
    """
    if beta_over_nu is None:
        beta_over_nu = 5.0 / 48.0
    gamma_over_nu = 43.0 / 24.0

    # (y-axis label, scaling exponent, direction: 1=divide, -1=multiply, 0=none)
    scale_info = {
        "chi":  ("chi / L^(gamma/nu)",  gamma_over_nu,  1),
        "Pinf": ("P∞ * L^(beta/nu)",    beta_over_nu,  -1),
        "R":    ("R(p)",                 0.0,            0),
    }
    if observable not in scale_info:
        raise ValueError(f"observable must be one of {list(scale_info)}")

    ylabel, exp, sign = scale_info[observable]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results_by_L)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for (L_val, res), col in zip(sorted(results_by_L.items()), colors):
        x = (res["p"] - pc) * (L_val ** (1.0 / nu))
        y = res[observable]
        if   sign ==  1: y = y / (L_val ** exp)
        elif sign == -1: y = y * (L_val ** exp)
        ax.plot(x, y, "o-", ms=3, lw=1.2, color=col, label=f"L = {L_val}")

    ax.axvline(0, color="grey", ls="--", lw=0.8)
    _style(ax, "(p - pc) * L^(1/nu)", ylabel,
           f"Data collapse — {observable}   (pc = {pc:.5f},  nu = {nu:.4f})")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Exponent convergence vs L
# ---------------------------------------------------------------------------

def plot_exponent_convergence(L_arr, estimates_by_L, uncertainties_by_L=None):
    """
    Four-panel figure: each exponent vs L with theory line and error bars.
    Percentage deviation from theory is annotated on each point.
    """
    L_arr = np.array(sorted(L_arr), dtype=float)
    exps  = [("beta",  THEORY["beta"],  "C0"),
             ("gamma", THEORY["gamma"], "C1"),
             ("tau",   THEORY["tau"],   "C2"),
             ("pc",    THEORY["pc"],    "C3")]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Exponent convergence with lattice size L", fontsize=13)

    for ax, (key, tv, col) in zip(axes.flatten(), exps):
        vals = np.array([estimates_by_L.get(int(L), {}).get(key, np.nan)
                         for L in L_arr], dtype=float)
        errs = np.array([(uncertainties_by_L or {}).get(int(L), {})
                          .get(key, {}).get("std", np.nan)
                          for L in L_arr], dtype=float)
        valid = np.isfinite(vals)

        if uncertainties_by_L and np.any(np.isfinite(errs)):
            ax.errorbar(L_arr[valid], vals[valid], yerr=errs[valid],
                        fmt="o-", ms=6, lw=1.4, color=col, capsize=4,
                        label="estimate +/- 1 sigma")
        else:
            ax.plot(L_arr[valid], vals[valid], "o-", ms=6,
                    lw=1.4, color=col, label="estimate")

        ax.axhline(tv, color="grey", ls="--", lw=1.2,
                   label=f"theory = {tv:.4f}")
        ax.axhspan(tv * 0.90, tv * 1.10, alpha=0.08, color="grey")

        for L_val, v in zip(L_arr[valid], vals[valid]):
            ax.annotate(f"{100*(v-tv)/tv:+.1f}%", xy=(L_val, v),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=col)

        _style(ax, "L", key, f"{key} vs L")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def plot_summary(results, pc, L, nu, uncertainties=None, results_by_L=None):
    """
    Six-panel publication figure:
      [0,0] R(p) sigmoid        [0,1] Data collapse (R across L values)
      [1,0] beta fit            [1,1] gamma fit
      [2,0] tau fit             [2,1] Exponent summary table
    """
    fig  = plt.figure(figsize=(14, 13))
    gs   = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.32)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)]
    p    = results["p"]

    # [0,0] R(p)
    ax = axes[0][0]
    ax.plot(p, results["R"], "o-", ms=4, color=_CD)
    ax.axvline(pc, color=_CF, ls="--", lw=1.2, label=f"pc = {pc:.5f}")
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    _style(ax, "p", "R(p)", "Spanning probability")

    # [0,1] Data collapse (R) or fallback P∞
    ax = axes[0][1]
    if results_by_L and len(results_by_L) >= 2:
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results_by_L)))
        for (L_val, res), col in zip(sorted(results_by_L.items()), colors):
            x = (res["p"] - pc) * (L_val ** (1.0 / nu))
            ax.plot(x, res["R"], "o-", ms=3, lw=1.0, color=col,
                    label=f"L={L_val}")
        ax.axvline(0, color="grey", ls="--", lw=0.8)
        _style(ax, "(p-pc)*L^(1/nu)", "R(p)",
               f"Data collapse  (nu = {nu:.3f})")
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.plot(p, results["Pinf"], "o-", ms=4, color="C1")
        ax.axvline(pc, color=_CF, ls="--", lw=1.2)
        _style(ax, "p", "P∞(p)", "Percolation strength")

    # [1,0] beta fit
    ax = axes[1][0]
    try:
        beta_v, c_b, mask_b = estimate_beta(p, results["Pinf"], pc)
        x = p - pc
        ax.loglog(x[x > 0], results["Pinf"][x > 0], "o", ms=4,
                  color=_CD, label="data")
        _fitline(ax, x[mask_b], beta_v, c_b,
                 label=_unc_label("beta", beta_v, uncertainties))
        _style(ax, "p - pc", "P∞", "beta fit", loglog=True)
    except Exception as e:
        ax.text(0.5, 0.5, f"beta fit failed:\n{e}",
                transform=ax.transAxes, ha="center", va="center")
        beta_v = None

    # [1,1] gamma fit
    ax = axes[1][1]
    try:
        gamma_v, c_g, mask_g = estimate_gamma(p, results["chi"], pc)
        x = np.abs(p - pc)
        ax.loglog(x[x > 0], results["chi"][x > 0], "o", ms=4,
                  color=_CD, label="data")
        _fitline(ax, x[mask_g], -gamma_v, c_g,
                 label=_unc_label("gamma", gamma_v, uncertainties))
        _style(ax, "|p - pc|", "chi", "gamma fit", loglog=True)
    except Exception as e:
        ax.text(0.5, 0.5, f"gamma fit failed:\n{e}",
                transform=ax.transAxes, ha="center", va="center")
        gamma_v = None

    # [2,0] tau fit
    ax = axes[2][0]
    try:
        k     = int(np.argmin(np.abs(p - pc)))
        smax  = L * L // 4
        tau_v, c_t, s_a, ns_a = estimate_tau(results["ns_density"][k],
                                              smin=2, smax=smax)
        mask_t = (s_a >= 2) & (ns_a > 0) & (s_a <= smax)
        ax.loglog(s_a, ns_a, "o", ms=3, color=_CD, label="data")
        _fitline(ax, s_a[mask_t], -tau_v, c_t,
                 label=_unc_label("tau", tau_v, uncertainties))
        _style(ax, "s", "n_s", "tau fit  (Fisher exponent)", loglog=True)
    except Exception as e:
        ax.text(0.5, 0.5, f"tau fit failed:\n{e}",
                transform=ax.transAxes, ha="center", va="center")
        tau_v = None

    # [2,1] Results table
    ax = axes[2][1]
    ax.axis("off")

    def _row(key, tv, pt):
        if uncertainties and key in uncertainties:
            m, s = uncertainties[key]["mean"], uncertainties[key]["std"]
            return [key, f"{m:.4f} +/- {s:.4f}", f"{tv:.4f}"]
        return [key, f"{pt:.4f}" if pt is not None else "N/A", f"{tv:.4f}"]

    pc_err = (f" +/- {uncertainties['pc']['std']:.5f}"
              if uncertainties and "pc" in uncertainties else "")

    table_data = [
        _row("beta",  THEORY["beta"],  beta_v),
        _row("gamma", THEORY["gamma"], gamma_v),
        _row("tau",   THEORY["tau"],   tau_v),
        ["nu", f"{nu:.4f}",          f"{THEORY['nu']:.4f}"],
        ["pc", f"{pc:.5f}{pc_err}", f"{THEORY['pc']:.5f}"],
    ]

    tbl = ax.table(cellText=table_data,
                   colLabels=["Exponent", "Estimated", "Theory (2-D)"],
                   cellLoc="center", loc="center",
                   bbox=[0.0, 0.05, 1.0, 0.90])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    for j in range(3):
        tbl[0, j].set_facecolor(_CD)
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Critical exponent summary", fontsize=10, pad=4)

    fig.suptitle(f"2-D Site Percolation — Critical Analysis   (L = {L})",
                 fontsize=13, fontweight="bold", y=0.995)
    return fig


# ---------------------------------------------------------------------------
# Fractal dimension
# ---------------------------------------------------------------------------

def plot_df_fit(L_arr, Smax_at_pc):
    """
    Log-log plot of mean largest cluster size vs L at pc.
    Slope = fractal dimension d_f.

    S_max ~ L^{d_f} at criticality because the spanning cluster is a
    fractal object whose mass grows sub-extensively with system size.
    """
    from analysis import estimate_df
    L_arr    = np.asarray(L_arr,      dtype=float)
    Smax_arr = np.asarray(Smax_at_pc, dtype=float)
    valid    = np.isfinite(Smax_arr) & (Smax_arr > 0)

    df_est, c, _ = estimate_df(L_arr[valid], Smax_arr[valid])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(L_arr[valid], Smax_arr[valid], "o", ms=7, color="C0",
              label="data")
    L_fit = np.logspace(np.log10(L_arr[valid].min()),
                        np.log10(L_arr[valid].max()), 200)
    ax.loglog(L_fit, np.exp(c + df_est * np.log(L_fit)), "--", color="C1",
              label=f"d_f = {df_est:.3f}  (theory {THEORY['df']:.3f})")

    _style(ax, "L", r"$\langle S_{\max} \rangle$",
           r"Fractal dimension: $\langle S_{\max} \rangle \sim L^{d_f}$ at $p_c$",
           loglog=True)
    fig.tight_layout()
    return fig, df_est


# ---------------------------------------------------------------------------
# Scaling correction window scan
# ---------------------------------------------------------------------------

def plot_scaling_window(scan_results, obs_type="beta", theory_val=None):
    """
    Visualise how the estimated exponent varies across fitting windows.

    Each point is one (xmin, xmax) pair from the window scan, coloured
    by xmax. A stable plateau where the estimate barely changes as the
    window varies identifies the reliable scaling regime.
    """
    if not scan_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No valid windows found",
                transform=ax.transAxes, ha="center")
        return fig

    xmins = np.array([r["xmin"]     for r in scan_results])
    exps  = np.array([r["exponent"] for r in scan_results])
    xmaxs = np.array([r["xmax"]     for r in scan_results])

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(xmins, exps, c=xmaxs, cmap="viridis",
                    s=30, alpha=0.7, zorder=3)
    plt.colorbar(sc, ax=ax, label="xmax")

    if theory_val is not None:
        ax.axhline(theory_val, color="red", ls="--", lw=1.2,
                   label=f"Theory = {theory_val:.4f}")
        ax.axhspan(theory_val * 0.95, theory_val * 1.05,
                   alpha=0.08, color="red", label="±5% band")

    sym = "β" if obs_type == "beta" else "γ"
    ax.set_xscale("log")
    _style(ax, r"$x_{\min} = |p - p_c|_{\min}$", f"Estimated {sym}",
           f"Scaling window scan for {sym} — stable plateau = scaling regime")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ML pc convergence vs L
# ---------------------------------------------------------------------------

def plot_ml_convergence(L_arr, pc_fss_by_L, pc_cnn_by_L,
                        pc_inf_fss=None, pc_inf_cnn=None):
    """
    CNN pc and FSS pc estimates plotted together vs L.

    Shows whether the CNN suffers from the same finite-size bias as FSS,
    and how quickly each method converges to the true pc as L grows.
    """
    L_arr    = np.array(sorted(L_arr), dtype=float)
    fss_vals = np.array([pc_fss_by_L.get(int(L), np.nan) for L in L_arr])
    cnn_vals = np.array([pc_cnn_by_L.get(int(L), np.nan) for L in L_arr])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(L_arr[np.isfinite(fss_vals)], fss_vals[np.isfinite(fss_vals)],
            "o-", ms=7, color="C0", label="FSS  (R=0.5 crossing)")
    ax.plot(L_arr[np.isfinite(cnn_vals)], cnn_vals[np.isfinite(cnn_vals)],
            "s--", ms=7, color="C1", label="CNN  (P=0.5 crossing)")

    if pc_inf_fss is not None:
        ax.axhline(pc_inf_fss, color="C0", ls=":", lw=1,
                   label=f"FSS extrap. pc = {pc_inf_fss:.5f}")
    if pc_inf_cnn is not None:
        ax.axhline(pc_inf_cnn, color="C1", ls=":", lw=1,
                   label=f"CNN extrap. pc = {pc_inf_cnn:.5f}")

    ax.axhline(THEORY["pc"], color="grey", ls="--", lw=1.2,
               label=f"Theory pc = {THEORY['pc']:.5f}")

    _style(ax, "L", r"$p_c$ estimate",
           r"FSS vs CNN $p_c$ convergence with lattice size $L$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# One-shot: all standard individual plots
# ---------------------------------------------------------------------------

def plot_all(results, pc, L, show=True):
    """Generate all standard individual plots. Returns list of (fig, name)."""
    figs = [(plot_basic_observables(results, L), "basic_observables")]

    for name, fn, kwargs in [
        ("beta_fit",  plot_beta_fit,  {}),
        ("gamma_fit", plot_gamma_fit, {}),
        ("tau_fit",   plot_tau_fit,   {"L": L}),
    ]:
        try:
            fig, _ = fn(results, pc, **kwargs)
            figs.append((fig, name))
        except Exception as e:
            print(f"  Skipping {name}: {e}")

    try:
        _, tau = plot_tau_fit(results, pc, L=L)
        figs.append((plot_ns_at_pc(results, pc, tau=tau), "ns_at_pc"))
    except Exception as e:
        print(f"  Skipping ns_at_pc: {e}")

    if show:
        plt.show()
    return figs
