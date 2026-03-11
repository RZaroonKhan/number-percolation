"""
refit_only.py
-------------
Re-runs ONLY the fine sweep and then regenerates the fit figures
with corrected fitting windows. Use this when you have good FSS
results already but want to improve beta, gamma, tau estimates
without re-running the full overnight simulation.

Steps run:
  1. Fine sweep at L_MAIN (30-40 mins at L=256, 1000 runs)
  2. Bootstrap uncertainties
  3. Exponent report with corrected windows
  4. beta_fit, gamma_fit, tau_fit figures
  5. Scaling window scan figures
  6. Updated summary figure (with corrected exponents)
  7. Newman-Ziff

Steps SKIPPED (keep your existing figures for these):
  - Coarse sweep
  - FSS loop (data collapse, pc_fss, exponent convergence, fractal dimension)
  - ML

Edit the config block below. Make sure L_MAIN, P_FINE, RUNS_FINE and
BASE_SEED match what you used in your original overnight run so the
fine sweep is consistent with your other figures.

The key things to tune are BETA_XMIN/XMAX and GAMMA_XMIN/XMAX.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

from simulation import estimate_observables_over_p
from analysis   import (
    estimate_pc_single_L, bootstrap_uncertainties,
    full_exponent_report, scaling_window_scan, optimal_window, THEORY,
)
from plotting import (
    plot_basic_observables, plot_beta_fit, plot_gamma_fit,
    plot_tau_fit, plot_nz_curve, plot_summary,
    plot_scaling_window,
)
from algorithms import newman_ziff

# =============================================================================
# Configuration — match your original run settings
# =============================================================================

L_MAIN     = 256
RUNS_FINE  = 1000
P_FINE     = np.linspace(0.57, 0.62, 40)
SPAN_MODE  = "LR"
BASE_SEED  = 42
N_BOOT     = 1000
FIG_DIR    = "figures"       # saves into same figures folder — overwrites old fits

# Your FSS results from the overnight run — paste in here
PC_OVERRIDE = 0.59296        # pc from your FSS extrapolation
NU_OVERRIDE = 1.378          # nu from your FSS fit

# -----------------------------------------------------------------------
# FITTING WINDOWS — this is the main thing to adjust
# The window [xmin, xmax] defines the range of |p - pc| used for fitting.
# Too small: noisy, too close to pc, finite-size effects dominate
# Too large: higher-order corrections kick in, slope drifts
#
# Look at your beta_fit and gamma_fit figures:
#   - Find the straight portion of the log-log plot
#   - Set xmin/xmax to bracket only that straight region
# -----------------------------------------------------------------------
BETA_XMIN,  BETA_XMAX  = 0.01,  0.04
GAMMA_XMIN, GAMMA_XMAX = 0.003, 0.03

# Scaling window scan range
SCAN_XMINS = np.logspace(-3, -1, 25)
SCAN_XMAXS = np.logspace(-2, np.log10(0.15), 25)

# =============================================================================
# Helpers
# =============================================================================

def _save(fig, name):
    if FIG_DIR:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")


def _run(desc=None):
    return estimate_observables_over_p(
        L=L_MAIN, p_values=P_FINE, runs=RUNS_FINE,
        span_mode=SPAN_MODE, n_jobs=-1,
        base_seed=BASE_SEED + 10000,
        return_runs=True,
        desc=desc,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  Refit only — fine sweep + corrected fit windows")
    print("=" * 60)
    print(f"  Using pc = {PC_OVERRIDE}  nu = {NU_OVERRIDE}  (from your FSS run)")
    print(f"  Beta  window: [{BETA_XMIN},  {BETA_XMAX}]")
    print(f"  Gamma window: [{GAMMA_XMIN}, {GAMMA_XMAX}]")
    print()

    # ------------------------------------------------------------------
    # Fine sweep
    # ------------------------------------------------------------------
    print(f"[1/4] Fine sweep  L={L_MAIN}, runs={RUNS_FINE} ...")
    fine = _run(desc=f"Fine sweep L={L_MAIN}")
    pc_fine = estimate_pc_single_L(fine["p"], fine["R"])
    print(f"  pc (fine sweep) = {pc_fine:.5f}  [using override: {PC_OVERRIDE:.5f}]")
    _save(plot_basic_observables(fine, L_MAIN, "Fine: "), "fine_observables")

    # Use the FSS pc — more accurate than single-L crossing
    pc_best = PC_OVERRIDE
    nu_fit  = NU_OVERRIDE

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    print(f"\n[2/4] Bootstrap uncertainties  (n_boot={N_BOOT}) ...")
    unc = bootstrap_uncertainties(
        fine, pc_best, n_boot=N_BOOT,
        beta_xmin=BETA_XMIN,  beta_xmax=BETA_XMAX,
        gamma_xmin=GAMMA_XMIN, gamma_xmax=GAMMA_XMAX,
        L=L_MAIN, rng=np.random.default_rng(BASE_SEED + 50000),
    )

    # ------------------------------------------------------------------
    # Exponent report + fit figures
    # ------------------------------------------------------------------
    print(f"\n[3/4] Extracting critical exponents ...")
    full_exponent_report(fine, pc=pc_best, L=L_MAIN,
                         beta_xmin=BETA_XMIN,  beta_xmax=BETA_XMAX,
                         gamma_xmin=GAMMA_XMIN, gamma_xmax=GAMMA_XMAX,
                         uncertainties=unc)

    for name, fn, kw in [
        ("beta_fit",  plot_beta_fit,  dict(xmin=BETA_XMIN,  xmax=BETA_XMAX)),
        ("gamma_fit", plot_gamma_fit, dict(xmin=GAMMA_XMIN, xmax=GAMMA_XMAX)),
        ("tau_fit",   plot_tau_fit,   dict(L=L_MAIN)),
    ]:
        print(f"  Plotting {name} ...")
        fig, _ = fn(fine, pc_best, **kw)
        _save(fig, name)

    # ------------------------------------------------------------------
    # Scaling window scan
    # ------------------------------------------------------------------
    print(f"\n[3b/4] Scaling correction window scan ...")
    for obs_type, obs_arr, sym, tv in [
        ("beta",  fine["Pinf"], "beta",  THEORY["beta"]),
        ("gamma", fine["chi"],  "gamma", THEORY["gamma"]),
    ]:
        scan = scaling_window_scan(
            fine["p"], obs_arr, pc_best,
            obs_type=obs_type,
            xmins=SCAN_XMINS, xmaxs=SCAN_XMAXS,
        )
        best = optimal_window(scan, tv)
        if best:
            print(f"  Best {sym} window: xmin={best['xmin']:.4f}, "
                  f"xmax={best['xmax']:.4f}  ->  {sym}={best['exponent']:.4f}"
                  f"  (theory {tv:.4f})")
        _save(plot_scaling_window(scan, obs_type=obs_type, theory_val=tv),
              f"scaling_window_{obs_type}")

    # ------------------------------------------------------------------
    # Summary + Newman-Ziff
    # ------------------------------------------------------------------
    print(f"\n[4/4] Summary figure and Newman-Ziff ...")
    _save(plot_summary(fine, pc=pc_best, L=L_MAIN, nu=nu_fit,
                       uncertainties=unc, results_by_L={}),
          "summary")
    print(f"  Running Newman-Ziff ...")
    largest = newman_ziff(L_MAIN, rng=np.random.default_rng(BASE_SEED + 99999))
    _save(plot_nz_curve(largest, L_MAIN, pc=pc_best), "newman_ziff")

    print(f"\nDone. Updated figures saved to: {FIG_DIR}")
    print(f"Keep your existing figures for: data collapse, pc_fss,")
    print(f"exponent_convergence, fractal_dimension — those are unchanged.")


if __name__ == "__main__":
    main()
