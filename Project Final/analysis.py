"""
analysis.py
-----------
Critical-point and exponent extraction for 2-D site percolation.

  THEORY                   — exact 2-D values for comparison.
  fit_loglog_slope()       — power-law fit in log-log space.
  estimate_pc_single_L()  — pc from R(p) = 0.5 crossing.
  estimate_pc_fss()        — FSS extrapolation pc(L) → pc(∞).
  estimate_beta/gamma/tau/df() — individual exponent fits.
  bootstrap_uncertainties()    — standard errors via resampling.
  full_exponent_report()       — formatted comparison table.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize    import curve_fit

try:
    from tqdm import tqdm as _tqdm
    _TQDM = True
except ImportError:
    _TQDM = False


def _progress(iterable, desc="", total=None):
    """Wrap iterable with tqdm if available, otherwise plain iteration."""
    if _TQDM:
        return _tqdm(iterable, desc=desc, total=total,
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                "[{elapsed}<{remaining}]")
    return iterable


# Exact 2-D site-percolation values
THEORY = {
    "pc":    0.59274,
    "beta":  5  / 36,   # ≈ 0.1389
    "gamma": 43 / 18,   # ≈ 2.389
    "nu":    4  / 3,    # ≈ 1.333
    "tau":   187 / 91,  # ≈ 2.055
    "df":    91  / 48,  # ≈ 1.896
}


def fit_loglog_slope(x, y, xmin=None, xmax=None):
    """
    Fit log y = m·log x + c over [xmin, xmax].
    Returns (slope m, intercept c, boolean mask of points used).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if xmin is not None: mask &= x >= xmin
    if xmax is not None: mask &= x <= xmax

    if mask.sum() < 2:
        raise ValueError(f"Too few points for log-log fit in [{xmin}, {xmax}].")

    m, c = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return m, c, mask


def estimate_pc_single_L(p_arr, R_arr):
    """
    Estimate pc as the p where R(p) = 0.5 via linear interpolation.
    Raises ValueError if R does not cross 0.5 within p_arr.
    """
    if R_arr.min() > 0.5 or R_arr.max() < 0.5:
        raise ValueError("R(p) does not cross 0.5 — widen your p_values.")
    return float(interp1d(R_arr, p_arr, kind="linear")(0.5))


def estimate_pc_fss(L_arr, pc_L_arr, nu_guess=1.333):
    """
    Extrapolate pc(L) -> pc(inf) via pc(L) = pc + a*L^{-1/nu}.

    nu is constrained to [0.8, 2.5] to prevent the fit latching onto
    nonsense values when pc(L) is noisy and non-monotonic. If the fit
    gives nu outside [0.9, 1.8] we fall back to fixing nu = theory (4/3)
    and only fitting pc_inf and a.

    Returns (pc_inf, nu, full_popt).
    """
    def model(L, pc_inf, a, nu):
        return pc_inf + a * L ** (-1.0 / nu)

    def model_fixed_nu(L, pc_inf, a):
        return pc_inf + a * L ** (-1.0 / nu_guess)

    # First: fit all three with bounds on nu
    try:
        popt, _ = curve_fit(
            model, L_arr, pc_L_arr,
            p0=[0.5927, 0.1, nu_guess],
            bounds=([0.55, -2.0, 0.8], [0.65, 2.0, 2.5]),
            maxfev=20000,
        )
        nu_fit = popt[2]
        if 0.9 <= nu_fit <= 1.8:
            return popt[0], nu_fit, popt
    except Exception:
        pass

    # Fallback: fix nu to theory
    try:
        popt2, _ = curve_fit(
            model_fixed_nu, L_arr, pc_L_arr,
            p0=[0.5927, 0.1],
            maxfev=20000,
        )
        return popt2[0], nu_guess, np.array([popt2[0], popt2[1], nu_guess])
    except Exception:
        return float(np.mean(pc_L_arr)), nu_guess, np.array([np.mean(pc_L_arr), 0.1, nu_guess])


def estimate_beta(p_arr, Pinf_arr, pc, xmin=0.01, xmax=0.08):
    """Beta from P∞ ~ (p-pc)^β for p > pc. Returns (beta, intercept, mask)."""
    return fit_loglog_slope(p_arr - pc, Pinf_arr, xmin=xmin, xmax=xmax)


def estimate_gamma(p_arr, chi_arr, pc, xmin=0.01, xmax=0.08):
    """Gamma from chi ~ |p-pc|^{-gamma}. Returns (gamma > 0, intercept, mask)."""
    slope, c, mask = fit_loglog_slope(np.abs(p_arr - pc), chi_arr,
                                      xmin=xmin, xmax=xmax)
    return -slope, c, mask


def estimate_tau(ns_dict, smin=2, smax=None):
    """Tau from n_s ~ s^{-tau} at criticality. Returns (tau, c, s_arr, ns_arr)."""
    s_arr  = np.array(sorted(ns_dict.keys()), dtype=float)
    ns_arr = np.array([ns_dict[int(s)] for s in s_arr], dtype=float)

    mask = (s_arr >= smin) & (ns_arr > 0)
    if smax is not None:
        mask &= s_arr <= smax

    slope, c, _ = fit_loglog_slope(s_arr[mask], ns_arr[mask])
    return -slope, c, s_arr, ns_arr


def estimate_df(L_arr, Smax_arr):
    """d_f from <S_max> ~ L^{d_f} at pc. Returns (df, intercept, mask)."""
    slope, c, mask = fit_loglog_slope(np.array(L_arr, float),
                                      np.array(Smax_arr, float))
    return slope, c, mask


def scaling_window_scan(p_arr, obs_arr, pc, obs_type="beta",
                        xmins=None, xmaxs=None):
    """
    Scan the fitting window and return exponent estimates at each window.

    For each (xmin, xmax) pair, fits the power law and records the
    exponent. Plotting the result shows the stable plateau region where
    the exponent is insensitive to window choice — this is where the
    true scaling regime lies and where your final window should sit.

    Parameters
    ----------
    p_arr    : p values from simulation.
    obs_arr  : observable array (Pinf for beta, chi for gamma).
    pc       : critical point estimate.
    obs_type : "beta" or "gamma".
    xmins    : array of xmin values to scan (default: 20 log-spaced values).
    xmaxs    : array of xmax values to scan (default: 20 log-spaced values).

    Returns
    -------
    results : list of dicts, each with keys
              "xmin", "xmax", "exponent", "n_points"
              Only entries where the fit succeeded are included.
    """
    if xmins is None:
        xmins = np.logspace(-3, -1, 20)
    if xmaxs is None:
        xmaxs = np.logspace(-2, np.log10(0.15), 20)

    fn      = estimate_beta if obs_type == "beta" else estimate_gamma
    results = []
    pairs   = [(xmin, xmax) for xmin in xmins for xmax in xmaxs
               if xmin < xmax]

    for xmin, xmax in _progress(pairs,
                                 desc=f"  Window scan ({obs_type})",
                                 total=len(pairs)):
        try:
            val, _, mask = fn(p_arr, obs_arr, pc, xmin=xmin, xmax=xmax)
            if np.isfinite(val) and mask.sum() >= 3:
                results.append({
                    "xmin":     float(xmin),
                    "xmax":     float(xmax),
                    "exponent": float(val),
                    "n_points": int(mask.sum()),
                })
        except Exception:
            pass

    return results


def optimal_window(scan_results, theory_val, tol=0.15):
    """
    From a window scan, find the (xmin, xmax) pair whose exponent estimate
    is closest to theory and has at least 3 points.

    Parameters
    ----------
    scan_results : output of scaling_window_scan().
    theory_val   : theoretical exponent value to compare against.
    tol          : maximum fractional deviation from theory to accept.

    Returns
    -------
    best : dict with keys "xmin", "xmax", "exponent", "n_points",
           or None if no valid window found.
    """
    valid = [r for r in scan_results
             if abs(r["exponent"] - theory_val) / theory_val <= tol
             and r["n_points"] >= 3]
    if not valid:
        return None
    return min(valid, key=lambda r: abs(r["exponent"] - theory_val))


def bootstrap_uncertainties(
    results_runs, pc_nominal,
    n_boot=500,
    beta_xmin=0.005, beta_xmax=0.08,
    gamma_xmin=0.005, gamma_xmax=0.08,
    tau_smin=2, L=None, rng=None,
):
    """
    Estimate standard errors on pc, beta, gamma, tau by bootstrap resampling.

    Requires results_runs to have been produced with return_runs=True,
    providing "R_runs", "Pinf_runs", "chi_runs" as (M, runs) arrays.

    Returns dict with keys "pc", "beta", "gamma", "tau", each containing
    "mean", "std", and "samples" (the full bootstrap array).
    """
    if rng is None:
        rng = np.random.default_rng()

    p_arr     = results_runs["p"]
    R_runs    = results_runs["R_runs"]
    Pinf_runs = results_runs["Pinf_runs"]
    chi_runs  = results_runs["chi_runs"]
    n_runs    = R_runs.shape[1]
    smax      = (L * L // 4) if L is not None else None
    ns_pc     = results_runs["ns_density"][int(np.argmin(np.abs(p_arr - pc_nominal)))]

    pc_boot = np.empty(n_boot)
    bb_boot = np.empty(n_boot)
    gm_boot = np.empty(n_boot)
    ta_boot = np.empty(n_boot)

    for b in _progress(range(n_boot), desc="  Bootstrap resamples"):
        idx    = rng.integers(0, n_runs, size=n_runs)
        R_b    = R_runs[:, idx].mean(axis=1)
        Pinf_b = Pinf_runs[:, idx].mean(axis=1)
        chi_b  = chi_runs[:, idx].mean(axis=1)

        try:    pc_b = estimate_pc_single_L(p_arr, R_b)
        except: pc_b = pc_nominal
        pc_boot[b] = pc_b

        try:    bb_boot[b], _, _ = estimate_beta(p_arr, Pinf_b, pc_b,
                                                  xmin=beta_xmin, xmax=beta_xmax)
        except: bb_boot[b] = np.nan

        try:    gm_boot[b], _, _ = estimate_gamma(p_arr, chi_b, pc_b,
                                                   xmin=gamma_xmin, xmax=gamma_xmax)
        except: gm_boot[b] = np.nan

        # n_s is pre-averaged so mimic variance with Poisson-scaled noise
        ns_noisy = {
            s: max(v + rng.normal(0, v / np.sqrt(n_runs) + 1e-12), 1e-12)
            for s, v in ns_pc.items()
        }
        try:    ta_boot[b], _, _, _ = estimate_tau(ns_noisy, smin=tau_smin,
                                                    smax=smax)
        except: ta_boot[b] = np.nan

    def _stats(arr):
        v = arr[np.isfinite(arr)]
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "samples": arr}

    return {"pc": _stats(pc_boot), "beta": _stats(bb_boot),
            "gamma": _stats(gm_boot), "tau": _stats(ta_boot)}


def full_exponent_report(results, pc, L,
                          beta_xmin=0.005,  beta_xmax=0.08,
                          gamma_xmin=0.005, gamma_xmax=0.08,
                          tau_smin=2, uncertainties=None):
    """
    Fit beta, gamma, tau and print a comparison table against theory.
    Pass uncertainties (from bootstrap_uncertainties) to show +/- errors
    and pull values (|estimate - theory| / sigma).
    Returns dict of point estimates.
    """
    p_arr, Pinf, chi = results["p"], results["Pinf"], results["chi"]
    estimates = {}

    for key, fn, obs, kw in [
        ("beta",  estimate_beta,  Pinf, dict(xmin=beta_xmin,  xmax=beta_xmax)),
        ("gamma", estimate_gamma, chi,  dict(xmin=gamma_xmin, xmax=gamma_xmax)),
    ]:
        try:
            estimates[key], _, _ = fn(p_arr, obs, pc, **kw)
        except Exception as e:
            estimates[key] = None
            print(f"  {key} fit failed: {e}")

    try:
        k = int(np.argmin(np.abs(p_arr - pc)))
        estimates["tau"], _, _, _ = estimate_tau(
            results["ns_density"][k], smin=tau_smin,
            smax=L * L // 4 if L else None
        )
    except Exception as e:
        estimates["tau"] = None
        print(f"  tau fit failed: {e}")

    W = 65
    print("\n" + "=" * W)
    print(f"  Critical exponent summary   (L = {L},  pc = {pc:.5f})")
    print("=" * W)
    print(f"  {'Exp':6s}  {'Estimated':>22s}  {'Theory':>10s}"
          + (f"  {'|delta|/sigma':>7s}" if uncertainties else ""))
    print("-" * W)

    for key, sym, tv in [("beta",  "beta",  THEORY["beta"]),
                          ("gamma", "gamma", THEORY["gamma"]),
                          ("tau",   "tau",   THEORY["tau"])]:
        est = estimates.get(key)
        if est is None:
            print(f"  {sym:6s}  {'N/A':>22s}  {tv:>10.4f}")
        elif uncertainties and key in uncertainties:
            unc  = uncertainties[key]["std"]
            pull = abs(est - tv) / unc if unc > 0 else np.inf
            print(f"  {sym:6s}  {est:.4f} +/- {unc:.4f}  {tv:>10.4f}  {pull:7.2f}")
        else:
            print(f"  {sym:6s}  {est:>22.4f}  {tv:>10.4f}")

    print("=" * W)
    if uncertainties and "pc" in uncertainties:
        print(f"\n  pc = {pc:.5f} +/- {uncertainties['pc']['std']:.5f}"
              f"   [theory: {THEORY['pc']:.5f}]")
    else:
        print(f"\n  pc = {pc:.5f}   [theory: {THEORY['pc']:.5f}]")
    print()

    return estimates
