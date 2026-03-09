"""
simulation.py
-------------
Parallelised Monte Carlo sweep over occupation probabilities p.

estimate_observables_over_p()
    Runs `runs` HK realisations at each p in p_values using all
    available CPU cores (joblib). Returns averaged observables and,
    when return_runs=True, the per-run arrays needed for bootstrap
    uncertainty estimation.
"""

import numpy as np
from joblib import Parallel, delayed

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

from algorithms  import hoshen_kopelman
from observables import (
    generate_lattice,
    spanning_info,
    percolation_strength,
    mean_finite_cluster_size,
    cluster_size_distribution,
    clusters_per_site,
)


# ---------------------------------------------------------------------------
# Single-run worker (top-level for pickling)
# ---------------------------------------------------------------------------

def _single_run(L: int, p: float, span_mode: str, seed: int) -> dict:
    """Run one HK realisation and return all scalar observables."""
    rng     = np.random.default_rng(seed)
    lattice = generate_lattice(L, p, rng=rng)
    labels, cluster_sizes = hoshen_kopelman(lattice)

    spans_lr, spans_tb, spanning_roots = spanning_info(labels)

    if span_mode == "LR":
        spans = spans_lr
    elif span_mode == "TB":
        spans = spans_tb
    elif span_mode == "ANY":
        spans = spans_lr or spans_tb
    else:
        raise ValueError(f"Unknown span_mode: {span_mode!r}")

    pinf = percolation_strength(labels, cluster_sizes, mode=span_mode)
    chi  = mean_finite_cluster_size(labels, cluster_sizes, mode=span_mode)
    nc   = clusters_per_site(cluster_sizes, L)
    ns   = cluster_size_distribution(cluster_sizes, exclude_roots=spanning_roots)

    return {
        "span":            int(spans),
        "pinf":            float(pinf),
        "chi":             float(chi),
        "nc":              float(nc),
        "ns":              ns,
        "largest_cluster": int(max(cluster_sizes.values())) if cluster_sizes else 0,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_observables_over_p(
    L:           int,
    p_values:    np.ndarray,
    runs:        int,
    span_mode:   str  = "LR",
    n_jobs:      int  = -1,
    base_seed:   int  = 0,
    return_runs: bool = False,
    desc:        str  = None,
) -> dict:
    """
    Average observables over `runs` realisations at each p in p_values.

    Parameters
    ----------
    L            : lattice side length.
    p_values     : 1-D array of occupation probabilities.
    runs         : number of independent realisations per p value.
    span_mode    : "LR", "TB", or "ANY".
    n_jobs       : joblib workers (-1 = all cores).
    base_seed    : base random seed.
    return_runs  : if True, also return per-run arrays for bootstrap
                   (keys: "R_runs", "Pinf_runs", "chi_runs").
    desc         : label shown on the progress bar (auto-set if None).

    Returns
    -------
    dict with keys:
        "p"          : (M,) float
        "R"          : (M,) spanning probability
        "Pinf"       : (M,) percolation strength P∞
        "chi"        : (M,) mean finite cluster size χ
        "nc_per_site": (M,) clusters per site
        "ns_density" : list of M dicts {s: density_per_site}
        [if return_runs=True]
        "R_runs"     : (M, runs) bool
        "Pinf_runs"  : (M, runs) float
        "chi_runs"   : (M, runs) float
    """
    M   = len(p_values)
    N   = L * L
    bar_desc = desc or f"L={L}, runs={runs}"

    R           = np.zeros(M)
    Pinf        = np.zeros(M)
    Chi         = np.zeros(M)
    nc_per_site = np.zeros(M)
    Smax        = np.zeros(M)   # mean largest cluster size at each p
    ns_density  = [dict() for _ in range(M)]

    # Per-run storage (only filled if return_runs=True)
    R_runs    = np.zeros((M, runs), dtype=np.float32) if return_runs else None
    Pinf_runs = np.zeros((M, runs), dtype=np.float32) if return_runs else None
    chi_runs  = np.zeros((M, runs), dtype=np.float32) if return_runs else None

    if _TQDM:
        pbar_ctx = tqdm(total=M, desc=bar_desc, unit="p-point",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} p-points "
                                   "[{elapsed}<{remaining}, {rate_fmt}]")
    else:
        from contextlib import nullcontext
        pbar_ctx = nullcontext()

    with pbar_ctx as pbar:
        for k, p in enumerate(p_values):
            seeds = base_seed + k * runs + np.arange(runs, dtype=int)

            raw = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_single_run)(L, p, span_mode, int(s)) for s in seeds
            )

            span_arr  = np.array([r["span"]            for r in raw], dtype=np.float32)
            pinf_arr  = np.array([r["pinf"]            for r in raw], dtype=np.float32)
            chi_arr   = np.array([r["chi"]             for r in raw], dtype=np.float32)
            nc_arr    = np.array([r["nc"]              for r in raw], dtype=np.float32)
            smax_arr  = np.array([r["largest_cluster"] for r in raw], dtype=np.float32)

            R[k]           = span_arr.mean()
            Pinf[k]        = pinf_arr.mean()
            Chi[k]         = chi_arr.mean()
            nc_per_site[k] = nc_arr.mean()
            Smax[k]        = smax_arr.mean()

            if return_runs:
                R_runs[k]    = span_arr
                Pinf_runs[k] = pinf_arr
                chi_runs[k]  = chi_arr

            ns_total: dict[int, int] = {}
            for r in raw:
                for s, c in r["ns"].items():
                    ns_total[s] = ns_total.get(s, 0) + c
            for s, c in ns_total.items():
                ns_density[k][s] = c / (runs * N)

            if _TQDM and pbar is not None:
                pbar.set_postfix(p=f"{p:.4f}", R=f"{R[k]:.3f}")
                pbar.update(1)

    out = {
        "p":           np.array(p_values, dtype=float),
        "R":           R,
        "Pinf":        Pinf,
        "chi":         Chi,
        "nc_per_site": nc_per_site,
        "Smax":        Smax,
        "ns_density":  ns_density,
    }

    if return_runs:
        out["R_runs"]    = R_runs
        out["Pinf_runs"] = Pinf_runs
        out["chi_runs"]  = chi_runs

    return out
