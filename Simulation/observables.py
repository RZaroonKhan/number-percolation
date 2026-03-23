"""
observables.py
--------------
Per-configuration observables computed from Hoshen-Kopelman output:

  spanning_info()            — which clusters span and which boundaries.
  percolation_strength()     — P∞: fraction of sites in spanning cluster(s).
  mean_finite_cluster_size() — χ: susceptibility proxy.
  cluster_size_distribution()— n_s counts for one configuration.
  clusters_per_site()        — number of clusters / N.
  generate_lattice()         — convenience lattice generator.

Monte Carlo averaging over p is done in simulation.py.
"""

import numpy as np


# Spanning detection

def spanning_info(labels: np.ndarray):
    """
    Determine which clusters span the lattice (left→right and/or top→bottom).

    Parameters
    ----------
    labels : (L, L) int array from HK (0 = empty, otherwise root+1).

    Returns
    -------
    spans_lr       : bool
    spans_tb       : bool
    spanning_roots : set[int]  — 0-based root ids that span in either direction.
    """
    L = labels.shape[0]

    left_col  = labels[:, 0]
    right_col = labels[:, L - 1]
    top_row   = labels[0, :]
    bot_row   = labels[L - 1, :]

    # Roots touching each boundary (exclude 0 = empty)
    left_roots  = set(left_col[left_col   > 0] - 1)
    right_roots = set(right_col[right_col > 0] - 1)
    top_roots   = set(top_row[top_row     > 0] - 1)
    bot_roots   = set(bot_row[bot_row     > 0] - 1)

    lr_roots = left_roots & right_roots
    tb_roots = top_roots  & bot_roots

    spanning_roots = lr_roots | tb_roots
    spans_lr       = len(lr_roots) > 0
    spans_tb       = len(tb_roots) > 0

    return spans_lr, spans_tb, spanning_roots


# Percolation strength P∞

def percolation_strength(labels: np.ndarray,
                         cluster_sizes: dict,
                         mode: str = "LR") -> float:
    """
    Fraction of lattice sites belonging to spanning cluster(s).

    mode : "LR" | "TB" | "ANY"
    """
    L = labels.shape[0]
    N = L * L

    spans_lr, spans_tb, spanning_roots = spanning_info(labels)

    if mode == "LR"  and not spans_lr:           return 0.0
    if mode == "TB"  and not spans_tb:           return 0.0
    if mode == "ANY" and not (spans_lr or spans_tb): return 0.0

    total = sum(cluster_sizes.get(r, 0) for r in spanning_roots)
    return total / N


# Mean finite cluster size χ

def mean_finite_cluster_size(labels: np.ndarray,
                              cluster_sizes: dict,
                              mode: str = "LR") -> float:
    """
    χ = Σ' s² n_s / Σ' s n_s

    The prime means the sum excludes spanning clusters, which is the
    physically correct definition (they are absorbed into P∞).

    Returns 0.0 if there are no finite clusters.
    """
    spans_lr, spans_tb, spanning_roots = spanning_info(labels)

    if mode == "LR"  and not spans_lr:           spanning_roots = set()
    if mode == "TB"  and not spans_tb:           spanning_roots = set()
    if mode == "ANY" and not (spans_lr or spans_tb): spanning_roots = set()

    finite = np.array([s for r, s in cluster_sizes.items()
                       if r not in spanning_roots], dtype=np.float64)

    if finite.size == 0:
        return 0.0

    return float((finite ** 2).sum() / finite.sum())


# Cluster size distribution n_s

def cluster_size_distribution(cluster_sizes: dict,
                               exclude_roots=None) -> dict:
    """
    Return {s: count_of_clusters_of_size_s}, optionally excluding certain roots.
    """
    if exclude_roots is None:
        exclude_roots = set()
    counts: dict[int, int] = {}
    for r, s in cluster_sizes.items():
        if r not in exclude_roots:
            counts[s] = counts.get(s, 0) + 1
    return counts


def cluster_size_density_per_site(size_counts: dict, L: int) -> dict:
    """Convert {s: count} → {s: count / N}."""
    N = L * L
    return {s: c / N for s, c in size_counts.items()}


# Clusters per site

def clusters_per_site(cluster_sizes: dict, L: int) -> float:
    return len(cluster_sizes) / (L * L)


# Lattice generator

def generate_lattice(L: int, p: float, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    return (rng.random((L, L)) < p).astype(np.int8)
