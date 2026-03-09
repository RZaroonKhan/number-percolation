"""
algorithms.py
-------------
Percolation algorithms:
  - Hoshen-Kopelman (HK): labels clusters on a fixed lattice snapshot.
  - Newman-Ziff (NZ):     incrementally adds sites and tracks the largest
                          cluster as a function of occupation number n.

Both use the UnionFind class from union_find.py.
"""

import numpy as np
from union_find import UnionFind


# ---------------------------------------------------------------------------
# Hoshen-Kopelman
# ---------------------------------------------------------------------------

def hoshen_kopelman(lattice: np.ndarray):
    """
    Label clusters on a 2-D binary lattice using Hoshen-Kopelman.

    Parameters
    ----------
    lattice : (L, L) int array  — 1 = occupied, 0 = empty.

    Returns
    -------
    labels        : (L, L) int array — each occupied site carries its
                    cluster's root label (root + 1; 0 means empty).
    cluster_sizes : dict {root (int) -> size (int)}
    """
    L  = lattice.shape[0]
    N  = L * L
    uf = UnionFind(N)

    labels = np.zeros(N, dtype=np.int32)

    # Flatten lattice for fast indexing
    flat = lattice.ravel()

    for idx in range(N):
        if flat[idx] == 0:
            continue

        i, j = divmod(idx, L)
        root  = idx

        # Check up-neighbour
        if i > 0:
            up = idx - L
            if flat[up]:
                root = uf.union(root, up)

        # Check left-neighbour
        if j > 0:
            left = idx - 1
            if flat[left]:
                root = uf.union(root, left)

        labels[idx] = uf.find(root) + 1   # provisional; fixed in pass 2

    # ------------------------------------------------------------------
    # Second pass: flatten all labels to their true root and count sizes
    # ------------------------------------------------------------------
    cluster_sizes: dict[int, int] = {}

    for idx in range(N):
        if labels[idx] == 0:
            continue
        root = uf.find(idx)
        labels[idx] = root + 1
        cluster_sizes[root] = cluster_sizes.get(root, 0) + 1

    return labels.reshape(L, L), cluster_sizes


# ---------------------------------------------------------------------------
# Newman-Ziff
# ---------------------------------------------------------------------------

def newman_ziff(L: int, rng=None) -> np.ndarray:
    """
    Newman-Ziff algorithm: add sites one at a time in random order and
    record the largest cluster size after each addition.

    Parameters
    ----------
    L   : lattice side length.
    rng : numpy Generator (optional) for reproducibility.

    Returns
    -------
    largest_cluster : (N+1,) int array
        largest_cluster[n] = size of largest cluster after n sites occupied.
        largest_cluster[0] = 0 by convention.
    """
    if rng is None:
        rng = np.random.default_rng()

    N        = L * L
    uf       = UnionFind(N)
    occupied = np.zeros(N, dtype=bool)
    order    = rng.permutation(N)

    # Precompute neighbour lists as a flat array for speed
    # neigh_ptr[i] : start index in neigh_flat for site i
    # neigh_flat   : concatenated neighbour lists
    neigh_counts = np.zeros(N, dtype=np.int32)
    rows, cols   = np.divmod(np.arange(N), L)

    # Each site has at most 4 neighbours
    neigh_buf = np.full((N, 4), -1, dtype=np.int32)
    cnt       = np.zeros(N, dtype=np.int32)

    up    = np.where(rows > 0,     np.arange(N) - L, -1)
    down  = np.where(rows < L - 1, np.arange(N) + L, -1)
    left  = np.where(cols > 0,     np.arange(N) - 1, -1)
    right = np.where(cols < L - 1, np.arange(N) + 1, -1)

    for direction in (up, down, left, right):
        mask = direction >= 0
        neigh_buf[mask, cnt[mask]] = direction[mask]
        cnt[mask] += 1

    largest_cluster           = np.zeros(N + 1, dtype=np.int32)
    current_largest: int      = 0

    for n in range(1, N + 1):
        site          = int(order[n - 1])
        occupied[site] = True
        root           = site

        for k in range(cnt[site]):
            nb = int(neigh_buf[site, k])
            if occupied[nb]:
                root = uf.union(root, nb)

        root            = uf.find(root)
        sz              = int(uf.size[root])
        if sz > current_largest:
            current_largest = sz
        largest_cluster[n] = current_largest

    return largest_cluster
