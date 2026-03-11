"""
union_find.py
-------------
Union-Find (Disjoint Set Union) data structure with union-by-size.

Path compression is applied in find() to keep the tree flat and
amortise future lookups. This is the core engine used by both
Hoshen-Kopelman and Newman-Ziff algorithms.
"""

import numpy as np


class UnionFind:
    """
    Union-Find with union-by-size and full path compression.

    Parameters
    ----------
    n : int
        Number of elements (sites on the lattice, indexed 0..n-1).
    """

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.size   = np.ones(n, dtype=np.int32)

    def find(self, x: int) -> int:
        """Return root of x with full path compression (iterative)."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression: point every node on the path directly to root
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> int:
        """
        Merge the sets containing a and b (union by size).
        Returns the root of the merged set.
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        # Attach smaller tree under larger
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra]  += self.size[rb]
        return ra
