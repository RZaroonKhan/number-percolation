"""
generate_lattice_figure.py
--------------------------
Produces two publication-quality figures:

  lattice_comparison.png  — three panels at p < pc, p ≈ pc, p > pc
                            on a 200x200 lattice (original figure).

  lattice_sizes.png       — five panels all at p ≈ pc, showing lattices
                            of increasing size (L = 10, 25, 50, 100, 200)
                            so the scale-free cluster structure is visible
                            across different system sizes.

Standalone script — no imports from the rest of the project.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class _UF:
    def __init__(self, n):
        self.p = np.arange(n, dtype=np.int32)
        self.s = np.ones(n,  dtype=np.int32)

    def find(self, x):
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.s[ra] < self.s[rb]:
            ra, rb = rb, ra
        self.p[rb]  = ra
        self.s[ra] += self.s[rb]
        return ra


# ---------------------------------------------------------------------------
# HK cluster labelling
# ---------------------------------------------------------------------------

def _hk(lattice):
    L    = lattice.shape[0]
    flat = lattice.ravel()
    uf   = _UF(L * L)
    labs = np.zeros(L * L, dtype=np.int32)

    for idx in range(L * L):
        if not flat[idx]:
            continue
        i, j = divmod(idx, L)
        root = idx
        if i > 0 and flat[idx - L]: root = uf.union(root, idx - L)
        if j > 0 and flat[idx - 1]: root = uf.union(root, idx - 1)
        labs[idx] = uf.find(root) + 1

    sizes = {}
    for idx in range(L * L):
        if labs[idx] == 0:
            continue
        r = uf.find(idx)
        labs[idx] = r + 1
        sizes[r]  = sizes.get(r, 0) + 1

    return labs.reshape(L, L), sizes


def _spanning_roots(labels):
    L  = labels.shape[0]
    lr = set(labels[:, 0][labels[:, 0] > 0] - 1) & \
         set(labels[:, L-1][labels[:, L-1] > 0] - 1)
    tb = set(labels[0, :][labels[0, :] > 0] - 1) & \
         set(labels[L-1, :][labels[L-1, :] > 0] - 1)
    return lr | tb


def _make_image(lattice, labels, span_roots):
    img = np.ones((*lattice.shape, 3), dtype=np.float32)
    img[lattice == 1] = [0.15, 0.15, 0.15]
    for r in span_roots:
        img[labels == r + 1] = [0.85, 0.10, 0.10]
    return img


def _render_panel(ax, L, p, rng, show_stats=True):
    """Generate a lattice and render it onto ax. Returns span count."""
    lattice        = (rng.random((L, L)) < p).astype(np.int8)
    labels, sizes  = _hk(lattice)
    span           = _spanning_roots(labels)
    img            = _make_image(lattice, labels, span)

    # Use nearest-neighbour for small lattices so pixels are crisp,
    # bilinear for large ones so they don't look jagged when scaled down
    interp = "nearest" if L <= 50 else "nearest"
    ax.imshow(img, origin="upper", interpolation=interp,
              extent=[0, L, 0, L])
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("#888888")

    if show_stats:
        spans_str = f"{len(span)} spanning" if span else "no spanning"
        ax.set_xlabel(
            f"occ. {lattice.sum()/L**2:.3f}  |  "
            f"{len(sizes)} clusters  |  {spans_str}",
            fontsize=7.5, color="#444444",
        )
    return span


def _legend(fig):
    handles = [
        mpatches.Patch(color="#262626", label="Occupied site"),
        mpatches.Patch(color="white",   label="Empty site",
                       ec="#aaaaaa", linewidth=0.8),
        mpatches.Patch(color="#d91919", label="Spanning cluster"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, frameon=True, framealpha=0.95,
               edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.01))


# ---------------------------------------------------------------------------
# Figure 1 — three p values on a 200x200 lattice
# ---------------------------------------------------------------------------

def make_lattice_figure(L=200, pc=0.5927, seed=42,
                        p_below=0.50, p_above=0.65,
                        save_path="lattice_comparison.png"):
    """
    Three-panel figure: p < pc, p ≈ pc, p > pc on a single lattice size.
    """
    rng    = np.random.default_rng(seed)
    p_vals = [p_below, pc, p_above]
    titles = [
        f"$p = {p_below}$   (below $p_c$)",
        f"$p = p_c \\approx {pc}$   (critical)",
        f"$p = {p_above}$   (above $p_c$)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    fig.patch.set_facecolor("white")

    for ax, p, title in zip(axes, p_vals, titles):
        _render_panel(ax, L, p, rng)
        ax.set_title(title, fontsize=13, pad=8)

    _legend(fig)
    fig.suptitle(f"Site percolation on a {L}×{L} square lattice",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — increasing lattice sizes, all at p = pc
# ---------------------------------------------------------------------------

def make_sizes_figure(L_values=(10, 25, 50, 100, 200),
                      pc=0.5927, seed=42,
                      save_path="lattice_sizes.png"):
    """
    Five-panel figure showing lattices of increasing size, all at p = pc.

    This illustrates finite-size effects: at small L the lattice is too
    small to reliably host a spanning cluster, while at large L the
    scale-free cluster structure characteristic of the critical point
    becomes clearly visible.
    """
    rng = np.random.default_rng(seed)
    n   = len(L_values)

    # Make panels equal in display size regardless of L
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.8))
    fig.patch.set_facecolor("white")

    for ax, L in zip(axes, L_values):
        span = _render_panel(ax, L, pc, rng, show_stats=True)
        ax.set_title(f"$L = {L}$", fontsize=13, pad=8)

    _legend(fig)
    fig.suptitle(
        f"Critical lattices ($p = p_c \\approx {pc}$) at increasing system size",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Run both figures
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    make_lattice_figure(
        L=200, pc=0.5927, seed=42,
        p_below=0.50, p_above=0.65,
        save_path="lattice_comparison.png",
    )
    make_sizes_figure(
        L_values=(10, 25, 50, 100, 200),
        pc=0.5927, seed=42,
        save_path="lattice_sizes.png",
    )
