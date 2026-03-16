"""
percolation_cnn.py
------------------
Standalone ML script for 2D site percolation phase detection.

Scientific question
-------------------
Can a convolutional neural network learn the percolation transition
directly from raw lattice configurations, and can its output be used
as an independent estimator of the critical point?

What the network does
---------------------
Input  : binary L×L lattice (occupied = 1, empty = 0)
Target : 0 or 1 from direct left-to-right spanning test
Output : P_CNN(spanning | configuration) — probability of LR spanning

The mean network output <y>(p, L) averaged over many configurations
behaves like a finite-size crossover curve, directly analogous to the
spanning probability R(p, L). The crossing point <y> = 0.5 gives an
ML estimate of the pseudocritical point pc^ML(L).

Training data strategy
----------------------
Configurations are drawn from three p-regions with emphasis on the
critical region, and the dataset is explicitly balanced to 50/50
spanning vs non-spanning labels. This forces the network to learn
geometric features of the transition rather than exploiting trivial
occupation density differences.

  below    : p in [0.55, 0.585]   — mostly non-spanning
  critical : p in [0.585, 0.600]  — mixed, many ambiguous cases
  above    : p in [0.600, 0.635]  — mostly spanning

Separate CNNs are trained for each lattice size L.

Figures produced
----------------
A. cnn_output_vs_p.png   — <y>(p,L) vs p, multi-L
B. cnn_pc_vs_L.png       — pc^ML(L) vs L^(-1/nu), FSS comparison
C. cnn_collapse.png      — approximate collapse of <y> vs (p-pc)L^(1/nu)
D. cnn_vs_R.png          — CNN <y> vs true R(p,L) overlaid
E. cnn_saliency.png      — saliency maps at p<pc, p=pc, p>pc
F. cnn_training.png      — training curves (supplementary)

Dependencies
------------
torch, numpy, matplotlib, scipy, tqdm
No imports from your main project — fully standalone.

Usage
-----
    python3 percolation_cnn.py

Checkpointing
-------------
Trained model saved to CKPT_DIR/model_L{L}.pt
Evaluation data saved to CKPT_DIR/eval_L{L}.npz
Delete files to force re-run, or set FORCE_RETRAIN = True.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize    import brentq

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm as _tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

# =============================================================================
# Configuration
# =============================================================================

PC_FSS  = 0.59296           # your FSS pc — used for reference lines
NU      = 4/3               # exact theoretical value

# L=32 added for better FSS extrapolation in Figure B
L_VALUES = [32, 64, 128, 256]

# Training data — balanced labels, critical region emphasis
# Regions chosen to straddle pc = 0.59296
P_BELOW    = (0.550, 0.585)  # mostly non-spanning
P_CRITICAL = (0.585, 0.600)  # mixed — hardest cases near pc
P_ABOVE    = (0.600, 0.635)  # mostly spanning

# Fraction of candidates drawn from each region
FRAC_BELOW    = 0.25
FRAC_CRITICAL = 0.50         # over-sample critical region
FRAC_ABOVE    = 0.25

N_TRAIN   = 8000             # total balanced training configurations per L
N_VAL     = 2000             # total balanced validation configurations per L

# Evaluation — fine grid around pc
P_EVAL    = np.linspace(0.560, 0.630, 50)
N_EVAL    = 500              # configurations per p value per L

# Training hyperparameters
N_EPOCHS     = 30
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4

BASE_SEED     = 42
FIG_DIR       = "figures_ml"
CKPT_DIR      = "checkpoints_ml"
FORCE_RETRAIN = False           # set True after any change to training data
                                # or delete checkpoints_ml/ to force re-run

# Validate fractions sum to 1
if not np.isclose(FRAC_BELOW + FRAC_CRITICAL + FRAC_ABOVE, 1.0):
    raise ValueError(
        f"FRAC_BELOW + FRAC_CRITICAL + FRAC_ABOVE must equal 1.0, "
        f"got {FRAC_BELOW + FRAC_CRITICAL + FRAC_ABOVE:.6f}"
    )

# Line styles — consistent with percolation_figures.py
LINE_STYLES = {
    32:  (0, (1, 3)),
    64:  (0, (1, 1)),
    128: (0, (3, 1, 1, 1)),
    256: (0, (5, 2)),
    512: "-",
}
LINE_LABELS = {L: f"$L = {L}$" for L in L_VALUES}

# Device — MPS on Apple Silicon, CUDA, or CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# =============================================================================
# Lattice generation and LR spanning test — fully self-contained
# =============================================================================

def generate_lattice(L, p, rng):
    """Generate a random binary L×L lattice with occupation probability p."""
    return (rng.random((L, L)) < p).astype(np.float32)


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent, rank, x, y):
    rx, ry = _find(parent, x), _find(parent, y)
    if rx == ry:
        return
    if rank[rx] < rank[ry]:
        rx, ry = ry, rx
    parent[ry] = rx
    if rank[rx] == rank[ry]:
        rank[rx] += 1


def spans_lr(lattice):
    """
    Return True if any cluster connects the left and right boundaries.

    This is the target label for all training and evaluation.
    The network is trained exclusively on left-to-right spanning labels.
    Union-find with path compression and union by rank — O(L^2 alpha(L^2)).
    """
    L      = lattice.shape[0]
    parent = list(range(L * L))
    rank   = [0] * (L * L)

    for i in range(L):
        for j in range(L):
            if lattice[i, j] == 0:
                continue
            idx = i * L + j
            if j + 1 < L and lattice[i, j+1]:
                _union(parent, rank, idx, i*L + j+1)
            if i + 1 < L and lattice[i+1, j]:
                _union(parent, rank, idx, (i+1)*L + j)

    left_roots  = {_find(parent, i*L)       for i in range(L)
                   if lattice[i, 0]}
    right_roots = {_find(parent, i*L + L-1) for i in range(L)
                   if lattice[i, L-1]}
    return bool(left_roots & right_roots)


# =============================================================================
# Balanced dataset with critical region emphasis
# =============================================================================

def _sample_one_p(rng):
    """
    Draw one p-value from the three training regions using categorical
    region selection. Each call independently picks a region with
    probability FRAC_BELOW / FRAC_CRITICAL / FRAC_ABOVE, then draws
    uniformly within that region.

    This is the correct single-draw version — the previous batch
    version broke for n=1 because int(1 * 0.25) = 0.
    """
    u = rng.random()
    if u < FRAC_BELOW:
        return float(rng.uniform(*P_BELOW))
    elif u < FRAC_BELOW + FRAC_CRITICAL:
        return float(rng.uniform(*P_CRITICAL))
    else:
        return float(rng.uniform(*P_ABOVE))


class PercolationDataset(Dataset):
    """
    Balanced dataset: exactly 50% spanning, 50% non-spanning labels.
    Configurations are drawn with emphasis on the critical region.

    Strategy
    --------
    Generate candidate configurations from the stratified p distribution
    and accept them into the spanning or non-spanning pool until both
    pools reach n_samples // 2. This guarantees label balance while
    keeping most samples near pc.
    """
    def __init__(self, L, n_samples, seed):
        self.L = L
        target_each = n_samples // 2    # target per class

        rng      = np.random.default_rng(seed)
        spanning     = []   # (lattice, label=1) pairs
        non_spanning = []   # (lattice, label=0) pairs

        # Generate candidates until both classes are full
        # Safety limit prevents infinite loop in degenerate cases
        max_attempts = n_samples * 20
        attempts     = 0

        while (len(spanning) < target_each or
               len(non_spanning) < target_each):
            if attempts >= max_attempts:
                print(f"  Warning: reached max_attempts for L={L}, "
                      f"spanning={len(spanning)}, "
                      f"non_spanning={len(non_spanning)}")
                break
            attempts += 1

            # Draw one p from the stratified distribution
            p   = _sample_one_p(rng)
            lat = generate_lattice(L, p, rng)
            s   = spans_lr(lat)

            if s and len(spanning) < target_each:
                spanning.append(lat)
            elif not s and len(non_spanning) < target_each:
                non_spanning.append(lat)

        # Combine and shuffle
        all_lats    = spanning[:target_each] + non_spanning[:target_each]
        all_labels  = [1.0] * len(spanning[:target_each]) + \
                      [0.0] * len(non_spanning[:target_each])

        idx = np.random.default_rng(seed + 1).permutation(len(all_lats))
        self.lattices = np.array([all_lats[i]   for i in idx],
                                 dtype=np.float32)[:, None, :, :]
        self.labels   = np.array([all_labels[i] for i in idx],
                                 dtype=np.float32)

        n_actual = len(self.labels)
        n_span   = int(self.labels.sum())

        if n_actual < n_samples:
            raise RuntimeError(
                f"Could not build full balanced dataset for L={L}. "
                f"Requested {n_samples}, got {n_actual}. "
                f"Try increasing max_attempts or widening the p-regions."
            )

        print(f"    L={L:4d}: {n_actual} samples  "
              f"({n_span} spanning, {n_actual-n_span} non-spanning)  "
              f"balance={n_span/n_actual:.3f}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.lattices[idx]),
                torch.tensor(self.labels[idx]))


# =============================================================================
# CNN architecture
# =============================================================================

class PercolationCNN(nn.Module):
    """
    CNN for left-to-right spanning cluster classification.

    Architecture: three conv blocks with batch norm and ReLU, followed
    by global average pooling and a sigmoid output. Global average
    pooling makes the architecture size-agnostic — the same model
    structure is used for all L values, although separate models are
    trained per L.

    Input  : (batch, 1, L, L) binary lattice
    Output : (batch,) probability of left-to-right spanning
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.features(x))).squeeze(1)


# =============================================================================
# Training
# =============================================================================

def train_model(L, seed_offset=0):
    """Train CNN for lattice size L. Returns (model, history)."""
    ckpt = os.path.join(CKPT_DIR, f"model_L{L}.pt")
    hist = os.path.join(CKPT_DIR, f"history_L{L}.npz")

    if not FORCE_RETRAIN and os.path.exists(ckpt) and os.path.exists(hist):
        print(f"  Loading trained model L={L} from checkpoint ...")
        model = PercolationCNN().to(DEVICE)
        # Fix issue 4: compatible torch.load without weights_only kwarg issues
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        h       = np.load(hist)
        history = {k: h[k].tolist() for k in h.files}
        return model, history

    print(f"\n  Training CNN for L={L} ...")
    print(f"  Generating balanced training dataset ...")
    train_ds = PercolationDataset(L, N_TRAIN, seed=BASE_SEED + seed_offset)
    val_ds   = PercolationDataset(L, N_VAL,
                                  seed=BASE_SEED + seed_offset + 99999)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

    model     = PercolationCNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01)

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    epochs = range(N_EPOCHS)
    if _TQDM:
        epochs = _tqdm(epochs, desc=f"  L={L:4d} training",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                  "[{elapsed}<{remaining}]")

    for epoch in epochs:
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * len(y)
            t_correct += ((out > 0.5) == y.bool()).sum().item()
            t_total   += len(y)
        scheduler.step()

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out   = model(x)
                loss  = criterion(out, y)
                v_loss    += loss.item() * len(y)
                v_correct += ((out > 0.5) == y.bool()).sum().item()
                v_total   += len(y)

        history["train_loss"].append(t_loss / t_total)
        history["val_loss"].append(v_loss   / v_total)
        history["train_acc"].append(t_correct / t_total)
        history["val_acc"].append(v_correct   / v_total)

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    np.savez(hist, **{k: np.array(v) for k, v in history.items()})
    print(f"  Saved: {ckpt}")
    print(f"  Final val acc  = {history['val_acc'][-1]:.4f}  "
          f"val loss = {history['val_loss'][-1]:.4f}")
    return model, history


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(model, L, seed_offset=0):
    """Compute <y>(p,L) and true R(p,L) over P_EVAL."""
    ckpt = os.path.join(CKPT_DIR, f"eval_L{L}.npz")
    if not FORCE_RETRAIN and os.path.exists(ckpt):
        print(f"  Loading evaluation data L={L} from checkpoint ...")
        d = np.load(ckpt)
        return {k: d[k] for k in d.files}

    print(f"  Evaluating L={L} over {len(P_EVAL)} p values, "
          f"{N_EVAL} configs each ...")
    model.eval()

    y_mean = np.zeros(len(P_EVAL))
    y_std  = np.zeros(len(P_EVAL))
    R      = np.zeros(len(P_EVAL))
    rng    = np.random.default_rng(BASE_SEED + seed_offset + 500000)

    p_iter = _tqdm(enumerate(P_EVAL),
                   total=len(P_EVAL),
                   desc=f"  Eval   L={L:4d}",
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                              "[{elapsed}<{remaining}]") \
             if _TQDM else enumerate(P_EVAL)

    for j, p in p_iter:
        lats   = np.zeros((N_EVAL, 1, L, L), dtype=np.float32)
        labels = np.zeros(N_EVAL)
        for k in range(N_EVAL):
            lat       = generate_lattice(L, p, rng)
            lats[k,0] = lat
            labels[k] = float(spans_lr(lat))

        with torch.no_grad():
            outs = model(torch.from_numpy(lats).to(DEVICE)).cpu().numpy()

        y_mean[j] = outs.mean()
        y_std[j]  = outs.std()
        R[j]      = labels.mean()

    pc_ml = _find_crossing(P_EVAL, y_mean)

    result = dict(p=P_EVAL, y_mean=y_mean, y_std=y_std,
                  R=R, pc_ml=np.array(pc_ml))
    os.makedirs(CKPT_DIR, exist_ok=True)
    np.savez(ckpt, **result)
    print(f"  pc^ML(L={L}) = {pc_ml:.5f}  "
          f"(FSS pc = {PC_FSS:.5f}  "
          f"delta = {pc_ml - PC_FSS:+.5f})")
    return result


def _find_crossing(p_arr, y_arr, threshold=0.5):
    """Find p where y first crosses threshold via root finding."""
    try:
        f     = interp1d(p_arr, y_arr - threshold, kind="linear")
        signs = np.sign(y_arr - threshold)
        idx   = np.where(np.diff(signs) != 0)[0]
        if len(idx) == 0:
            return np.nan
        crossings = []
        for i in idx:
            try:
                crossings.append(brentq(f, p_arr[i], p_arr[i+1]))
            except Exception:
                crossings.append((p_arr[i] + p_arr[i+1]) / 2.0)
        return float(min(crossings, key=lambda x: abs(x - PC_FSS)))
    except Exception:
        return np.nan


# =============================================================================
# Saliency maps
# =============================================================================

def compute_saliency(model, L, p, n_samples=10, seed=0):
    """
    Gradient-based saliency: mean |d output / d input| over n_samples.
    Shows which lattice sites most influence the network's decision.
    Note: saliency reflects local gradient sensitivity, not necessarily
    the spanning path — interpret as exploratory rather than definitive.
    """
    model.eval()
    rng  = np.random.default_rng(seed)
    maps = np.zeros((L, L))

    for _ in range(n_samples):
        lat = generate_lattice(L, p, rng)
        x   = torch.from_numpy(lat[None, None]).to(DEVICE)
        x.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(x).backward()
        maps += x.grad.abs().squeeze().cpu().numpy()

    return maps / n_samples


# =============================================================================
# Figure A: <y>(p, L) vs p, multi-L
# =============================================================================

def fig_cnn_output_vs_p(eval_data):
    fig, ax = plt.subplots(figsize=(7, 5))

    for L in L_VALUES:
        d = eval_data[L]
        ax.plot(d["p"], d["y_mean"],
                linestyle=LINE_STYLES.get(L, "-"),
                color="black", lw=1.3,
                label=LINE_LABELS.get(L, f"L={L}"))

    ax.axvline(PC_FSS, color="black", lw=0.9, ls=":",
               label=rf"$p_c^{{\rm FSS}} = {PC_FSS:.5f}$")
    ax.axhline(0.5, color="black", lw=0.7, ls="--", alpha=0.5,
               label=r"$\langle y \rangle = 0.5$")

    ax.set_xlabel(r"$p$", fontsize=13)
    ax.set_ylabel(r"$\langle y \rangle (p;\,L)$", fontsize=13)
    ax.set_xlim(P_EVAL.min(), P_EVAL.max())
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)
    ax.set_title("CNN mean output vs occupation probability", fontsize=10)
    fig.tight_layout()
    return fig


# =============================================================================
# Figure B: pc^ML(L) vs L^(-1/nu)
# =============================================================================

def fig_pc_vs_L(eval_data):
    fig, ax = plt.subplots(figsize=(6, 5))

    L_arr  = np.array([L for L in L_VALUES
                       if np.isfinite(float(eval_data[L]["pc_ml"]))],
                      dtype=float)
    pc_arr = np.array([float(eval_data[L]["pc_ml"]) for L in L_arr])

    if len(L_arr) < 2:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center")
        fig.tight_layout()
        return fig

    x = L_arr ** (-1.0 / NU)

    ax.plot(x, pc_arr, "o", color="black", ms=8, zorder=5,
            label=r"$p_c^{\rm ML}(L)$")

    for xi, pci, L in zip(x, pc_arr, L_arr):
        ax.annotate(f"$L={int(L)}$", (xi, pci),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)

    # Linear fit: pc(L) = pc_inf + a * L^(-1/nu)
    try:
        coeffs = np.polyfit(x, pc_arr, 1)
        x_fit  = np.linspace(0, x.max() * 1.1, 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "--", color="black",
                lw=1.2,
                label=rf"Linear fit: $p_c^\infty = {coeffs[1]:.5f}$")
        ax.plot(0, coeffs[1], "*", color="black", ms=12, zorder=6,
                label=r"Extrapolated $p_c^\infty$")
    except Exception:
        pass

    ax.axhline(PC_FSS, color="black", lw=0.9, ls=":",
               label=rf"$p_c^{{\rm FSS}} = {PC_FSS:.5f}$")

    ax.set_xlabel(r"$L^{-1/\nu}$", fontsize=13)
    ax.set_ylabel(r"$p_c^{\rm ML}(L)$", fontsize=13)
    ax.set_xlim(-0.002, x.max() * 1.15)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)
    ax.set_title(r"ML pseudocritical point vs $L^{-1/\nu}$", fontsize=10)
    fig.tight_layout()
    return fig


# =============================================================================
# Figure C: Approximate data collapse <y> vs (p-pc) L^(1/nu)
# =============================================================================

def fig_cnn_collapse(eval_data):
    """
    Approximate collapse of CNN output.
    Since <y> is dimensionless the rescaled x-axis (p-pc)*L^(1/nu) is
    sufficient. Note: this is an empirical collapse, not strict FSS —
    the CNN output is not formally guaranteed to obey a scaling ansatz.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for L in L_VALUES:
        d = eval_data[L]
        x = (d["p"] - PC_FSS) * (float(L) ** (1.0 / NU))
        ax.plot(x, d["y_mean"],
                linestyle=LINE_STYLES.get(L, "-"),
                color="black", lw=1.3,
                label=LINE_LABELS.get(L, f"L={L}"))

    ax.axvline(0,   color="black", lw=0.8, ls=":", alpha=0.6)
    ax.axhline(0.5, color="black", lw=0.7, ls="--", alpha=0.5)

    ax.set_xlabel(r"$(p - p_c)\, L^{1/\nu}$", fontsize=13)
    ax.set_ylabel(r"$\langle y \rangle$", fontsize=13)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)
    ax.set_title("Approximate data collapse of CNN output", fontsize=10)
    fig.tight_layout()
    return fig


# =============================================================================
# Figure D: CNN <y> vs true R(p, L) overlaid
# =============================================================================

def fig_cnn_vs_R(eval_data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(r"CNN output $\langle y \rangle$ vs true spanning "
                 r"probability $R(p;\,L)$", fontsize=11)

    # Left: overlay for smallest and largest L
    ax = axes[0]
    for L in [L_VALUES[0], L_VALUES[-1]]:
        d  = eval_data[L]
        ls = LINE_STYLES.get(L, "-")
        ax.plot(d["p"], d["y_mean"], linestyle=ls, color="black",
                lw=1.3, label=rf"CNN $\langle y \rangle$, $L={L}$")
        ax.plot(d["p"], d["R"],      linestyle=ls, color="grey",
                lw=1.3, alpha=0.7,  label=rf"True $R$, $L={L}$")
    ax.axvline(PC_FSS, color="black", lw=0.8, ls=":")
    ax.axhline(0.5,    color="black", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel(r"$p$", fontsize=13)
    ax.set_ylabel("Probability", fontsize=13)
    ax.set_xlim(P_EVAL.min(), P_EVAL.max())
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=10)
    ax.set_title("Overlay  (black = CNN,  grey = true $R$)", fontsize=10)

    # Right: scatter CNN vs R — should lie on y=x
    ax = axes[1]
    for L in L_VALUES:
        d = eval_data[L]
        ax.scatter(d["R"], d["y_mean"], s=12, alpha=0.6,
                   label=LINE_LABELS.get(L, f"L={L}"))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="$y = x$")
    ax.set_xlabel(r"True $R(p;\,L)$", fontsize=13)
    ax.set_ylabel(r"CNN $\langle y \rangle(p;\,L)$", fontsize=13)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=10)
    ax.set_title("CNN output vs true spanning probability", fontsize=10)

    fig.tight_layout()
    return fig


# =============================================================================
# Figure E: Saliency maps
# =============================================================================

def fig_saliency(models, L=None):
    """
    Gradient saliency at three p values for the largest L.
    Top row: lattice configuration.
    Bottom row: mean |gradient| — which sites drive the output.
    Treat as exploratory: saliency shows local gradient sensitivity,
    not necessarily the spanning cluster itself.
    """
    if L is None:
        L = L_VALUES[-1]
    model = models[L]

    p_vals = [PC_FSS - 0.03, PC_FSS, PC_FSS + 0.03]
    titles = [r"$p = p_c - 0.03$  (subcritical)",
              r"$p = p_c$  (critical)",
              r"$p = p_c + 0.03$  (supercritical)"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(f"Gradient saliency maps  (L={L})\n"
                 "Top: lattice configuration   "
                 "Bottom: mean |∂output/∂input|",
                 fontsize=10)

    rng = np.random.default_rng(BASE_SEED + 777777)

    for col, (p, title) in enumerate(zip(p_vals, titles)):
        lat = generate_lattice(L, p, rng)
        sal = compute_saliency(model, L, p, n_samples=10,
                               seed=BASE_SEED + col * 1000)

        ax = axes[0, col]
        ax.imshow(lat, cmap="binary", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        spanning = spans_lr(lat)
        ax.text(0.02, 0.98,
                "Spanning" if spanning else "Non-spanning",
                transform=ax.transAxes, fontsize=8,
                color="red" if spanning else "blue",
                va="top", fontweight="bold")

        ax = axes[1, col]
        im = ax.imshow(sal, cmap="hot", interpolation="nearest")
        ax.set_title("Saliency", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


# =============================================================================
# Figure F: Training curves (supplementary)
# =============================================================================

def fig_training_curves(histories):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("CNN training history  (black = train,  grey = val)",
                 fontsize=11)

    for L in L_VALUES:
        h  = histories[L]
        ls = LINE_STYLES.get(L, "-")
        ep = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], ls=ls, color="black", lw=1.0,
                     label=LINE_LABELS.get(L, f"L={L}"))
        axes[0].plot(ep, h["val_loss"],   ls=ls, color="grey",
                     lw=1.0, alpha=0.7)
        axes[1].plot(ep, h["train_acc"],  ls=ls, color="black", lw=1.0,
                     label=LINE_LABELS.get(L, f"L={L}"))
        axes[1].plot(ep, h["val_acc"],    ls=ls, color="grey",
                     lw=1.0, alpha=0.7)

    for ax, ylabel, title in [
        (axes[0], "Loss",     "Training and validation loss"),
        (axes[1], "Accuracy", "Training and validation accuracy"),
    ]:
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    fig.tight_layout()
    return fig


# =============================================================================
# Save helper
# =============================================================================

def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  Percolation CNN — LR spanning cluster detection")
    print("=" * 60)
    print(f"  Device     : {DEVICE}")
    print(f"  L_VALUES   : {L_VALUES}")
    print(f"  N_TRAIN    : {N_TRAIN}  N_VAL : {N_VAL}")
    print(f"  N_EPOCHS   : {N_EPOCHS}  BATCH : {BATCH_SIZE}")
    print(f"  PC_FSS     : {PC_FSS:.5f}  NU : {NU:.6f}")
    print(f"  P_BELOW    : {P_BELOW}  frac={FRAC_BELOW}")
    print(f"  P_CRITICAL : {P_CRITICAL}  frac={FRAC_CRITICAL}")
    print(f"  P_ABOVE    : {P_ABOVE}  frac={FRAC_ABOVE}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR,  exist_ok=True)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("\n[1/3] Training models (one per L) ...")
    models, histories = {}, {}
    for i, L in enumerate(L_VALUES):
        model, history   = train_model(L, seed_offset=i * 200000)
        models[L]        = model
        histories[L]     = history

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print("\n[2/3] Evaluating models ...")
    eval_data = {}
    for i, L in enumerate(L_VALUES):
        eval_data[L] = evaluate_model(models[L], L,
                                      seed_offset=i * 200000)

    print("\n  pc^ML summary:")
    print(f"  {'L':>6}  {'pc^ML':>10}  {'pc^FSS':>10}  {'delta':>10}")
    for L in L_VALUES:
        pc_ml = float(eval_data[L]["pc_ml"])
        delta = pc_ml - PC_FSS if np.isfinite(pc_ml) else np.nan
        print(f"  {L:>6}  {pc_ml:>10.5f}  {PC_FSS:>10.5f}  "
              f"{delta:>+10.5f}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\n[3/3] Saving figures ...")

    _save(fig_cnn_output_vs_p(eval_data),  "cnn_output_vs_p")
    _save(fig_pc_vs_L(eval_data),          "cnn_pc_vs_L")
    _save(fig_cnn_collapse(eval_data),     "cnn_collapse")
    _save(fig_cnn_vs_R(eval_data),         "cnn_vs_R")
    _save(fig_saliency(models),            "cnn_saliency")
    _save(fig_training_curves(histories),  "cnn_training")

    print(f"\nDone. Figures saved to: {FIG_DIR}/")
    print(f"Models saved to:         {CKPT_DIR}/")


if __name__ == "__main__":
    main()
