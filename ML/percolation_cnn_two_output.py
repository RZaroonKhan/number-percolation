"""
percolation_cnn_two_output.py
-----------------------------
Standalone two-output CNN for 2D site percolation phase detection.

Scientific question
-------------------
Can a convolutional neural network learn the percolation transition
directly from raw lattice configurations, and can its output be used
as an independent estimator of the critical point?

What the network does
---------------------
Input  : binary L×L lattice (occupied = 1, empty = 0)
Target : 0 or 1 from direct left-to-right spanning test
Output : two class probabilities
         y1 = P(non-spanning | configuration)
         y2 = P(spanning     | configuration)

Averaging over many configurations at fixed (p, L) gives:
    <y1>(p, L), <y2>(p, L)

From these we define:
    Delta y = <y2> - <y1>          (ML order parameter)
    chi_ML  = d(Delta y)/dp        (ML susceptibility analogue)

Figures produced
----------------
1. two_output_outputs_vs_p.png   — <y1>, <y2> vs p
2. two_output_pc_vs_L.png        — p_c^ML(L) vs L^(-1/nu)
3. two_output_collapse.png       — <y1>, <y2> collapse vs (p-p_c)L^(1/nu)
4. ml_delta_y_vs_p.png           — Delta y vs p
5. ml_delta_y_collapse.png       — Delta y collapse
6. ml_chi_ml_vs_p.png            — chi_ML vs p
7. ml_chi_ml_peak_vs_L.png       — peak position of chi_ML vs L^(-1/nu)
8. cnn_training.png              — training curves (supplementary)

Usage
-----
    python3 percolation_cnn_two_output.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq

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

PC_FSS = 0.59296
NU = 4 / 3

L_VALUES = [32, 64, 128, 256]

# Training regions
P_BELOW = (0.550, 0.585)
P_CRITICAL = (0.585, 0.600)
P_ABOVE = (0.600, 0.635)

FRAC_BELOW = 0.25
FRAC_CRITICAL = 0.50
FRAC_ABOVE = 0.25

N_TRAIN = 8000
N_VAL = 2000

P_EVAL = np.linspace(0.560, 0.630, 50)
N_EVAL = 500

N_EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4

BASE_SEED = 42
FIG_DIR = "figures_ml_two_output"
CKPT_DIR = "checkpoints_ml_two_output"
FORCE_RETRAIN = False

if not np.isclose(FRAC_BELOW + FRAC_CRITICAL + FRAC_ABOVE, 1.0):
    raise ValueError(
        "FRAC_BELOW + FRAC_CRITICAL + FRAC_ABOVE must equal 1.0"
    )

LINE_STYLES = {
    32:  (0, (1, 3)),
    64:  (0, (1, 1)),
    128: (0, (3, 1, 1, 1)),
    256: (0, (5, 2)),
    512: "-",
}
LINE_LABELS = {L: f"$L={L}$" for L in L_VALUES}

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
# Lattice generation and LR spanning test
# =============================================================================

def generate_lattice(L: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a random binary LxL lattice."""
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


def spans_lr(lattice: np.ndarray) -> bool:
    """
    Return True if any occupied cluster connects left and right boundaries.
    """
    L = lattice.shape[0]
    parent = list(range(L * L))
    rank = [0] * (L * L)

    for i in range(L):
        for j in range(L):
            if lattice[i, j] == 0:
                continue
            idx = i * L + j
            if j + 1 < L and lattice[i, j + 1]:
                _union(parent, rank, idx, i * L + j + 1)
            if i + 1 < L and lattice[i + 1, j]:
                _union(parent, rank, idx, (i + 1) * L + j)

    left_roots = {_find(parent, i * L) for i in range(L) if lattice[i, 0]}
    right_roots = {_find(parent, i * L + L - 1) for i in range(L) if lattice[i, L - 1]}
    return bool(left_roots & right_roots)


# =============================================================================
# Balanced dataset with critical region emphasis
# =============================================================================

def _sample_one_p(rng: np.random.Generator) -> float:
    """Sample one p value from the three training regions."""
    u = rng.random()
    if u < FRAC_BELOW:
        return float(rng.uniform(*P_BELOW))
    elif u < FRAC_BELOW + FRAC_CRITICAL:
        return float(rng.uniform(*P_CRITICAL))
    return float(rng.uniform(*P_ABOVE))


class PercolationDataset(Dataset):
    """
    Balanced dataset:
      class 0 = non-spanning
      class 1 = spanning
    """
    def __init__(self, L: int, n_samples: int, seed: int):
        target_each = n_samples // 2
        rng = np.random.default_rng(seed)

        class0 = []
        class1 = []

        max_attempts = n_samples * 20
        attempts = 0

        while len(class0) < target_each or len(class1) < target_each:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"Could not build full balanced dataset for L={L}. "
                    f"Got class0={len(class0)}, class1={len(class1)}. "
                    f"Try widening the p-ranges or increasing max_attempts."
                )
            attempts += 1

            p = _sample_one_p(rng)
            lat = generate_lattice(L, p, rng)
            span = spans_lr(lat)

            if span and len(class1) < target_each:
                class1.append(lat)
            elif (not span) and len(class0) < target_each:
                class0.append(lat)

        lattices = class0 + class1
        labels = [0] * len(class0) + [1] * len(class1)

        perm = np.random.default_rng(seed + 1).permutation(len(labels))
        self.lattices = np.array([lattices[i] for i in perm], dtype=np.float32)[:, None, :, :]
        self.labels = np.array([labels[i] for i in perm], dtype=np.int64)

        print(
            f"    L={L:4d}: {len(self.labels)} samples "
            f"({np.sum(self.labels == 0)} non-spanning, "
            f"{np.sum(self.labels == 1)} spanning)"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.lattices[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# =============================================================================
# CNN with two outputs
# =============================================================================

class PercolationCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
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
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),  # two logits
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


# =============================================================================
# Training
# =============================================================================

def train_model(L: int, seed_offset: int = 0):
    """Train or load a CNN for one lattice size L."""
    ckpt = os.path.join(CKPT_DIR, f"model_L{L}.pt")
    hist = os.path.join(CKPT_DIR, f"history_L{L}.npz")

    if (not FORCE_RETRAIN) and os.path.exists(ckpt) and os.path.exists(hist):
        print(f"  Loading trained model L={L} from checkpoint ...")
        model = PercolationCNN2().to(DEVICE)
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        h = np.load(hist)
        history = {k: h[k].tolist() for k in h.files}
        return model, history

    print(f"\n  Training CNN for L={L} ...")
    train_ds = PercolationDataset(L, N_TRAIN, seed=BASE_SEED + seed_offset)
    val_ds = PercolationDataset(L, N_VAL, seed=BASE_SEED + seed_offset + 99999)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = PercolationCNN2().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    epoch_iter = range(N_EPOCHS)
    if _TQDM:
        epoch_iter = _tqdm(epoch_iter, desc=f"  L={L:4d} training")

    for _ in epoch_iter:
        model.train()
        t_loss = 0.0
        t_correct = 0
        t_total = 0

        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            t_loss += loss.item() * len(y)
            t_correct += (preds == y).sum().item()
            t_total += len(y)

        scheduler.step()

        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0

        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                preds = logits.argmax(dim=1)

                v_loss += loss.item() * len(y)
                v_correct += (preds == y).sum().item()
                v_total += len(y)

        history["train_loss"].append(t_loss / t_total)
        history["val_loss"].append(v_loss / v_total)
        history["train_acc"].append(t_correct / t_total)
        history["val_acc"].append(v_correct / v_total)

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    np.savez(hist, **{k: np.array(v) for k, v in history.items()})

    print(
        f"  Saved: {ckpt}\n"
        f"  Final val acc = {history['val_acc'][-1]:.4f}  "
        f"val loss = {history['val_loss'][-1]:.4f}"
    )
    return model, history


# =============================================================================
# Evaluation
# =============================================================================

def _find_equal_prob_crossing(p_arr, y1_arr, y2_arr):
    """
    Find p where y1 = y2, i.e. y2 - y1 = 0.
    """
    try:
        diff = y2_arr - y1_arr
        f = interp1d(p_arr, diff, kind="linear")
        signs = np.sign(diff)
        idx = np.where(np.diff(signs) != 0)[0]
        if len(idx) == 0:
            return np.nan

        crossings = []
        for i in idx:
            try:
                crossings.append(brentq(f, p_arr[i], p_arr[i + 1]))
            except Exception:
                crossings.append(0.5 * (p_arr[i] + p_arr[i + 1]))

        return float(min(crossings, key=lambda x: abs(x - PC_FSS)))
    except Exception:
        return np.nan


def evaluate_model(model, L: int, seed_offset: int = 0):
    """
    Evaluate model across P_EVAL.
    Returns y1_mean, y2_mean, true R, and p_c^ML.
    """
    ckpt = os.path.join(CKPT_DIR, f"eval_L{L}.npz")
    if (not FORCE_RETRAIN) and os.path.exists(ckpt):
        print(f"  Loading evaluation data L={L} from checkpoint ...")
        d = np.load(ckpt)
        return {k: d[k] for k in d.files}

    print(f"  Evaluating L={L} over {len(P_EVAL)} p values, {N_EVAL} configs each ...")
    model.eval()
    rng = np.random.default_rng(BASE_SEED + seed_offset + 500000)

    y1_mean = np.zeros(len(P_EVAL))
    y2_mean = np.zeros(len(P_EVAL))
    R = np.zeros(len(P_EVAL))

    p_iter = enumerate(P_EVAL)
    if _TQDM:
        p_iter = _tqdm(p_iter, total=len(P_EVAL), desc=f"  L={L:4d} eval")

    for j, p in p_iter:
        lats = np.zeros((N_EVAL, 1, L, L), dtype=np.float32)
        labels = np.zeros(N_EVAL, dtype=np.float32)

        for k in range(N_EVAL):
            lat = generate_lattice(L, p, rng)
            lats[k, 0] = lat
            labels[k] = float(spans_lr(lat))

        with torch.no_grad():
            x = torch.from_numpy(lats).to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        y1_mean[j] = probs[:, 0].mean()
        y2_mean[j] = probs[:, 1].mean()
        R[j] = labels.mean()

    pc_ml = _find_equal_prob_crossing(P_EVAL, y1_mean, y2_mean)

    result = {
        "p": P_EVAL,
        "y1_mean": y1_mean,
        "y2_mean": y2_mean,
        "R": R,
        "pc_ml": np.array(pc_ml),
    }

    os.makedirs(CKPT_DIR, exist_ok=True)
    np.savez(ckpt, **result)

    print(f"  p_c^ML(L={L}) = {pc_ml:.5f}  (FSS p_c = {PC_FSS:.5f})")
    return result


# =============================================================================
# Derived ML observables
# =============================================================================

def _delta_y(d):
    """
    ML order parameter:
        Delta y = <y_span> - <y_nonspan> = <y2> - <y1>
    """
    return d["y2_mean"] - d["y1_mean"]


def _chi_ml(d):
    """
    ML susceptibility analogue:
        chi_ML = d(Delta y)/dp
    """
    return np.gradient(_delta_y(d), d["p"])


# =============================================================================
# Figures
# =============================================================================

def _save(fig, name: str):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_outputs_vs_p(eval_data):
    fig, ax = plt.subplots(figsize=(8, 5))

    for L in L_VALUES:
        d = eval_data[L]
        ls = LINE_STYLES.get(L, "-")

        ax.plot(d["p"], d["y1_mean"], linestyle=ls, color="black", lw=1.2)
        ax.plot(d["p"], d["y2_mean"], linestyle=ls, color="black", lw=1.2, label=LINE_LABELS[L])

    ax.axvline(PC_FSS, color="orange", lw=2, alpha=0.8, label=rf"$p_c={PC_FSS:.5f}$")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)

    ax.set_xlabel(r"$p$", fontsize=13)
    ax.set_ylabel(r"$\langle y_1\rangle,\ \langle y_2\rangle$", fontsize=13)
    ax.set_xlim(P_EVAL.min(), P_EVAL.max())
    ax.set_ylim(0, 1)
    ax.set_title("Two-output CNN probabilities vs occupation probability", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_pc_vs_L(eval_data):
    fig, ax = plt.subplots(figsize=(6, 5))

    L_arr = np.array(
        [L for L in L_VALUES if np.isfinite(float(eval_data[L]["pc_ml"]))],
        dtype=float
    )
    pc_arr = np.array([float(eval_data[L]["pc_ml"]) for L in L_arr])

    if len(L_arr) < 2:
        ax.text(0.5, 0.5, "Insufficient data for ML FSS fit",
                transform=ax.transAxes, ha="center")
        fig.tight_layout()
        return fig

    x = L_arr ** (-1.0 / NU)

    ax.plot(x, pc_arr, "o", color="green", ms=8, label=r"$p_c^{\rm ML}(L)$")

    for xi, yi, L in zip(x, pc_arr, L_arr):
        ax.annotate(f"$L={int(L)}$", (xi, yi),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)

    try:
        coeffs = np.polyfit(x, pc_arr, 1)
        x_fit = np.linspace(0, x.max() * 1.1, 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "--", color="brown", lw=1.5,
                label=rf"fit: $p_c^{{\infty}}={coeffs[1]:.5f}$")
        ax.plot(0, coeffs[1], "d", color="green", ms=8)
    except Exception:
        pass

    ax.axhline(PC_FSS, color="gray", ls="--", lw=1.0,
               label=rf"$p_c^{{\rm FSS}}={PC_FSS:.5f}$")

    ax.set_xlabel(r"$L^{-1/\nu}$", fontsize=13)
    ax.set_ylabel(r"$p_c^{\rm ML}(L)$", fontsize=13)
    ax.set_title("ML pseudocritical point scaling", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_output_collapse(eval_data):
    fig, ax = plt.subplots(figsize=(8, 5))

    for L in L_VALUES:
        d = eval_data[L]
        x = (d["p"] - PC_FSS) * (L ** (1.0 / NU))
        ls = LINE_STYLES.get(L, "-")

        ax.plot(x, d["y1_mean"], linestyle=ls, color="black", lw=1.2)
        ax.plot(x, d["y2_mean"], linestyle=ls, color="black", lw=1.2, label=LINE_LABELS[L])

    ax.axvline(0, color="orange", lw=2, alpha=0.8)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)

    ax.set_xlabel(r"$(p-p_c)L^{1/\nu}$", fontsize=13)
    ax.set_ylabel(r"$\langle y_1\rangle,\ \langle y_2\rangle$", fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_title("Approximate collapse of two CNN outputs", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_delta_y_vs_p(eval_data):
    fig, ax = plt.subplots(figsize=(8, 5))

    for L in L_VALUES:
        d = eval_data[L]
        dy = _delta_y(d)
        ax.plot(
            d["p"], dy,
            linestyle=LINE_STYLES.get(L, "-"),
            color="black", lw=1.3,
            label=LINE_LABELS[L]
        )

    ax.axvline(PC_FSS, color="orange", lw=2, alpha=0.8, label=rf"$p_c={PC_FSS:.5f}$")
    ax.axhline(0, color="gray", ls="--", lw=0.8)

    ax.set_xlabel(r"$p$", fontsize=13)
    ax.set_ylabel(r"$\Delta y = \langle y_2\rangle - \langle y_1\rangle$", fontsize=13)
    ax.set_xlim(P_EVAL.min(), P_EVAL.max())
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("ML order parameter vs occupation probability", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_delta_y_collapse(eval_data):
    fig, ax = plt.subplots(figsize=(8, 5))

    for L in L_VALUES:
        d = eval_data[L]
        x = (d["p"] - PC_FSS) * (L ** (1.0 / NU))
        dy = _delta_y(d)

        ax.plot(
            x, dy,
            linestyle=LINE_STYLES.get(L, "-"),
            color="black", lw=1.3,
            label=LINE_LABELS[L]
        )

    ax.axvline(0, color="orange", lw=2, alpha=0.8)
    ax.axhline(0, color="gray", ls="--", lw=0.8)

    ax.set_xlabel(r"$(p-p_c)L^{1/\nu}$", fontsize=13)
    ax.set_ylabel(r"$\Delta y = \langle y_2\rangle - \langle y_1\rangle$", fontsize=13)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Approximate collapse of ML order parameter", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_chi_ml_vs_p(eval_data):
    fig, ax = plt.subplots(figsize=(8, 5))

    for L in L_VALUES:
        d = eval_data[L]
        chi_ml = _chi_ml(d)

        ax.plot(
            d["p"], chi_ml,
            linestyle=LINE_STYLES.get(L, "-"),
            color="black", lw=1.3,
            label=LINE_LABELS[L]
        )

    ax.axvline(PC_FSS, color="orange", lw=2, alpha=0.8, label=rf"$p_c={PC_FSS:.5f}$")

    ax.set_xlabel(r"$p$", fontsize=13)
    ax.set_ylabel(r"$\chi_{\rm ML} = d(\Delta y)/dp$", fontsize=13)
    ax.set_xlim(P_EVAL.min(), P_EVAL.max())
    ax.set_title("ML susceptibility analogue", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_chi_ml_peak_vs_L(eval_data):
    fig, ax = plt.subplots(figsize=(6, 5))

    L_arr = []
    p_peak_arr = []

    for L in L_VALUES:
        d = eval_data[L]
        chi_ml = _chi_ml(d)
        idx = int(np.argmax(chi_ml))
        L_arr.append(L)
        p_peak_arr.append(d["p"][idx])

    L_arr = np.array(L_arr, dtype=float)
    p_peak_arr = np.array(p_peak_arr, dtype=float)
    x = L_arr ** (-1.0 / NU)

    ax.plot(x, p_peak_arr, "o", color="purple", ms=8, label=r"$p_{\rm peak}^{\chi_{\rm ML}}(L)$")

    for xi, yi, L in zip(x, p_peak_arr, L_arr):
        ax.annotate(f"$L={int(L)}$", (xi, yi),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)

    try:
        coeffs = np.polyfit(x, p_peak_arr, 1)
        x_fit = np.linspace(0, x.max() * 1.1, 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "--", color="black", lw=1.2,
                label=rf"fit: $p^{{\infty}}={coeffs[1]:.5f}$")
    except Exception:
        pass

    ax.axhline(PC_FSS, color="gray", ls="--", lw=1.0,
               label=rf"$p_c^{{\rm FSS}}={PC_FSS:.5f}$")

    ax.set_xlabel(r"$L^{-1/\nu}$", fontsize=13)
    ax.set_ylabel(r"$p_{\rm peak}^{\chi_{\rm ML}}(L)$", fontsize=13)
    ax.set_title("Scaling of ML susceptibility peak", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_training_curves(histories):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("CNN training history  (black = train, grey = val)", fontsize=11)

    for L in L_VALUES:
        h = histories[L]
        ls = LINE_STYLES.get(L, "-")
        ep = range(1, len(h["train_loss"]) + 1)

        axes[0].plot(ep, h["train_loss"], ls=ls, color="black", lw=1.0, label=LINE_LABELS[L])
        axes[0].plot(ep, h["val_loss"], ls=ls, color="grey", lw=1.0, alpha=0.7)

        axes[1].plot(ep, h["train_acc"], ls=ls, color="black", lw=1.0, label=LINE_LABELS[L])
        axes[1].plot(ep, h["val_acc"], ls=ls, color="grey", lw=1.0, alpha=0.7)

    for ax, ylabel, title in [
        (axes[0], "Loss", "Training and validation loss"),
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
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  Percolation CNN — two-output LR spanning classifier")
    print("=" * 60)
    print(f"  Device     : {DEVICE}")
    print(f"  L_VALUES   : {L_VALUES}")
    print(f"  N_TRAIN    : {N_TRAIN}  N_VAL : {N_VAL}")
    print(f"  N_EPOCHS   : {N_EPOCHS}  BATCH : {BATCH_SIZE}")
    print(f"  PC_FSS     : {PC_FSS:.5f}  NU : {NU:.6f}")
    print(f"  P_BELOW    : {P_BELOW}  frac={FRAC_BELOW}")
    print(f"  P_CRITICAL : {P_CRITICAL}  frac={FRAC_CRITICAL}")
    print(f"  P_ABOVE    : {P_ABOVE}  frac={FRAC_ABOVE}")
    print(f"  FORCE_RETRAIN = {FORCE_RETRAIN}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("\n[1/3] Training models (one per L) ...")
    models = {}
    histories = {}
    for i, L in enumerate(L_VALUES):
        model, history = train_model(L, seed_offset=i * 200000)
        models[L] = model
        histories[L] = history

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print("\n[2/3] Evaluating models ...")
    eval_data = {}
    for i, L in enumerate(L_VALUES):
        eval_data[L] = evaluate_model(models[L], L, seed_offset=i * 200000)

    print("\n  p_c^ML summary:")
    print(f"  {'L':>6}  {'p_c^ML':>10}  {'p_c^FSS':>10}  {'delta':>10}")
    for L in L_VALUES:
        pc_ml = float(eval_data[L]["pc_ml"])
        delta = pc_ml - PC_FSS if np.isfinite(pc_ml) else np.nan
        print(f"  {L:>6}  {pc_ml:>10.5f}  {PC_FSS:>10.5f}  {delta:>+10.5f}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\n[3/3] Saving figures ...")
    _save(fig_outputs_vs_p(eval_data), "two_output_outputs_vs_p")
    _save(fig_pc_vs_L(eval_data), "two_output_pc_vs_L")
    _save(fig_output_collapse(eval_data), "two_output_collapse")
    _save(fig_delta_y_vs_p(eval_data), "ml_delta_y_vs_p")
    _save(fig_delta_y_collapse(eval_data), "ml_delta_y_collapse")
    _save(fig_chi_ml_vs_p(eval_data), "ml_chi_ml_vs_p")
    _save(fig_chi_ml_peak_vs_L(eval_data), "ml_chi_ml_peak_vs_L")
    _save(fig_training_curves(histories), "cnn_training")

    print(f"\nDone. Figures saved to: {FIG_DIR}/")
    print(f"Models/checkpoints saved to: {CKPT_DIR}/")


if __name__ == "__main__":
    main()
