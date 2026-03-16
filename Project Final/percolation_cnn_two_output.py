"""
percolation_cnn_two_output.py
-----------------------------
Two-output CNN for 2D site percolation.

Goal
----
Train a classifier with two outputs:
  y1 = P(non-spanning | configuration)
  y2 = P(spanning     | configuration)

Then analyse:
  1. <y1>(p,L), <y2>(p,L) vs p
  2. p_c^ML(L) from <y1> = <y2> = 0.5
  3. Collapse vs (p - p_c)L^(1/nu)

This is the closest analogue of the figure style you sent.
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
    raise ValueError("Training fractions must sum to 1.")

LINE_STYLES = {
    32:  (0, (1, 3)),
    64:  (0, (1, 1)),
    128: (0, (3, 1, 1, 1)),
    256: (0, (5, 2)),
}
LINE_LABELS = {L: f"$L={L}$" for L in L_VALUES}

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# =============================================================================
# Lattice generation and LR spanning
# =============================================================================

def generate_lattice(L, p, rng):
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
# Balanced dataset
# =============================================================================

def _sample_one_p(rng):
    u = rng.random()
    if u < FRAC_BELOW:
        return float(rng.uniform(*P_BELOW))
    elif u < FRAC_BELOW + FRAC_CRITICAL:
        return float(rng.uniform(*P_CRITICAL))
    return float(rng.uniform(*P_ABOVE))


class PercolationDataset(Dataset):
    """
    Balanced two-class dataset:
      class 0 = non-spanning
      class 1 = spanning
    """
    def __init__(self, L, n_samples, seed):
        target_each = n_samples // 2
        rng = np.random.default_rng(seed)

        class0 = []
        class1 = []

        max_attempts = n_samples * 20
        attempts = 0

        while len(class0) < target_each or len(class1) < target_each:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"Could not build full dataset for L={L}. "
                    f"Got class0={len(class0)}, class1={len(class1)}."
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
            f"L={L:4d}: {len(self.labels)} samples "
            f"({np.sum(self.labels==0)} non-spanning, {np.sum(self.labels==1)} spanning)"
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
            nn.Linear(64, 2),   # two logits
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


# =============================================================================
# Training
# =============================================================================

def train_model(L, seed_offset=0):
    ckpt = os.path.join(CKPT_DIR, f"model_L{L}.pt")
    hist = os.path.join(CKPT_DIR, f"history_L{L}.npz")

    if (not FORCE_RETRAIN) and os.path.exists(ckpt) and os.path.exists(hist):
        model = PercolationCNN2().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        h = np.load(hist)
        history = {k: h[k].tolist() for k in h.files}
        return model, history

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
        epoch_iter = _tqdm(epoch_iter, desc=f"L={L:4d} training")

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

    return model, history


# =============================================================================
# Evaluation
# =============================================================================

def _find_equal_prob_crossing(p_arr, y1_arr, y2_arr):
    """
    Find p where y1 = y2, i.e. y2 - y1 = 0.
    Equivalent to y2 = 0.5, but this is more natural for 2-output plots.
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


def evaluate_model(model, L, seed_offset=0):
    ckpt = os.path.join(CKPT_DIR, f"eval_L{L}.npz")
    if (not FORCE_RETRAIN) and os.path.exists(ckpt):
        d = np.load(ckpt)
        return {k: d[k] for k in d.files}

    model.eval()
    rng = np.random.default_rng(BASE_SEED + seed_offset + 500000)

    y1_mean = np.zeros(len(P_EVAL))
    y2_mean = np.zeros(len(P_EVAL))
    R = np.zeros(len(P_EVAL))

    p_iter = enumerate(P_EVAL)
    if _TQDM:
        p_iter = _tqdm(p_iter, total=len(P_EVAL), desc=f"L={L:4d} eval")

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
    return result


# =============================================================================
# Figures matching the style you want
# =============================================================================

def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_outputs_vs_p(eval_data):
    fig, ax = plt.subplots(figsize=(8, 5))

    for L in L_VALUES:
        d = eval_data[L]
        ls = LINE_STYLES.get(L, "-")
        ax.plot(d["p"], d["y1_mean"], linestyle=ls, color="black", lw=1.2)
        ax.plot(d["p"], d["y2_mean"], linestyle=ls, color="black", lw=1.2, label=LINE_LABELS[L])

    ax.axvline(PC_FSS, color="orange", lw=2, alpha=0.8)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)

    ax.set_xlabel(r"$p$", fontsize=13)
    ax.set_ylabel(r"$\langle y_1\rangle,\ \langle y_2\rangle$", fontsize=13)
    ax.set_xlim(P_EVAL.min(), P_EVAL.max())
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.set_title("CNN outputs vs occupation probability")
    fig.tight_layout()
    return fig


def fig_pc_vs_L(eval_data):
    fig, ax = plt.subplots(figsize=(6, 5))

    L_arr = np.array([L for L in L_VALUES if np.isfinite(float(eval_data[L]["pc_ml"]))], dtype=float)
    pc_arr = np.array([float(eval_data[L]["pc_ml"]) for L in L_arr])

    x = L_arr ** (-1.0 / NU)

    ax.plot(x, pc_arr, "o", color="green", ms=8, label=r"$p_c^{ML}(L)$")

    for xi, yi, L in zip(x, pc_arr, L_arr):
        ax.annotate(f"$L={int(L)}$", (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=8)

    if len(x) >= 2:
        coeffs = np.polyfit(x, pc_arr, 1)
        x_fit = np.linspace(0, x.max() * 1.1, 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "--", color="brown", lw=1.5)
        ax.axhline(PC_FSS, color="gray", ls="--", lw=1.0)
        ax.plot(0, coeffs[1], "d", color="green", ms=8)

    ax.set_xlabel(r"$L^{-1/\nu}$", fontsize=13)
    ax.set_ylabel(r"$p_c^{ML}(L)$", fontsize=13)
    ax.set_title("ML pseudocritical point")
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
    ax.set_title("Approximate collapse of CNN outputs")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"L_VALUES = {L_VALUES}")

    models = {}
    histories = {}
    eval_data = {}

    for i, L in enumerate(L_VALUES):
        model, history = train_model(L, seed_offset=i * 200000)
        models[L] = model
        histories[L] = history

    for i, L in enumerate(L_VALUES):
        eval_data[L] = evaluate_model(models[L], L, seed_offset=i * 200000)

    _save(fig_outputs_vs_p(eval_data), "two_output_outputs_vs_p")
    _save(fig_pc_vs_L(eval_data), "two_output_pc_vs_L")
    _save(fig_output_collapse(eval_data), "two_output_collapse")

    print("Done.")


if __name__ == "__main__":
    main()