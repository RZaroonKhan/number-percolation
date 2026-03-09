"""
ml_analysis.py
--------------
Interpretation and analysis of the trained percolation CNN.

  run_ml_pipeline()          — end-to-end: train, evaluate, compare to FSS.
  plot_training_history()    — loss and accuracy curves.
  plot_cnn_pc_curve()        — CNN sigmoid vs p with pc comparison.
  plot_saliency()            — which pixels drive the classification.
  plot_ml_summary()          — combined ML results figure.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, random_split

from observables import generate_lattice
from analysis    import THEORY
from ml_model    import (
    PercolationDataset, PercolationCNN,
    train_model, estimate_pc_from_cnn, save_model,
)


# ---------------------------------------------------------------------------
# Saliency map
# ---------------------------------------------------------------------------

def compute_saliency(model, lattice, device=None):
    """
    Compute a saliency map for a single lattice by backpropagating
    the gradient of the predicted class score to the input pixels.

    Bright pixels in the saliency map are the ones the network is most
    sensitive to — i.e. the sites whose occupation most influences
    the phase prediction. For percolation, you expect the spanning
    cluster to light up.

    Parameters
    ----------
    model   : trained PercolationCNN.
    lattice : (L, L) numpy array (binary).
    device  : torch device.

    Returns
    -------
    saliency : (L, L) numpy array, values in [0, 1].
    pred_class : int (0 = sub-critical, 1 = super-critical).
    confidence : float, probability of predicted class.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    x = torch.from_numpy(
        lattice.astype(np.float32)[np.newaxis, np.newaxis]  # (1, 1, L, L)
    ).to(device)
    x.requires_grad_(True)

    logits     = model(x)
    pred_class = int(logits.argmax(1).item())
    confidence = float(torch.softmax(logits, dim=1)[0, pred_class].item())

    # Backprop the score of the predicted class to the input
    model.zero_grad()
    logits[0, pred_class].backward()

    saliency = x.grad.data.abs().squeeze().cpu().numpy()
    # Normalise to [0, 1]
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    return saliency, pred_class, confidence


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_history(history):
    """Loss and accuracy curves over training epochs."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"], "o-", ms=4, label="train")
    ax1.plot(epochs, history["val_loss"],   "o-", ms=4, label="validation")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Training and validation loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_acc"], "o-", ms=4, color="C2")
    ax2.axhline(1.0, color="grey", ls="--", lw=0.8)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation accuracy")
    ax2.set_ylim(0, 1.05); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_cnn_pc_curve(p_values, prob_super, pc_cnn, pc_fss=None):
    """
    CNN super-critical probability vs p, with pc estimates marked.

    The S-shaped curve is the network's analogue of the spanning
    probability R(p). Its crossing at 0.5 is the CNN's pc estimate.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(p_values, prob_super, "o-", ms=4, color="#2166ac",
            label="CNN P(super-critical)")
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.axvline(pc_cnn, color="#d6604d", ls="--", lw=1.5,
               label=f"CNN pc = {pc_cnn:.5f}")

    if pc_fss is not None:
        ax.axvline(pc_fss, color="green", ls="--", lw=1.5,
                   label=f"FSS pc = {pc_fss:.5f}")

    ax.axvline(THEORY["pc"], color="black", ls=":", lw=1,
               label=f"Theory pc = {THEORY['pc']:.5f}")

    ax.set_xlabel("p")
    ax.set_ylabel("P(super-critical)")
    ax.set_title("CNN phase probability vs p")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_saliency(model, L, pc, device=None, seed=42):
    """
    Three-panel saliency figure: one lattice at each of
    p < pc, p ≈ pc, p > pc.

    Each panel shows the lattice with occupied sites coloured by
    their saliency score — brighter = more influential on the prediction.
    """
    rng    = np.random.default_rng(seed)
    p_vals = [pc - 0.06, pc, pc + 0.06]
    labels = [f"p = {pc-0.06:.3f}  (sub-critical)",
              f"p ≈ pc = {pc:.4f}  (critical)",
              f"p = {pc+0.06:.3f}  (super-critical)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Saliency maps — which sites drive the CNN prediction?",
                 fontsize=12)

    for ax, p, title in zip(axes, p_vals, labels):
        lattice          = generate_lattice(L, p, rng=rng)
        sal, pred, conf  = compute_saliency(model, lattice, device=device)

        # Show saliency only on occupied sites; empty sites stay grey
        img = np.full((*lattice.shape, 3), 0.85)   # light grey background
        occ = lattice == 1
        # Map saliency to a red colourmap on occupied sites
        cmap    = plt.cm.hot
        sal_rgb = cmap(sal)[:, :, :3]
        img[occ] = sal_rgb[occ]

        ax.imshow(img, origin="upper", interpolation="nearest")
        phase = "super-critical" if pred == 1 else "sub-critical"
        ax.set_title(f"{title}\nPredicted: {phase}  ({conf:.1%} confidence)",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    return fig


def plot_ml_summary(history, p_values, prob_super, pc_cnn, pc_fss=None):
    """
    Three-panel ML summary: training curves, CNN pc curve, pc comparison bar.
    """
    fig = plt.figure(figsize=(15, 5))
    gs  = fig.add_gridspec(1, 3, wspace=0.35)

    # --- Panel 1: loss curves ---
    ax = fig.add_subplot(gs[0])
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "o-", ms=3, label="train loss")
    ax.plot(epochs, history["val_loss"],   "o-", ms=3, label="val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training history"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Panel 2: CNN pc curve ---
    ax = fig.add_subplot(gs[1])
    ax.plot(p_values, prob_super, "o-", ms=3, color="#2166ac")
    ax.axhline(0.5,         color="grey",   ls=":", lw=0.8)
    ax.axvline(pc_cnn,      color="#d6604d",ls="--", lw=1.5,
               label=f"CNN pc = {pc_cnn:.5f}")
    if pc_fss is not None:
        ax.axvline(pc_fss,  color="green",  ls="--", lw=1.5,
                   label=f"FSS pc = {pc_fss:.5f}")
    ax.axvline(THEORY["pc"],color="black",  ls=":", lw=1,
               label=f"Theory = {THEORY['pc']:.5f}")
    ax.set_xlabel("p"); ax.set_ylabel("P(super-critical)")
    ax.set_title("CNN phase probability"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # --- Panel 3: pc comparison ---
    ax = fig.add_subplot(gs[2])
    methods = ["Theory"]
    values  = [THEORY["pc"]]
    colours = ["#444444"]
    if pc_fss is not None:
        methods.append("FSS");  values.append(pc_fss);  colours.append("green")
    methods.append("CNN");  values.append(pc_cnn);  colours.append("#d6604d")

    bars = ax.barh(methods, values, color=colours, height=0.4)
    ax.axvline(THEORY["pc"], color="#444444", ls="--", lw=1)
    for bar, val in zip(bars, values):
        ax.text(val + 0.0002, bar.get_y() + bar.get_height()/2,
                f"{val:.5f}", va="center", fontsize=9)
    ax.set_xlabel("pc estimate")
    ax.set_title("pc comparison")
    ax.set_xlim(min(values) - 0.005, max(values) + 0.008)
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("CNN Phase Classification — Results Summary",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def run_ml_pipeline(
    L,
    pc_fss,
    p_train_low  = None,
    p_train_high = None,
    p_gap        = 0.04,
    samples_per_p= 300,
    val_fraction = 0.2,
    batch_size   = 64,
    n_epochs     = 25,
    lr           = 1e-3,
    n_filters    = 32,
    dropout      = 0.3,
    p_eval       = None,
    eval_samples = 300,
    seed         = 1234,
    device       = None,
    save_path    = "percolation_cnn.pt",
):
    """
    Full ML pipeline: generate data, train CNN, estimate pc, return results.

    The training set uses p values away from pc (outside a gap of ±p_gap)
    so the network sees unambiguous examples. The eval set uses a dense
    grid across the full transition for the pc curve.

    Parameters
    ----------
    L             : lattice side length (use same as L_MAIN).
    pc_fss        : FSS pc estimate (used to centre the training gap
                    and as the class boundary).
    p_train_low   : lower bound of training p range (default pc-0.12).
    p_train_high  : upper bound of training p range (default pc+0.12).
    p_gap         : half-width of the gap around pc excluded from training.
                    (Larger gap = harder problem but sharper pc estimate.)
    samples_per_p : lattice realisations per training p value.
    val_fraction  : fraction of training data held out for validation.
    batch_size    : mini-batch size.
    n_epochs      : training epochs.
    lr            : Adam learning rate.
    n_filters     : CNN filter count.
    dropout       : dropout probability.
    p_eval        : dense p grid for pc estimation (default: 40 points
                    in [pc-0.08, pc+0.08]).
    eval_samples  : lattices per p for pc estimation curve.
    seed          : master random seed.
    device        : torch device (auto-detected if None).
    save_path     : path to save trained model weights.

    Returns
    -------
    dict with keys:
        "model"      : trained PercolationCNN
        "history"    : training history dict
        "p_eval"     : p values used for pc curve
        "prob_super" : CNN super-critical probability at each p
        "pc_cnn"     : CNN pc estimate
        "device"     : device used
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    if p_train_low  is None: p_train_low  = pc_fss - 0.12
    if p_train_high is None: p_train_high = pc_fss + 0.12
    if p_eval is None:
        p_eval = np.linspace(pc_fss - 0.08, pc_fss + 0.08, 40)

    # Build training p values: exclude gap around pc
    p_all  = np.linspace(p_train_low, p_train_high, 30)
    p_train_vals = p_all[np.abs(p_all - pc_fss) >= p_gap]
    print(f"  Training p values: {len(p_train_vals)} points "
          f"(gap ±{p_gap} around pc={pc_fss:.4f})")

    # --- Dataset ---
    dataset = PercolationDataset(
        L, p_train_vals, samples_per_p,
        pc_threshold=pc_fss, seed=seed,
    )
    n_val   = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    print(f"  Dataset: {n_train} train, {n_val} val samples")

    # --- Model ---
    model = PercolationCNN(L, n_filters=n_filters, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # --- Train ---
    print(f"  Training for {n_epochs} epochs ...")
    history = train_model(model, train_loader, val_loader,
                          n_epochs=n_epochs, lr=lr, device=device)

    # --- Save ---
    if save_path:
        save_model(model, save_path)

    # --- Estimate pc ---
    print(f"  Estimating pc from CNN output ...")
    p_arr, prob_super, pc_cnn = estimate_pc_from_cnn(
        model, L, p_eval,
        samples_per_p=eval_samples, seed=seed + 1, device=device,
    )
    print(f"  CNN pc estimate: {pc_cnn:.5f}  "
          f"[FSS: {pc_fss:.5f},  theory: {THEORY['pc']:.5f}]")

    return {
        "model":       model,
        "history":     history,
        "p_eval":      p_arr,
        "prob_super":  prob_super,
        "pc_cnn":      pc_cnn,
        "device":      device,
    }
