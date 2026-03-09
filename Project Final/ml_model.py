"""
ml_model.py
-----------
CNN-based phase classifier for 2-D site percolation.

The network takes raw lattice snapshots (L x L binary images) as input
and learns to classify them as sub-critical (p < pc) or super-critical
(p > pc). No physics is given to the network — it discovers the order
parameter from the images alone.

Once trained, the network's output confidence as a function of p gives
an independent estimate of pc: the crossing point where it is equally
unsure whether a configuration is sub- or super-critical.

Classes / functions
-------------------
  PercolationDataset   — PyTorch Dataset wrapping generated lattices.
  PercolationCNN       — the convolutional classifier.
  train_model()        — training loop with validation tracking.
  estimate_pc_from_cnn() — extract pc from the network's sigmoid output.
  save_model() / load_model() — persistence helpers.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from observables import generate_lattice


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PercolationDataset(Dataset):
    """
    Generates labelled lattice snapshots on the fly.

    Each sample is a (1, L, L) float32 tensor (single-channel image).
    The label is 1 (super-critical) if p > pc_threshold, else 0.

    Training data is drawn from p values away from pc so the network
    sees clear examples. The ambiguous region near pc is used only
    at evaluation time to locate the transition.

    Parameters
    ----------
    L             : lattice side length.
    p_values      : array of p values to sample from.
    samples_per_p : number of lattice realisations per p value.
    pc_threshold  : boundary between class 0 and class 1.
    seed          : random seed for reproducibility.
    """

    def __init__(self, L, p_values, samples_per_p, pc_threshold, seed=0):
        self.L         = L
        self.threshold = pc_threshold

        rng = np.random.default_rng(seed)
        lattices, labels = [], []

        for p in p_values:
            label = 1 if p > pc_threshold else 0
            for _ in range(samples_per_p):
                lat = generate_lattice(L, p, rng=rng).astype(np.float32)
                lattices.append(lat[np.newaxis])   # add channel dim → (1, L, L)
                labels.append(label)

        # Store as tensors
        self.X = torch.from_numpy(np.stack(lattices))   # (N, 1, L, L)
        self.y = torch.tensor(labels, dtype=torch.long) # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# CNN architecture
# ---------------------------------------------------------------------------

class PercolationCNN(nn.Module):
    """
    Convolutional classifier for L x L percolation lattices.

    Architecture
    ------------
    Two convolutional blocks (conv → batch norm → ReLU → max pool),
    followed by two fully-connected layers with dropout.

    The conv layers learn local spatial features (cluster connectivity).
    The FC layers combine them into a global phase prediction.

    Output is a 2-class logit vector; apply softmax to get probabilities.

    Parameters
    ----------
    L        : lattice side length (determines FC input size).
    n_filters: number of filters in each conv layer (default 32).
    dropout  : dropout probability in FC layers (default 0.3).
    """

    def __init__(self, L, n_filters=32, dropout=0.3):
        super().__init__()

        self.conv_block = nn.Sequential(
            # Block 1: (1, L, L) → (n_filters, L/2, L/2)
            nn.Conv2d(1, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: → (n_filters*2, L/4, L/4)
            nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Size of flattened conv output
        conv_out = n_filters * 2 * (L // 4) * (L // 4)

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),   # 2 classes: sub-critical, super-critical
        )

    def forward(self, x):
        return self.fc_block(self.conv_block(x))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader,
                n_epochs=20, lr=1e-3, device=None):
    """
    Train the CNN with cross-entropy loss and Adam optimiser.

    Parameters
    ----------
    model        : PercolationCNN instance.
    train_loader : DataLoader for training data.
    val_loader   : DataLoader for validation data.
    n_epochs     : number of training epochs.
    lr           : learning rate.
    device       : torch device (auto-detected if None).

    Returns
    -------
    history : dict with keys "train_loss", "val_loss", "val_acc"
              — one value per epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model   = model.to(device)
    optim   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, n_epochs + 1):

        # --- Training ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optim.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optim.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                val_loss += loss_fn(logits, y_batch).item() * len(y_batch)
                correct  += (logits.argmax(1) == y_batch).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc   = correct / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch:3d}/{n_epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.3f}")

    return history


# ---------------------------------------------------------------------------
# pc estimation from CNN output
# ---------------------------------------------------------------------------

def estimate_pc_from_cnn(model, L, p_values, samples_per_p=200,
                          seed=9999, device=None, batch_size=64):
    """
    Estimate pc as the p where the network's super-critical probability = 0.5.

    For each p in p_values, generate samples_per_p lattices, pass them
    through the network, and record the mean P(super-critical). The
    crossing point of this curve at 0.5 is the CNN's estimate of pc.

    Parameters
    ----------
    model         : trained PercolationCNN.
    L             : lattice side length.
    p_values      : dense array of p values covering the transition.
    samples_per_p : lattices per p value (more = smoother curve).
    seed          : random seed.
    device        : torch device.

    Returns
    -------
    p_values  : the input p array.
    prob_super: mean P(super-critical) at each p.
    pc_cnn    : estimated critical point (float).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    rng        = np.random.default_rng(seed)
    prob_super = np.zeros(len(p_values))
    softmax    = nn.Softmax(dim=1)

    for k, p in enumerate(p_values):
        lattices = np.stack([
            generate_lattice(L, p, rng=rng).astype(np.float32)[np.newaxis]
            for _ in range(samples_per_p)
        ])
        X = torch.from_numpy(lattices)

        probs = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch  = X[i:i + batch_size].to(device)
                logits = model(batch)
                probs.append(softmax(logits)[:, 1].cpu().numpy())

        prob_super[k] = np.concatenate(probs).mean()

    # Find crossing point via linear interpolation
    from scipy.interpolate import interp1d
    try:
        if prob_super.min() < 0.5 < prob_super.max():
            f     = interp1d(prob_super, p_values, kind="linear")
            pc_cnn = float(f(0.5))
        else:
            pc_cnn = float(p_values[np.argmin(np.abs(prob_super - 0.5))])
    except Exception:
        pc_cnn = float(p_values[np.argmin(np.abs(prob_super - 0.5))])

    return np.array(p_values), prob_super, pc_cnn


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, path="percolation_cnn.pt"):
    """Save model weights to disk."""
    torch.save(model.state_dict(), path)
    print(f"  Model saved: {path}")


def load_model(L, path="percolation_cnn.pt", n_filters=32, dropout=0.3):
    """Load model weights from disk. Returns model in eval mode."""
    model = PercolationCNN(L, n_filters=n_filters, dropout=dropout)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"  Model loaded: {path}")
    return model
