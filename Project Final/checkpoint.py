"""
checkpoint.py
-------------
Save and load simulation results between runs so you can stop and resume
at any step, or split the physics and ML steps across two machines.

All data is stored as .npz files in CHECKPOINT_DIR (set in main.py).

Usage
-----
From main.py, each expensive step follows this pattern:

    data = ckpt_load("step_name")
    if data is None:
        data = run_expensive_thing(...)
        ckpt_save("step_name", data)

If the checkpoint file exists the simulation is skipped entirely.
Delete a checkpoint file to force that step to re-run.
"""

import os
import numpy as np


_CKPT_DIR = "checkpoints"   # overridden by main.py via set_checkpoint_dir()


def set_checkpoint_dir(path: str):
    global _CKPT_DIR
    _CKPT_DIR = path
    os.makedirs(_CKPT_DIR, exist_ok=True)


def _path(name: str) -> str:
    return os.path.join(_CKPT_DIR, f"{name}.npz")


# ---------------------------------------------------------------------------
# Generic save / load
# ---------------------------------------------------------------------------

def ckpt_save(name: str, data: dict):
    """
    Save a dict of numpy arrays (and scalars) to a checkpoint file.
    Lists of dicts (like ns_density) are serialised via object arrays.
    """
    os.makedirs(_CKPT_DIR, exist_ok=True)
    flat = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            flat[k] = v
        elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], dict):
            # e.g. ns_density: list of dicts → object array
            flat[k] = np.array(v, dtype=object)
        elif isinstance(v, dict):
            # e.g. nested dicts — serialise as object array of length 1
            flat[k] = np.array([v], dtype=object)
        else:
            flat[k] = np.array(v)
    np.savez_compressed(_path(name), **flat)
    print(f"  [ckpt] Saved: {_path(name)}")


def ckpt_load(name: str):
    """
    Load a checkpoint. Returns None if the file doesn't exist.
    Reconstructs object arrays back into lists/dicts as appropriate.
    """
    p = _path(name)
    if not os.path.exists(p):
        return None
    print(f"  [ckpt] Loading: {p}")
    raw  = np.load(p, allow_pickle=True)
    data = {}
    for k in raw.files:
        v = raw[k]
        if v.dtype == object:
            if v.ndim == 1 and len(v) == 1 and isinstance(v[0], dict):
                data[k] = v[0]          # single dict
            else:
                data[k] = list(v)       # list of dicts (e.g. ns_density)
        elif v.ndim == 0:
            data[k] = v.item()          # scalar
        else:
            data[k] = v
    return data


def ckpt_exists(name: str) -> bool:
    return os.path.exists(_path(name))


def ckpt_delete(name: str):
    p = _path(name)
    if os.path.exists(p):
        os.remove(p)
        print(f"  [ckpt] Deleted: {p}")


# ---------------------------------------------------------------------------
# Specialised helpers for the results_by_L dict (FSS loop)
# ---------------------------------------------------------------------------

def ckpt_save_fss(results_by_L: dict, pc_of_L: list,
                  estimates_by_L: dict, uncertainties_by_L: dict,
                  Smax_at_pc: dict):
    """Save the full FSS loop results as a single checkpoint."""
    payload = {
        "L_vals":    np.array(list(results_by_L.keys()), dtype=int),
        "pc_of_L":   np.array(pc_of_L, dtype=float),
        "Smax_at_pc_keys":   np.array(list(Smax_at_pc.keys()),   dtype=int),
        "Smax_at_pc_values": np.array(list(Smax_at_pc.values()), dtype=float),
        "estimates_by_L":    np.array([estimates_by_L],   dtype=object),
        "uncertainties_by_L":np.array([uncertainties_by_L], dtype=object),
    }
    # Save each L's results dict separately so arrays are stored efficiently
    for L_val, res in results_by_L.items():
        for key, val in res.items():
            arr_key = f"fss_{L_val}_{key}"
            if isinstance(val, list) and val and isinstance(val[0], dict):
                payload[arr_key] = np.array(val, dtype=object)
            elif isinstance(val, np.ndarray):
                payload[arr_key] = val
            else:
                payload[arr_key] = np.array(val)

    os.makedirs(_CKPT_DIR, exist_ok=True)
    np.savez_compressed(_path("fss_loop"), **payload)
    print(f"  [ckpt] Saved FSS loop: {_path('fss_loop')}")


def ckpt_load_fss():
    """Load the FSS loop checkpoint. Returns None if not found."""
    p = _path("fss_loop")
    if not os.path.exists(p):
        return None
    print(f"  [ckpt] Loading FSS loop: {p}")
    raw = np.load(p, allow_pickle=True)

    L_vals           = raw["L_vals"].tolist()
    pc_of_L          = raw["pc_of_L"].tolist()
    estimates_by_L   = raw["estimates_by_L"][0]
    uncertainties_by_L = raw["uncertainties_by_L"][0]
    Smax_at_pc       = dict(zip(raw["Smax_at_pc_keys"].tolist(),
                                raw["Smax_at_pc_values"].tolist()))

    results_by_L = {}
    for L_val in L_vals:
        res = {}
        prefix = f"fss_{L_val}_"
        for key in raw.files:
            if key.startswith(prefix):
                obs = key[len(prefix):]
                v   = raw[key]
                if v.dtype == object:
                    res[obs] = list(v)
                elif v.ndim == 0:
                    res[obs] = v.item()
                else:
                    res[obs] = v
        results_by_L[L_val] = res

    return {
        "results_by_L":      results_by_L,
        "pc_of_L":           pc_of_L,
        "estimates_by_L":    estimates_by_L,
        "uncertainties_by_L":uncertainties_by_L,
        "Smax_at_pc":        Smax_at_pc,
    }


# ---------------------------------------------------------------------------
# List all checkpoints
# ---------------------------------------------------------------------------

def ckpt_list():
    """Print all existing checkpoint files and their sizes."""
    if not os.path.exists(_CKPT_DIR):
        print("  No checkpoint directory found.")
        return
    files = sorted(f for f in os.listdir(_CKPT_DIR) if f.endswith(".npz"))
    if not files:
        print("  No checkpoints found.")
        return
    print(f"\n  Checkpoints in '{_CKPT_DIR}':")
    for f in files:
        size = os.path.getsize(os.path.join(_CKPT_DIR, f)) / 1024 / 1024
        print(f"    {f:<35s}  {size:.1f} MB")
