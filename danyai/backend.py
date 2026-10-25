import os
import numpy as np

try:
    import cupy as cp
    _HAS_CP = True
except Exception:
    _HAS_CP = False

GPU = os.environ.get("DANYAI_GPU", "0") == "1" and _HAS_CP
xp = cp if GPU else np


def to_numpy(a):
    if _HAS_CP and isinstance(a, cp.ndarray):
        return a.get()
    return np.asarray(a)


def to_scalar(a):
    return float(to_numpy(a).item() if hasattr(to_numpy(a), "item") else a)


def scatter_add(arr, idx, val):
    if GPU:
        cp.scatter_add(arr, idx, val)
    else:
        np.add.at(arr, idx, val)