from __future__ import annotations
import numpy as np
import pandas as pd

def laplace_noise(scale: float, size=None):
    return np.random.laplace(loc=0.0, scale=scale, size=size)

def clip_series(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.clip(lower=lo, upper=hi)

def dp_count(n: int, epsilon: float) -> float:
    # sensitivity=1 for count; Laplace(b=1/epsilon)
    b = 1.0 / max(epsilon, 1e-6)
    return float(n + np.random.laplace(0.0, b))

def dp_sum(s: pd.Series, epsilon: float, clip_lo: float, clip_hi: float) -> float:
    # sensitivity = (clip_hi-clip_lo)
    s = clip_series(s, clip_lo, clip_hi)
    sens = (clip_hi - clip_lo)
    b = sens / max(epsilon, 1e-6)
    return float(s.sum() + np.random.laplace(0.0, b))

def dp_mean(s: pd.Series, epsilon: float, clip_lo: float, clip_hi: float) -> float:
    s = clip_series(s, clip_lo, clip_hi)
    # mean sensitivity = (clip_hi-clip_lo) / n
    n = max(len(s), 1)
    sens = (clip_hi - clip_lo) / n
    b = sens / max(epsilon, 1e-6)
    return float(s.mean() + np.random.laplace(0.0, b))

def suppress_small_cells(df: pd.DataFrame, group_cols, k: int) -> pd.DataFrame:
    sizes = df.groupby(group_cols).size().rename("n")
    keep_idx = sizes[sizes >= k].index
    mask = df.set_index(group_cols).index.isin(keep_idx)
    return df.loc[mask].copy()
