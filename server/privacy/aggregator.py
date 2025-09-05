from __future__ import annotations
import pandas as pd
from pathlib import Path
from .dp import dp_count, dp_mean, dp_sum, suppress_small_cells

REPO = Path(__file__).resolve().parents[2]
METRICS_5S = REPO / "data" / "metrics" / "metrics_5s.parquet"

DEFAULTS = {
    "epsilon": 1.0,    # tune in dashboard
    "clip_lo": 0.0,
    "clip_hi": 1.0,
    "k": 5
}

def load_metrics() -> pd.DataFrame:
    return pd.read_parquet(METRICS_5S).copy()

def dp_group_aggregate(group_by, metric: str, agg: str, epsilon: float, k: int, clip_lo: float, clip_hi: float):
    df = load_metrics()
    # enforce k-anon first
    df2 = suppress_small_cells(df, group_by, k=k)
    if df2.empty:
        return pd.DataFrame(columns=group_by + [f"{agg}_{metric}_dp", "n_dp"])

    out_rows = []
    for keys, g in df2.groupby(group_by, sort=False):
        s = g[metric].dropna()
        n = len(s)
        if agg == "mean":
            val = dp_mean(s, epsilon=epsilon, clip_lo=clip_lo, clip_hi=clip_hi)
        elif agg == "sum":
            val = dp_sum(s, epsilon=epsilon, clip_lo=clip_lo, clip_hi=clip_hi)
        elif agg == "count":
            val = dp_count(n, epsilon=epsilon)
        else:
            raise ValueError("agg must be one of: mean,sum,count")
        rec = dict(zip(group_by, keys if isinstance(keys, tuple) else (keys,)))
        rec[f"{agg}_{metric}_dp"] = val
        rec["n_dp"] = dp_count(n, epsilon=epsilon)
        out_rows.append(rec)
    return pd.DataFrame(out_rows)
