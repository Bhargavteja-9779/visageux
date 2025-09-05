from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

REPO = Path(__file__).resolve().parents[2]
RAW_EVENTS = REPO / "data" / "features" / "events_sessionized.parquet"
WIN5 = REPO / "data" / "features" / "windows_5s.parquet"

FEATURES = [
    "speed_mean",
    "speed_max",
    "speed_std",
    "rage_clicks",
    "dead_clicks",
    "hover_stall",
    "scroll_velocity",
    "scroll_oscillations",
    "scroll_depth",
    "clicks",
    "moves",
]

def load_sources() -> Tuple[pd.DataFrame, pd.DataFrame]:
    ev = pd.read_parquet(RAW_EVENTS).copy()
    ev["ts"] = pd.to_numeric(ev["ts"], errors="coerce")
    ev = ev.dropna(subset=["ts"]).sort_values(["sess_key","ts"]).reset_index(drop=True)

    w = pd.read_parquet(WIN5).copy()
    for c in ["w_start","w_end"]:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w.dropna(subset=["w_start","w_end"]).sort_values(["sess_key","w_start"]).reset_index(drop=True)
    return ev, w

def compute_next_event_gap(ev: pd.DataFrame, w: pd.DataFrame) -> pd.Series:
    # map for quick lookup per sess
    gaps = []
    by_sess = {k: g["ts"].to_numpy() for k,g in ev.groupby("sess_key", sort=False)}
    for sess_key, wstart, wend in w[["sess_key","w_start","w_end"]].itertuples(index=False):
        ts = by_sess.get(sess_key)
        if ts is None or ts.size == 0:
            gaps.append(np.inf)
            continue
        # next raw event strictly AFTER w_end
        nxt = ts[ts > wend]
        if nxt.size == 0:
            gaps.append(np.inf)
        else:
            gaps.append(float(nxt[0] - wend))
    return pd.Series(gaps, index=w.index, dtype=float)

def make_labels(w: pd.DataFrame, gaps: pd.Series, horizon_sec: float = 10.0) -> pd.Series:
    # 1 if next event is >= horizon away (or none), else 0
    y = (gaps >= horizon_sec).astype(int)
    return y

def zscore_fit(X: pd.DataFrame) -> Dict[str, Tuple[float,float]]:
    stats = {}
    for c in FEATURES:
        s = float(X[c].std(ddof=0) or 1.0)
        m = float(X[c].mean())
        stats[c] = (m, s)
    return stats

def zscore_apply(X: pd.DataFrame, stats: Dict[str, Tuple[float,float]]) -> np.ndarray:
    arr = []
    for c in FEATURES:
        m, s = stats[c]
        arr.append(((X[c].values - m) / (s if s != 0 else 1.0)).astype("float32"))
    return np.vstack(arr).T  # [N, F]

def build_sequences(w: pd.DataFrame, y: pd.Series, L: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X_seq: [N, L, F]
      y_seq: [N] label for the LAST window in each sequence
    """
    rows = []
    labels = []
    for sk, g in w.reset_index().groupby("sess_key", sort=False):
        idxs = g["index"].to_numpy()
        # sliding window indices
        for i in range(len(idxs) - L + 1):
            win = idxs[i:i+L]
            rows.append(win)
            labels.append(int(y.iloc[win[-1]]))
    return np.array(rows, dtype=int), np.array(labels, dtype=int)
