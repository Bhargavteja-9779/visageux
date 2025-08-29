from pathlib import Path
import pandas as pd
import numpy as np
import json

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "parquet"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION_GAP = 30 * 60  # 30 minutes in seconds

def load_events() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("events_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw event parquet files in {RAW_DIR}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # normalize columns that may be missing
    for col in ["sid","uid","ts","ev","x","y","el","dom","view","aff","perf"]:
        if col not in df.columns: df[col] = np.nan
    # ensure numeric
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    # sort
    df = df.sort_values(["uid","sid","ts"], kind="mergesort").reset_index(drop=True)
    return df

def sessionize(df: pd.DataFrame) -> pd.DataFrame:
    # new session when gap > SESSION_GAP or uid/sid changes
    grp = df.groupby(["uid","sid"], sort=False)
    parts = []
    for (uid,sid), g in grp:
        g = g.copy()
        gap = g["ts"].diff().fillna(0.0)
        new_flag = (gap > SESSION_GAP).astype(int)
        sess_incr = new_flag.cumsum()
        g["sess_key"] = g["uid"].astype(str) + "@" + g["sid"].astype(str) + ":" + sess_incr.astype(str)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    return out

def main():
    df = load_events()
    sess = sessionize(df)
    path = OUT_DIR / "events_sessionized.parquet"
    sess.to_parquet(path, index=False)
    print(f"[sessionize] wrote {len(sess)} rows → {path}")

if __name__ == "__main__":
    main()
