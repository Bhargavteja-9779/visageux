from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EV_PATH = REPO / "data" / "features" / "events_sessionized.parquet"
WIN_PATH = REPO / "data" / "features" / "windows_5s.parquet"
MET_PATH = REPO / "data" / "metrics" / "metrics_5s.parquet"
OUT = REPO / "data" / "reports" / "baseline_vs_metrics.csv"

def build_targets(ev: pd.DataFrame, w: pd.DataFrame):
    # drop-off proxy: last window of each session where next gap >=10s is labeled 1 (already used in training)
    # here we'll compute per-session: whether the session ends with a long idle (>=10s)
    # approximate: if (max(ts) - last window end) >= 10s
    ends = ev.groupby("sess_key")["ts"].max().rename("last_ts")
    w_end = w.groupby("sess_key")["w_end"].max().rename("last_w_end")
    tgt = pd.concat([ends, w_end], axis=1).dropna()
    tgt["drop_10s"] = (tgt["last_ts"] - tgt["last_w_end"] >= 10.0).astype(int)
    return tgt

def ga_baselines(ev: pd.DataFrame):
    g = ev.groupby("sess_key")
    first = g["ts"].min().rename("first_ts")
    last  = g["ts"].max().rename("last_ts")
    n_ev  = g.size().rename("events_n")
    # bounce: very short sessions; use <10s and <=1 click as proxy
    clicks = ev[ev["ev"]=="click"].groupby("sess_key").size().rename("clicks_n")
    df = pd.concat([first, last, n_ev, clicks], axis=1).fillna(0)
    df["dwell_s"] = df["last_ts"] - df["first_ts"]
    df["bounce"]  = ((df["dwell_s"] < 10) & (df["clicks_n"] <= 1)).astype(int)
    return df

def our_metrics(met: pd.DataFrame):
    # mean UFI/RCS per session; MIV median ignoring NaN
    g = met.groupby("sess_key")
    out = pd.DataFrame({
        "UFI_mean": g["UFI"].mean(),
        "RCS_mean": g["RCS"].mean(),
        "MIV_median": g["MIV"].median()
    })
    return out

def main():
    ev = pd.read_parquet(EV_PATH)
    w  = pd.read_parquet(WIN_PATH)
    met= pd.read_parquet(MET_PATH)

    tgt = build_targets(ev, w)
    ga  = ga_baselines(ev)
    ours= our_metrics(met)

    df = ga.join(ours, how="inner").join(tgt["drop_10s"], how="inner")
    df = df.dropna()

    # correlations with drop_10s
    # Spearman rank corr is robust
    from scipy.stats import spearmanr
    rows=[]
    for col in ["bounce","dwell_s","events_n","UFI_mean","RCS_mean","MIV_median"]:
        try:
            r,p = spearmanr(df[col], df["drop_10s"])
            rows.append({"feature": col, "spearman_r": float(r), "pval": float(p)})
        except Exception as e:
            rows.append({"feature": col, "spearman_r": np.nan, "pval": np.nan, "err": str(e)})
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[report] wrote {OUT}:\n", out.to_string(index=False))

if __name__ == "__main__":
    main()
