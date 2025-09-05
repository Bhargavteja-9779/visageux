from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WIN_PATH = REPO / "data" / "features" / "windows_5s.parquet"
MET_PATH = REPO / "data" / "metrics" / "metrics_5s.parquet"
OUT = REPO / "data" / "reports" / "ablation_stability.csv"

def compute_metrics_from_windows(w: pd.DataFrame, drop_cursor=False, drop_scroll=False):
    # Recompute UFI/RCS with terms zeroed depending on ablation
    out = pd.DataFrame()
    out["sess_key"] = w["sess_key"]
    # terms from metrics.py formula:
    rage = np.clip(w["rage_clicks"]/2.0, 0, 1)
    dead = np.clip(w["dead_clicks"]/2.0, 0, 1)
    stall= np.clip(w["hover_stall"]/1.0, 0, 1)
    osc  = np.clip(w["scroll_oscillations"]/3.0, 0, 1)
    jitter = 1.0 / (1.0 + np.exp(-(w["speed_std"].fillna(0.0))/150.0)) * (w["clicks"]==0).astype(float)

    if drop_cursor:
      jitter = 0.0
    if drop_scroll:
      osc = 0.0
      v = 0.0
    else:
      v = w["scroll_velocity"].fillna(0.0)

    UFI = 0.35*rage + 0.20*dead + 0.15*stall + 0.15*osc + 0.15*jitter
    rcs_base = 0.6*(1.0/(1.0+np.exp(-( -w["speed_std"].fillna(0.0)/150.0)))) + 0.4*(1.0/(1.0+np.exp(-(v/200.0))))
    RCS = np.clip(rcs_base, 0, 1) * (1.0 - osc)
    return pd.DataFrame({"sess_key":w["sess_key"], "UFI":UFI, "RCS":RCS})

def main():
    w = pd.read_parquet(WIN_PATH)
    met = pd.read_parquet(MET_PATH)[["sess_key","UFI","RCS"]].copy()
    base = met.groupby("sess_key")[["UFI","RCS"]].mean()

    cur0 = compute_metrics_from_windows(w, drop_cursor=True, drop_scroll=False).groupby("sess_key")[["UFI","RCS"]].mean()
    scr0 = compute_metrics_from_windows(w, drop_cursor=False, drop_scroll=True).groupby("sess_key")[["UFI","RCS"]].mean()

    # Spearman correlation between base and ablated (session-level)
    from scipy.stats import spearmanr
    rows=[]
    for name, df2 in [("drop_cursor", cur0), ("drop_scroll", scr0)]:
        for col in ["UFI","RCS"]:
            a = base[col].reindex(df2.index).dropna()
            b = df2[col].reindex(a.index).dropna()
            r,p = spearmanr(a, b)
            rows.append({"ablation":name, "metric":col, "spearman_r": float(r), "pval": float(p)})
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[ablation] wrote {OUT}:\n", out.to_string(index=False))

if __name__ == "__main__":
    main()
