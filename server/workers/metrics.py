from pathlib import Path
import pandas as pd
import numpy as np

IN_PATH = Path(__file__).resolve().parents[2] / "data" / "features" / "windows_5s.parquet"
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "metrics" / "metrics_5s.parquet"

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main():
    w = pd.read_parquet(IN_PATH).copy()

    # --- UFI components ---
    rage_c = np.clip(w["rage_clicks"] / 2.0, 0, 1)
    dead_c = np.clip(w["dead_clicks"] / 2.0, 0, 1)
    stall_c = np.clip(w["hover_stall"] / 1.0, 0, 1)
    osc_c = np.clip(w["scroll_oscillations"] / 3.0, 0, 1)
    jitter_c = sigmoid((w["speed_std"].fillna(0.0)) / 200.0) * (w["clicks"] == 0).astype(float)

    ufi = (
        0.35 * rage_c +
        0.20 * dead_c +
        0.15 * stall_c +
        0.15 * osc_c +
        0.15 * jitter_c
    )
    ufi = np.clip(ufi, 0, 1)

    # --- RCS ---
    # higher when speed_std low and scroll_velocity positive & smooth; penalize oscillations
    rcs = (0.6 * sigmoid(-(w["speed_std"].fillna(0.0)) / 150.0) +
           0.4 * sigmoid((w["scroll_velocity"].fillna(0.0)) / 200.0))
    rcs = np.clip(rcs, 0, 1) * (1.0 - np.clip(w["scroll_oscillations"] / 3.0, 0, 1))

    # --- MIV proxy ---
    # decision latency: if clicks>0 in window, we estimate "scroll stop" as low |scroll_velocity|
    # For this v1, use: if |scroll_velocity| < 20 px/s → assume stopped at w_start; latency = time to first click center of window.
    # NOTE: This is a coarse proxy; later we’ll refine using raw events.
    stopped = (w["scroll_velocity"].abs() < 20.0).astype(float)
    # assume first click happens mid-window when clicks>0; better: carry over click timestamps later
    approx_first_click_at = w["w_start"] + 2.5
    miv = np.where(w["clicks"] > 0,
                   np.where(stopped > 0, approx_first_click_at - w["w_start"], np.nan),
                   np.nan)

    out = w[["sess_key","w_start","w_end"]].copy()
    out["UFI"] = ufi
    out["RCS"] = rcs
    out["MIV"] = miv
    # Keep useful counts for debugging
    out["rage_clicks"] = w["rage_clicks"]
    out["dead_clicks"] = w["dead_clicks"]
    out["hover_stall"] = w["hover_stall"]
    out["scroll_oscillations"] = w["scroll_oscillations"]
    out["scroll_velocity"] = w["scroll_velocity"]
    out["speed_std"] = w["speed_std"]
    out["clicks"] = w["clicks"]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"[metrics] wrote {len(out)} rows → {OUT_PATH}")

if __name__ == "__main__":
    main()
