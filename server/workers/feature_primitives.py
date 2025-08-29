from pathlib import Path
import pandas as pd
import numpy as np
from math import hypot

IN_PATH = Path(__file__).resolve().parents[2] / "data" / "features" / "events_sessionized.parquet"
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "features" / "windows_5s.parquet"

WINDOW = 5.0  # seconds
RAGE_MS = 0.6
RAGE_RADIUS = 50.0
LOW_SPEED = 40.0  # px/s for hover-stall
STALL_MS = 0.7
NEAR_PX = 120.0

ACTIONABLE_PREFIXES = ("button", "a", "input", "select", "textarea", "label")

def _click_bursts(clicks):
    # clicks: list of (t,x,y)
    if len(clicks) < 3:
        return 0
    clicks = sorted(clicks)
    rage = 0
    for i in range(len(clicks)-2):
        t0,x0,y0 = clicks[i]
        t1,x1,y1 = clicks[i+1]
        t2,x2,y2 = clicks[i+2]
        if (t2 - t0) <= RAGE_MS and hypot(x2-x0, y2-y0) <= RAGE_RADIUS:
            rage += 1
    return rage

def _direction_changes(seq):
    # seq of scroll y positions with time; count sign changes in dy
    if len(seq) < 3:
        return 0
    dirs = []
    for i in range(1,len(seq)):
        dy = seq[i][1] - seq[i-1][1]
        if dy > 0: dirs.append(1)
        elif dy < 0: dirs.append(-1)
        else: dirs.append(0)
    # count nonzero sign flips
    flips = 0
    last = 0
    for d in dirs:
        if d == 0: 
            continue
        if last != 0 and np.sign(d) != np.sign(last):
            flips += 1
        last = d
    return flips

def main():
    df = pd.read_parquet(IN_PATH)
    # ensure types
    for c in ["x","y"]: 
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values(["sess_key","ts"]).reset_index(drop=True)

    # compute windows
    rows = []
    for sess_key, g in df.groupby("sess_key", sort=False):
        start = g["ts"].min()
        end = g["ts"].max()
        win_starts = np.arange(start, end + 1e-6, WINDOW)

        # pre-extract per-type
        moves = g[g["ev"]=="mousemove"][["ts","x","y"]].dropna().to_numpy()
        clicks = g[g["ev"]=="click"][["ts","x","y","el"]].to_numpy()
        scrolls = []
        gv = g[g["ev"]=="scroll"]
        for _, row in gv.iterrows():
            vy = None
            if isinstance(row.get("view"), dict):
                vy = row["view"].get("y", None)
            else:
                # some rows might store JSON-like strings; ignore for now
                pass
            if vy is not None:
                scrolls.append((row["ts"], float(vy)))

        # precompute cursor speed
        speeds = []
        for i in range(1, len(moves)):
            t0,x0,y0 = moves[i-1]
            t1,x1,y1 = moves[i]
            dt = max((t1 - t0), 1e-6)
            d = hypot((x1-x0) or 0.0, (y1-y0) or 0.0)
            speeds.append((t1, d/dt))  # speed at t1

        # loop windows
        for w in win_starts:
            w0, w1 = w, w + WINDOW
            # speeds
            win_speeds = [s for (t,s) in speeds if w0 <= t < w1]
            sp_mean = float(np.mean(win_speeds)) if win_speeds else 0.0
            sp_max  = float(np.max(win_speeds))  if win_speeds else 0.0
            sp_std  = float(np.std(win_speeds))  if win_speeds else 0.0

            # clicks
            cks = [(float(t), float(x or 0.0), float(y or 0.0), str(el) if el is not None else "") 
                   for (t,x,y,el) in clicks if w0 <= t < w1]
            rage = _click_bursts([(t,x,y) for (t,x,y,_) in cks])

            dead = 0
            for (_, _, _, el) in cks:
                el_str = (el or "").lower()
                if not el_str.startswith(ACTIONABLE_PREFIXES):
                    dead += 1

            # hover-stall proxy:
            # if we have a click in window, look back 1s before it: if speeds < LOW_SPEED for >= STALL_MS and cursor near click coords
            stall = 0
            for (t,x,y,_) in cks:
                near_samples = [ (tt,ss) for (tt,ss) in speeds if (t-1.0) <= tt < t and ss < LOW_SPEED ]
                if near_samples:
                    # assume near if we had any recent mousemove (we don't track distance to element center reliably)
                    dur = len(near_samples) * 0.05  # approx if we sampled ~20Hz; conservative proxy
                    if dur >= STALL_MS:
                        stall += 1

            # scroll
            sw = [(t,vy) for (t,vy) in scrolls if w0 <= t < w1]
            osc = _direction_changes(sw)
            vel = 0.0
            if len(sw) >= 2:
                dt = (sw[-1][0] - sw[0][0]) or 1e-6
                dy = (sw[-1][1] - sw[0][1])
                vel = float(dy / dt)
            depth = float(max([vy for (_,vy) in sw], default=0.0))

            rows.append({
                "sess_key": sess_key,
                "w_start": w0,
                "w_end": w1,
                "speed_mean": sp_mean,
                "speed_max": sp_max,
                "speed_std": sp_std,
                "rage_clicks": rage,
                "dead_clicks": dead,
                "hover_stall": stall,
                "scroll_velocity": vel,
                "scroll_oscillations": osc,
                "scroll_depth": depth,
                "clicks": len(cks),
                "moves": int(np.sum([1 for (t,_) in speeds if w0 <= t < w1])),
            })

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_PATH, index=False)
    print(f"[feature_primitives] wrote {len(out)} window rows → {OUT_PATH}")

if __name__ == "__main__":
    main()
