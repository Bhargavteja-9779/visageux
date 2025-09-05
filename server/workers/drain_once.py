import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import redis

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = REPO_ROOT / "data" / "parquet"
OUTDIR.mkdir(parents=True, exist_ok=True)

r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)
QUEUE = "events"

def main():
    batch = []
    # Pop everything currently in Redis
    while True:
        raw = r.lpop(QUEUE)
        if raw is None:
            break
        try:
            batch.append(json.loads(raw))
        except Exception as e:
            print("skip bad json:", e)

    if not batch:
        print("[drain] queue empty — nothing to write.")
        return

    df = pd.DataFrame(batch)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    path = OUTDIR / f"events_{stamp}.parquet"
    df.to_parquet(path, index=False)
    print(f"[drain] wrote {len(df)} rows → {path}")

if __name__ == "__main__":
    main()
