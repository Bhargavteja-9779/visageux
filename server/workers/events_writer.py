import json
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import redis

# ---------- Paths ----------
# .../visageux/server/workers/events_writer.py → repo root is 3 levels up
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = REPO_ROOT / "data" / "parquet"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- Queue ----------
QUEUE_NAME = "events"
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)

# ---------- Flush policy ----------
BATCH_SIZE = 100          # write every 100 events
FLUSH_SECONDS = 10        # or every 10s, whichever first

def write_batch(batch):
    if not batch:
        return
    df = pd.DataFrame(batch)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    path = OUTDIR / f"events_{stamp}.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    print(f"[writer] wrote {len(batch)} → {path}")

def run():
    print(f"[writer] watching Redis list '{QUEUE_NAME}'…")
    buf = []
    last = time.time()

    while True:
        # Blocking pop with timeout so we can time-flush
        item = r.blpop(QUEUE_NAME, timeout=1)
        if item:
            _, raw = item
            try:
                ev = json.loads(raw)
                buf.append(ev)
            except Exception as e:
                print(f"[writer] JSON decode error: {e!r}")

        if buf and (len(buf) >= BATCH_SIZE or (time.time() - last) >= FLUSH_SECONDS):
            write_batch(buf)
            buf.clear()
            last = time.time()

if __name__ == "__main__":
    run()
