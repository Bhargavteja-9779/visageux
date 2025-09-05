import redis, json, pandas as pd
from pathlib import Path

r = redis.Redis(host="localhost", port=6379, db=0)

outdir = Path(__file__).resolve().parent.parent / "../data/parquet"
outdir.mkdir(parents=True, exist_ok=True)

def run_worker():
    buffer = []
    while True:
        _, raw = r.blpop("events")   # blocking pop
        event = json.loads(raw)
        buffer.append(event)
        if len(buffer) >= 10:   # write every 10 events
            df = pd.DataFrame(buffer)
            df.to_parquet(outdir / "events.parquet", engine="pyarrow", append=True)
            print(f"Wrote {len(buffer)} events")
            buffer.clear()

if __name__ == "__main__":
    run_worker()
