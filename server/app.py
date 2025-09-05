from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Union
import redis

from .events import Event

app = FastAPI(title="VisageUX API", version="0.1.0")

# CORS so the extension/front-end can POST later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # tighten later
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to local Redis once at startup
def _redis():
    return redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)

r = _redis()

@app.get("/health")
def health():
    ok = True
    redis_ok = False
    try:
        r.ping()
        redis_ok = True
    except Exception:
        pass
    return {"ok": ok, "service": "visageux-api", "redis": redis_ok}

@app.post("/ingest")
def ingest(payload: Union[Event, List[Event]] = Body(...)):
    """
    Accept either a single Event object or a list of Events.
    Push each JSON string to Redis list 'events'.
    """
    try:
        if isinstance(payload, list):
            for ev in payload:
                # Pydantic v2
                r.rpush("events", ev.model_dump_json())
            return {"status": "queued", "count": len(payload)}
        else:
            r.rpush("events", payload.model_dump_json())
            return {"status": "queued", "count": 1}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
