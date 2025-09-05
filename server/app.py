from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Union
import redis
from fastapi import Query
from typing import List, Literal
from .privacy.aggregator import dp_group_aggregate, DEFAULTS

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
@app.get("/privacy/config")
def privacy_config():
    return DEFAULTS

@app.post("/privacy/aggregate")
def privacy_aggregate(
    group_by: List[str] = Query(..., description="columns to group by, e.g., sess_key or page_section (later)"),
    metric: Literal["UFI","RCS","MIV"] = Query("UFI"),
    agg: Literal["mean","sum","count"] = Query("mean"),
    epsilon: float = Query(DEFAULTS["epsilon"]),
    k: int = Query(DEFAULTS["k"]),
    clip_lo: float = Query(DEFAULTS["clip_lo"]),
    clip_hi: float = Query(DEFAULTS["clip_hi"]),
):
    try:
        df = dp_group_aggregate(group_by, metric, agg, epsilon, k, clip_lo, clip_hi)
        return {"rows": df.to_dict(orient="records")}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    
@app.get("/privacy/aggregate")
def privacy_aggregate_get(
    group_by: List[str] = Query(..., description="columns to group by, e.g., sess_key"),
    metric: Literal["UFI","RCS","MIV"] = Query("UFI"),
    agg: Literal["mean","sum","count"] = Query("mean"),
    epsilon: float = Query(DEFAULTS["epsilon"]),
    k: int = Query(DEFAULTS["k"]),
    clip_lo: float = Query(DEFAULTS["clip_lo"]),
    clip_hi: float = Query(DEFAULTS["clip_hi"]),
):
    try:
        df = dp_group_aggregate(group_by, metric, agg, epsilon, k, clip_lo, clip_hi)
        return {"rows": df.to_dict(orient="records")}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
