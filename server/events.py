from pydantic import BaseModel, Field
from typing import Optional, Dict

class Event(BaseModel):
    # minimal, privacy-first
    sid: str = Field(..., description="hashed session id")
    uid: str = Field(..., description="hashed user id")
    ts: float = Field(..., description="epoch seconds")
    ev: str = Field(..., description="event type e.g. click/scroll/move")
    x: Optional[int] = None
    y: Optional[int] = None
    el: Optional[str] = None
    dom: Optional[str] = None
    view: Optional[Dict] = None
    aff: Optional[str] = None
    perf: Optional[Dict] = None
