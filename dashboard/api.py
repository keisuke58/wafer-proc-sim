"""
FastAPI backend — live sensor data for the dicing dashboard.

Endpoints:
    GET  /api/history?n=60   last N WaferReading records (JSON array)
    GET  /api/latest         most recent reading
    POST /api/reset          restart from wafer 0
    POST /api/config         update anomaly_rate / interval_s / n_life
    GET  /api/stream         SSE — one JSON event per new wafer

Run:
    uvicorn dashboard.api:app --reload --port 8000
"""
import asyncio
import json
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dashboard.simulator import get_simulator

app = FastAPI(title="wafer-proc-sim", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sim = get_simulator()
sim.start()


class Config(BaseModel):
    anomaly_rate: float = 0.15
    interval_s:   float = 1.0
    n_life:       int   = 200


@app.get("/api/history")
def history(n: int = 60):
    return sim.get_history(n)


@app.get("/api/latest")
def latest():
    return sim.get_latest() or {}


@app.post("/api/reset")
def reset():
    sim.reset()
    return {"status": "reset"}


@app.post("/api/config")
def configure(cfg: Config):
    sim.configure(
        anomaly_rate=cfg.anomaly_rate,
        interval_s=cfg.interval_s,
        n_life=cfg.n_life,
    )
    return {"status": "ok", "config": cfg.model_dump()}


@app.get("/api/stream")
async def stream():
    """SSE endpoint — sends one JSON data event per new wafer."""
    async def gen():
        last_id = -1
        while True:
            r = sim.get_latest()
            if r and r["wafer_id"] != last_id:
                last_id = r["wafer_id"]
                yield f"data: {json.dumps(r)}\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
