from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, List

from .model import ModelService
from .config import MAX_SCATTER_ROWS

app = FastAPI(title="Spotify Genre Segmentation API", version="1.0")
service = None

class RecommendRequest(BaseModel):
    track_name: Optional[str] = Field(default=None, description="Exact track name to seed recommendations")
    features: Optional[Dict[str, float]] = Field(default=None, description="Raw features dict (acousticness, energy, etc.)")
    k: int = 10

@app.on_event("startup")
def on_startup():
    global service
    try:
        service = ModelService()
    except Exception as e:
        # Delay failure; allow health endpoint to report reason
        service = None
        print("Startup error:", str(e))

@app.get("/health")
def health():
    if service is None:
        return {"status": "error", "message": "Model not initialized. Check dataset path and server logs."}
    return {"status": "ok"}

@app.get("/summary")
def summary():
    if service is None:
        raise HTTPException(500, "Service not ready")
    return service.summary()

@app.get("/correlation")
def correlation():
    if service is None:
        raise HTTPException(500, "Service not ready")
    return service.correlation()

@app.get("/scatter")
def scatter(limit: int = Query(default=MAX_SCATTER_ROWS, ge=100, le=50000)):
    if service is None:
        raise HTTPException(500, "Service not ready")
    return service.scatter(limit=limit)

@app.get("/search_tracks")
def search_tracks(q: str, limit: int = 20):
    if service is None:
        raise HTTPException(500, "Service not ready")
    return {"results": service.search_tracks(q=q, limit=limit)}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    if service is None:
        raise HTTPException(500, "Service not ready")
    try:
        results = service.recommend(track_name=req.track_name, features=req.features, k=req.k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(400, str(e))
