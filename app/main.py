"""SymptomSense API (FastAPI)."""
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app import db
from app.model_service import ModelService, UnknownSymptomError

service = ModelService()
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "dnn")


@asynccontextmanager
async def lifespan(_app):
    db.init_db()
    yield


app = FastAPI(title="SymptomSense API", version="1.0.0",
              description="Prediksi kategori diagnosis dari daftar gejala (decision support untuk SayMed/BPRC).",
              lifespan=lifespan)


class PredictRequest(BaseModel):
    symptoms: list[str] = Field(..., min_length=1, max_length=30, examples=[["back pain", "leg pain", "hip pain"]])
    model: Literal["dnn", "logreg"] = DEFAULT_MODEL
    top_k: int = Field(3, ge=1, le=10)
    save: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "models": ["dnn", "logreg"], "n_symptoms": len(service.symptoms),
            "n_classes": len(service.classes)}


@app.get("/symptoms")
def symptoms():
    return {"symptoms": service.symptoms}


@app.get("/models")
def models():
    return service.metrics


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = service.predict(req.symptoms, model=req.model, top_k=req.top_k)
    except UnknownSymptomError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    result["saved_id"] = db.save_prediction(req.model, req.symptoms, result) if req.save else None
    return result


@app.get("/history")
def history(limit: int = Query(50, ge=1, le=500)):
    return {"items": db.list_predictions(limit)}


@app.get("/stats")
def stats():
    return db.stats()
