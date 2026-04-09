import tempfile, os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
from uvicorn import lifespan
from app.model import load_model
from app.inference import preprocess, postprocess, mask_to_geojson
import rasterio, numpy as np
from app.schema import PredictionResponse, HealthResponse, ErrorResponse


app = FastAPI(
    title="Field Boundary Detection API - v1.0 (Base Model), 30CH Input",
    description="Initial production release featuring the 30-channel NetCDF inference engine. This version establishes the core FastAPI infrastructure, handles file-based inference requests, and implements the foundational raster-to-GeoJSON vectorization logic for agricultural field extraction.",
    version="1.0.1"
)

# Load model once at startup — critical for performance
model = load_model()

from fastapi.responses import JSONResponse
import json

@app.post("/api/v1/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        with torch.no_grad():
            tensor = preprocess(tmp_path)          # (1, 30, 256, 256)
            logits = model(tensor)

        pred_mask = postprocess(logits)         # (H, W) uint8

        geojson_path = mask_to_geojson(
            pred_mask,
            reference_nc_path=tmp_path,
            output_dir=tempfile.gettempdir()
        )

        with open(geojson_path) as f:
            geojson_data = json.load(f)

        return JSONResponse(content=geojson_data)

    finally:
        os.unlink(tmp_path)   # always clean up upload


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        device=str(next(model.parameters()).device)
    )
    
    
