import tempfile, os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
from model import load_model
from inference import preprocess, postprocess, mask_to_geojson
import rasterio, numpy as np
from schema import PredictionResponse, HealthResponse, ErrorResponse


app = FastAPI()

# Load model once at startup — critical for performance
model = load_model()

from fastapi.responses import JSONResponse
import json

@app.post("/predict")
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
    
    
