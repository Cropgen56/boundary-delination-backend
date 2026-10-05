import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.model import load_model
from app.inference import extract_fields
from app.schema import PredictRequest, HealthResponse, ErrorResponse

app = FastAPI(
    title="Field Boundary Detection API — v2.0",
    description=(
        "Triple-head EfficientNet-B4 UNet model (fine-tuned on Kurankhed data). "
        "Pass a centre coordinate and an AOI size (2 – 10 km²) to get a "
        "GeoJSON FeatureCollection of delineated field boundaries."
    ),
    version="2.0.0",
)

# ── Load model once at startup (critical for performance) ─────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = load_model()
model.to(device)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/predict",
    summary="Delineate field boundaries for a given location & AOI size",
    responses={
        200: {"description": "GeoJSON FeatureCollection of predicted field polygons"},
        422: {"model": ErrorResponse, "description": "Validation error (e.g. box_km out of range)"},
        500: {"model": ErrorResponse, "description": "Internal inference error"},
    },
)
async def predict(req: PredictRequest):
    """
    **Request body fields**

    | Field | Type | Required | Default | Description |
    |-------|------|----------|---------|-------------|
    | `center_lat` | float | ✅ | — | Latitude of AOI centre (WGS84) |
    | `center_lon` | float | ✅ | — | Longitude of AOI centre (WGS84) |
    | `box_km` | float | ❌ | `3.0` | AOI side length in km — **2 to 10** |

    **Example**
    ```json
    { "center_lat": 17.702059, "center_lon": 76.006878, "box_km": 5.0 }
    ```
    """
    try:
        geojson = extract_fields(
            center_lat=req.center_lat,
            center_lon=req.center_lon,
            box_km=req.box_km,
            model=model,
            device=device,
        )
        return JSONResponse(content=geojson)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        device=str(next(model.parameters()).device),
    )
