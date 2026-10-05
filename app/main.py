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
        "Pass village GeoJSON from the boundary service to get field boundaries "
        "clipped to each village polygon. Multiple villages can be processed together."
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
    summary="Delineate field boundaries within village GeoJSON",
    responses={
        200: {"description": "Per-village predictions clipped to supplied village boundaries"},
        422: {"model": ErrorResponse, "description": "Invalid village GeoJSON"},
        404: {"model": ErrorResponse, "description": "No villages match the requested taluka"},
        500: {"model": ErrorResponse, "description": "Internal inference error"},
    },
)
async def predict(req: PredictRequest):
    """
    **Request body fields**

    | Field | Type | Required | Default | Description |
    |-------|------|----------|---------|-------------|
    | `geojson` | object or array | ✅ | — | Village Feature, FeatureCollection, or array of Features |
    | `taluka` | string | ❌ | — | Filter supplied village Features by `properties.taluka` |

    **Example**
    ```json
    { "geojson": { "type": "Feature", "geometry": { "type": "Polygon", "coordinates": [] }, "properties": { "name": "Mangrud", "taluka": "Bhiwapur" } }, "taluka": "Bhiwapur" }
    ```
    """
    try:
        if isinstance(req.geojson, list):
            villages = req.geojson
        elif req.geojson.get("type") == "Feature":
            villages = [req.geojson]
        elif req.geojson.get("type") == "FeatureCollection":
            villages = req.geojson.get("features", [])
        else:
            raise ValueError("geojson must be a Feature, FeatureCollection, or array of Features")

        if not villages or any(village.get("type") != "Feature" for village in villages):
            raise ValueError("geojson must contain at least one GeoJSON Feature")

        if req.taluka:
            requested_taluka = req.taluka.strip().casefold()
            villages = [
                village for village in villages
                if str((village.get("properties") or {}).get("taluka", "")).strip().casefold()
                == requested_taluka
            ]
            if not villages:
                raise HTTPException(
                    status_code=404,
                    detail=f"No village features found for taluka '{req.taluka}'.",
                )

        results = []
        for village in villages:
            properties = village.get("properties") or {}
            results.append({
                "village": properties.get("name"),
                "taluka": properties.get("taluka"),
                "geojson": extract_fields(village, model=model, device=device),
            })
        return JSONResponse(content={"taluka": req.taluka, "villages": results})
    except HTTPException:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        device=str(next(model.parameters()).device),
    )
