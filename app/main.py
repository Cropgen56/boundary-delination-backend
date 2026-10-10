import asyncio
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.model import load_model
from app.inference import (
    ImageryFetchError,
    InvalidVillageGeometryError,
    extract_fields,
)
from app.schema import PredictRequest, HealthResponse, ErrorResponse

logger = logging.getLogger(__name__)

# ── P1-3: one inference at a time — prevents GPU OOM under concurrent load ────
_inference_semaphore = asyncio.Semaphore(1)

# Module-level refs populated by lifespan (typed for static analysis)
device: torch.device
model:  torch.nn.Module


# ── P1-2: lifespan replaces bare module-level load_model() call ───────────────
# Errors here (e.g. missing checkpoint) produce a clear log message and prevent
# the server from starting, rather than crashing mid-import with a stack trace.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading field boundary model on %s …', device)
    model = load_model()
    model.to(device)
    logger.info('Field boundary model ready on %s.', device)
    yield
    # nothing to release for a pure-inference model, but the hook is here
    # if GPU memory or other resources need explicit cleanup in future.


app = FastAPI(
    title="Field Boundary Detection API — v2.0",
    description=(
        "Triple-head EfficientNet-B4 UNet model (fine-tuned on Kurankhed data). "
        "Pass village GeoJSON from the boundary service to get field boundaries "
        "clipped to each village polygon. Multiple villages can be processed together."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/predict",
    summary="Delineate field boundaries within village GeoJSON",
    responses={
        200: {"description": "GeoJSON FeatureCollection of predictions clipped to supplied village boundaries"},
        # P2-2: typed error responses so OpenAPI spec shows the exact error schema
        404: {"model": ErrorResponse, "description": "No villages match the requested taluka"},
        422: {"model": ErrorResponse, "description": "Invalid village GeoJSON or geometry too large"},
        502: {"model": ErrorResponse, "description": "Imagery provider request failed"},
        504: {"model": ErrorResponse, "description": "Prediction timed out — village may be too large"},
        500: {"model": ErrorResponse, "description": "Internal inference error"},
    },
)
async def predict(req: PredictRequest):
    """
    **Request body fields**

    The body can be a raw GeoJSON Feature, FeatureCollection, or feature array.
    To filter by taluka, use `{ "geojson": <GeoJSON>, "taluka": "Bhiwapur" }`.

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
            # P1-4: raise 422 directly — ValueError fell through to the 500 handler
            raise HTTPException(
                status_code=422,
                detail="geojson must be a Feature, FeatureCollection, or array of Features",
            )

        if not villages or any(
            not isinstance(village, dict) or village.get("type") != "Feature"
            for village in villages
        ):
            raise HTTPException(
                status_code=422,
                detail="geojson must contain at least one GeoJSON Feature",
            )

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

        logger.info(
            'Prediction request started: villages=%d, taluka=%s.',
            len(villages), req.taluka or 'not specified',
        )
        predicted_features = []
        for village in villages:
            properties = village.get("properties") or {}
            village_name = properties.get("name", "unnamed")
            logger.info('Starting prediction for village %s.', village_name)
            # P0:  asyncio.to_thread → blocking work off the event loop
            # P1-1: asyncio.wait_for → hard 5-minute cap per village (HTTP 504)
            # P1-3: _inference_semaphore → one GPU/CPU inference at a time
            try:
                async with _inference_semaphore:
                    prediction = await asyncio.wait_for(
                        asyncio.to_thread(
                            extract_fields, village, model=model, device=device
                        ),
                        timeout=300,  # seconds — override via P1-1 config if needed
                    )
            except asyncio.TimeoutError:
                logger.error(
                    'Prediction timed out for village %s after 300 s.', village_name
                )
                raise HTTPException(
                    status_code=504,
                    detail=(
                        f"Prediction for village '{village_name}' timed out. "
                        "The village may be too large — try splitting it."
                    ),
                )
            for feature in prediction.get("features", []):
                feature_properties = dict(feature.get("properties") or {})
                feature_properties["village"] = properties.get("name")
                feature_properties["taluka"] = properties.get("taluka")
                feature["properties"] = feature_properties
                predicted_features.append(feature)
            logger.info(
                'Finished prediction for village %s: fields=%d.',
                village_name, len(prediction.get('features', [])),
            )
        return JSONResponse(content={
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:OGC:1.3:CRS84",
                },
            },
            "features": predicted_features,
        })
    except HTTPException:
        raise
    except InvalidVillageGeometryError as exc:
        logger.warning('Invalid prediction request: %s', exc)
        raise HTTPException(status_code=422, detail=f"Invalid village GeoJSON: {exc}") from exc
    except ImageryFetchError as exc:
        logger.exception('Imagery provider failed during prediction.')
        raise HTTPException(
            status_code=502,
            detail='Unable to fetch imagery from the ESRI provider. Please try again later.',
        ) from exc
    except Exception as exc:
        logger.exception('Unexpected error during field boundary prediction.')
        raise HTTPException(
            status_code=500,
            detail='Prediction failed due to an internal server error. Check server logs.',
        ) from exc


@app.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        device=str(next(model.parameters()).device),
    )
