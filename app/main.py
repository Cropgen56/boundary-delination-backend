import asyncio
import json
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

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
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
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

# Allow browser clients (e.g. the test UI) to call the API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Serve the test UI and example GeoJSON
import os
example_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example")
app.mount("/ui", StaticFiles(directory=example_dir), name="ui")


# ── Shared helpers ────────────────────────────────────────────────────────────────

def _parse_villages(req: PredictRequest) -> list[dict]:
    """Extract and validate the list of village Features from a request.
    Raises HTTPException on invalid input."""
    geojson = req.geojson
    if isinstance(geojson, list):
        villages = geojson
    elif geojson.get("type") == "Feature":
        villages = [geojson]
    elif geojson.get("type") == "FeatureCollection":
        villages = geojson.get("features", [])
    else:
        raise HTTPException(
            status_code=422,
            detail="geojson must be a Feature, FeatureCollection, or array of Features",
        )

    if not villages or any(
        not isinstance(v, dict) or v.get("type") != "Feature" for v in villages
    ):
        raise HTTPException(
            status_code=422,
            detail="geojson must contain at least one GeoJSON Feature",
        )

    if req.taluka:
        tfilter = req.taluka.strip().casefold()
        villages = [
            v for v in villages
            if str((v.get("properties") or {}).get("taluka", "")).strip().casefold()
            == tfilter
        ]
        if not villages:
            raise HTTPException(
                status_code=404,
                detail=f"No village features found for taluka '{req.taluka}'.",
            )

    return villages


def _sse_json(data: dict) -> str:
    """Format a dict as a Server-Sent Event data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


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
    villages = _parse_villages(req)
    logger.info(
        'Prediction request started: villages=%d, taluka=%s.',
        len(villages), req.taluka or 'not specified',
    )
    try:
        predicted_features = []
        for village in villages:
            properties = village.get("properties") or {}
            village_name = properties.get("name", "unnamed")
            logger.info('Starting prediction for village %s.', village_name)
            try:
                async with _inference_semaphore:
                    prediction = await asyncio.wait_for(
                        asyncio.to_thread(
                            extract_fields, village, model=model, device=device
                        ),
                        timeout=300,
                    )
            except asyncio.TimeoutError:
                logger.error('Prediction timed out for village %s after 300 s.', village_name)
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
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
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


@app.post(
    "/api/v1/predict/stream",
    summary="Stream field boundary predictions with live progress events (SSE)",
    response_class=StreamingResponse,
)
async def predict_stream(req: PredictRequest):
    """
    Identical input to `/api/v1/predict` but returns a **text/event-stream** (SSE)
    response so clients can display real-time pipeline progress.

    Each event is a JSON object with a `type` field:
    - `start`        — inference is beginning
    - `village_start`— processing a specific village
    - `progress`     — pipeline stage update (`step`, `message`, `pct` 0–100)
    - `village_done` — one village finished (`field_count`)
    - `result`       — final GeoJSON FeatureCollection (last event)
    - `error`        — something went wrong (`detail`)
    """
    villages = _parse_villages(req)

    async def event_stream():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        all_features: list = []

        def _progress(step: str, message: str, pct: int):
            """Called from the thread-pool thread — must be thread-safe."""
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "progress", "step": step, "message": message, "pct": pct},
            )

        try:
            yield _sse_json({"type": "start", "village_count": len(villages)})

            for idx, village in enumerate(villages):
                props = village.get("properties") or {}
                vname = props.get("name", "unnamed")
                yield _sse_json({"type": "village_start", "index": idx, "name": vname})
                logger.info('SSE stream: starting village %s.', vname)

                # Sentinel placed in the queue when the thread finishes
                result_holder: list = []
                error_holder:  list = []

                async def _run(v=village):
                    try:
                        res = await asyncio.to_thread(
                            extract_fields, v,
                            model=model, device=device,
                            progress_callback=_progress,
                        )
                        result_holder.append(res)
                    except Exception as exc:  # noqa: BLE001
                        error_holder.append(exc)
                    finally:
                        # always signal completion so the consumer loop exits
                        loop.call_soon_threadsafe(queue.put_nowait, {"type": "__done__"})

                task = asyncio.create_task(_run())
                keepalive_tick = 0

                while True:
                    # Drain all queued items without blocking
                    while not queue.empty():
                        item = queue.get_nowait()
                        if item["type"] == "__done__":
                            break
                        yield _sse_json(item)
                    else:
                        # No __done__ sentinel seen yet — yield and sleep briefly
                        await asyncio.sleep(0.15)
                        keepalive_tick += 1
                        if keepalive_tick % 7 == 0:   # ~every 1 s
                            yield ": keepalive\n\n"
                        continue
                    break  # broke out of inner while on __done__

                await task  # ensure the task coroutine is fully finished

                if error_holder:
                    exc = error_holder[0]
                    if isinstance(exc, InvalidVillageGeometryError):
                        yield _sse_json({"type": "error", "detail": f"Invalid geometry: {exc}"})
                    elif isinstance(exc, ImageryFetchError):
                        yield _sse_json({"type": "error", "detail": "ESRI imagery fetch failed."})
                    else:
                        yield _sse_json({"type": "error", "detail": str(exc)})
                    return

                result = result_holder[0] if result_holder else {"features": []}
                for feat in result.get("features", []):
                    fp = dict(feat.get("properties") or {})
                    fp["village"] = props.get("name")
                    fp["taluka"]  = props.get("taluka")
                    feat["properties"] = fp
                    all_features.append(feat)

                field_count = len(result.get("features", []))
                logger.info('SSE stream: village %s done, fields=%d.', vname, field_count)
                yield _sse_json({
                    "type": "village_done",
                    "index": idx,
                    "name": vname,
                    "field_count": field_count,
                })

            yield _sse_json({
                "type": "result",
                "geojson": {
                    "type": "FeatureCollection",
                    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                    "features": all_features,
                },
            })

        except Exception as exc:  # noqa: BLE001
            logger.exception('Unexpected error in SSE stream.')
            yield _sse_json({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
