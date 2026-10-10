# Field Boundary Detection API

This project exposes a FastAPI service for detecting agricultural field boundaries from village GeoJSON polygons. It uses a deep learning model based on a triple-head UNet with EfficientNet-B4 encoder and returns GeoJSON field polygons clipped to each village boundary.

The server is designed to:

- accept one or more village features
- fetch RGB imagery from ESRI World Imagery
- run sliding-window inference over the village area
- generate field-instance predictions with watershed-based segmentation
- post-process, simplify, and filter polygons
- reproject results from UTM back to WGS84
- return GeoJSON ready for GIS or map UI consumption

---

## 1. Overview

The backend is implemented in Python and FastAPI. It loads a trained checkpoint file, performs inference on each village, and returns the detected field polygons as GeoJSON FeatureCollection objects.

### Main components

- `app/main.py` — API server, request validation, lifecycle management, SSE streaming endpoint
- `app/inference.py` — full field extraction pipeline and image processing logic
- `app/model.py` — model architecture and checkpoint loading
- `app/config.py` — global hyperparameters, safety limits, and cache settings
- `app/schema.py` — Pydantic request and response models
- `example/` — example UI and GeoJSON sample files
- `checkpoints/` — trained model checkpoint location

---

## 2. Architecture and runtime flow

The service is structured around a single inference pipeline that runs per village.

### Startup lifecycle

When the app starts, FastAPI runs `lifespan()` in `app/main.py`:

1. detects available accelerator (`cuda`, `mps`, or `cpu`)
2. loads the model checkpoint from `CHECKPOINT_PATH`
3. moves the model to the selected device
4. keeps a single global model reference for inference requests

This ensures the model is loaded once at startup rather than on every request.

### Request handling

The API accepts a request payload that can be any of the following:

- a single GeoJSON Feature
- a GeoJSON FeatureCollection
- a list of GeoJSON Features

The request is normalized by Pydantic and validated in `_parse_villages()`.

### Inference flow per village

For each village, the server runs the following steps:

1. Validate the geometry
2. Determine the correct UTM CRS from the village centroid
3. Reproject the village geometry to UTM
4. Check the bbox size against a safety limit
5. Fetch RGB imagery from ESRI World Imagery
6. Run sliding-window model inference
7. Apply village mask
8. Extract field instances via watershed-based segmentation
9. Convert instance labels into polygon candidates
10. Simplify and filter polygons
11. Reproject to WGS84
12. Return GeoJSON features

---

## 3. Deep learning model

The model is defined in `app/model.py` and is a custom triple-head UNet.

### Model structure

- encoder: EfficientNet-B4 backbone from `segmentation-models-pytorch`
- decoder: UNet decoder
- output heads:
  - `extent_head` → probability map for field presence
  - `boundary_head` → probability map for field boundaries
  - `distance_head` → distance-to-boundary map

The model returns three outputs per input patch:

- extent probability tensor
- boundary probability tensor
- distance tensor

These outputs are later combined to build a final field-instance segmentation map.

### Checkpoint loading

The model is loaded from `CHECKPOINT_PATH` using PyTorch `torch.load(...)` with `map_location='cpu'` and `weights_only=True`.

Default checkpoint path:

```bash
checkpoints/ft_best.pt
```

This path can be overridden with:

```bash
export CHECKPOINT_PATH=/path/to/model.pt
```

---

## 4. Pipeline details

The actual field extraction pipeline is defined in `app/inference.py`.

### 4.1 Geometry validation

The input village feature must be a valid GeoJSON `Feature` with a `Polygon` or `MultiPolygon` geometry.

The code rejects:

- missing geometry
- invalid GeoJSON
- empty geometry
- unsupported geometry types
- village bounding boxes above the configured safety threshold

### 4.2 UTM projection and bounds

The pipeline calculates the appropriate UTM CRS from longitude and latitude, then reprojects the village polygon into that local UTM system for accurate distance and area calculations.

This is important because:

- imagery tiling and raster alignment happen in a projected coordinate system
- field dimensions and simplification are done in metres
- area/perimeter values are calculated in metric units before being converted back to WGS84

### 4.3 Imagery fetching

Imagery is pulled from the ESRI World Imagery service.

Important implementation details:

- the request is made in Web Mercator (EPSG:3857) for the ESRI export endpoint
- the bounding box is split into tiles for memory efficiency
- tile chunks are stitched together into a single mosaic
- the mosaic is cached on disk in the `data/` directory as `.npz` files
- cached imagery is reused for repeated requests with the same village geometry
- a size-based cache eviction rule removes older files when total cache size exceeds `CACHE_MAX_GB`

### 4.4 Sliding-window inference

The imagery mosaic is processed as overlapping patches using a Hann-window blend to reduce seam artifacts.

The model runs in inference mode with `torch.no_grad()` enabled to reduce memory use and speed inference.

### 4.5 Post-processing and segmentation

The model produces extent, boundary, and distance predictions. These are combined to derive a field-instance label map through watershed segmentation.

This includes:

- normalization of the distance map
- Sobel edge extraction from the distance channel
- Gaussian smoothing of the boundary map
- morphological closing and hole filling
- h-minima watershed-based instance separation
- small object removal

### 4.6 Polygon refinement

Candidate polygons are processed using:

- dagger removal for narrow spikes
- Douglas-Peucker simplification
- clipping to the village polygon
- removal of invalid or empty geometries
- filtering by a relative area threshold based on median field size

The resulting polygons are assigned metrics:

- `area_m2`
- `area_ha`
- `perimeter_m`

Finally they are reprojected to EPSG:4326 and returned as GeoJSON.

---

## 5. API endpoints

Base URL is usually:

```text
http://localhost:8080
```

### 5.1 Health check

#### GET /health

Returns service health information and model status.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

`device` is the runtime device selected by the server, for example:

- `cuda`
- `mps`
- `cpu`

---

### 5.2 Predict fields

#### POST /api/v1/predict

This is the main inference endpoint.

Request body requirements:

- `geojson`: required
- `taluka`: optional string filter

Accepted `geojson` types:

- GeoJSON `Feature`
- GeoJSON `FeatureCollection`
- list of GeoJSON `Feature` objects

Example request:

```json
{
  "geojson": {
    "type": "Feature",
    "geometry": {
      "type": "Polygon",
      "coordinates": [
        [
          [73.72, 19.98],
          [73.73, 19.98],
          [73.73, 19.99],
          [73.72, 19.99],
          [73.72, 19.98]
        ]
      ]
    },
    "properties": {
      "name": "Borgaon",
      "taluka": "Bhiwapur"
    }
  },
  "taluka": "Bhiwapur"
}
```

Example response:

```json
{
  "type": "FeatureCollection",
  "crs": {
    "type": "name",
    "properties": {
      "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
    }
  },
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[...]]]
      },
      "properties": {
        "field_id": 1,
        "area_m2": 245.2,
        "area_ha": 0.024,
        "perimeter_m": 69.8,
        "village": "Borgaon",
        "taluka": "Bhiwapur"
      }
    }
  ]
}
```

This endpoint returns a complete GeoJSON FeatureCollection of detected field polygons.

---

### 5.3 Streaming progress endpoint

#### POST /api/v1/predict/stream

This endpoint streams progress events as Server-Sent Events (SSE). It is useful for UI dashboards that want to show live model progress.

The stream emits JSON messages with a `type` field.

Common event types:

- `start` — stream begins
- `village_start` — a village is beginning
- `progress` — stage updates such as validation, imagery fetch, inference, post-processing
- `village_done` — village finished with field count
- `result` — final GeoJSON FeatureCollection
- `error` — request or processing failure

Example SSE payload:

```json
{"type":"start","village_count":1}
{"type":"village_start","index":0,"name":"Borgaon"}
{"type":"progress","step":"imagery","message":"Fetching RGB imagery from ESRI World Imagery …","pct":15}
{"type":"village_done","index":0,"name":"Borgaon","field_count":42}
{"type":"result","geojson":{"type":"FeatureCollection","features":[...]}}
```

This is the endpoint used by the HTML test UI in `/ui`.

---

## 6. Test UI

The app mounts a static directory from the `example/` folder at `/ui`.

This allows the browser UI to load assets and call the API using a simple frontend. The example UI is useful for local testing before integrating the backend into a larger application.

Browse to:

```text
http://localhost:8080/ui
```

or with the example page path if available in your deployment environment.

---

## 7. Error handling and response codes

The API returns HTTP errors for malformed input, external failures, or processing problems.

### 7.1 400 / 404 / 422 / 502 / 504 / 500

The server explicitly maps common errors to these status codes:

#### 404 Not Found

Returned when the optional `taluka` filter matches no villages.

Example:

```json
{
  "detail": "No village features found for taluka 'X'."
}
```

#### 422 Unprocessable Entity

Used for invalid village GeoJSON or geometry problems.

Examples:

- malformed GeoJSON
- non-Feature object
- invalid polygon geometry
- bounding box exceeds configured maximum area

Typical detail messages include:

```json
{
  "detail": "geojson must be a Feature, FeatureCollection, or array of Features"
}
```

or

```json
{
  "detail": "Village bounding box (150.2 km²) exceeds the maximum allowed area of 100 km². Split the village into smaller tiles and retry."
}
```

#### 502 Bad Gateway

Returned when imagery cannot be fetched from the ESRI provider.

```json
{
  "detail": "Unable to fetch imagery from the ESRI provider. Please try again later."
}
```

#### 504 Gateway Timeout

Returned when inference for a village takes longer than the internal timeout window.

```json
{
  "detail": "Prediction for village 'X' timed out. The village may be too large — try splitting it."
}
```

#### 500 Internal Server Error

Returned when an unexpected application error occurs.

```json
{
  "detail": "Prediction failed due to an internal server error. Check server logs."
}
```

---

## 8. Request validation rules

### Accepted input

The server expects a GeoJSON village definition that is geographically valid and in WGS84 coordinates.

### Filter behavior

If `taluka` is specified, the service filters village features by:

```python
properties.taluka
```

Comparison is case-insensitive after trimming whitespace.

### Important safety constraint

Large village bounding boxes are rejected before any imagery fetch occurs.

The check is configured in `app/config.py`:

```python
MAX_BBOX_KM2 = float(os.environ.get('MAX_BBOX_KM2', 100))
```

This prevents memory blowups when a village is too large for the model and imagery pipeline.

---

## 9. Configuration

The main runtime settings live in `app/config.py`.

### Core parameters

```python
PATCH_PX = 256
RES_M = 1.0
OVERSEG_H = 0.18
EXTENT_THRESH = 0.5
SIMPLIFY_TOL_M = 2.0
DAGGER_WIDTH_M = 2.0
MIN_AREA_FRAC = 0.10
MAX_BBOX_KM2 = 100
CACHE_MAX_GB = 5
```

### Meaning of important settings

- `PATCH_PX` — model input tile size
- `RES_M` — spatial resolution in metres per pixel
- `OVERSEG_H` — watershed seeding depth
- `EXTENT_THRESH` — threshold for field mask creation
- `SIMPLIFY_TOL_M` — polygon simplification tolerance in metres
- `DAGGER_WIDTH_M` — spike removal tolerance
- `MIN_AREA_FRAC` — minimum allowed polygon area relative to median field size
- `MAX_BBOX_KM2` — maximum allowed village area before request rejection
- `CACHE_MAX_GB` — maximum disk usage of cached imagery

### Environment overrides

These values can be driven by environment variables in deployment environments:

```bash
export MAX_BBOX_KM2=100
export CACHE_MAX_GB=5
export CHECKPOINT_PATH=/path/to/checkpoint.pt
```

---

## 10. Local development and running the server

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the app

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### Run with Docker Compose

```bash
docker-compose up --build
```

The service is then available at:

```text
http://localhost:8080
```

### Health check

```bash
curl http://localhost:8080/health
```

---

## 11. Project structure

```text
.
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── inference.py
│   ├── main.py
│   ├── model.py
│   └── schema.py
├── checkpoints/
│   └── ft_best.pt
├── data/
│   └── cached imagery .npz files
├── example/
│   ├── borgaon_manju.geojson
│   └── test_ui.html
├── notebooks/
│   └── ft_inference.ipynb
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── FIXES.md
└── README.md
```

---

## 12. Operational notes

### Caching

The app caches imagery in `data/` to avoid repeated expensive downloads for the same village area.

This helps with:

- repeated inference on the same geometry
- local testing and UI iteration
- reducing external API usage

### Concurrency

The backend uses a single-inference semaphore to serialize heavy model execution and reduce GPU or MPS memory contention.

This is a pragmatic safeguard for a single-worker deployment model.

### Performance trade-offs

Model inference is computationally expensive, and large villages can take a long time to process. The service includes:

- bbox guardrails
- request timeout handling
- per-village inference serialization
- SSE progress updates to keep users informed

### Device support

The server detects available hardware automatically:

- CUDA GPU if available
- Apple MPS if available
- fallback CPU

---

## 13. Typical debugging workflow

If the request fails, the natural debugging sequence is:

1. confirm the village geometry is valid
2. check that the bounding box is under the configured limit
3. validate the `taluka` filter value if used
4. confirm the model checkpoint exists at `CHECKPOINT_PATH`
5. look at the service logs for imagery fetch or inference errors
6. check the `data/` cache directory for stale or corrupted `.npz` files

---

## 14. Best practices for API consumers

- split very large villages before sending them to the API
- prefer returning a single village polygon per request for easier debugging
- use the SSE streaming endpoint when you need a frontend progress indicator
- treat `422` as a request-shape problem rather than a model problem
- treat `502` as an external imagery provider problem
- treat `504` as a large-area processing timeout

---

## 15. Summary

This backend is a complete field boundary extraction service built around a trained UNet model and a geospatial data-processing pipeline. It transforms village GeoJSON into precise field polygons using remote imagery, deep learning inference, and polygon post-processing.

It is intended for:

- agricultural mapping use cases
- farm field delineation
- village-level parcel extraction
- frontend map integrations
- GIS and geospatial processing workflows

If you need to integrate this service with an external application, the recommended flow is:

1. send GeoJSON village features to `/api/v1/predict`
2. parse the returned FeatureCollection
3. optionally use `/api/v1/predict/stream` for live progress in UIs
4. visualize the resulting field polygons on a map or export them to GIS tools

---

## 16. Useful commands

Check the app health:

```bash
curl http://localhost:8080/health
```

Run the app locally:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Run via Docker:

```bash
docker-compose up --build
```

---

This README reflects the actual server implementation in this repository and is intended to document the real API behavior, pipeline, and operational constraints of the service.
