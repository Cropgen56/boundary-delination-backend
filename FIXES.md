# Field Boundary Detection API — Fix Tracker

> Analysed on: 2026-10-10  
> Server version: `2.0.0`

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Fixed |
| 🔲 | Pending |

---

## 🔴 P0 — Will cause failure in production

### P0-1 · Event loop blocked by `extract_fields` in `async def predict`

| Field | Detail |
|-------|--------|
| **File** | `app/main.py` · line 104 |
| **Status** | ✅ Fixed — 2026-10-10 |

**Root cause**  
`extract_fields` calls `requests.get()` (blocking I/O), `time.sleep()`, and heavy NumPy/PyTorch work inside an `async` handler — freezing the server for the duration of every prediction.

**Fix applied**
```diff
- prediction = extract_fields(village, model=model, device=device)
+ prediction = await asyncio.to_thread(
+     extract_fields, village, model=model, device=device
+ )
```

---

### P0-2 · `time.sleep()` inside tile fetcher blocks the event loop

| Field | Detail |
|-------|--------|
| **File** | `app/inference.py` · lines 178, 203 |
| **Status** | ✅ Fixed — 2026-10-10 (resolved by P0-1) |

**Root cause**  
`time.sleep(2 ** attempt)` (retry backoff) and `time.sleep(0.5)` (polite tile delay) ran on the event loop thread.

**Fix applied**  
No code change needed. Because `extract_fields` is now dispatched via `asyncio.to_thread`, all `time.sleep` calls run inside the thread-pool thread.

---

### P0-3 · No input size limit — OOM risk for large villages

| Field | Detail |
|-------|--------|
| **File** | `app/inference.py` · line 394  ·  `app/config.py` |
| **Status** | ✅ Fixed — 2026-10-10 |

**Root cause**  
With `RES_M = 1.0`, a 10×10 km village generates a 10,000×10,000 px mosaic (~300 MB of float32 arrays). No guard existed.

**Fix applied**

`app/config.py`:
```python
MAX_BBOX_KM2 = float(os.environ.get('MAX_BBOX_KM2', 100))  # km²
```

`app/inference.py` — guard before any network I/O:
```python
area_km2 = (maxx - minx) * (maxy - miny) / 1e6
if area_km2 > MAX_BBOX_KM2:
    raise InvalidVillageGeometryError(
        f'Village bounding box ({area_km2:.1f} km²) exceeds the '
        f'maximum allowed area of {MAX_BBOX_KM2:.0f} km². '
        'Split the village into smaller tiles and retry.'
    )
```
Returns **HTTP 422**. Limit is env-overridable via `MAX_BBOX_KM2`.

---

## 🟠 P1 — Reliability risk under normal use

### P1-1 · No request timeout — requests can hang indefinitely

| Field | Detail |
|-------|--------|
| **File** | `app/main.py` · `predict` handler |
| **Status** | ✅ Fixed — 2026-10-10 |

**Root cause**  
No `asyncio.wait_for` or middleware timeout. A slow ESRI server or large village holds the connection open forever.

**Planned fix**
```python
try:
    prediction = await asyncio.wait_for(
        asyncio.to_thread(extract_fields, village, model=model, device=device),
        timeout=300,  # 5 minutes per village
    )
except asyncio.TimeoutError:
    raise HTTPException(status_code=504, detail="Prediction timed out.")
```

---

### P1-2 · Model loaded at module import time — crash if checkpoint missing

| Field | Detail |
|-------|--------|
| **File** | `app/main.py` · lines 27–30 |
| **Status** | ✅ Fixed — 2026-10-10 |

**Root cause**  
`load_model()` runs at import time. If `ft_best.pt` is absent the app fails to start with no graceful message.

**Planned fix** — use FastAPI `lifespan`:
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model()
    model.to(device)
    logger.info('Field boundary model loaded on %s.', device)
    yield

app = FastAPI(..., lifespan=lifespan)
```

---

### P1-3 · No concurrency lock around model inference

| Field | Detail |
|-------|--------|
| **File** | `app/main.py` · `predict` handler |
| **Status** | ✅ Fixed — 2026-10-10 |

**Root cause**  
Safe today with `--workers 1`, but concurrent requests share the same model object. Scaling will risk GPU OOM or corrupted tensor state.

**Planned fix**
```python
_inference_semaphore = asyncio.Semaphore(1)

async with _inference_semaphore:
    prediction = await asyncio.to_thread(
        extract_fields, village, model=model, device=device
    )
```

---

### P1-4 · `ValueError` in GeoJSON branch returns HTTP 500 instead of 422

| Field | Detail |
|-------|--------|
| **File** | `app/main.py` · line 71 |
| **Status** | ✅ Fixed — 2026-10-10 |

**Root cause**  
`raise ValueError(...)` falls through to the bare `except Exception` block → HTTP 500 for a clear client error.

**Planned fix**
```diff
- raise ValueError("geojson must be a Feature, FeatureCollection, or array of Features")
+ raise HTTPException(
+     status_code=422,
+     detail="geojson must be a Feature, FeatureCollection, or array of Features",
+ )
```

---

## 🟡 P2 — Code quality / operational concerns

### P2-1 · Unbounded disk cache — no eviction or size limit

| Field | Detail |
|-------|--------|
| **File** | `app/inference.py` · `_save_cached_imagery` |
| **Status** | ✅ Fixed — 2026-10-10 |

**Root cause**  
`.npz` imagery files grow forever in `/data/`. Will eventually fill ephemeral container storage.

**Planned fix**  
After each save, scan the cache dir and delete oldest files when total size exceeds `CACHE_MAX_GB` (default 5 GB).

---

### P2-2 · `ErrorResponse` schema imported but never used in route specs

| Field | Detail |
|-------|--------|
| **File** | `app/main.py` · line 12  ·  `app/schema.py` · lines 37–38 |
| **Status** | ✅ Fixed — 2026-10-10 |

**Root cause**  
`ErrorResponse` is defined and imported but not wired into any `responses={}` dict — error shapes are absent from the OpenAPI spec.

**Planned fix**
```python
responses={
    422: {"model": ErrorResponse, "description": "Invalid village GeoJSON"},
    502: {"model": ErrorResponse, "description": "Imagery provider failed"},
    500: {"model": ErrorResponse, "description": "Internal inference error"},
}
```

---

### P2-3 · `np.ptp()` deprecated — removed in NumPy 2.0

| Field | Detail |
|-------|--------|
| **File** | `app/inference.py` · line 290 |
| **Status** | ✅ Fixed — 2026-10-10 |

**Root cause**  
`np.ptp()` was removed in NumPy 2.0. Raises `AttributeError` on any environment with NumPy ≥ 2.0.

**Planned fix**
```diff
- dist_norm = (distance_pred - distance_pred.min()) / (np.ptp(distance_pred) + 1e-9)
+ dist_norm = (distance_pred - distance_pred.min()) / ((distance_pred.max() - distance_pred.min()) + 1e-9)
```

---

## Progress Summary

| Severity | Total | ✅ Fixed | 🔲 Pending |
|----------|-------|---------|---------|
| 🔴 P0 | 3 | 3 | 0 |
| 🟠 P1 | 4 | 4 | 0 |
| 🟡 P2 | 3 | 3 | 0 |
| **Total** | **10** | **10** | **0** |
