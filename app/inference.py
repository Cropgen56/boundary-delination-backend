"""
inference.py — v3 field boundary pipeline
==========================================
Input  : village GeoJSON Feature(s) with Polygon or MultiPolygon geometry
Output : GeoJSON FeatureCollection of field polygons

Pipeline (v3 algorithm from village_field_polygons_v3.ipynb):
-------------------------------------------------------------
1. Derive UTM CRS from coordinates and validate bounds
2. Fetch RGB imagery from ESRI World Imagery (with S3 & disk caching, parallel chunking)
3. Sliding-window inference with Hann-window blending (channels-last, fp16 autocast)
4. Field instance extraction:
   - SEED_MODE='cores': field cores via robustly normalized distance & low boundary prob,
     selective extent-hole filling, disk(1) closing, flooding watershed, orphan rescue.
   - SEED_MODE='hminima': legacy watershed on blurred boundary/distance edge map.
5. Post-extraction refinement:
   - split_oversized: stepping-down h-minima ladder on bounding box of large regions
   - merge_weak_boundaries: region-adjacency graph merge for borders with boundary prob < MERGE_BND_T
   - rescue compact green patches, absorb small enclosed gaps, and smooth labels
6. Vector polygon post-processing:
   - VECTOR_MODE='topo': shared pixel-edge arcs simplified once with a per-field vertex budget.
   - VECTOR_MODE='coverage': GeoPandas simplify_coverage (shared boundaries simplified once).
   - VECTOR_MODE='legacy': dagger removal, RDP simplify, orthogonal square-up.
7. Funnel: area floor (min_field_m2, median area frac), exact village clip, QC tagging ('ok', 'small', 'oversize').
8. Reproject UTM → WGS84, preserve all input village properties, save to S3 cache, emit GeoJSON.
"""

import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

import numpy as np
import shapely
import torch
import geopandas as gpd
import pyproj

from PIL import Image
from shapely.geometry import shape, mapping, MultiPolygon, Polygon
from shapely.ops import unary_union
from rasterio.features import geometry_mask, shapes as rio_shapes
from rasterio.transform import from_bounds
from skimage.segmentation import watershed, expand_labels
from skimage.morphology import disk, h_minima, h_maxima
from skimage.filters import sobel
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter

from app.config import (
    PATCH_PX, RES_M, BATCH_SIZE, OVERLAP, USE_FP16, TTA,
    SEED_MODE, SEED_DIST_MIN, SEED_BND_MAX, SEED_MIN_PX,
    EDGE_SIGMA, DIST_WEIGHT, MERGE_BND_T, LABEL_SMOOTH_R,
    RESCUE_ORPHANS, MIN_REGION_PX, OVERSEG_H, EXTENT_THRESH,
    EXT_T, BND_NORM, BND_FLOOR, CORE_T, GAP_CLOSE_R, NECK_H,
    MERGE_T, SMOOTH_R, MAX_FIELD_PX,
    VECTOR_MODE, COVERAGE_TOL_M, DAGGER_WIDTH_M, SIMPLIFY_FIRST_M,
    SMOOTH_METHOD, ANGLE_TOL_DEG, SIMPLIFY_AFTER_M,
    SPLIT_OVERSIZED, OVERSIZE_PCTL, MIN_FIELD_M2, MIN_AREA_FRAC,
    SMALL_FIELD_HA, MAX_FIELD_HA, CLIP_TO_VILLAGE,
    MAX_BBOX_KM2, CACHE_MAX_GB,
    ESRI_MAX_WORKERS, ESRI_IMAGE_FORMAT, ESRI_JPEG_QUALITY,
    ESRI_TIMEOUT_SEC, ESRI_MAX_CHUNK_PX,
    ENABLE_ROAD_MASK, ENABLE_URBAN_MASK, ROAD_BARRIER_ELEV,
    TOPO_TOL_M, MAX_VERTICES, LABEL_GAUSS_SIGMA, GAUSS_SIGMA,
    EXTENT_HOLE_MAX_PX, EXTENT_HOLE_MAX_M2,
    TREE_FILL_MAX_PX, TREE_FILL_MAX_M2, TREE_MIN_ENCLOSED, BUND_OPEN_PX, BUND_OPEN_M,
    RESCUE_FIELDS, RESCUE_MIN_PX, RESCUE_MIN_M2, RESCUE_MIN_SOLIDITY,
    RESCUE_MIN_WIDTH_PX, RESCUE_MIN_WIDTH_M, RESCUE_MIN_EXG,
)
from app.result_cache import (
    geometry_cache_digest,
    get_cached_imagery,
    store_cached_imagery_async,
    store_cached_fields,
)
from app.auxiliary import fetch_road_and_urban_masks

# ── Feature detection for simplify_coverage ───────────────────────────────────
HAS_COVERAGE = hasattr(gpd.GeoSeries, 'simplify_coverage')

# ── ESRI World Imagery tile server ────────────────────────────────────────────
_ESRI_URL     = ('https://services.arcgisonline.com/arcgis/rest/services/'
                 'World_Imagery/MapServer/export')
_CACHE_DIR = Path(__file__).resolve().parent.parent / 'data'
logger = logging.getLogger(__name__)


def _village_cache_key(village_geom: object) -> str:
    """Create a stable cache filename for the village geometry."""
    return geometry_cache_digest(village_geom)


def _load_cached_imagery(cache_key: str, minx: float, miny: float, maxx: float, maxy: float):
    """Load cached ESRI imagery from the local data folder if it exists."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f'esri_{cache_key[:32]}.npz'
    if not cache_path.exists():
        return None

    try:
        payload = np.load(cache_path)
        mosaic = payload['mosaic']
        if mosaic.ndim != 3 or mosaic.shape[0] != 3:
            raise ValueError('Cached imagery array is invalid.')
        logger.info('Using cached ESRI imagery from %s.', cache_path)
        return mosaic, from_bounds(minx, miny, maxx, maxy, mosaic.shape[2], mosaic.shape[1])
    except Exception as exc:
        logger.warning('Cached imagery was unreadable at %s: %s. Re-fetching from ESRI.', cache_path, exc)
        try:
            cache_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def _save_cached_imagery(cache_key: str, mosaic: np.ndarray, minx: float, miny: float, maxx: float, maxy: float):
    """Persist fetched ESRI imagery and evict oldest files when the cache
    directory exceeds CACHE_MAX_GB total on-disk size."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f'esri_{cache_key[:32]}.npz'
    np.savez_compressed(
        cache_path,
        mosaic=mosaic,
        minx=np.float64(minx),
        miny=np.float64(miny),
        maxx=np.float64(maxx),
        maxy=np.float64(maxy),
    )
    logger.info('Saved ESRI imagery cache to %s.', cache_path)

    if CACHE_MAX_GB <= 0:
        return cache_path

    limit_bytes = CACHE_MAX_GB * 1024 ** 3
    entries = sorted(
        _CACHE_DIR.glob('esri_*.npz'),
        key=lambda p: p.stat().st_mtime,
    )
    total = sum(p.stat().st_size for p in entries)
    while total > limit_bytes and entries:
        oldest = entries.pop(0)
        freed = oldest.stat().st_size
        try:
            oldest.unlink()
            total -= freed
            logger.info(
                'Cache eviction: deleted %s (freed %.1f MB, total now %.1f MB).',
                oldest.name, freed / 1024**2, total / 1024**2,
            )
        except OSError as exc:
            logger.warning('Cache eviction failed for %s: %s', oldest.name, exc)
            break

    return cache_path


class ImageryFetchError(RuntimeError):
    """Raised when imagery cannot be fetched from the configured provider."""


class InvalidVillageGeometryError(ValueError):
    """Raised when a village feature does not contain valid polygon geometry."""


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def _utm_crs(lon: float, lat: float) -> str:
    """Return the UTM EPSG string appropriate for a given lon/lat."""
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f'EPSG:{epsg}'


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — imagery fetching
# ─────────────────────────────────────────────────────────────────────────────

def _get_http_session(pool_size: int = 16) -> requests.Session:
    """Construct a connection-pooled requests Session with automatic retries."""
    session = requests.Session()
    retries = Retry(
        total=4,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=retries,
    )
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


def fetch_imagery(minx: float, miny: float, maxx: float, maxy: float,
                  utm_crs: str, village_cache_key: str | None = None,
                  progress_callback=None) -> tuple[np.ndarray, object]:
    """
    Download RGB imagery from ESRI World Imagery for a UTM bounding box.
    Checks local disk cache and S3 imagery cache before downloading.
    Uses concurrent parallel chunk requests and persistent connection pooling.

    The image is requested directly in the village's UTM CRS (bboxSR = imageSR = UTM EPSG) on an
    exact RES_M grid anchored at the top-left corner, so `transform` describes every pixel exactly.
    (Before this fix the image was fetched in Web Mercator and then labelled with UTM bounds: the
    UTM grid is rotated up to ~1 deg against Mercator, which misplaced polygons by tens of metres
    for villages far from their zone's central meridian.)
    """
    # Exact RES_M grid: snap the right/bottom edges so pixels are exactly RES_M in the UTM CRS
    w_px = int(round((maxx - minx) / RES_M))
    h_px = int(round((maxy - miny) / RES_M))
    maxx, miny = minx + w_px * RES_M, maxy - h_px * RES_M
    # 'utm1-' prefix: imagery cached before this fix was fetched in Web Mercator -- never reuse it
    cache_key = f'utm1-{village_cache_key}' if village_cache_key else None

    # 1. Check local disk cache
    if cache_key:
        cached = _load_cached_imagery(cache_key, minx, miny, maxx, maxy)
        if cached is not None and cached[0].shape[1:] == (h_px, w_px):
            if progress_callback:
                progress_callback('imagery', 'Using cached RGB imagery from local disk …', 35)
            return cached

        # 2. Check S3 imagery cache
        cached_s3 = get_cached_imagery(cache_key, minx, miny, maxx, maxy)
        if cached_s3 is not None and cached_s3[0].shape[1:] == (h_px, w_px):
            if progress_callback:
                progress_callback('imagery', 'Using cached RGB imagery from S3 …', 35)
            return cached_s3

    started_at = time.perf_counter()
    epsg = int(utm_crs.split(':')[1])                      # _utm_crs() returns 'EPSG:326xx'
    cols = list(range(0, w_px, ESRI_MAX_CHUNK_PX)) + [w_px]
    rows = list(range(0, h_px, ESRI_MAX_CHUNK_PX)) + [h_px]
    nx, ny = len(cols) - 1, len(rows) - 1
    total_tiles = nx * ny
    logger.info(
        "Fetching ESRI imagery: CRS=%s bounds=(%.1f, %.1f, %.1f, %.1f), "
        "size=%dx%d px, tiles=%d (nx=%d, ny=%d, format=%s, workers=%d)",
        utm_crs, minx, miny, maxx, maxy, w_px, h_px, total_tiles, nx, ny,
        ESRI_IMAGE_FORMAT, min(ESRI_MAX_WORKERS, total_tiles),
    )

    session = _get_http_session(pool_size=max(10, min(ESRI_MAX_WORKERS, total_tiles) * 2))

    def _fetch_tile(coords, x0, y0, x1, y1, pw, ph, retries=4):
        params = {
            'bbox': f'{x0},{y0},{x1},{y1}',
            'bboxSR': epsg, 'imageSR': epsg,
            'size': f'{pw},{ph}',
            'format': ESRI_IMAGE_FORMAT,
            'f': 'image',
        }
        if ESRI_IMAGE_FORMAT in ('jpg', 'jpeg'):
            params['quality'] = ESRI_JPEG_QUALITY

        last_error = None
        for attempt in range(retries):
            try:
                r = session.get(_ESRI_URL, params=params, timeout=ESRI_TIMEOUT_SEC)
                r.raise_for_status()
                ct = r.headers.get('Content-Type', '')
                if 'image' not in ct:
                    raise ValueError(f'non-image response ({ct}): {r.text[:120]}')
                arr = np.array(
                    Image.open(io.BytesIO(r.content)).convert('RGB')
                ).transpose(2, 0, 1)
                if arr.shape != (3, ph, pw):
                    raise ValueError(f'unexpected chunk shape {arr.shape}, wanted {(3, ph, pw)}')
                return coords, arr
            except Exception as exc:
                last_error = exc
                logger.warning(
                    'ESRI imagery tile request failed (attempt %d/%d, tile=%s): %s',
                    attempt + 1, retries, coords, exc,
                )
                if attempt + 1 < retries:
                    time.sleep(2 ** attempt)
        raise ImageryFetchError(
            f'ESRI imagery tile {coords} failed after all retries.'
        ) from last_error

    # chunk (r0, c0) covers rows r0:r1 / cols c0:c1 of the mosaic; its bbox follows from the exact grid
    tile_jobs = []
    for c0, c1 in zip(cols[:-1], cols[1:]):
        for r0, r1 in zip(rows[:-1], rows[1:]):
            tile_jobs.append((
                (r0, c0),
                minx + c0 * RES_M, maxy - r1 * RES_M,
                minx + c1 * RES_M, maxy - r0 * RES_M,
                c1 - c0, r1 - r0,
            ))

    tiles: dict = {}
    max_workers = min(ESRI_MAX_WORKERS, total_tiles) if total_tiles > 1 else 1
    completed_count = 0

    if total_tiles == 1:
        job = tile_jobs[0]
        coords, arr = _fetch_tile(job[0], job[1], job[2], job[3], job[4], job[5], job[6])
        tiles[coords] = arr
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_coords = {
                executor.submit(_fetch_tile, job[0], job[1], job[2], job[3], job[4], job[5], job[6]): job[0]
                for job in tile_jobs
            }
            for future in as_completed(future_to_coords):
                coords, arr = future.result()
                tiles[coords] = arr
                completed_count += 1
                if progress_callback:
                    pct = int(15 + 20 * (completed_count / total_tiles))
                    progress_callback(
                        'imagery',
                        f'Fetching ESRI imagery ({completed_count}/{total_tiles} tiles complete) …',
                        pct,
                    )

    mosaic = np.zeros((3, h_px, w_px), dtype=np.uint8)
    for (r0, c0), arr in tiles.items():
        mosaic[:, r0:r0 + arr.shape[1], c0:c0 + arr.shape[2]] = arr
    transform = from_bounds(minx, miny, maxx, maxy, w_px, h_px)   # exactly RES_M per pixel

    if cache_key:
        _save_cached_imagery(cache_key, mosaic, minx, miny, maxx, maxy)
        store_cached_imagery_async(cache_key, mosaic, minx, miny, maxx, maxy, utm_crs=utm_crs)

    logger.info(
        "ESRI imagery fetched: mosaic=%s, elapsed=%.1fs",
        mosaic.shape, time.perf_counter() - started_at,
    )
    return mosaic, transform


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — sliding-window inference (fast: channels-last, autocast)
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, img_uint8: np.ndarray, device,
                  overlap: int = OVERLAP, batch_size: int = BATCH_SIZE,
                  fp16: bool = USE_FP16, tta: bool = TTA, progress_callback=None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Overlapping-patch inference with Hann-window blending.
    Supports TTA (averaging 4 flips), FP16 autocast, and channels-last memory layout.
    Accepts either a single TripleHeadModel or a dict {'extent': ext_m, 'boundary': bnd_m}.

    Parameters
    ----------
    model       : TripleHeadModel or dict of models
    img_uint8   : (3, H, W) uint8 RGB array
    device      : torch.device
    overlap     : patch overlap in pixels
    batch_size  : inference batch size
    fp16        : whether to use FP16 autocast
    tta         : whether to average 4 flips (none, LR, UD, both)
    progress_callback: optional UI callback

    Returns
    -------
    extent_prob   : (H, W) float32  — field extent probability
    boundary_prob : (H, W) float32  — boundary probability
    distance_pred : (H, W) float32  — distance-to-boundary prediction
    """
    if isinstance(model, dict):
        ext_model = model.get('extent')
        bnd_model = model.get('boundary', ext_model)
    else:
        ext_model = model
        bnd_model = model

    same_model = (ext_model is bnd_model)

    img    = img_uint8.astype('float32') / 255.0
    H, W   = img.shape[1], img.shape[2]
    stride = PATCH_PX - overlap

    ph = (stride - (H - PATCH_PX) % stride) % stride if H > PATCH_PX else PATCH_PX - H
    pw = (stride - (W - PATCH_PX) % stride) % stride if W > PATCH_PX else PATCH_PX - W
    ip = np.pad(img, ((0, 0), (0, ph + PATCH_PX), (0, pw + PATCH_PX)), mode='reflect')
    del img
    Hp, Wp = ip.shape[1], ip.shape[2]

    acc  = np.zeros((3, Hp, Wp), 'float32')
    wsum = np.zeros((Hp, Wp), 'float32')
    ramp = np.hanning(PATCH_PX)
    win  = (np.outer(ramp, ramp) + 1e-6).astype('float32')

    coords = [(y, x)
              for y in range(0, Hp - PATCH_PX + 1, stride)
              for x in range(0, Wp - PATCH_PX + 1, stride)]
    total_patches = len(coords)

    flips = [(), (3,), (2,), (2, 3)] if tta else [()]
    use_amp = bool(fp16 and device.type == 'cuda')

    with torch.inference_mode():
        for i in range(0, total_patches, batch_size):
            bc = coords[i:i + batch_size]
            bi = torch.from_numpy(np.stack([ip[:, y:y + PATCH_PX, x:x + PATCH_PX] for y, x in bc]))
            bi = bi.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)

            es, bs, ds = 0.0, 0.0, 0.0
            for f in flips:
                xb = torch.flip(bi, f) if f else bi
                if use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        if same_model:
                            e, b, d = ext_model(xb)
                        else:
                            e, _, _ = ext_model(xb)
                            _, b, d = bnd_model(xb)
                else:
                    if same_model:
                        e, b, d = ext_model(xb)
                    else:
                        e, _, _ = ext_model(xb)
                        _, b, d = bnd_model(xb)

                e = torch.sigmoid(e.float())
                b = torch.sigmoid(b.float())
                d = d.float()

                if f:
                    e = torch.flip(e, f)
                    b = torch.flip(b, f)
                    d = torch.flip(d, f)

                es = es + e
                bs = bs + b
                ds = ds + d

            num_flips = float(len(flips))
            e_np = (es / num_flips).cpu().numpy()[:, 0]
            b_np = (bs / num_flips).cpu().numpy()[:, 0]
            d_np = (ds / num_flips).cpu().numpy()[:, 0]

            for (y, x), ee, bb, dd in zip(bc, e_np, b_np, d_np):
                acc[0, y:y + PATCH_PX, x:x + PATCH_PX] += ee * win
                acc[1, y:y + PATCH_PX, x:x + PATCH_PX] += bb * win
                acc[2, y:y + PATCH_PX, x:x + PATCH_PX] += dd * win
                wsum[y:y + PATCH_PX, x:x + PATCH_PX]   += win

            if progress_callback and (i % (batch_size * 4) == 0 or i + batch_size >= total_patches):
                pct = int(40 + 18 * (min(i + batch_size, total_patches) / total_patches))
                progress_callback('inference', f'Running model inference ({min(i + batch_size, total_patches)}/{total_patches} patches) …', pct)

    acc /= np.maximum(wsum, 1e-6)
    return acc[0, :H, :W], acc[1, :H, :W], acc[2, :H, :W]


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — v5 instance extraction (boundary normalization, neck seeding, segment)
# ─────────────────────────────────────────────────────────────────────────────

def drop_small_regions(labels: np.ndarray, min_px: int) -> np.ndarray:
    """Zero out connected label regions with fewer than min_px pixels."""
    if labels.max() == 0:
        return labels
    counts = np.bincount(labels.ravel())
    small = counts < min_px
    small[0] = False
    labels = labels.copy()
    labels[small[labels]] = 0
    return labels


def _robust_norm(a: np.ndarray, mask: np.ndarray | None = None, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Robust min-max normalization using percentiles."""
    v = a[mask] if mask is not None and mask.any() else a
    p_lo, p_hi = np.percentile(v, [lo, hi])
    return np.clip((a - p_lo) / (p_hi - p_lo + 1e-9), 0.0, 1.0)


def fill_small_holes(mask: np.ndarray, max_hole_px: float | None) -> np.ndarray:
    """binary_fill_holes, but only for holes up to max_hole_px pixels.
    Larger holes (settlements, ponds surrounded by fields) are left open
    so they don't get merged into field extent."""
    if max_hole_px is None:
        return ndi.binary_fill_holes(mask)
    holes = ndi.binary_fill_holes(mask) & ~mask
    lab, n = ndi.label(holes)
    if n == 0:
        return mask
    small = np.bincount(lab.ravel()) <= max_hole_px
    small[0] = False
    return mask | small[lab]


def normalize_boundary(bp: np.ndarray, mask: np.ndarray | None = None,
                       pctl: float | None = BND_NORM, floor: float = BND_FLOOR) -> np.ndarray:
    """Rescale the boundary map so its pctl-th percentile (inside mask) becomes 1. The fine-tuned
    boundary head is under-confident (peaks ~0.45, never > 0.5), so fixed thresholds such as
    'boundary < 0.30' would mean something different on every checkpoint. floor caps the gain
    (at most 1/floor) so a block with almost no boundaries does not get its noise blown up."""
    if pctl is None:
        return bp.astype(np.float32)
    v = bp[mask] if mask is not None and mask.any() else bp
    p = float(np.percentile(v, pctl)) if v.size else 1.0
    return np.clip(bp / max(p, floor), 0.0, 1.0).astype(np.float32)


def make_seeds(extent_mask: np.ndarray, bp: np.ndarray, dist: np.ndarray | None = None,
               mode: str = SEED_MODE, seed_bnd_max: float = SEED_BND_MAX,
               seed_dist_min: float = SEED_DIST_MIN, seed_min_px: int = SEED_MIN_PX,
               seed_open_r: int = 1, neck_h: float = NECK_H,
               road_mask: np.ndarray | None = None) -> np.ndarray:
    """Seed markers, one per field.
    'cores'    : pixels away from boundaries AND with high predicted distance (v2/v3 rule). Best when
                 the distance head is good; merges fields when it is weak.
    'neck'     : areas enclosed by the boundary map, split wherever they narrow into a neck (a gap
                 in a bund) -- peaks of the area's own distance-to-boundary that stand >= neck_h px
                 above the neck between them. Does not rely on the distance head.
    'neck_dist': as 'neck', with the peaks weighted by the predicted distance.
    Over-splits from the neck modes are undone by merge_weak_boundaries, which keeps a split only
    when the shared border has boundary evidence."""
    bcore = extent_mask & (bp < seed_bnd_max)
    if road_mask is not None and road_mask.any():
        bcore &= ~road_mask

    if mode == 'cores':
        d = np.zeros_like(bp) if dist is None else dist
        sm = ndi.binary_opening(bcore & (d > seed_dist_min), structure=disk(1))
        markers, _ = ndi.label(sm)
        return drop_small_regions(markers, seed_min_px)

    if mode not in ('neck', 'neck_dist'):
        raise ValueError(f"unknown seed mode {mode!r}")

    if seed_open_r > 0:
        bcore = ndi.binary_opening(bcore, structure=disk(seed_open_r))
    mb, _ = ndi.label(bcore)
    mb = drop_small_regions(mb, seed_min_px)
    if mb.max() == 0:
        return mb.astype(np.int32)

    geo = ndi.distance_transform_edt(mb > 0).astype(np.float32)
    surf = geo * (0.5 + dist) if (mode == 'neck_dist' and dist is not None) else geo
    peaks = h_maxima(gaussian_filter(surf, 1.0), neck_h) > 0
    markers, _ = ndi.label(ndi.binary_dilation(peaks, structure=disk(1)) & (mb > 0))

    # a boundary core with no peak at all (tiny / odd shape) still gets one seed
    has = np.zeros(int(mb.max()) + 1, bool)
    has[np.unique(mb[markers > 0])] = True
    orphan = (mb > 0) & ~has[mb]
    if orphan.any():
        ol, _ = ndi.label(orphan)
        markers = np.where(ol > 0, ol + markers.max(), markers)

    return markers.astype(np.int32)


def segment(extent_prob: np.ndarray, boundary_prob: np.ndarray, village_mask: np.ndarray,
            distance_pred: np.ndarray | None = None, img: np.ndarray | None = None,
            road_mask: np.ndarray | None = None, urban_mask: np.ndarray | None = None,
            post: bool = True, **cfg) -> tuple[np.ndarray, list[int], dict]:
    """
    v5 model outputs -> field label image (from village_fields_inference.ipynb).
    Combines extent thresholding, boundary normalization, neck-based watershed seeding,
    weak boundary merging, compact patch rescue, tree gap filling, and Gaussian smoothing.
    """
    c = {
        'EXT_T': EXT_T,
        'EDGE_SIGMA': EDGE_SIGMA,
        'BND_NORM': BND_NORM,
        'BND_FLOOR': BND_FLOOR,
        'CORE_T': CORE_T,
        'GAP_CLOSE_R': GAP_CLOSE_R,
        'SEED_MODE': SEED_MODE,
        'NECK_H': NECK_H,
        'SEED_MIN_PX': SEED_MIN_PX,
        'MERGE_T': MERGE_T,
        'MIN_REGION_PX': MIN_REGION_PX,
        'EXTENT_HOLE_MAX_PX': EXTENT_HOLE_MAX_PX,
        'MAX_FIELD_PX': MAX_FIELD_PX,
        'RESCUE': RESCUE_FIELDS,
        'RESCUE_MIN_PX': RESCUE_MIN_PX,
        'RESCUE_MIN_SOLIDITY': RESCUE_MIN_SOLIDITY,
        'RESCUE_MIN_WIDTH_PX': RESCUE_MIN_WIDTH_PX,
        'RESCUE_MIN_EXG': RESCUE_MIN_EXG,
        'TREE_FILL_MAX_PX': TREE_FILL_MAX_PX,
        'TREE_MIN_ENCLOSED': TREE_MIN_ENCLOSED,
        'BUND_OPEN_PX': BUND_OPEN_PX,
        'SMOOTH_R': SMOOTH_R,
        'GAUSS_SIGMA': GAUSS_SIGMA,
    }
    c.update(cfg)

    # 1. Extent closing and small hole filling
    max_hole = c['EXTENT_HOLE_MAX_PX']
    ext = fill_small_holes(
        ndi.binary_closing(extent_prob > c['EXT_T'], structure=disk(1)),
        max_hole
    ) & village_mask

    if urban_mask is not None and urban_mask.any():
        ext &= ~urban_mask

    # 2. Boundary smoothing and percentile normalization
    b = boundary_prob.astype(np.float32)
    if c['EDGE_SIGMA']:
        b = gaussian_filter(b, c['EDGE_SIGMA'])
    b = normalize_boundary(b, village_mask, c['BND_NORM'], c['BND_FLOOR'])

    # 3. Wall and free areas
    wall = b >= c['CORE_T']
    if road_mask is not None and road_mask.any():
        wall |= road_mask
    if c['GAP_CLOSE_R'] > 0:
        wall = ndi.binary_dilation(wall, structure=disk(int(c['GAP_CLOSE_R'])))
    free = ext & ~wall

    # 4. Marker seeding
    if c['SEED_MODE'] == 'cores':
        markers = make_seeds(free, b, dist=distance_pred, mode='cores', seed_bnd_max=2.0,
                             seed_dist_min=SEED_DIST_MIN, seed_min_px=c['SEED_MIN_PX'],
                             road_mask=road_mask)
    elif c['SEED_MODE'] in ('neck', 'neck_dist'):
        markers = make_seeds(free, b, dist=distance_pred, mode=c['SEED_MODE'], seed_bnd_max=2.0,
                             seed_min_px=c['SEED_MIN_PX'], seed_open_r=1, neck_h=c['NECK_H'],
                             road_mask=road_mask)
    else:
        raise ValueError(f"unknown SEED_MODE {c['SEED_MODE']!r}")

    n_seeds = int(len(np.unique(markers)) - (1 if (markers == 0).any() else 0))

    # 5. Watershed flooding on normalized boundary
    elev = b.copy()
    if road_mask is not None and road_mask.any():
        elev = np.where(road_mask, ROAD_BARRIER_ELEV, elev)

    labels = watershed(elev, markers, mask=ext).astype(np.int32)

    # 6. Unseeded field land recovery
    ol, _ = ndi.label(ext & (labels == 0))
    cnt = np.bincount(ol.ravel())
    cnt[0] = 0
    keep = cnt >= c['MIN_REGION_PX']
    if keep.any():
        ol[~keep[ol]] = 0
        labels = np.where(ol > 0, ol + labels.max(), labels).astype(np.int32)

    # 7. Merge weak boundaries (region-adjacency merge)
    n_merged = 0
    if c['MERGE_T'] and c['MERGE_T'] > 0:
        bp_merge = np.where(road_mask, 1.0, b) if road_mask is not None else b
        labels, n_merged = merge_weak_boundaries(
            labels, bp_merge, thresh=c['MERGE_T'], max_area_px=c['MAX_FIELD_PX']
        )

    # 8. Post-processing passes
    rescued: list[int] = []
    n_filled = 0
    if post:
        if c['RESCUE']:
            labels, rescued = rescue_compact_patches(
                labels, b, village_mask, img=img,
                min_px=c['RESCUE_MIN_PX'], max_px=c['MAX_FIELD_PX'],
                min_solidity=c['RESCUE_MIN_SOLIDITY'],
                min_width_px=c['RESCUE_MIN_WIDTH_PX'],
                min_exg=c['RESCUE_MIN_EXG']
            )
        if c['TREE_FILL_MAX_PX'] > 0:
            labels, n_filled = fill_small_gaps(
                labels, village_mask, max_px=c['TREE_FILL_MAX_PX'],
                min_enclosed=c['TREE_MIN_ENCLOSED'],
                bund_r=int(c['BUND_OPEN_PX'])
            )
        if c['SMOOTH_R'] > 0:
            labels = smooth_labels(labels, int(c['SMOOTH_R']))
        if c['GAUSS_SIGMA'] > 0:
            labels = smooth_labels_gaussian(labels, c['GAUSS_SIGMA'], valid=village_mask)

    labels = drop_small_regions(labels, c['MIN_REGION_PX'])
    info = {
        'seeds': n_seeds,
        'merged': int(n_merged),
        'rescued': len(rescued),
        'filled': int(n_filled),
        'fields': int(len(np.unique(labels)) - (1 if (labels == 0).any() else 0)),
    }
    return labels, rescued, info


def extract_instances_v2(extent_prob: np.ndarray, boundary_prob: np.ndarray,
                         distance_pred: np.ndarray, village_mask: np.ndarray,
                         road_mask: np.ndarray | None = None,
                         urban_mask: np.ndarray | None = None,
                         **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Backward-compatibility wrapper returning (labels, elev, bp, info)."""
    labels, rescued, info = segment(
        extent_prob, boundary_prob, village_mask,
        distance_pred=distance_pred, road_mask=road_mask, urban_mask=urban_mask,
        post=False, **kwargs
    )
    bp = normalize_boundary(boundary_prob, village_mask)
    return labels, bp, bp, info


def extract_instances_hminima(extent_prob: np.ndarray, boundary_prob: np.ndarray,
                              distance_pred: np.ndarray, village_mask: np.ndarray,
                              extent_thresh: float = EXTENT_THRESH, overseg_h: float = OVERSEG_H,
                              min_px: int = MIN_REGION_PX, rescue_orphans: bool = RESCUE_ORPHANS
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Legacy h-minima seeded watershed with fence & orphan rescue."""
    d_range = distance_pred.max() - distance_pred.min()
    dist_norm = (distance_pred - distance_pred.min()) / (d_range + 1e-9)
    dist_edges = sobel(gaussian_filter(dist_norm, sigma=1.0))
    dist_edges /= (dist_edges.max() + 1e-9)
    edge_strength = np.maximum(boundary_prob / (boundary_prob.max() + 1e-9), dist_edges)
    edge_smooth = gaussian_filter(edge_strength, sigma=1.5)

    extent_mask = ndi.binary_fill_holes(
        ndi.binary_closing(extent_prob > extent_thresh, structure=disk(2))
    )
    extent_mask &= village_mask

    fenced = np.where(village_mask, edge_smooth, edge_smooth.max())
    markers, _ = ndi.label(h_minima(fenced, overseg_h) & extent_mask)
    labels = watershed(edge_smooth, markers, mask=extent_mask)
    n_seeded = len(np.unique(labels)) - (1 if (labels == 0).any() else 0)

    n_rescued = 0
    if rescue_orphans:
        ol, _ = ndi.label(extent_mask & (labels == 0))
        cnt = np.bincount(ol.ravel())
        cnt[0] = 0
        keep = cnt >= min_px
        n_rescued = int(keep.sum())
        if n_rescued:
            ol[~keep[ol]] = 0
            labels = np.where(ol > 0, ol + labels.max(), labels).astype(labels.dtype)

    labels = drop_small_regions(labels, min_px)
    info = {
        'extent_pct_of_village': 100 * extent_mask.sum() / max(village_mask.sum(), 1),
        'labelled_pct_of_extent': 100 * (labels > 0).sum() / max(extent_mask.sum(), 1),
        'seeded_regions': n_seeded,
        'rescued_regions': n_rescued,
        'final_regions': len(np.unique(labels)) - (1 if (labels == 0).any() else 0),
    }
    return labels.astype(np.int32), edge_smooth, boundary_prob, info



# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — post-extraction operations (split oversized, merge weak, smooth)
# ─────────────────────────────────────────────────────────────────────────────

def split_oversized(labels: np.ndarray, edge_img: np.ndarray, thresh_m2: float, px_area: float,
                    h_ladder: tuple = (0.12, 0.08, 0.05, 0.03, 0.02), min_piece_px: int = 200,
                    verbose: bool = False) -> np.ndarray:
    """Retry h-minima seeding inside each oversized region (on its bounding-box crop only),
    stepping h down until it splits into 2+ pieces. Regions that never split stay intact."""
    out = labels.copy()
    next_id = int(labels.max()) + 1
    counts = np.bincount(labels.ravel())
    slices = ndi.find_objects(labels)
    high = edge_img.max()
    targets = [l for l in range(1, len(counts)) if counts[l] * px_area > thresh_m2]
    n_split = 0
    for lid in targets:
        sl = slices[lid - 1]
        if sl is None:
            continue
        m = labels[sl] == lid
        sub_edge = np.where(m, edge_img[sl], high)
        for h in h_ladder:
            mk, n = ndi.label(h_minima(sub_edge, h) & m)
            if n < 2:
                continue
            sub = watershed(sub_edge, mk, mask=m)
            pieces = [s for s in np.unique(sub) if s != 0 and (sub == s).sum() >= min_piece_px]
            if len(pieces) < 2:
                continue
            view = out[sl]
            for s in pieces:
                view[(sub == s) & m] = next_id
                next_id += 1
            n_split += 1
            break
    if verbose:
        logger.info("Split %d of %d oversized regions", n_split, len(targets))
    return out


def merge_weak_boundaries(labels: np.ndarray, boundary_prob: np.ndarray, thresh: float = MERGE_BND_T,
                          min_contact_px: int = 10, max_area_px: float | None = None) -> tuple[np.ndarray, int]:
    """Region-adjacency merge: two touching regions are merged when the boundary prob along their
    shared border is weak (mean < thresh). Weakest borders first; never grows a region past max_area_px."""
    if thresh <= 0 or labels.max() == 0:
        return labels, 0
    L, B = labels, boundary_prob
    los, his, vals = [], [], []
    for a, b, va, vb in ((L[:, :-1], L[:, 1:], B[:, :-1], B[:, 1:]),
                         (L[:-1, :], L[1:, :], B[:-1, :], B[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        los.append(np.minimum(a[m], b[m]))
        his.append(np.maximum(a[m], b[m]))
        vals.append(np.maximum(va[m], vb[m]))
    lo, hi, v = (np.concatenate(x) for x in (los, his, vals))
    if lo.size == 0:
        return labels, 0
    n = int(L.max()) + 1
    key = lo.astype(np.int64) * n + hi
    uniq, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv)
    mean = np.bincount(inv, weights=v) / cnt
    cand = np.where((cnt >= min_contact_px) & (mean < thresh))[0]
    cand = cand[np.argsort(mean[cand])]

    parent = np.arange(n)
    area = np.bincount(L.ravel(), minlength=n).astype(np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merged = 0
    for c in cand:
        ra, rb = find(int(uniq[c] // n)), find(int(uniq[c] % n))
        if ra == rb or (max_area_px is not None and area[ra] + area[rb] > max_area_px):
            continue
        parent[rb] = ra
        area[ra] += area[rb]
        merged += 1
    root = np.array([find(i) for i in range(n)])
    root[0] = 0
    return root[L].astype(L.dtype), merged


def smooth_labels(labels: np.ndarray, radius: int = LABEL_SMOOTH_R) -> np.ndarray:
    """Raster spike ('dagger') removal that keeps the tiling: open each region with a disk, then
    hand the removed pixels to the nearest surviving neighbour."""
    if radius <= 0 or labels.max() == 0:
        return labels
    fp = disk(radius)
    out = np.zeros_like(labels)
    for i, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        sl = tuple(slice(max(s.start - radius, 0), s.stop + radius) for s in sl)
        m = labels[sl] == i
        o = ndi.binary_opening(m, structure=fp)
        if not o.any():
            o = m
        out[sl][o] = i
    fg = labels > 0
    filled = expand_labels(out, distance=radius + 1)
    out = np.where(fg & (out == 0), filled, out)
    out[~fg] = 0
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 (v3) — missed-field rescue, tree absorption, Gaussian label smoothing
# ──────────────────────────────────────────────────────────────────────────────

def rescue_compact_patches(
    labels: np.ndarray, boundary_prob: np.ndarray, valid: np.ndarray,
    img: np.ndarray | None = None,
    min_px: float = 2000, max_px: float = 220_000,
    min_solidity: float = 0.88, min_width_px: float = 20,
    min_exg: float = 0.02, bnd_cut: float = 0.5, open_r: int = 2,
) -> tuple[np.ndarray, list]:
    """Fields the model scored as non-crop (dense sugarcane / banana / orchard blocks
    that look like trees): a leftover non-field patch that is big, compact, field-shaped
    and green becomes a field. Returns (labels, list_of_new_label_ids)."""
    from skimage.measure import regionprops
    bg = (labels == 0) & valid & (boundary_prob < bnd_cut)
    if open_r > 0:
        bg = ndi.binary_opening(bg, structure=disk(open_r))    # detach roads / bunds
    comp, n = ndi.label(bg)
    if n == 0:
        return labels, []

    exg = None
    if img is not None and min_exg is not None:
        f = img.astype(np.float32)
        s = f.sum(0) + 1e-6
        exg = (2 * f[1] - f[0] - f[2]) / s                   # excess-green, -1..2

    out = labels.copy()
    nid = int(labels.max()) + 1
    new_ids = []
    for p in regionprops(comp):
        if not (min_px <= p.area <= max_px):
            continue
        minor_axis_length = getattr(p, 'axis_minor_length', None)
        if minor_axis_length is None:
            minor_axis_length = p.minor_axis_length
        if p.solidity < min_solidity or minor_axis_length < min_width_px:
            continue
        if exg is not None:
            rr, cc = p.coords[:, 0], p.coords[:, 1]
            if float(exg[rr, cc].mean()) < min_exg:
                continue
        out[p.coords[:, 0], p.coords[:, 1]] = nid
        new_ids.append(nid)
        nid += 1
    return out, new_ids


def fill_small_gaps(
    labels: np.ndarray, valid: np.ndarray,
    max_px: float = 1500, min_enclosed: float = 0.6, bund_r: int = 2,
) -> tuple[np.ndarray, int]:
    """Trees, bushes and shadows: a small non-field blob inside the village that is mostly
    surrounded by fields is given to the surrounding field(s) — each pixel to the nearest
    field. The non-field mask is first opened with disk(bund_r) so thin bunds don't connect
    every tree into one huge network."""
    bg = (labels == 0) & valid
    blobs = ndi.binary_opening(bg, structure=disk(bund_r)) if bund_r > 0 else bg
    comp, n = ndi.label(blobs)
    if n == 0:
        return labels, 0
    sizes = np.bincount(comp.ravel(), minlength=n + 1)
    dil = ndi.grey_dilation(comp, size=3)
    ring = (dil > 0) & (comp == 0)
    rc = dil[ring]
    enclosed = (labels[ring] > 0) | (bg[ring] & ~blobs[ring])
    tot = np.bincount(rc, minlength=n + 1).astype(float)
    enc = np.bincount(rc, weights=enclosed.astype(float), minlength=n + 1)
    frac = enc / np.maximum(tot, 1)
    ok = (sizes <= max_px) & (frac >= min_enclosed)
    ok[0] = False
    if not ok.any():
        return labels, 0
    out = labels.copy()
    m = int(np.ceil(np.sqrt(max_px))) + 2 * bund_r + 2
    slices = ndi.find_objects(comp)
    grow = disk(bund_r) if bund_r > 0 else None
    for k in np.nonzero(ok)[0]:
        sl = slices[k - 1]
        sl = tuple(slice(max(s.start - m, 0), s.stop + m) for s in sl)
        tgt = comp[sl] == k
        if grow is not None:
            tgt = ndi.binary_dilation(tgt, structure=grow) & bg[sl]
        sub = labels[sl]
        if not (sub > 0).any():
            continue
        near = expand_labels(sub, distance=2 * m)
        out[sl][tgt] = near[tgt]
    return out, int(ok.sum())


def smooth_labels_gaussian(labels: np.ndarray, sigma: float = 1.5,
                           valid: np.ndarray | None = None) -> np.ndarray:
    """Smooth every region's outline without breaking the tiling: blur each region's
    binary mask with a Gaussian and give each pixel to the highest blurred value.
    Removes wiggles shorter than ~3*sigma px; corners round by ~sigma px."""
    if sigma <= 0 or labels.max() == 0:
        return labels
    m = int(np.ceil(3 * sigma))
    best = gaussian_filter((labels == 0).astype(np.float32), sigma)
    out = np.zeros_like(labels)
    for i, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        sl = tuple(slice(max(s.start - m, 0), s.stop + m) for s in sl)
        b = gaussian_filter((labels[sl] == i).astype(np.float32), sigma)
        bv, ov = best[sl], out[sl]
        upd = b > bv
        bv[upd] = b[upd]
        ov[upd] = i
    if valid is not None:
        out[~valid] = 0
    return out



# ─────────────────────────────────────────────────────────────────────────────
# Step 5 (v3) — topology-preserving arc-tracing vectoriser
# ─────────────────────────────────────────────────────────────────────────────

_E, _W, _S, _N = (0, 1), (0, -1), (1, 0), (-1, 0)


def trace_arcs(L: np.ndarray) -> list:
    """L: 2-D int label array (0 = background).
    Returns a list of arc dicts:
      {'pts': list of (row, col) corner coords,
       'left': label, 'right': label, 'closed': bool}.
    'left' is the label on the left when walking the arc in screen orientation (row down)."""
    H, W = L.shape
    P = np.zeros((H + 2, W + 2), dtype=np.int64)
    P[1:-1, 1:-1] = L
    R, C = H + 3, W + 3                                   # corner grid size
    # hcr[i, j]: horizontal crack between P[i-1, j] (above) and P[i, j] (below)
    hcr = np.zeros((R, C), bool)
    hcr[1:H + 2, 0:W + 2] = P[0:H + 1, :] != P[1:H + 2, :]
    # vcr[i, j]: vertical crack between P[i, j-1] (left) and P[i, j] (right)
    vcr = np.zeros((R, C), bool)
    vcr[0:H + 2, 1:W + 2] = P[:, 0:W + 1] != P[:, 1:W + 2]

    deg = np.zeros((R, C), np.int8)
    deg += hcr;          deg[:, 1:] += hcr[:, :-1]        # E / W
    deg += vcr;          deg[1:, :] += vcr[:-1, :]         # S / N
    node = deg >= 3

    Pl = P.tolist()
    hv = np.zeros((R, C), bool)
    vv = np.zeros((R, C), bool)
    hl, vl = hcr.tolist(), vcr.tolist()
    hvl, vvl, nodel = hv.tolist(), vv.tolist(), node.tolist()

    def step(i, j, d):
        if d == _E:  return hl[i][j],     hvl, i,     j,     (i, j + 1), Pl[i-1][j],   Pl[i][j]
        if d == _W:  return hl[i][j-1],   hvl, i,     j - 1, (i, j - 1), Pl[i][j-1],   Pl[i-1][j-1]
        if d == _S:  return vl[i][j],     vvl, i,     j,     (i + 1, j), Pl[i][j],     Pl[i][j-1]
        return           vl[i-1][j],   vvl, i - 1, j,     (i - 1, j), Pl[i-1][j-1], Pl[i-1][j]

    def crack_ok(i, j, d):
        if d == _E: return 0 <= j < C - 1 and hl[i][j]
        if d == _W: return j >= 1 and hl[i][j - 1]
        if d == _S: return 0 <= i < R - 1 and vl[i][j]
        return i >= 1 and vl[i - 1][j]

    def walk(i0, j0, d0, closed_start=False):
        pts = [(i0, j0)]
        _, vis, ci, cj, nxt, left, right = step(i0, j0, d0)
        vis[ci][cj] = True
        d = d0
        i, j = nxt
        while True:
            pts.append((i, j))
            if nodel[i][j] or (closed_start and (i, j) == (i0, j0)):
                break
            back = (-d[0], -d[1])
            for nd in (d, (-d[1], d[0]), (d[1], -d[0])):
                if nd == back or not crack_ok(i, j, nd):
                    continue
                ok2, vis, ci, cj, nxt2, _, _ = step(i, j, nd)
                if vis[ci][cj]:
                    continue
                vis[ci][cj] = True
                d = nd
                i, j = nxt2
                break
            else:
                break
        pts = [(a - 1, b - 1) for a, b in pts]
        return {'pts': pts, 'left': left, 'right': right, 'closed': pts[0] == pts[-1]}

    arcs = []
    ni, nj = np.nonzero(node)
    for i, j in zip(ni.tolist(), nj.tolist()):
        for d in (_E, _W, _S, _N):
            if not crack_ok(i, j, d):
                continue
            _, vis, ci, cj, _, _, _ = step(i, j, d)
            if not vis[ci][cj]:
                arcs.append(walk(i, j, d))
    # closed loops with no node (a field surrounded by one other label / background)
    for arr, vis, d in ((hcr, hvl, _E), (vcr, vvl, _S)):
        ii, jj = np.nonzero(arr)
        for i, j in zip(ii.tolist(), jj.tolist()):
            if not vis[i][j]:
                arcs.append(walk(i, j, d, closed_start=True))
    return arcs


def _drop_collinear(pts: np.ndarray) -> np.ndarray:
    """Remove interior points lying on a straight run (keeps endpoints)."""
    a = np.asarray(pts, dtype=np.float64)
    if len(a) <= 2:
        return a
    d1 = a[1:-1] - a[:-2]
    d2 = a[2:] - a[1:-1]
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    keep = np.r_[True, cross != 0, True]
    return a[keep]


def _dp_importance(a: np.ndarray, floor: float) -> np.ndarray:
    """Douglas-Peucker importance for each point of open polyline a (endpoints = inf).
    Importance is made monotone (child <= parent) so thresholding keeps a valid DP subset."""
    n = len(a)
    imp = np.zeros(n)
    imp[0] = imp[-1] = np.inf
    stack = [(0, n - 1, np.inf)]
    while stack:
        s, e, parent = stack.pop()
        if e - s < 2:
            continue
        seg = a[e] - a[s]
        Lseg = np.hypot(seg[0], seg[1])
        q = a[s + 1:e] - a[s]
        if Lseg == 0:
            dist = np.hypot(q[:, 0], q[:, 1])
        else:
            dist = np.abs(q[:, 0] * seg[1] - q[:, 1] * seg[0]) / Lseg
        k = int(np.argmax(dist))
        dk = float(dist[k])
        if dk < floor:
            continue
        k += s + 1
        imp[k] = min(dk, parent)
        stack.append((s, k, imp[k]))
        stack.append((k, e, imp[k]))
    return imp


def _chain_rings(arcs: list) -> dict:
    """Group half-arcs into closed rings per label.
    Returns {label: [[(arc_idx, reversed), ...], ...]}."""
    chains: dict = {}
    halfs: dict = {}
    for idx, a in enumerate(arcs):
        for l, rev in ((a['left'], False), (a['right'], True)):
            if l == 0:
                continue
            if a['closed']:
                chains.setdefault(l, []).append([(idx, rev)])
            else:
                halfs.setdefault(l, []).append((idx, rev))
    for l, hs in halfs.items():
        start_map: dict = {}
        for h in hs:
            p = arcs[h[0]]['p']
            start_map.setdefault(tuple(p[-1]) if h[1] else tuple(p[0]), []).append(h)
        used: set = set()
        for h0 in hs:
            if h0 in used:
                continue
            chain, h = [], h0
            while h is not None and h not in used:
                used.add(h)
                chain.append(h)
                p = arcs[h[0]]['p']
                end = tuple(p[0]) if h[1] else tuple(p[-1])
                cands = [c for c in start_map.get(end, []) if c not in used]
                if not cands:
                    break
                if len(cands) == 1:
                    h = cands[0]
                    continue
                # several ways out of this corner (diagonal touch): sharpest LEFT turn
                din = np.sign((p[0] - p[1]) if h[1] else (p[-1] - p[-2]))
                left = np.array([-din[1], din[0]])

                def rank(c, _p=arcs, _din=din, _left=left):
                    q = _p[c[0]]['p']
                    dout = np.sign((q[-2] - q[-1]) if c[1] else (q[1] - q[0]))
                    return 0 if np.array_equal(dout, _left) else (1 if np.array_equal(dout, _din) else 2)
                h = min(cands, key=rank)
            chains.setdefault(l, []).append(chain)
    return chains


def vectorize_labels(labels: np.ndarray, tol_px: float = 2.0, max_vertices: int = 0) -> dict:
    """Returns {label: [ring, ...]}; each ring is an (N, 2) float64 array of (row, col) corner
    coords, closed (first == last). Shells have negative signed_area, holes positive.
    tol_px: Douglas-Peucker tolerance in pixels. max_vertices: per-label vertex budget (0 = none)."""
    arcs = trace_arcs(labels)
    floor = tol_px if tol_px > 0 else 1e-9
    for a in arcs:
        p = _drop_collinear(a['pts'])
        if a['closed']:
            p = p[:-1]
            if len(p) >= 3:
                k = int(np.argmax(np.hypot(*(p - p[0]).T)))
                i1 = _dp_importance(p[:k + 1], floor)
                i2 = _dp_importance(np.vstack([p[k:], p[:1]]), floor)
                imp = np.r_[i1[:-1], i2[:-1]]
            else:
                imp = np.full(len(p), np.inf)
        else:
            imp = _dp_importance(p, floor)
        a['p'], a['imp'] = p, imp
        a['min_interior'] = 0

    chains = _chain_rings(arcs)
    for l, cs in chains.items():
        for ch in cs:
            if len(ch) <= 2:
                for idx, _ in ch:
                    if not arcs[idx]['closed']:
                        arcs[idx]['min_interior'] = 1

    labels_present = set(chains)
    thr = {l: tol_px for l in labels_present}
    if max_vertices and max_vertices > 0:
        for l, cs in chains.items():
            arc_ids = {idx for ch in cs for idx, _ in ch}
            fixed, cand = 0, []
            for idx in arc_ids:
                a = arcs[idx]
                fixed += int(np.isinf(a['imp']).sum()) - (0 if a['closed'] else 1)
                f = a['imp'][np.isfinite(a['imp'])]
                cand.append(f[f >= tol_px])
            v = np.sort(np.concatenate(cand))[::-1] if cand else np.empty(0)
            budget = max(max_vertices - fixed, 0)
            if len(v) > budget:
                thr[l] = max(tol_px, float(v[budget]) + 1e-9)

    for a in arcs:
        t = max(thr.get(a['left'], tol_px), thr.get(a['right'], tol_px))
        keep = a['imp'] >= t
        n_int = int(keep.sum()) - int(np.isinf(a['imp']).sum())
        if a['closed']:
            if keep.sum() < 4:
                keep[np.argsort(-a['imp'])[:min(4, len(keep))]] = True
            q = a['p'][keep]
            a['s'] = np.vstack([q, q[:1]])
        else:
            if n_int < a['min_interior'] and len(a['imp']) > 2:
                inner = a['imp'].copy(); inner[[0, -1]] = -1
                keep[int(np.argmax(inner))] = True
            a['s'] = a['p'][keep]

    out: dict = {}
    for l, cs in chains.items():
        rs = []
        for ch in cs:
            pieces = []
            for idx, rev in ch:
                s = arcs[idx]['s'][::-1] if rev else arcs[idx]['s']
                pieces.append(s if not pieces else s[1:])
            ring = np.vstack(pieces)
            if not np.array_equal(ring[0], ring[-1]):
                ring = np.vstack([ring, ring[:1]])
            if len(ring) >= 4 and abs(_signed_area(ring)) > 1e-9:
                rs.append(ring)
        out[l] = rs
    return out


def _signed_area(ring: np.ndarray) -> float:
    """Shoelace in (x=col, y=row) coords. Shells (row-down screen) have negative area."""
    x, y = ring[:, 1], ring[:, 0]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _rings_to_world(ring: np.ndarray, transform) -> np.ndarray:
    """(row, col) corner coords → (x, y) map coords using rasterio affine transform."""
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    r, cc = ring[:, 0], ring[:, 1]
    return np.column_stack([c + cc * a + r * b, f + cc * d + r * e])


def _as_polygonal(geom):
    """Valid Polygon/MultiPolygon, or None."""
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.geom_type in ('Polygon', 'MultiPolygon'):
        return geom
    parts = [_as_polygonal(g) for g in getattr(geom, 'geoms', [])]
    parts = [p for p in parts if p is not None]
    return unary_union(parts) if parts else None


def polygonize_labels_topo(
    labels: np.ndarray, transform, crs: str,
    tol_m: float = TOPO_TOL_M, max_vertices: int = MAX_VERTICES,
) -> gpd.GeoDataFrame:
    """Label raster → simplified, gap-free GeoDataFrame polygons.
    Topology-preserving: each shared edge is traced and simplified exactly once,
    so neighbouring fields always share identical geometry (zero gaps / overlaps)."""
    tol_px = tol_m / abs(transform.a)
    rings = vectorize_labels(labels, tol_px=tol_px, max_vertices=max_vertices)
    geoms, labs = [], []
    for lab, rs in rings.items():
        shells = [_as_polygonal(Polygon(_rings_to_world(r, transform)))
                  for r in rs if _signed_area(r) < 0]
        holes  = [_as_polygonal(Polygon(_rings_to_world(r, transform)))
                  for r in rs if _signed_area(r) > 0]
        shells = [g for g in shells if g is not None]
        holes  = [g for g in holes  if g is not None]
        if not shells:
            continue
        try:
            g = unary_union(shells)
            if holes:
                g = g.difference(unary_union(holes))
        except Exception:
            g = unary_union([s.buffer(0) for s in shells])
        g = _as_polygonal(g)
        if g is None:
            continue
        for part in (g.geoms if g.geom_type == 'MultiPolygon' else [g]):
            if part.area > 0:
                geoms.append(part)
                labs.append(int(lab))
    return gpd.GeoDataFrame({'label': labs}, geometry=geoms, crs=crs)


def _remove_daggers(poly, max_w: float = DAGGER_WIDTH_M, min_notch_area: float = 20.0):
    """Fill narrow inward spikes (daggers) via convex-hull notch detection."""
    if poly.is_empty or not poly.is_valid:
        return poly
    notches = poly.convex_hull.difference(poly)
    if notches.is_empty:
        return poly
    parts = list(notches.geoms) if hasattr(notches, 'geoms') else [notches]
    fill  = [n for n in parts
             if n.area >= min_notch_area and n.buffer(-max_w / 2).is_empty]
    if not fill:
        return poly
    out = unary_union([poly] + fill)
    return (max(out.geoms, key=lambda g: g.area)
            if isinstance(out, MultiPolygon) else out)


def orthogonal_regularize(poly, angle_tol_deg: float = ANGLE_TOL_DEG):
    """Snap near-axis edges to exactly axis-aligned relative to the polygon's dominant orientation."""
    if poly.is_empty or poly.geom_type != 'Polygon':
        return poly
    mrr = list(poly.minimum_rotated_rectangle.exterior.coords)
    e = np.array(mrr[1]) - np.array(mrr[0])
    base_angle = np.arctan2(e[1], e[0])
    coords = list(poly.exterior.coords)[:-1]
    new_coords = [np.array(coords[0], dtype=float)]
    for i in range(1, len(coords)):
        p0 = new_coords[-1]
        p1 = np.array(coords[i], dtype=float)
        v = p1 - p0
        if np.linalg.norm(v) < 1e-6:
            continue
        ang_deg = np.degrees(np.arctan2(v[1], v[0]) - base_angle) % 180
        nearest_axis = round(ang_deg / 90) * 90
        if abs(ang_deg - nearest_axis) <= angle_tol_deg:
            snapped = base_angle + np.radians(nearest_axis)
            nd = np.array([np.cos(snapped), np.sin(snapped)])
            p1 = p0 + nd * np.dot(v, nd)
        new_coords.append(p1)
    if len(new_coords) < 3:
        return poly
    out = Polygon([tuple(c) for c in new_coords])
    if not out.is_valid:
        out = out.buffer(0)
    if out.geom_type == 'MultiPolygon':
        out = max(out.geoms, key=lambda g: g.area)
    return out if (out.is_valid and not out.is_empty and out.area > poly.area * 0.3) else poly


def postprocess_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    'topo'     (v3): already simplified while tracing — only validates geometry.
    'coverage' (v2): geopandas simplify_coverage, shared edges simplified once → no slivers.
    'legacy'       : dagger removal → simplify → square-up → simplify, per polygon.
    """
    if VECTOR_MODE == 'topo':
        return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.is_valid]

    if VECTOR_MODE == 'coverage' and HAS_COVERAGE and len(gdf):
        gdf = gdf.copy()
        try:
            gdf['geometry'] = gdf.geometry.simplify_coverage(COVERAGE_TOL_M, simplify_boundary=True)
            gdf['geometry'] = gdf.geometry.make_valid()
            gdf = gdf.explode(index_parts=False)
            gdf = gdf[gdf.geom_type == 'Polygon']
            return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.is_valid]
        except Exception as exc:
            logger.warning('coverage simplify failed (%s); using legacy post-processing', exc)

    gdf = gdf.copy()
    gdf['geometry'] = [_remove_daggers(g, DAGGER_WIDTH_M) for g in gdf.geometry]
    gdf['geometry'] = [g.simplify(SIMPLIFY_FIRST_M, preserve_topology=True) for g in gdf.geometry]
    if SMOOTH_METHOD == 'orthogonal':
        gdf['geometry'] = [orthogonal_regularize(g, ANGLE_TOL_DEG) for g in gdf.geometry]
    if SIMPLIFY_AFTER_M > 0:
        gdf['geometry'] = [g.simplify(SIMPLIFY_AFTER_M, preserve_topology=True) for g in gdf.geometry]
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.is_valid]


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — public entry point
# ─────────────────────────────────────────────────────────────────────────────

def extract_fields(village_feature: dict, model, device,
                   progress_callback=None) -> dict:
    """
    End-to-end field boundary extraction.

    Parameters
    ----------
    village_feature   : GeoJSON Feature with a Polygon or MultiPolygon in WGS84
    model             : loaded TripleHeadModel
    device            : torch.device
    progress_callback : optional callable(step: str, message: str, pct: int)
                        called at every major pipeline stage for live UI updates.

    Returns
    -------
    GeoJSON FeatureCollection dict (WGS84 coordinates)
    """
    def _emit(step: str, message: str, pct: int = 0):
        if progress_callback:
            try:
                progress_callback(step, message, pct)
            except Exception:
                pass

    if (not isinstance(village_feature, dict)
            or village_feature.get('type') != 'Feature'
            or not village_feature.get('geometry')):
        raise InvalidVillageGeometryError(
            'Each village must be a GeoJSON Feature with a geometry.'
        )

    _emit('geometry', 'Validating village geometry …', 5)
    try:
        village_wgs84 = shape(village_feature['geometry'])
    except (TypeError, ValueError) as exc:
        raise InvalidVillageGeometryError('Village geometry is not valid GeoJSON.') from exc
    if (village_wgs84.is_empty or not village_wgs84.is_valid
            or village_wgs84.geom_type not in ('Polygon', 'MultiPolygon')):
        raise InvalidVillageGeometryError(
            'Village geometry must be a valid Polygon or MultiPolygon.'
        )

    representative_point = village_wgs84.representative_point()
    utm_crs = _utm_crs(representative_point.x, representative_point.y)
    projector = pyproj.Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True)
    village_utm = shapely.transform(
        village_wgs84,
        lambda xy: np.column_stack(projector.transform(xy[:, 0], xy[:, 1])),
    )
    minx, miny, maxx, maxy = village_utm.bounds
    area_km2 = (maxx - minx) * (maxy - miny) / 1e6
    if area_km2 > MAX_BBOX_KM2:
        raise InvalidVillageGeometryError(
            f'Village bounding box ({area_km2:.1f} km²) exceeds the '
            f'maximum allowed area of {MAX_BBOX_KM2:.0f} km². '
            'Split the village into smaller tiles and retry.'
        )

    village_cache_key = _village_cache_key(village_wgs84)

    # 1. Imagery
    _emit('imagery', 'Checking cached imagery or fetching from ESRI …', 15)
    logger.info('Starting imagery retrieval for village in %s.', utm_crs)
    mosaic, transform = fetch_imagery(
        minx, miny, maxx, maxy, utm_crs,
        village_cache_key=village_cache_key,
        progress_callback=progress_callback,
    )

    # 1b. Auxiliary road and urban masks
    road_mask, urban_mask = None, None
    if ENABLE_ROAD_MASK or ENABLE_URBAN_MASK:
        _emit('auxiliary', 'Retrieving road barrier and urban masks …', 35)
        road_mask, urban_mask = fetch_road_and_urban_masks(
            minx, miny, maxx, maxy, utm_crs,
            shape=mosaic.shape[1:],
            transform=transform,
            village_cache_key=village_cache_key,
            progress_callback=progress_callback,
        )

    # 2. Inference
    _emit('inference', 'Running sliding-window model inference …', 40)
    inference_started_at = time.perf_counter()
    logger.info('Running model inference on imagery mosaic %s.', mosaic.shape)
    extent_prob, boundary_prob, distance_pred = run_inference(
        model, mosaic, device, overlap=OVERLAP, batch_size=BATCH_SIZE, fp16=USE_FP16,
        progress_callback=progress_callback,
    )
    logger.info(
        'Model inference complete: elapsed=%.1fs',
        time.perf_counter() - inference_started_at,
    )

    _emit('masking', 'Applying village boundary mask …', 58)
    village_mask = geometry_mask(
        [mapping(village_utm)],
        out_shape=extent_prob.shape,
        transform=transform,
        invert=True,
    )

    # 3. Instance extraction via segment (v5 neck seeding & sharp boundary maps)
    _emit('watershed', f'Extracting field instances (mode={SEED_MODE}) …', 65)
    logger.info('Extracting field instances (mode=%s).', SEED_MODE)

    labels, rescued_ids, info = segment(
        extent_prob, boundary_prob, village_mask,
        distance_pred=distance_pred, img=mosaic,
        road_mask=road_mask, urban_mask=urban_mask,
        post=True,
    )

    # 4. Optional oversized splitting if enabled and using legacy mode
    if SPLIT_OVERSIZED and SEED_MODE not in ('neck', 'neck_dist') and labels.max() > 0:
        _emit('splitting', 'Splitting oversized regions …', 72)
        sizes = np.bincount(labels.ravel())[1:]
        sizes = sizes[sizes > 0] * px_area
        if len(sizes):
            pctl_thresh = float(np.percentile(sizes, OVERSIZE_PCTL))
            bp_s = normalize_boundary(boundary_prob, village_mask)
            labels = split_oversized(labels, bp_s, pctl_thresh, px_area, verbose=False)
            labels = split_oversized(labels, bp_s, MAX_FIELD_HA * 1e4, px_area, verbose=False,
                                     h_ladder=(0.12, 0.08, 0.05, 0.03, 0.02, 0.01))

    logger.info(
        'Post-extraction: seeds=%d, merged=%d, rescued=%d, tree-filled=%d, final_fields=%d',
        info.get('seeds', 0), info.get('merged', 0), len(rescued_ids), info.get('filled', 0), info.get('fields', 0),
    )

    # 5. Polygonization
    _emit('polygonize', f'Vectorising labels (mode={VECTOR_MODE}) …', 80)
    if VECTOR_MODE == 'topo':
        # Topology-preserving arc-tracer: shared edges simplified once, zero gaps/overlaps
        gdf = polygonize_labels_topo(labels, transform, utm_crs)
        if gdf.empty:
            logger.info('No field instances detected inside village boundary.')
            empty_result = {'type': 'FeatureCollection', 'features': []}
            store_cached_fields(village_feature, empty_result)
            _emit('complete', 'Done — no field instances detected.', 100)
            return empty_result
        gdf = postprocess_polygons(gdf)  # topo: only validates
    else:
        raw_polys = [
            shape(g)
            for g, v in rio_shapes(labels.astype(np.int32), mask=labels > 0, transform=transform)
            if v > 0
        ]
        if not raw_polys:
            logger.info('No field instances detected inside village boundary.')
            empty_result = {'type': 'FeatureCollection', 'features': []}
            store_cached_fields(village_feature, empty_result)
            _emit('complete', 'Done — no field instances detected.', 100)
            return empty_result
        gdf = gpd.GeoDataFrame({'id': range(len(raw_polys))}, geometry=raw_polys, crs=utm_crs)
        gdf = postprocess_polygons(gdf)

    if gdf.empty:
        logger.info('No field polygons remained after post-processing.')
        empty_result = {'type': 'FeatureCollection', 'features': []}
        store_cached_fields(village_feature, empty_result)
        _emit('complete', 'Done — all polygons filtered out in post-processing.', 100)
        return empty_result

    # Tag rescued fields with source='rescued'; all others get source='model'
    rescued_set = set(rescued_ids)
    if 'label' in gdf.columns:
        gdf['source'] = gdf['label'].apply(lambda lb: 'rescued' if lb in rescued_set else 'model')
    else:
        gdf['source'] = 'model'

    # 6. Funnel: area filtering, exact village clipping, and QC attributes
    _emit('funnel', 'Applying area filtering, exact village clipping, and QC …', 86)
    logger.info('Post-processing %d candidate field polygons (vector_mode=%s).', len(gdf), VECTOR_MODE)
    min_area_m2 = max(MIN_FIELD_M2, gdf.geometry.area.median() * MIN_AREA_FRAC)
    gdf = gdf[gdf.geometry.area >= min_area_m2]

    if CLIP_TO_VILLAGE and village_utm is not None:
        gdf['geometry'] = gdf.geometry.intersection(village_utm)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].explode(index_parts=False)
        gdf = gdf[(gdf.geom_type == 'Polygon') & gdf.geometry.is_valid & (gdf.geometry.area >= min_area_m2)]

    if gdf.empty:
        logger.info('No field polygons remained after village clipping and area filter.')
        empty_result = {'type': 'FeatureCollection', 'features': []}
        store_cached_fields(village_feature, empty_result)
        _emit('complete', 'Done — all polygons filtered out after clipping.', 100)
        return empty_result

    gdf = gdf.reset_index(drop=True)
    gdf['id'] = np.arange(1, len(gdf) + 1)
    gdf['area_m2'] = gdf.geometry.area.round(1)
    gdf['area_ha'] = (gdf.geometry.area / 1e4).round(3)
    gdf['perimeter_m'] = gdf.geometry.length.round(1)
    gdf['n_vertices'] = gdf.geometry.map(lambda geom: len(geom.exterior.coords) - 1)
    gdf['qc'] = np.select([gdf.area_ha < SMALL_FIELD_HA, gdf.area_ha > MAX_FIELD_HA], ['small', 'oversize'], 'ok')

    # 7. Reproject → WGS84 and build GeoJSON
    _emit('reproject', 'Reprojecting UTM → WGS84 and building GeoJSON …', 92)
    gdf_wgs84 = gdf.to_crs('EPSG:4326')
    village_props = dict(village_feature.get('properties') or {})
    features = [
        {
            'type': 'Feature',
            'geometry': mapping(row.geometry),
            'properties': {
                **village_props,
                'village':     village_props.get('name', village_props.get('village')),
                'field_id':    int(row['id']),
                'area_m2':     float(row['area_m2']),
                'area_ha':     float(row['area_ha']),
                'perimeter_m': float(row['perimeter_m']),
                'n_vertices':  int(row['n_vertices']),
                'qc':          str(row['qc']),
                'source':      str(row['source']),
            },
        }
        for _, row in gdf_wgs84.iterrows()
    ]

    result = {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'},
        },
        'features': features,
    }
    # Persist prediction to S3
    store_cached_fields(village_feature, result)
    logger.info('Village inference complete: %d field polygons (QC: %s).',
                len(features), dict(gdf['qc'].value_counts()))
    _emit('complete', f'Done — {len(features)} field polygons detected.', 100)
    return result