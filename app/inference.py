"""
inference.py — v2 field boundary pipeline
==========================================
Input  : village GeoJSON Feature(s) with Polygon or MultiPolygon geometry
Output : GeoJSON FeatureCollection of field polygons

Pipeline (v2 algorithm from village_field_polygons_v2_.ipynb):
-------------------------------------------------------------
1. Derive UTM CRS from coordinates and validate bounds
2. Fetch RGB imagery from ESRI World Imagery (with S3 & disk caching, parallel chunking)
3. Sliding-window inference with Hann-window blending (channels-last, fp16 autocast)
4. Field instance extraction:
   - SEED_MODE='cores': field cores via robustly normalized distance & low boundary prob,
     disk(1) closing (prevents bridging 1-3m bunds), flooding watershed, orphan rescue.
   - SEED_MODE='hminima': legacy watershed on blurred boundary/distance edge map.
5. Post-extraction refinement:
   - split_oversized: stepping-down h-minima ladder on bounding box of large regions
   - merge_weak_boundaries: region-adjacency graph merge for borders with boundary prob < MERGE_BND_T
   - smooth_labels: raster spike/dagger removal via disk opening and expand_labels
6. Vector polygon post-processing:
   - VECTOR_MODE='coverage': GeoPandas simplify_coverage (shared boundaries simplified once,
     eliminating slivers/staircases).
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
from skimage.morphology import disk, h_minima
from skimage.filters import sobel
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter

from app.config import (
    PATCH_PX, RES_M, BATCH_SIZE, OVERLAP, USE_FP16,
    SEED_MODE, SEED_DIST_MIN, SEED_BND_MAX, SEED_MIN_PX,
    EDGE_SIGMA, DIST_WEIGHT, MERGE_BND_T, LABEL_SMOOTH_R,
    RESCUE_ORPHANS, MIN_REGION_PX, OVERSEG_H, EXTENT_THRESH,
    VECTOR_MODE, COVERAGE_TOL_M, DAGGER_WIDTH_M, SIMPLIFY_FIRST_M,
    SMOOTH_METHOD, ANGLE_TOL_DEG, SIMPLIFY_AFTER_M,
    SPLIT_OVERSIZED, OVERSIZE_PCTL, MIN_FIELD_M2, MIN_AREA_FRAC,
    SMALL_FIELD_HA, MAX_FIELD_HA, CLIP_TO_VILLAGE,
    MAX_BBOX_KM2, CACHE_MAX_GB,
    ESRI_MAX_WORKERS, ESRI_IMAGE_FORMAT, ESRI_JPEG_QUALITY,
    ESRI_TIMEOUT_SEC, ESRI_MAX_CHUNK_PX,
)
from app.result_cache import (
    geometry_cache_digest,
    get_cached_imagery,
    store_cached_imagery_async,
    store_cached_fields,
)

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
    """
    # 1. Check local disk cache
    if village_cache_key:
        cached = _load_cached_imagery(village_cache_key, minx, miny, maxx, maxy)
        if cached is not None:
            if progress_callback:
                progress_callback('imagery', 'Using cached RGB imagery from local disk …', 35)
            return cached

        # 2. Check S3 imagery cache
        cached_s3 = get_cached_imagery(village_cache_key, minx, miny, maxx, maxy)
        if cached_s3 is not None:
            if progress_callback:
                progress_callback('imagery', 'Using cached RGB imagery from S3 …', 35)
            return cached_s3

    started_at = time.perf_counter()
    w_px = int(round((maxx - minx) / RES_M))
    h_px = int(round((maxy - miny) / RES_M))

    # Reproject corners to Web Mercator (ESRI accepts 3857)
    proj = pyproj.Transformer.from_crs(utm_crs, 'EPSG:3857', always_xy=True)
    xs, ys = zip(*[proj.transform(x, y) for x, y in
                   [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]])
    web_minx, web_maxx = min(xs), max(xs)
    web_miny, web_maxy = min(ys), max(ys)

    nx = int(np.ceil(w_px / ESRI_MAX_CHUNK_PX))
    ny = int(np.ceil(h_px / ESRI_MAX_CHUNK_PX))
    total_tiles = nx * ny
    logger.info(
        "Fetching ESRI imagery: CRS=%s bounds=(%.1f, %.1f, %.1f, %.1f), "
        "size=%dx%d px, tiles=%d (nx=%d, ny=%d, format=%s, workers=%d)",
        utm_crs, minx, miny, maxx, maxy, w_px, h_px, total_tiles, nx, ny,
        ESRI_IMAGE_FORMAT, min(ESRI_MAX_WORKERS, total_tiles),
    )

    cw    = (web_maxx - web_minx) / nx
    ch    = (web_maxy - web_miny) / ny
    cw_px = int(round(w_px / nx))
    ch_px = int(round(h_px / ny))

    session = _get_http_session(pool_size=max(10, min(ESRI_MAX_WORKERS, total_tiles) * 2))

    def _fetch_tile(coords, x0, y0, x1, y1, pw, ph, retries=4):
        params = {
            'bbox': f'{x0},{y0},{x1},{y1}',
            'bboxSR': 3857, 'imageSR': 3857,
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

    tile_jobs = []
    for i in range(nx):
        for j in range(ny):
            tile_jobs.append((
                (i, j),
                web_minx + i * cw,       web_miny + j * ch,
                web_minx + (i + 1) * cw, web_miny + (j + 1) * ch,
                cw_px, ch_px,
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

    # rows are bottom-up in world coords → reverse j for array order
    mosaic = np.concatenate(
        [np.concatenate([tiles[(i, j)] for j in reversed(range(ny))], axis=1)
         for i in range(nx)],
        axis=2,
    )
    transform = from_bounds(minx, miny, maxx, maxy, mosaic.shape[2], mosaic.shape[1])

    if village_cache_key:
        _save_cached_imagery(village_cache_key, mosaic, minx, miny, maxx, maxy)
        store_cached_imagery_async(village_cache_key, mosaic, minx, miny, maxx, maxy, utm_crs=utm_crs)

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
                  fp16: bool = USE_FP16, progress_callback=None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Overlapping-patch inference with Hann-window blending.
    Supports FP16 autocast on CUDA and channels-last memory layout.

    Parameters
    ----------
    model       : TripleHeadModel (eval mode)
    img_uint8   : (3, H, W) uint8 RGB array
    device      : torch.device

    Returns
    -------
    extent_prob   : (H, W) float32  — field extent probability
    boundary_prob : (H, W) float32  — boundary probability
    distance_pred : (H, W) float32  — distance-to-boundary prediction
    """
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

    use_amp = bool(fp16 and device.type == 'cuda')
    with torch.inference_mode():
        for i in range(0, total_patches, batch_size):
            bc = coords[i:i + batch_size]
            bi = torch.from_numpy(np.stack([ip[:, y:y + PATCH_PX, x:x + PATCH_PX] for y, x in bc]))
            bi = bi.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    e, b, d = model(bi)
            else:
                e, b, d = model(bi)
            e = torch.sigmoid(e.float()).cpu().numpy()[:, 0]
            b = torch.sigmoid(b.float()).cpu().numpy()[:, 0]
            d = d.float().cpu().numpy()[:, 0]
            for (y, x), ee, bb, dd in zip(bc, e, b, d):
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
# Step 3 — instance extraction (v2 cores seeding & legacy h-minima)
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


def extract_instances_v2(extent_prob: np.ndarray, boundary_prob: np.ndarray,
                         distance_pred: np.ndarray, village_mask: np.ndarray,
                         extent_thresh: float = EXTENT_THRESH, seed_bnd_max: float = SEED_BND_MAX,
                         seed_dist_min: float = SEED_DIST_MIN, seed_min_px: int = SEED_MIN_PX,
                         edge_sigma: float = EDGE_SIGMA, dist_weight: float = DIST_WEIGHT,
                         min_px: int = MIN_REGION_PX, rescue_orphans: bool = RESCUE_ORPHANS
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    v2 instance extraction using field cores.
    Returns (labels, elev [flooding surface], bp [smoothed boundary prob], info).
    """
    # closing with disk(1), not disk(2): a 2-px closing bridges the 1-3 m bunds between fields
    extent_mask = ndi.binary_fill_holes(
        ndi.binary_closing(extent_prob > extent_thresh, structure=disk(1))
    )
    extent_mask &= village_mask

    bp = gaussian_filter(boundary_prob.astype('float32'), sigma=edge_sigma) if edge_sigma else boundary_prob.astype('float32')
    dist = _robust_norm(gaussian_filter(distance_pred.astype('float32'), 1.0), extent_mask)

    seed_mask = extent_mask & (bp < seed_bnd_max) & (dist > seed_dist_min)
    seed_mask = ndi.binary_opening(seed_mask, structure=disk(1))     # break 1-px bridges between cores
    markers, _ = ndi.label(seed_mask)
    markers = drop_small_regions(markers, seed_min_px)
    n_seeded = len(np.unique(markers)) - 1

    elev = bp + dist_weight * (1.0 - dist)
    labels = watershed(elev, markers, mask=extent_mask)

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
    return labels.astype(np.int32), elev, bp, info


def extract_instances_hminima(extent_prob: np.ndarray, boundary_prob: np.ndarray,
                              distance_pred: np.ndarray, village_mask: np.ndarray,
                              extent_thresh: float = EXTENT_THRESH, overseg_h: float = OVERSEG_H,
                              min_px: int = MIN_REGION_PX, rescue_orphans: bool = RESCUE_ORPHANS
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Legacy h-minima seeded watershed with fence & orphan rescue.
    Returns (labels, edge_smooth, bp, info).
    """
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
    'coverage': simplify the polygons as ONE tiling, so a shared edge is
    simplified once for both neighbours -> no slivers/overlaps, and staircase becomes straight line.
    (Spikes were already removed in raster space by smooth_labels.)
    'legacy': dagger removal -> simplify -> square-up -> simplify, per polygon.
    """
    if VECTOR_MODE == 'coverage' and HAS_COVERAGE and len(gdf):
        gdf = gdf.copy()
        try:
            gdf['geometry'] = gdf.geometry.simplify_coverage(COVERAGE_TOL_M, simplify_boundary=True)
            gdf['geometry'] = gdf.geometry.make_valid()
            gdf = gdf.explode(index_parts=False)
            gdf = gdf[gdf.geom_type == 'Polygon']
            return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.is_valid]
        except Exception as exc:
            logger.warning("coverage simplify failed (%s); using legacy post-processing", exc)

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

    # 3. Instance extraction
    _emit('watershed', f'Extracting field instances (mode={SEED_MODE}) …', 65)
    logger.info('Extracting field instances (mode=%s).', SEED_MODE)
    px_area = abs(transform.a) * abs(transform.e)

    if SEED_MODE == 'cores':
        labels, edge, bp_s, info = extract_instances_v2(
            extent_prob, boundary_prob, distance_pred, village_mask,
            extent_thresh=EXTENT_THRESH, seed_bnd_max=SEED_BND_MAX,
            seed_dist_min=SEED_DIST_MIN, seed_min_px=SEED_MIN_PX,
            edge_sigma=EDGE_SIGMA, dist_weight=DIST_WEIGHT,
            min_px=MIN_REGION_PX, rescue_orphans=RESCUE_ORPHANS,
        )
    else:
        labels, edge, bp_s, info = extract_instances_hminima(
            extent_prob, boundary_prob, distance_pred, village_mask,
            extent_thresh=EXTENT_THRESH, overseg_h=OVERSEG_H,
            min_px=MIN_REGION_PX, rescue_orphans=RESCUE_ORPHANS,
        )

    # 4. Oversized splitting and weak boundary merge
    if SPLIT_OVERSIZED and labels.max() > 0:
        _emit('splitting', 'Splitting oversized regions …', 72)
        sizes = np.bincount(labels.ravel())[1:]
        sizes = sizes[sizes > 0] * px_area
        if len(sizes):
            pctl_thresh = float(np.percentile(sizes, OVERSIZE_PCTL))
            labels = split_oversized(labels, edge, pctl_thresh, px_area, verbose=False)
            labels = split_oversized(labels, edge, MAX_FIELD_HA * 1e4, px_area, verbose=False,
                                     h_ladder=(0.12, 0.08, 0.05, 0.03, 0.02, 0.01))

    if SEED_MODE == 'cores':
        _emit('merging', 'Merging weak boundaries & smoothing labels …', 76)
        labels, n_merged = merge_weak_boundaries(
            labels, bp_s, thresh=MERGE_BND_T,
            max_area_px=MAX_FIELD_HA * 1e4 / px_area,
        )
        if LABEL_SMOOTH_R > 0:
            labels = drop_small_regions(smooth_labels(labels, LABEL_SMOOTH_R), MIN_REGION_PX)
        info['merged_regions'] = n_merged

    # 5. Polygonization
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

    gdf = gpd.GeoDataFrame(
        {'id': range(len(raw_polys))}, geometry=raw_polys, crs=utm_crs
    )

    # 6. Post-process & simplify
    _emit('postprocess', f'Post-processing {len(raw_polys)} candidate polygons (mode={VECTOR_MODE}) …', 80)
    logger.info('Post-processing %d candidate field polygons (vector_mode=%s).', len(raw_polys), VECTOR_MODE)
    gdf = postprocess_polygons(gdf)
    if gdf.empty:
        logger.info('No field polygons remained after post-processing.')
        empty_result = {'type': 'FeatureCollection', 'features': []}
        store_cached_fields(village_feature, empty_result)
        _emit('complete', 'Done — all polygons filtered out in post-processing.', 100)
        return empty_result

    # 7. Funnel: area filtering, exact village clipping, and QC attributes
    _emit('funnel', 'Applying area filtering, exact village clipping, and QC …', 86)
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
    gdf['qc'] = np.select([gdf.area_ha < SMALL_FIELD_HA, gdf.area_ha > MAX_FIELD_HA], ['small', 'oversize'], 'ok')

    # 8. Reproject → WGS84 and build GeoJSON
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
                'qc':          str(row['qc']),
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