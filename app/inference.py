"""
inference.py — v2 field boundary pipeline
==========================================
Input  : village GeoJSON Feature(s) with Polygon or MultiPolygon geometry
Output : GeoJSON FeatureCollection of field polygons

Pipeline
--------
1. Derive UTM CRS from coordinates
2. Fetch RGB imagery from ESRI World Imagery (chunked to stay under tile limits)
3. Sliding-window inference with Hann-window blending (avoids seam artefacts)
4. H-minima seeded watershed on combined edge/distance map → instance labels
5. Post-process polygons (dagger removal, RDP simplification, area filter)
6. Reproject UTM → WGS84, emit GeoJSON
"""

import hashlib
import io
import json
import logging
import time
from pathlib import Path

import requests

import numpy as np
import shapely
import torch
import geopandas as gpd
import pyproj

from PIL import Image
from shapely.geometry import shape, mapping, MultiPolygon
from shapely.ops import unary_union
from rasterio.features import geometry_mask, shapes as rio_shapes
from rasterio.transform import from_bounds
from skimage.segmentation import watershed
from skimage.morphology import closing, disk, remove_small_objects, h_minima
from skimage.filters import sobel
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter

from app.config import (
    PATCH_PX, RES_M,
    OVERSEG_H, EXTENT_THRESH,
    SIMPLIFY_TOL_M, DAGGER_WIDTH_M, MIN_AREA_FRAC,
    MAX_BBOX_KM2, CACHE_MAX_GB,
)

# ── ESRI World Imagery tile server ────────────────────────────────────────────
_ESRI_URL     = ('https://services.arcgisonline.com/arcgis/rest/services/'
                 'World_Imagery/MapServer/export')
_MAX_CHUNK_PX = 4000      # max pixels per tile request (ESRI limit)
_CACHE_DIR = Path(__file__).resolve().parent.parent / 'data'
logger = logging.getLogger(__name__)


def _village_cache_key(village_geom: object) -> str:
    """Create a stable cache filename for the village geometry."""
    if village_geom is None:
        raise ValueError('Village geometry is required for cache key generation.')
    digest = hashlib.sha256(village_geom.wkb).hexdigest()
    return digest[:32]


def _load_cached_imagery(cache_key: str, minx: float, miny: float, maxx: float, maxy: float):
    """Load cached ESRI imagery from the local data folder if it exists."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f'esri_{cache_key}.npz'
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
    cache_path = _CACHE_DIR / f'esri_{cache_key}.npz'
    np.savez_compressed(
        cache_path,
        mosaic=mosaic,
        minx=np.float64(minx),
        miny=np.float64(miny),
        maxx=np.float64(maxx),
        maxy=np.float64(maxy),
    )
    logger.info('Saved ESRI imagery cache to %s.', cache_path)

    # ── P2-1: LRU eviction ───────────────────────────────────────────────────────
    if CACHE_MAX_GB <= 0:
        return cache_path   # eviction disabled

    limit_bytes = CACHE_MAX_GB * 1024 ** 3
    entries = sorted(
        _CACHE_DIR.glob('esri_*.npz'),
        key=lambda p: p.stat().st_mtime,   # oldest first
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

def fetch_imagery(minx: float, miny: float, maxx: float, maxy: float,
                  utm_crs: str, village_cache_key: str | None = None) -> tuple[np.ndarray, object]:
    """
    Download RGB imagery from ESRI World Imagery for a UTM bounding box.

    Returns
    -------
    mosaic    : (3, H, W) uint8 numpy array
    transform : rasterio Affine transform (UTM coords)
    """
    if village_cache_key:
        cached = _load_cached_imagery(village_cache_key, minx, miny, maxx, maxy)
        if cached is not None:
            return cached

    started_at = time.perf_counter()
    w_px = int(round((maxx - minx) / RES_M))
    h_px = int(round((maxy - miny) / RES_M))

    # Reproject corners to Web Mercator (ESRI accepts 3857)
    proj = pyproj.Transformer.from_crs(utm_crs, 'EPSG:3857', always_xy=True)
    xs, ys = zip(*[proj.transform(x, y) for x, y in
                   [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]])
    web_minx, web_maxx = min(xs), max(xs)
    web_miny, web_maxy = min(ys), max(ys)

    def _fetch_tile(x0, y0, x1, y1, pw, ph, retries=4):
        params = {
            'bbox': f'{x0},{y0},{x1},{y1}',
            'bboxSR': 3857, 'imageSR': 3857,
            'size': f'{pw},{ph}',
            'format': 'png', 'f': 'image',
        }
        last_error = None
        for attempt in range(retries):
            try:
                r = requests.get(_ESRI_URL, params=params, timeout=60)
                r.raise_for_status()
                ct = r.headers.get('Content-Type', '')
                if 'image' not in ct:
                    raise ValueError(f'non-image response ({ct}): {r.text[:120]}')
                return np.array(
                    Image.open(io.BytesIO(r.content)).convert('RGB')
                ).transpose(2, 0, 1)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    'ESRI imagery tile request failed (attempt %d/%d): %s',
                    attempt + 1, retries, exc,
                )
                if attempt + 1 < retries:
                    time.sleep(2 ** attempt)
        raise ImageryFetchError(
            'ESRI imagery tile request failed after all retries.'
        ) from last_error

    nx = int(np.ceil(w_px / _MAX_CHUNK_PX))
    ny = int(np.ceil(h_px / _MAX_CHUNK_PX))
    logger.info(
        "Fetching ESRI imagery: CRS=%s bounds=(%.1f, %.1f, %.1f, %.1f), "
        "size=%dx%d px, tiles=%d",
        utm_crs, minx, miny, maxx, maxy, w_px, h_px, nx * ny,
    )
    cw    = (web_maxx - web_minx) / nx
    ch    = (web_maxy - web_miny) / ny
    cw_px = int(round(w_px / nx))
    ch_px = int(round(h_px / ny))

    tiles: dict = {}
    for i in range(nx):
        for j in range(ny):
            tiles[(i, j)] = _fetch_tile(
                web_minx + i * cw,       web_miny + j * ch,
                web_minx + (i + 1) * cw, web_miny + (j + 1) * ch,
                cw_px, ch_px,
            )
            time.sleep(0.5)   # be polite to the tile server

    # rows are bottom-up in world coords → reverse j for array order
    mosaic = np.concatenate(
        [np.concatenate([tiles[(i, j)] for j in reversed(range(ny))], axis=1)
         for i in range(nx)],
        axis=2,
    )
    transform = from_bounds(minx, miny, maxx, maxy, mosaic.shape[2], mosaic.shape[1])
    if village_cache_key:
        _save_cached_imagery(village_cache_key, mosaic, minx, miny, maxx, maxy)
    logger.info(
        "ESRI imagery fetched: mosaic=%s, elapsed=%.1fs",
        mosaic.shape, time.perf_counter() - started_at,
    )
    return mosaic, transform


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — sliding-window inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, img_uint8: np.ndarray, device,
                  overlap: int = 64, batch_size: int = 8
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Overlapping-patch inference with Hann-window blending.

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
    Hp, Wp = ip.shape[1], ip.shape[2]

    acc  = np.zeros((3, Hp, Wp), 'float32')
    wsum = np.zeros((Hp, Wp), 'float32')
    win  = (np.outer(np.hanning(PATCH_PX), np.hanning(PATCH_PX)) + 1e-6).astype('float32')

    coords = [(y, x)
              for y in range(0, Hp - PATCH_PX + 1, stride)
              for x in range(0, Wp - PATCH_PX + 1, stride)]

    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            bc = coords[i:i + batch_size]
            bi = np.stack([ip[:, y:y + PATCH_PX, x:x + PATCH_PX] for y, x in bc])
            e, b, d = model(torch.from_numpy(bi).to(device))
            e = torch.sigmoid(e).cpu().numpy()[:, 0]
            b = torch.sigmoid(b).cpu().numpy()[:, 0]
            d = d.cpu().numpy()[:, 0]
            for (y, x), ee, bb, dd in zip(bc, e, b, d):
                acc[0, y:y + PATCH_PX, x:x + PATCH_PX] += ee * win
                acc[1, y:y + PATCH_PX, x:x + PATCH_PX] += bb * win
                acc[2, y:y + PATCH_PX, x:x + PATCH_PX] += dd * win
                wsum[y:y + PATCH_PX, x:x + PATCH_PX]   += win

    acc /= np.maximum(wsum, 1e-6)
    return acc[0, :H, :W], acc[1, :H, :W], acc[2, :H, :W]


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — instance extraction (h-minima seeded watershed)
# ─────────────────────────────────────────────────────────────────────────────

def extract_instances(extent_prob: np.ndarray, boundary_prob: np.ndarray,
                      distance_pred: np.ndarray) -> np.ndarray:
    """
    Combine boundary and distance maps into a joint edge strength image,
    then seed watershed from h-minima to produce an instance label map.

    Returns label array (H, W) int32 — 0 = background.
    """
    # P2-3: np.ptp() was removed in NumPy 2.0 — use explicit max-min instead
    dist_norm   = (distance_pred - distance_pred.min()) / (
        (distance_pred.max() - distance_pred.min()) + 1e-9
    )
    dist_edges  = sobel(gaussian_filter(dist_norm, sigma=1.0))
    dist_edges /= (dist_edges.max() + 1e-9)

    edge_strength = np.maximum(boundary_prob / (boundary_prob.max() + 1e-9), dist_edges)
    edge_smooth   = gaussian_filter(edge_strength, sigma=1.5)

    extent_mask = ndi.binary_fill_holes(
        closing(extent_prob > EXTENT_THRESH, disk(2))
    )
    markers, _ = ndi.label(h_minima(edge_smooth, OVERSEG_H))
    labels      = watershed(edge_smooth, markers, mask=extent_mask)
    # max_size is inclusive, so 199 keeps the old min_size=200 behaviour
    labels      = remove_small_objects(labels, max_size=199)

    return labels.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — polygon post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _remove_daggers(poly, max_w: float = 2.0, min_notch_area: float = 20.0):
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


def postprocess_polygons(gdf: gpd.GeoDataFrame, clip_geometry=None) -> gpd.GeoDataFrame:
    """Apply dagger removal, RDP simplification and median-area filter."""
    gdf = gdf.copy()
    gdf['geometry'] = [_remove_daggers(g, DAGGER_WIDTH_M) for g in gdf.geometry]
    gdf['geometry'] = [g.simplify(SIMPLIFY_TOL_M, preserve_topology=True)
                       for g in gdf.geometry]
    if clip_geometry is not None:
        gdf['geometry'] = [g.intersection(clip_geometry) for g in gdf.geometry]
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.is_valid]
    gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]

    if gdf.empty:
        return gdf

    med_area = gdf.geometry.area.median()
    gdf = gdf[gdf.geometry.area >= med_area * MIN_AREA_FRAC].reset_index(drop=True)

    gdf['area_m2']     = gdf.geometry.area.round(1)
    gdf['area_ha']     = (gdf.geometry.area / 10000).round(3)
    gdf['perimeter_m'] = gdf.geometry.length.round(1)
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def extract_fields(village_feature: dict, model, device) -> dict:
    """
    End-to-end field boundary extraction.

    Parameters
    ----------
    village_feature : GeoJSON Feature with a Polygon or MultiPolygon in WGS84
    model      : loaded TripleHeadModel
    device     : torch.device

    Returns
    -------
    GeoJSON FeatureCollection dict (WGS84 coordinates)
    """
    if (not isinstance(village_feature, dict)
            or village_feature.get('type') != 'Feature'
            or not village_feature.get('geometry')):
        raise InvalidVillageGeometryError(
            'Each village must be a GeoJSON Feature with a geometry.'
        )

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

    # ── P0 guard: reject inputs whose bounding box would cause OOM ──────────
    area_km2 = (maxx - minx) * (maxy - miny) / 1e6
    if area_km2 > MAX_BBOX_KM2:
        raise InvalidVillageGeometryError(
            f'Village bounding box ({area_km2:.1f} km²) exceeds the '
            f'maximum allowed area of {MAX_BBOX_KM2:.0f} km². '
            'Split the village into smaller tiles and retry.'
        )

    village_cache_key = _village_cache_key(village_wgs84)

    # 1. Imagery
    logger.info('Starting imagery retrieval for village in %s.', utm_crs)
    mosaic, transform = fetch_imagery(minx, miny, maxx, maxy, utm_crs, village_cache_key=village_cache_key)

    # 2. Inference
    inference_started_at = time.perf_counter()
    logger.info('Running model inference on imagery mosaic %s.', mosaic.shape)
    extent_prob, boundary_prob, distance_pred = run_inference(model, mosaic, device)
    logger.info(
        'Model inference complete: elapsed=%.1fs',
        time.perf_counter() - inference_started_at,
    )
    village_mask = geometry_mask(
        [mapping(village_utm)],
        out_shape=extent_prob.shape,
        transform=transform,
        invert=True,
    )
    extent_prob = np.where(village_mask, extent_prob, 0)

    # 3. Instances
    logger.info('Extracting field instances from model predictions.')
    labels = extract_instances(extent_prob, boundary_prob, distance_pred)

    raw_polys = [
        shape(g)
        for g, v in rio_shapes(labels, mask=labels > 0, transform=transform)
        if v > 0
    ]
    if not raw_polys:
        logger.info('No field instances detected inside village boundary.')
        return {'type': 'FeatureCollection', 'features': []}

    gdf = gpd.GeoDataFrame(
        {'id': range(len(raw_polys))}, geometry=raw_polys, crs=utm_crs
    )

    # 4. Post-process
    logger.info('Post-processing %d candidate field polygons.', len(raw_polys))
    gdf = postprocess_polygons(gdf, clip_geometry=village_utm)
    if gdf.empty:
        logger.info('No field polygons remained after post-processing.')
        return {'type': 'FeatureCollection', 'features': []}

    # 5. Reproject → WGS84 and build GeoJSON
    gdf_wgs84 = gdf.to_crs('EPSG:4326')
    features = [
        {
            'type': 'Feature',
            'geometry': mapping(row.geometry),
            'properties': {
                'field_id':    int(row['id']) + 1,
                'area_m2':     row['area_m2'],
                'area_ha':     row['area_ha'],
                'perimeter_m': row['perimeter_m'],
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
    logger.info('Village inference complete: %d field polygons.', len(features))
    return result