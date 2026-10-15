"""
auxiliary.py — Road and Urban Masking Service for Field Boundary Delineation
=============================================================================
Provides road barrier enforcement and urban/settlement masking to eliminate
false fields in village settlements and prevent merging across roads.

Strategy
--------
Tier 1 — Local GeoPackage (fastest, zero network):
  If OSM_GPKG_PATH / OSM_URBAN_GPKG_PATH are configured, the village bbox is
  queried directly via GDAL's SQLite R-Tree index (~10 ms, no RAM overhead).

Tier 2 — Overpass API (OSM vector data, class-aware buffering):
  Fetches actual road/urban *vector* geometries from OpenStreetMap via the
  public Overpass API and rasterizes them with road-class-appropriate buffer
  widths (motorway=10m, primary=6m, secondary=4m, tertiary/residential=3m,
  track/path=2.5m).  Results are cached to disk so each village only hits
  the API once.

Tier 3 — Disk & S3 caching:
  Generated masks are persisted locally and (optionally) to S3 so repeat
  runs for the same village are instant.

NOTE: The previous ESRI World Transportation visual-tile fallback has been
removed.  That approach rasterised the map *rendering* (coloured lines for
every road class, labels, icons) rather than geometry, producing over-wide
masks that degraded segmentation rather than improving it.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pyproj
import requests
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union

from app.config import (
    BASE_DIR,
    ENABLE_ROAD_MASK,
    ENABLE_URBAN_MASK,
    OSM_GPKG_PATH,
    OSM_URBAN_GPKG_PATH,
)
from app.result_cache import (
    get_cached_auxiliary,
    store_cached_auxiliary_async,
)

logger = logging.getLogger(__name__)
_LOCAL_DATA_DIR = Path(BASE_DIR) / 'data'

# Overpass API endpoints (tried in order)
_OVERPASS_ENDPOINTS = [
    'https://overpass-api.de/api/interpreter',
    'https://overpass.karte.io/api/interpreter',
    'https://maps.mail.ru/osm/tools/overpass/api/interpreter',
]

# Buffer widths per highway class (metres, half-width each side of centreline)
_ROAD_BUFFER_M: dict[str, float] = {
    'motorway':       10.0,
    'motorway_link':   8.0,
    'trunk':           8.0,
    'trunk_link':      6.0,
    'primary':         6.0,
    'primary_link':    5.0,
    'secondary':       4.5,
    'secondary_link':  4.0,
    'tertiary':        3.5,
    'tertiary_link':   3.0,
    'residential':     3.0,
    'unclassified':    3.0,
    'living_street':   2.5,
    'service':         2.5,
    'track':           2.5,
    'path':            2.0,
    'footway':         1.5,
    'cycleway':        1.5,
}
_ROAD_BUFFER_DEFAULT_M = 3.0


# ──────────────────────────────────────────────────────────────────────────────
# Local disk cache helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_disk_aux_cache(cache_key: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    """Load cached road and urban masks from local data/ directory."""
    _LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _LOCAL_DATA_DIR / f'aux_{cache_key[:32]}.npz'
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path)
        road_mask = data['road_mask']
        urban_mask = data['urban_mask']
        if road_mask.shape == shape and urban_mask.shape == shape:
            logger.info('Loaded auxiliary masks from local disk: %s', cache_path)
            return road_mask, urban_mask
        logger.warning('Cached aux mask shape mismatch (%s vs %s) — regenerating.', road_mask.shape, shape)
    except Exception as exc:
        logger.warning('Failed reading local auxiliary cache %s: %s', cache_path, exc)
    return None


def _save_disk_aux_cache(cache_key: str, road_mask: np.ndarray, urban_mask: np.ndarray):
    """Save road and urban masks to local data/ directory."""
    _LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _LOCAL_DATA_DIR / f'aux_{cache_key[:32]}.npz'
    try:
        np.savez_compressed(
            cache_path,
            road_mask=road_mask.astype(bool),
            urban_mask=urban_mask.astype(bool),
        )
        logger.info('Saved auxiliary masks to local disk: %s', cache_path)
    except Exception as exc:
        logger.warning('Failed saving local auxiliary cache %s: %s', cache_path, exc)


# ──────────────────────────────────────────────────────────────────────────────
# Tier 1 — Local GeoPackage (fastest)
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_roads_from_gpkg(minx: float, miny: float, maxx: float, maxy: float,
                           utm_crs: str, shape: tuple[int, int], transform) -> np.ndarray | None:
    """Read roads from an indexed GeoPackage file without loading the whole file into RAM."""
    if not OSM_GPKG_PATH or not Path(OSM_GPKG_PATH).is_file():
        return None
    try:
        import geopandas as gpd
        gdf = gpd.read_file(OSM_GPKG_PATH, bbox=(minx, miny, maxx, maxy))
        if gdf.empty:
            return np.zeros(shape, dtype=bool)
        if str(gdf.crs) != utm_crs:
            gdf = gdf.to_crs(utm_crs)
        buffers = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom and not geom.is_empty:
                hw = str(row.get('highway', '')).lower()
                buf_m = _ROAD_BUFFER_M.get(hw, _ROAD_BUFFER_DEFAULT_M)
                buffers.append((geom.buffer(buf_m), 1))
        if not buffers:
            return np.zeros(shape, dtype=bool)
        mask = rasterize(buffers, out_shape=shape, transform=transform,
                         fill=0, dtype=np.uint8) > 0
        logger.info('GPKG roads: rasterized %d road geometries into mask (%.2f%% coverage).',
                    len(buffers), mask.mean() * 100)
        return mask
    except Exception as exc:
        logger.warning('Error reading roads from GPKG (%s): %s', OSM_GPKG_PATH, exc)
        return None


def _fetch_urban_from_gpkg(minx: float, miny: float, maxx: float, maxy: float,
                           utm_crs: str, shape: tuple[int, int], transform) -> np.ndarray | None:
    """Read urban/settlement areas from an indexed GeoPackage."""
    if not OSM_URBAN_GPKG_PATH or not Path(OSM_URBAN_GPKG_PATH).is_file():
        return None
    try:
        import geopandas as gpd
        gdf = gpd.read_file(OSM_URBAN_GPKG_PATH, bbox=(minx, miny, maxx, maxy))
        if gdf.empty:
            return np.zeros(shape, dtype=bool)
        if str(gdf.crs) != utm_crs:
            gdf = gdf.to_crs(utm_crs)
        shapes = [(geom, 1) for geom in gdf.geometry if geom and not geom.is_empty]
        if not shapes:
            return np.zeros(shape, dtype=bool)
        mask = rasterize(shapes, out_shape=shape, transform=transform,
                         fill=0, dtype=np.uint8) > 0
        logger.info('GPKG urban: rasterized %d polygons (%.2f%% coverage).', len(shapes), mask.mean() * 100)
        return mask
    except Exception as exc:
        logger.warning('Error reading urban layer from GPKG (%s): %s', OSM_URBAN_GPKG_PATH, exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Tier 2 — Overpass API (OSM vector geometry, class-aware buffering)
# ──────────────────────────────────────────────────────────────────────────────

def _overpass_query(ql: str, timeout: int = 30) -> dict | None:
    """Run an Overpass QL query, trying multiple public mirrors in order."""
    for url in _OVERPASS_ENDPOINTS:
        try:
            r = requests.post(url, data={'data': ql}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.warning('Overpass mirror %s failed: %s', url, exc)
    return None


def _wgs84_bbox(minx: float, miny: float, maxx: float, maxy: float,
                utm_crs: str, margin_m: float = 50.0) -> tuple[float, float, float, float]:
    """Convert a UTM bbox (+ margin) to WGS84 (S, W, N, E) for Overpass queries."""
    proj = pyproj.Transformer.from_crs(utm_crs, 'EPSG:4326', always_xy=True)
    corners = [
        proj.transform(minx - margin_m, miny - margin_m),
        proj.transform(maxx + margin_m, miny - margin_m),
        proj.transform(maxx + margin_m, maxy + margin_m),
        proj.transform(minx - margin_m, maxy + margin_m),
    ]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    return min(lats), min(lons), max(lats), max(lons)   # S, W, N, E


def _geojson_lines_to_shapely(elements: list[dict]) -> list:
    """Convert Overpass way elements (with geometry) to Shapely LineStrings."""
    from shapely.geometry import LineString
    lines = []
    for el in elements:
        if el.get('type') != 'way':
            continue
        geom = el.get('geometry', [])
        if len(geom) < 2:
            continue
        try:
            lines.append((LineString([(g['lon'], g['lat']) for g in geom]),
                          el.get('tags', {}).get('highway', '')))
        except Exception:
            pass
    return lines


def _geojson_polys_to_shapely(elements: list[dict]) -> list:
    """Convert Overpass way elements to Shapely Polygons (closed ways)."""
    from shapely.geometry import Polygon
    polys = []
    for el in elements:
        if el.get('type') != 'way':
            continue
        geom = el.get('geometry', [])
        if len(geom) < 3:
            continue
        try:
            pts = [(g['lon'], g['lat']) for g in geom]
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            poly = Polygon(pts)
            if poly.is_valid and not poly.is_empty:
                polys.append(poly)
        except Exception:
            pass
    return polys


def _fetch_roads_from_overpass(
    minx: float, miny: float, maxx: float, maxy: float,
    utm_crs: str, shape: tuple[int, int], transform,
) -> np.ndarray:
    """Fetch road centrelines from OSM via Overpass, buffer by class, rasterize."""
    s, w, n, e = _wgs84_bbox(minx, miny, maxx, maxy, utm_crs)
    ql = f"""
    [out:json][timeout:30];
    (
      way["highway"~"^(motorway|motorway_link|trunk|trunk_link|
                         primary|primary_link|secondary|secondary_link|
                         tertiary|tertiary_link|residential|unclassified|
                         living_street|service|track|path|footway|cycleway)$"]
          ({s:.6f},{w:.6f},{n:.6f},{e:.6f});
    );
    out geom;
    """

    t0 = time.perf_counter()
    data = _overpass_query(ql)
    if data is None:
        logger.warning('Overpass road query failed — using empty road mask.')
        return np.zeros(shape, dtype=bool)

    elements = data.get('elements', [])
    if not elements:
        logger.info('Overpass: no roads found in bbox.')
        return np.zeros(shape, dtype=bool)

    # Reproject from WGS84 to UTM
    proj = pyproj.Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True)
    lines_hw = _geojson_lines_to_shapely(elements)

    import shapely
    buffered_shapes = []
    for line_wgs84, hw in lines_hw:
        try:
            line_utm = shapely.ops.transform(lambda x, y: proj.transform(x, y), line_wgs84)
            buf_m = _ROAD_BUFFER_M.get(hw.lower(), _ROAD_BUFFER_DEFAULT_M)
            buffered_shapes.append((line_utm.buffer(buf_m), 1))
        except Exception:
            pass

    if not buffered_shapes:
        return np.zeros(shape, dtype=bool)

    mask = rasterize(buffered_shapes, out_shape=shape, transform=transform,
                     fill=0, dtype=np.uint8) > 0
    logger.info(
        'Overpass roads: %d ways in %.1fs → mask coverage %.2f%%.',
        len(lines_hw), time.perf_counter() - t0, mask.mean() * 100,
    )
    return mask


def _fetch_urban_from_overpass(
    minx: float, miny: float, maxx: float, maxy: float,
    utm_crs: str, shape: tuple[int, int], transform,
) -> np.ndarray:
    """Fetch settlement/urban polygons from OSM via Overpass."""
    s, w, n, e = _wgs84_bbox(minx, miny, maxx, maxy, utm_crs)
    ql = f"""
    [out:json][timeout:30];
    (
      way["landuse"~"^(residential|commercial|industrial|retail|construction)$"]
          ({s:.6f},{w:.6f},{n:.6f},{e:.6f});
      way["place"~"^(village|town|city|suburb|hamlet|neighbourhood)$"]
          ({s:.6f},{w:.6f},{n:.6f},{e:.6f});
      way["building"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});
    );
    out geom;
    """

    t0 = time.perf_counter()
    data = _overpass_query(ql)
    if data is None:
        logger.warning('Overpass urban query failed — using empty urban mask.')
        return np.zeros(shape, dtype=bool)

    elements = data.get('elements', [])
    polys_wgs84 = _geojson_polys_to_shapely(elements)
    if not polys_wgs84:
        logger.info('Overpass: no urban polygons found in bbox.')
        return np.zeros(shape, dtype=bool)

    proj = pyproj.Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True)
    import shapely.ops
    shapes = []
    for poly in polys_wgs84:
        try:
            p_utm = shapely.ops.transform(lambda x, y: proj.transform(x, y), poly)
            if p_utm.is_valid and not p_utm.is_empty:
                shapes.append((p_utm, 1))
        except Exception:
            pass

    if not shapes:
        return np.zeros(shape, dtype=bool)

    mask = rasterize(shapes, out_shape=shape, transform=transform,
                     fill=0, dtype=np.uint8) > 0
    logger.info(
        'Overpass urban: %d polygons in %.1fs → mask coverage %.2f%%.',
        len(polys_wgs84), time.perf_counter() - t0, mask.mean() * 100,
    )
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def fetch_road_and_urban_masks(
    minx: float, miny: float, maxx: float, maxy: float,
    utm_crs: str, shape: tuple[int, int], transform,
    village_cache_key: str | None = None,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve road-barrier mask and urban/settlement exclusion mask.

    Resolution order
    ----------------
    1. Local disk cache  (instant)
    2. S3 cache          (fast)
    3. Local GeoPackage  (GPKG — if OSM_GPKG_PATH configured)
    4. Overpass API      (OSM vector geometry, class-aware buffering)

    Returns
    -------
    road_mask  : (H, W) bool — True where road centreline buffers exist
    urban_mask : (H, W) bool — True where settlement/building polygons exist
    """

    # ── 1. Local disk cache ───────────────────────────────────────────────────
    if village_cache_key:
        cached = _load_disk_aux_cache(village_cache_key, shape)
        if cached is not None:
            if progress_callback:
                progress_callback('auxiliary', 'Loaded road and urban masks from local disk …', 38)
            return cached

        # ── 2. S3 cache ───────────────────────────────────────────────────────
        cached_s3 = get_cached_auxiliary(village_cache_key, shape)
        if cached_s3 is not None:
            if progress_callback:
                progress_callback('auxiliary', 'Loaded road and urban masks from S3 …', 38)
            _save_disk_aux_cache(village_cache_key, cached_s3[0], cached_s3[1])
            return cached_s3

    if progress_callback:
        progress_callback('auxiliary', 'Fetching OSM road & urban vectors (Overpass API) …', 37)

    # ── 3 + 4. Road mask ─────────────────────────────────────────────────────
    road_mask: np.ndarray
    if ENABLE_ROAD_MASK:
        # Tier 1: local GPKG (fast, zero network)
        road_mask = _fetch_roads_from_gpkg(minx, miny, maxx, maxy, utm_crs, shape, transform)
        if road_mask is None:
            # Tier 2: Overpass API (OSM vector, class-aware buffers)
            road_mask = _fetch_roads_from_overpass(minx, miny, maxx, maxy, utm_crs, shape, transform)
    else:
        road_mask = np.zeros(shape, dtype=bool)

    # ── 3 + 4. Urban mask ────────────────────────────────────────────────────
    urban_mask: np.ndarray
    if ENABLE_URBAN_MASK:
        urban_mask = _fetch_urban_from_gpkg(minx, miny, maxx, maxy, utm_crs, shape, transform)
        if urban_mask is None:
            urban_mask = _fetch_urban_from_overpass(minx, miny, maxx, maxy, utm_crs, shape, transform)
    else:
        urban_mask = np.zeros(shape, dtype=bool)

    # ── 5. Persist ────────────────────────────────────────────────────────────
    if village_cache_key:
        _save_disk_aux_cache(village_cache_key, road_mask, urban_mask)
        store_cached_auxiliary_async(village_cache_key, road_mask, urban_mask)

    logger.info(
        'Auxiliary masks ready — road: %.2f%% | urban: %.2f%% of bbox.',
        road_mask.mean() * 100, urban_mask.mean() * 100,
    )
    return road_mask, urban_mask
