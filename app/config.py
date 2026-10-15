import os
from pathlib import Path

# ── Load local .env if present ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_env_path = Path(BASE_DIR) / '.env'
if _env_path.is_file():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except Exception:
        with open(_env_path, 'r', encoding='utf-8') as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith('#') and '=' in _line:
                    _k, _v = _line.split('=', 1)
                    os.environ.setdefault(_k.strip(), _v.strip())

# ── Inference hyperparameters (v5 notebook: sliding window, TTA) ──────────────
PATCH_PX         = 256
RES_M            = 1.0    # imagery resolution in metres per pixel
BATCH_SIZE       = int(os.environ.get('BATCH_SIZE', 16))
OVERLAP          = int(os.environ.get('OVERLAP', 64))
USE_FP16         = os.environ.get('USE_FP16', 'true').lower() in ('true', '1', 'yes')
TTA              = os.environ.get('TTA', 'true').lower() in ('true', '1', 'yes')

# ── Extraction & Seeding (v5 'neck' mode & sharp boundary maps) ───────────────
SEED_MODE        = os.environ.get('SEED_MODE', 'neck')   # 'neck' (v5 default), 'cores' (v2/v3), or 'neck_dist'
EXT_T            = float(os.environ.get('EXT_T', 0.5))   # extent probability above this = field land
EXTENT_THRESH    = float(os.environ.get('EXTENT_THRESH', EXT_T)) # backward compatibility alias
EDGE_SIGMA       = float(os.environ.get('EDGE_SIGMA', 0.7)) # px blur on boundary map
_bnd_norm_env    = os.environ.get('BND_NORM', '99').strip()
BND_NORM         = None if _bnd_norm_env.lower() in ('none', '0', '') else float(_bnd_norm_env)
BND_FLOOR        = float(os.environ.get('BND_FLOOR', 0.15))
CORE_T           = float(os.environ.get('CORE_T', 0.2))  # boundary (rescaled) at/above this is a wall between fields
GAP_CLOSE_R      = float(os.environ.get('GAP_CLOSE_R', 0)) # px; thicken walls before seeding so gaps separate fields
NECK_H           = float(os.environ.get('NECK_H', 2.0))  # px neck height for h_maxima in 'neck' mode
SEED_MIN_PX      = int(os.environ.get('SEED_MIN_PX', 30))
MERGE_T          = float(os.environ.get('MERGE_T', 0.3))  # merge touching fields whose shared border averages below this (0 = off)
MERGE_BND_T      = float(os.environ.get('MERGE_BND_T', MERGE_T)) # backward compatibility alias
DIST_WEIGHT      = float(os.environ.get('DIST_WEIGHT', 0.3)) # used in cores/legacy mode
SEED_DIST_MIN    = float(os.environ.get('SEED_DIST_MIN', 0.25))
SEED_BND_MAX     = float(os.environ.get('SEED_BND_MAX', 0.30))
LABEL_SMOOTH_R   = int(os.environ.get('LABEL_SMOOTH_R', 1))
SMOOTH_R         = int(os.environ.get('SMOOTH_R', LABEL_SMOOTH_R))
RESCUE_ORPHANS   = os.environ.get('RESCUE_ORPHANS', 'true').lower() in ('true', '1', 'yes')
MIN_REGION_PX    = int(os.environ.get('MIN_REGION_PX', 200))
OVERSEG_H        = float(os.environ.get('OVERSEG_H', 0.18))
MAX_FIELD_PX     = int(os.environ.get('MAX_FIELD_PX', 220_000))

# ── Polygon post-processing & simplification ──────────────────────────────────
VECTOR_MODE      = os.environ.get('VECTOR_MODE', 'topo')   # 'topo' (v3/v5), 'coverage' (v2), or 'legacy'
COVERAGE_TOL_M   = float(os.environ.get('COVERAGE_TOL_M', 1.5))
DAGGER_WIDTH_M   = float(os.environ.get('DAGGER_WIDTH_M', 2.0))
SIMPLIFY_FIRST_M = float(os.environ.get('SIMPLIFY_FIRST_M', 2.0))
SMOOTH_METHOD    = os.environ.get('SMOOTH_METHOD', 'orthogonal')
ANGLE_TOL_DEG    = float(os.environ.get('ANGLE_TOL_DEG', 15))
SIMPLIFY_AFTER_M = float(os.environ.get('SIMPLIFY_AFTER_M', 1.0))

# ── v3/v5: topo vectoriser ────────────────────────────────────────────────────
# Traces shared pixel-edges as arcs, simplifies each arc once → zero gaps/overlaps.
TOPO_TOL_M       = float(os.environ.get('TOPO_TOL_M', 2.5))    # Douglas-Peucker tolerance (m)
MAX_VERTICES     = int(os.environ.get('MAX_VERTICES', 16))      # per-field vertex budget (0 = none)
LABEL_GAUSS_SIGMA = float(os.environ.get('LABEL_GAUSS_SIGMA', 1.0))  # Gaussian blur on label raster (0=off)
GAUSS_SIGMA      = float(os.environ.get('GAUSS_SIGMA', LABEL_GAUSS_SIGMA))

# ── v3/v5: extent mask hole filling ───────────────────────────────────────────
# Fill holes in the field-extent mask only up to this size; larger holes
# (settlements, ponds) are left open so they don't become part of a field.
_hole_env = os.environ.get('EXTENT_HOLE_MAX_PX', os.environ.get('EXTENT_HOLE_MAX_M2', '1500'))
EXTENT_HOLE_MAX_PX = None if _hole_env.lower() in ('none', '') else float(_hole_env)
EXTENT_HOLE_MAX_M2 = EXTENT_HOLE_MAX_PX

# ── v3/v5: tree / bush gap absorption ─────────────────────────────────────────
TREE_FILL_MAX_PX  = float(os.environ.get('TREE_FILL_MAX_PX', os.environ.get('TREE_FILL_MAX_M2', 1500)))
TREE_FILL_MAX_M2  = TREE_FILL_MAX_PX
TREE_MIN_ENCLOSED = float(os.environ.get('TREE_MIN_ENCLOSED', 0.6))   # fraction of perimeter touching fields
BUND_OPEN_PX      = int(os.environ.get('BUND_OPEN_PX', int(round(float(os.environ.get('BUND_OPEN_M', 2.0))))))
BUND_OPEN_M       = float(BUND_OPEN_PX)

# ── v3/v5: missed-field rescue ────────────────────────────────────────────────
# Field-shaped non-crop patches (dense sugarcane / banana / orchards that the
# model scores as trees) become fields tagged source='rescued'.
RESCUE_FIELDS        = os.environ.get('RESCUE_FIELDS', 'true').lower() in ('true', '1', 'yes')
RESCUE_MIN_PX        = float(os.environ.get('RESCUE_MIN_PX', os.environ.get('RESCUE_MIN_M2', 2000)))
RESCUE_MIN_M2        = RESCUE_MIN_PX
RESCUE_MIN_SOLIDITY  = float(os.environ.get('RESCUE_MIN_SOLIDITY', 0.88))  # area / convex-hull area
RESCUE_MIN_WIDTH_PX  = float(os.environ.get('RESCUE_MIN_WIDTH_PX', os.environ.get('RESCUE_MIN_WIDTH_M', 20)))
RESCUE_MIN_WIDTH_M   = RESCUE_MIN_WIDTH_PX
RESCUE_MIN_EXG       = float(os.environ.get('RESCUE_MIN_EXG', 0.02))       # min excess-green (excludes rooftops/bare soil)

# ── Oversized splitting & area rules ──────────────────────────────────────────
SPLIT_OVERSIZED  = os.environ.get('SPLIT_OVERSIZED', 'false').lower() in ('true', '1', 'yes')
OVERSIZE_PCTL    = float(os.environ.get('OVERSIZE_PCTL', 95))
MIN_FIELD_M2     = float(os.environ.get('MIN_FIELD_M2', 500))  # 0.05 ha minimum floor
MIN_AREA_FRAC    = float(os.environ.get('MIN_AREA_FRAC', 0.10))
SMALL_FIELD_HA   = float(os.environ.get('SMALL_FIELD_HA', 0.10))
MAX_FIELD_HA     = float(os.environ.get('MAX_FIELD_HA', 22.0))
CLIP_TO_VILLAGE  = os.environ.get('CLIP_TO_VILLAGE', 'true').lower() in ('true', '1', 'yes')

# ── Road & Urban auxiliary masking ────────────────────────────────────────────
ENABLE_ROAD_MASK     = os.environ.get('ENABLE_ROAD_MASK', 'true').lower() in ('true', '1', 'yes')
ENABLE_URBAN_MASK    = os.environ.get('ENABLE_URBAN_MASK', 'true').lower() in ('true', '1', 'yes')
ROAD_BARRIER_ELEV    = float(os.environ.get('ROAD_BARRIER_ELEV', 1.5))
OSM_GPKG_PATH        = os.environ.get('OSM_GPKG_PATH', '').strip()
OSM_URBAN_GPKG_PATH  = os.environ.get('OSM_URBAN_GPKG_PATH', '').strip()

# ── Input safety limits ───────────────────────────────────────────────────────
MAX_BBOX_KM2    = float(os.environ.get('MAX_BBOX_KM2', 100))  # km²

# ── Imagery disk cache ────────────────────────────────────────────────────────
CACHE_MAX_GB    = float(os.environ.get('CACHE_MAX_GB', 5))    # gigabytes

# ── ESRI Imagery Fetcher performance tuning ───────────────────────────────────
ESRI_MAX_WORKERS     = int(os.environ.get('ESRI_MAX_WORKERS', 6))
ESRI_IMAGE_FORMAT    = os.environ.get('ESRI_IMAGE_FORMAT', 'png').lower()  # 'png' or 'jpg'
ESRI_JPEG_QUALITY    = int(os.environ.get('ESRI_JPEG_QUALITY', 90))
ESRI_TIMEOUT_SEC     = float(os.environ.get('ESRI_TIMEOUT_SEC', 45))
ESRI_MAX_CHUNK_PX    = int(os.environ.get('ESRI_MAX_CHUNK_PX', 1500))

# ── S3 Cache configuration (imagery and boundary predictions) ─────────────────
S3_BUCKET            = os.environ.get('S3_BUCKET', '').strip()
S3_PREFIX            = os.environ.get('S3_PREFIX', 'boundary-results').strip('/')
S3_REGION            = os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION') or 'eu-north-1'
RESULT_CACHE_VERSION = os.environ.get('RESULT_CACHE_VERSION', 'v2')
AWS_ACCESS_KEY_ID    = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY= os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_SESSION_TOKEN    = os.environ.get('AWS_SESSION_TOKEN')
S3_STORE_IMAGERY     = os.environ.get('S3_STORE_IMAGERY', 'true').lower() in ('true', '1', 'yes')

# ── Model checkpoints ─────────────────────────────────────────────────────────
# Defaults to ft21_best.pt, falls back to ft_best.pt if missing.
# Also allows separate checkpoints for extent vs boundary models.
_default_ckpt = os.path.join(BASE_DIR, "checkpoints", "ft21_best.pt")
if not os.path.exists(_default_ckpt):
    _default_ckpt = os.path.join(BASE_DIR, "checkpoints", "ft_best.pt")

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", _default_ckpt)
EXTENT_CHECKPOINT_PATH = os.environ.get("EXTENT_CHECKPOINT_PATH", CHECKPOINT_PATH)
BOUNDARY_CHECKPOINT_PATH = os.environ.get("BOUNDARY_CHECKPOINT_PATH", CHECKPOINT_PATH)


