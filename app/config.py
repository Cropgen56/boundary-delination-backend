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

# ── Inference hyperparameters (tuned from village_field_polygons_v2_) ─────────
PATCH_PX         = 256
RES_M            = 1.0    # imagery resolution in metres per pixel
BATCH_SIZE       = int(os.environ.get('BATCH_SIZE', 16))
OVERLAP          = int(os.environ.get('OVERLAP', 64))
USE_FP16         = os.environ.get('USE_FP16', 'true').lower() in ('true', '1', 'yes')

# ── Extraction & Seeding (v2 'cores' mode) ────────────────────────────────────
SEED_MODE        = os.environ.get('SEED_MODE', 'cores')   # 'cores' or 'hminima'
SEED_DIST_MIN    = float(os.environ.get('SEED_DIST_MIN', 0.25))
SEED_BND_MAX     = float(os.environ.get('SEED_BND_MAX', 0.30))
SEED_MIN_PX      = int(os.environ.get('SEED_MIN_PX', 30))
EDGE_SIGMA       = float(os.environ.get('EDGE_SIGMA', 0.7))
DIST_WEIGHT      = float(os.environ.get('DIST_WEIGHT', 0.3))
MERGE_BND_T      = float(os.environ.get('MERGE_BND_T', 0.15))
LABEL_SMOOTH_R   = int(os.environ.get('LABEL_SMOOTH_R', 2))
RESCUE_ORPHANS   = os.environ.get('RESCUE_ORPHANS', 'true').lower() in ('true', '1', 'yes')
MIN_REGION_PX    = int(os.environ.get('MIN_REGION_PX', 200))
OVERSEG_H        = float(os.environ.get('OVERSEG_H', 0.18))
EXTENT_THRESH    = float(os.environ.get('EXTENT_THRESH', 0.5))

# ── Polygon post-processing & simplification ──────────────────────────────────
VECTOR_MODE      = os.environ.get('VECTOR_MODE', 'coverage')  # 'coverage' or 'legacy'
COVERAGE_TOL_M   = float(os.environ.get('COVERAGE_TOL_M', 1.5))
DAGGER_WIDTH_M   = float(os.environ.get('DAGGER_WIDTH_M', 2.0))
SIMPLIFY_FIRST_M = float(os.environ.get('SIMPLIFY_FIRST_M', 2.0))
SMOOTH_METHOD    = os.environ.get('SMOOTH_METHOD', 'orthogonal')
ANGLE_TOL_DEG    = float(os.environ.get('ANGLE_TOL_DEG', 15))
SIMPLIFY_AFTER_M = float(os.environ.get('SIMPLIFY_AFTER_M', 1.0))

# ── Oversized splitting & area rules ──────────────────────────────────────────
SPLIT_OVERSIZED  = os.environ.get('SPLIT_OVERSIZED', 'true').lower() in ('true', '1', 'yes')
OVERSIZE_PCTL    = float(os.environ.get('OVERSIZE_PCTL', 95))
MIN_FIELD_M2     = float(os.environ.get('MIN_FIELD_M2', 500))  # 0.05 ha minimum floor
MIN_AREA_FRAC    = float(os.environ.get('MIN_AREA_FRAC', 0.10))
SMALL_FIELD_HA   = float(os.environ.get('SMALL_FIELD_HA', 0.10))
MAX_FIELD_HA     = float(os.environ.get('MAX_FIELD_HA', 22.0))
CLIP_TO_VILLAGE  = os.environ.get('CLIP_TO_VILLAGE', 'true').lower() in ('true', '1', 'yes')

# ── Input safety limits ───────────────────────────────────────────────────────
# Villages whose UTM bounding box exceeds this area are rejected with HTTP 422
# before any imagery is fetched, preventing OOM on extremely large inputs.
MAX_BBOX_KM2    = float(os.environ.get('MAX_BBOX_KM2', 100))  # km²

# ── Imagery disk cache ────────────────────────────────────────────────────────
# After each save the cache dir is scanned; oldest files are deleted once the
# total on-disk size exceeds this limit. Set to 0 to disable eviction.
CACHE_MAX_GB    = float(os.environ.get('CACHE_MAX_GB', 5))    # gigabytes

# ── ESRI Imagery Fetcher performance tuning ───────────────────────────────────
# Fetch tiles in parallel rather than serially to drastically decrease latency
ESRI_MAX_WORKERS     = int(os.environ.get('ESRI_MAX_WORKERS', 6))
ESRI_IMAGE_FORMAT    = os.environ.get('ESRI_IMAGE_FORMAT', 'png').lower()  # 'png' or 'jpg'
ESRI_JPEG_QUALITY    = int(os.environ.get('ESRI_JPEG_QUALITY', 90))
ESRI_TIMEOUT_SEC     = float(os.environ.get('ESRI_TIMEOUT_SEC', 45))
ESRI_MAX_CHUNK_PX    = int(os.environ.get('ESRI_MAX_CHUNK_PX', 1500))

# ── S3 Cache configuration (imagery and boundary predictions) ─────────────────
S3_BUCKET            = os.environ.get('S3_BUCKET', '').strip()
S3_PREFIX            = os.environ.get('S3_PREFIX', 'boundary-results').strip('/')
S3_REGION            = os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION') or 'eu-north-1'
RESULT_CACHE_VERSION = os.environ.get('RESULT_CACHE_VERSION', 'v1')
AWS_ACCESS_KEY_ID    = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY= os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_SESSION_TOKEN    = os.environ.get('AWS_SESSION_TOKEN')
S3_STORE_IMAGERY     = os.environ.get('S3_STORE_IMAGERY', 'true').lower() in ('true', '1', 'yes')

# ── Model checkpoint ──────────────────────────────────────────────────────────
# Place ft_best.pt in the checkpoints directory,
# or point CHECKPOINT_PATH at it (e.g. a mounted volume in Docker)
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", os.path.join(BASE_DIR, "checkpoints", "ft_best.pt")
)

