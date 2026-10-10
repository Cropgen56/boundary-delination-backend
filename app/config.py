import os

# ── Inference hyperparameters (tuned from held-out evaluation) ────────────────
PATCH_PX        = 256
RES_M           = 1.0    # imagery resolution in metres per pixel

OVERSEG_H       = 0.18   # h-minima depth for watershed seeding (lower = more regions)
EXTENT_THRESH   = 0.5    # binary threshold on extent probability map
SIMPLIFY_TOL_M  = 2.0    # Douglas-Peucker simplification tolerance in metres
DAGGER_WIDTH_M  = 2.0    # max spike width to fill in dagger-removal step
MIN_AREA_FRAC   = 0.10   # drop polygons below this fraction of median field area

# ── Input safety limits ───────────────────────────────────────────────────────
# Villages whose UTM bounding box exceeds this area are rejected with HTTP 422
# before any imagery is fetched, preventing OOM on extremely large inputs.
MAX_BBOX_KM2    = float(os.environ.get('MAX_BBOX_KM2', 100))  # km²

# ── Imagery disk cache ────────────────────────────────────────────────────────
# After each save the cache dir is scanned; oldest files are deleted once the
# total on-disk size exceeds this limit. Set to 0 to disable eviction.
CACHE_MAX_GB    = float(os.environ.get('CACHE_MAX_GB', 5))    # gigabytes

# ── Model checkpoint ──────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Place ft_best.pt in the checkpoints directory,
# or point CHECKPOINT_PATH at it (e.g. a mounted volume in Docker)
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", os.path.join(BASE_DIR, "checkpoints", "ft_best.pt")
)
