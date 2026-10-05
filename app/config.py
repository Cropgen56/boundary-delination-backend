import os

# ── Inference hyperparameters (tuned from held-out evaluation) ────────────────
PATCH_PX        = 256
RES_M           = 1.0    # imagery resolution in metres per pixel

OVERSEG_H       = 0.18   # h-minima depth for watershed seeding (lower = more regions)
EXTENT_THRESH   = 0.5    # binary threshold on extent probability map
SIMPLIFY_TOL_M  = 2.0    # Douglas-Peucker simplification tolerance in metres
DAGGER_WIDTH_M  = 2.0    # max spike width to fill in dagger-removal step
MIN_AREA_FRAC   = 0.10   # drop polygons below this fraction of median field area

# ── Model checkpoint ──────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Place ft_best.pt in the checkpoints/ folder (downloaded from Drive)
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "ft_best.pt")
