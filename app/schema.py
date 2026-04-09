from pydantic import BaseModel, Field
from typing import Optional

# --- Request ---
# No request schema needed for file upload.
# FastAPI handles UploadFile directly in the route — Pydantic can't validate binary files.

# --- Response (for JSON endpoint if you add one) ---
class PredictionResponse(BaseModel):
    filename: str
    num_field_pixels: int        # count of pixels predicted as field
    total_pixels: int            # always 256*256 = 65536
    field_coverage_pct: float    # num_field_pixels / total_pixels * 100

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str                  # "cuda" or "cpu"

# --- Error ---
class ErrorResponse(BaseModel):
    detail: str