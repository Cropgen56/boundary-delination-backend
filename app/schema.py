from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for /api/v1/predict"""

    geojson: dict[str, Any] | list[dict[str, Any]] = Field(
        ...,
        description=(
            "A village GeoJSON Feature, FeatureCollection, or array of Features "
            "returned by the boundary service. Coordinates must be WGS84."
        ),
    )
    taluka: str | None = Field(
        None,
        description="Optionally process only features whose properties.taluka matches this name.",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str   # 'cuda' or 'cpu'


class ErrorResponse(BaseModel):
    detail: str