from typing import Any

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    """Wrapped request, or raw GeoJSON normalized into this model."""

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
    force_rerun: bool = Field(
        default=False,
        description="If True, bypasses cached S3 field boundaries and re-runs model inference and postprocessing.",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_raw_geojson(cls, value: Any) -> Any:
        if isinstance(value, list):
            return {"geojson": value}
        if isinstance(value, dict) and value.get("type") in ("Feature", "FeatureCollection"):
            return {"geojson": value}
        return value


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str   # 'cuda' or 'cpu'


class ErrorResponse(BaseModel):
    detail: str