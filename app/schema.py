from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for /api/v1/predict"""

    center_lat: float = Field(
        ...,
        description="Latitude of the AOI centre point (WGS84, decimal degrees).",
        ge=-90.0,
        le=90.0,
        examples=[17.702059],
    )
    center_lon: float = Field(
        ...,
        description="Longitude of the AOI centre point (WGS84, decimal degrees).",
        ge=-180.0,
        le=180.0,
        examples=[76.006878],
    )
    box_km: float = Field(
        3.0,
        description=(
            "Side length of the square AOI in kilometres. "
            "Select between 2 km × 2 km and 10 km × 10 km."
        ),
        ge=2.0,
        le=10.0,
        examples=[2.0, 3.0, 5.0, 7.5, 10.0],
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str   # 'cuda' or 'cpu'


class ErrorResponse(BaseModel):
    detail: str