FROM python:3.11-slim

# CPU-only PyTorch by default (keeps the image ~1 GB instead of ~6 GB).
# For a GPU image: --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1000

WORKDIR /app

# rasterio, pyproj, shapely and geopandas ship manylinux wheels with GDAL/GEOS/PROJ
# bundled, so no system GDAL or compilers are needed.
RUN pip install --index-url ${TORCH_INDEX_URL} torch

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser app/ /app/app/

# Model weights are mounted at runtime (see docker-compose.yml) rather than baked in.
ENV CHECKPOINT_PATH=/app/checkpoints/ft_best.pt

USER appuser

EXPOSE 8000

# Docker marks the container unhealthy if /health stops responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Single worker: each worker loads its own copy of the model into memory.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
