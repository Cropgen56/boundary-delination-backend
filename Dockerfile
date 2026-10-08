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
# bundled, so no system GDAL or compilers are needed. rasterio still needs libexpat.
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# torch and torchvision must come from the same index, or torchvision's C++ ops
# fail to load ("operator torchvision::nms does not exist").
RUN pip install --index-url ${TORCH_INDEX_URL} torch torchvision

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser app/ /app/app/

# Model weights are baked into the image so it runs standalone (e.g. on Cloud Run).
# ft_best.pt must be in the project root when building (it is not in git).
COPY --chown=appuser:appuser ft_best.pt /app/checkpoints/ft_best.pt
ENV CHECKPOINT_PATH=/app/checkpoints/ft_best.pt

USER appuser

EXPOSE 8080

# Docker marks the container unhealthy if /health stops responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=5)" || exit 1

# Single worker: each worker loads its own copy of the model into memory.
# Cloud Run sets PORT (8080 by default).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
