
FROM python:3.11-slim

# setting work directory
WORKDIR /app

# dependency for rasterio and shapely
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libnetcdf-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*



COPY requirements.txt /app/

RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt


COPY . .


EXPOSE 8000

# healthcheck — Docker will mark container unhealthy if /health stops responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]