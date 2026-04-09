# Use an official Python runtime as a parent image
FROM python:3.10-slim

# setting work directory
WORKDIR /app

# dependency for rasterio and shapely
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*


# copy the requirements file into the container at /app
COPY requirements.txt /app/
# install dependencies
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# copy the current directory contents into the container at /app
COPY . .

# expose port 8000
EXPOSE 8000

# run the command to start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]