import os

import numpy as np
import xarray as xr
import torch
from app.config import BAND_VARS, N_TIMESTEPS, REFLECTANCE_SCALE, MEAN, STD
from rasterio.transform import from_bounds

def load_nc_as_array(nc_path):
    """
    Load .nc file → (H, W, 30) float32 array.
    Channel order per timestep: [B4, B3, B2, B8, NDVI]
    Reflectance bands scaled to 0-1. NDVI kept as-is (already 0-1).
    """
    ds = xr.open_dataset(nc_path)
    time_slices = []

    for t in range(N_TIMESTEPS):
        bands = []
        for var in BAND_VARS:
            band = ds[var].isel(time=t).values.astype(np.float32)   # (H, W)
            if var != 'NDVI':
                band = band / REFLECTANCE_SCALE                     # scale to 0–1
            bands.append(band)
        time_slices.append(np.stack(bands, axis=-1))                # (H, W, 5)

    image = np.concatenate(time_slices, axis=-1)                    # (H, W, 30)
    ds.close()
    return image

def replace_nans(image, fill_value=0.0):
    return np.nan_to_num(image, nan=fill_value, posinf=fill_value, neginf=fill_value)

def preprocess(nc_path) -> torch.Tensor:
    img = replace_nans(load_nc_as_array(nc_path))        # (256, 256, 30)
    img = (img - MEAN) / STD                             # normalize
    img = torch.from_numpy(img).permute(2, 0, 1)         # (30, 256, 256)
    return img.unsqueeze(0)                              # (1, 30, 256, 256)

def postprocess(logits: torch.Tensor, threshold=0.5) -> np.ndarray:
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (256, 256)
    return (probs > threshold).astype(np.uint8)



# Step 5b — Convert binary mask to GeoJSON (field polygons)
import json
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import pyproj
from functools import partial

def mask_to_geojson(pred_mask, reference_nc_path, output_dir,
                    min_area_px=50, simplify_tolerance=1.5):
    """
    Convert binary predicted mask → GeoJSON of field polygons.

    Steps:
      1. Vectorize raster mask (rasterio.features.shapes)
      2. Filter out tiny noise polygons (< min_area_px pixels)
      3. Simplify polygon vertices (reduce file size)
      4. Reproject from EPSG:3035 → WGS84 (lat/lon) for GeoJSON standard
      5. Write FeatureCollection to .geojson

    Args:
        pred_mask        : (H, W) uint8 binary mask (0=background, 1=field)
        reference_nc_path: original .nc path to read spatial extent from
        output_dir       : where to save the .geojson
        min_area_px      : drop polygons smaller than this (noise removal)
        simplify_tolerance: vertex simplification in CRS units (metres here)

    Returns:
        path to saved .geojson file
    """
    # Read spatial extent from the .nc file (EPSG:3035 coords)
    ds = xr.open_dataset(reference_nc_path)
    x  = ds['x'].values   # (W,)
    y  = ds['y'].values   # (H,)
    ds.close()

    transform = from_bounds(
        x.min(), y.min(), x.max(), y.max(),
        pred_mask.shape[1], pred_mask.shape[0]
    )

    # Vectorize: rasterio.features.shapes yields (geometry, value) pairs
    # Only keep shapes where value == 1 (field pixels)
    field_shapes = [
        shape(geom)
        for geom, val in shapes(pred_mask, transform=transform)
        if val == 1
    ]

    print(f'\nVectorization: {len(field_shapes)} raw polygons extracted')

    # Filter noise (tiny fragments from imperfect predictions)
    # 1 pixel = 10m × 10m = 100 m² at Sentinel-2 resolution
    pixel_area_m2 = abs(transform.a * transform.e)   # pixel width × height
    min_area_m2   = min_area_px * pixel_area_m2
    field_shapes  = [s for s in field_shapes if s.area >= min_area_m2]
    print(f'After noise filter (>{min_area_px}px): {len(field_shapes)} polygons')

    # Simplify to reduce vertex count (tolerance in metres, CRS:3035)
    field_shapes = [s.simplify(simplify_tolerance, preserve_topology=True)
                    for s in field_shapes]

    # Reproject EPSG:3035 → WGS84 for standard GeoJSON
    transformer = pyproj.Transformer.from_crs(
        'EPSG:3035', 'EPSG:4326', always_xy=True
    )

    def reproject_shape(geom):
        coords = np.array(geom.exterior.coords)
        lons, lats = transformer.transform(coords[:, 0], coords[:, 1])
        from shapely.geometry import Polygon
        return Polygon(zip(lons, lats))

    reprojected = [reproject_shape(s) for s in field_shapes
                   if s.geom_type == 'Polygon' and not s.is_empty]

    # Build GeoJSON FeatureCollection
    features = []
    for i, geom in enumerate(reprojected):
        features.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {
                "field_id": i + 1,
                "area_m2": round(field_shapes[i].area, 2),   # in EPSG:3035 metres
                "area_ha": round(field_shapes[i].area / 10000, 4),
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": features
    }

    out_path = os.path.join(output_dir, 'predicted_fields.geojson')
    with open(out_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    total_area_ha = sum(f['properties']['area_ha'] for f in features)
    print(f'GeoJSON saved        → {out_path}')
    print(f'Total fields         : {len(features)}')
    print(f'Total field area     : {total_area_ha:.2f} ha')

    return out_path