import os

import numpy as np
import xarray as xr
import torch
from app.config import  MEAN, STD
from rasterio.transform import from_bounds

def load_nc_as_4ch(nc_path):
    """
    Load .nc file → (H, W, 4) float32 array.
    Takes the median across time for B2, B3, B4, B8.
    """
    ds = xr.open_dataset(nc_path)
    bands = []

    for var in ['B2', 'B3', 'B4', 'B8']:              # (time, H, W) → median → (H, W)
        arr = ds[var].values
        median = np.median(arr, axis=0)
        bands.append(median.astype(np.float32))
    ds.close()

    return np.stack(bands, axis=-1)                    # (H, W, 4)

def replace_nans(image, fill_value=0.0):
    return np.nan_to_num(image, nan=fill_value, posinf=fill_value, neginf=fill_value)

def preprocess(nc_path) -> torch.Tensor:
    img = replace_nans(load_nc_as_4ch(nc_path))        # (256, 256, 4)
    img = (img - MEAN) / (STD + 1e-8)                         # normalize
    img = torch.from_numpy(img).permute(2, 0, 1)         # (4, 256, 256)
    return img.unsqueeze(0)                              # (1, 4, 256, 256)

def postprocess(logits: torch.Tensor) -> np.ndarray:
    """
    Convert raw logits (1, 3, H, W) → per-pixel class map (H, W) uint8.
    Classes: 0=Background  1=Interior  2=Boundary
    """
    probs     = torch.softmax(logits, dim=1)          # (1, 3, H, W)
    class_map = torch.argmax(probs, dim=1)            # (1, H, W)
    return class_map.squeeze().cpu().numpy().astype(np.uint8)  # (H, W)



# Step 5b — Convert 3-class mask to GeoJSON (field polygons)
import json
from rasterio.features import shapes
from shapely.geometry import shape, mapping, Polygon
import pyproj
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def _empty_geojson(output_dir):
    """Write and return path to an empty FeatureCollection."""
    out_path = os.path.join(output_dir, 'predicted_fields.geojson')
    with open(out_path, 'w') as f:
        json.dump({"type": "FeatureCollection", "features": []}, f)
    return out_path

def mask_to_geojson(pred_mask, reference_nc_path, output_dir,
                    min_area_px=50, simplify_tolerance=1.5,
                    watershed_min_distance=8):
    """
    Convert 3-class predicted mask → GeoJSON of field polygons.
    Internally extracts Interior pixels (class=1) as the field binary mask.

    Steps:
      1. Build binary mask from Interior class (pred_mask == 1)
      2. Vectorize raster mask (rasterio.features.shapes)
      3. Filter out tiny noise polygons (< min_area_px pixels)
      4. Simplify polygon vertices (reduce file size)
      5. Reproject from EPSG:3035 → WGS84 (lat/lon) for GeoJSON standard
      6. Write FeatureCollection to .geojson

    Args:
        pred_mask        : (H, W) uint8 class map — 0=Background, 1=Interior, 2=Boundary
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

    # ── Step 1: Watershed instance separation ────────────────────────────────
    # Interior mask: only Class 1 pixels define field bodies
    interior_mask = (pred_mask == 1).astype(np.uint8)

    # Distance transform: each pixel's value = distance to nearest background
    distance = ndi.distance_transform_edt(interior_mask)

    # Find the deepest center peak of every distinct field
    peaks = peak_local_max(distance, min_distance=watershed_min_distance,
                           labels=interior_mask)

    if len(peaks) == 0:
        print('No field peaks detected — returning empty GeoJSON.')
        return _empty_geojson(output_dir)

    peak_mask = np.zeros(distance.shape, dtype=bool)
    peak_mask[tuple(peaks.T)] = True
    markers, _ = ndi.label(peak_mask)

    # Flood from peaks outward; build 1-px dams where floods meet
    separated_labels = watershed(-distance, markers, mask=interior_mask)

    # ── Step 2: Vectorize the labeled map ────────────────────────────────────
    pixel_area_m2 = abs(transform.a * transform.e)
    min_area_m2   = min_area_px * pixel_area_m2

    field_shapes = []
    for geom, val in shapes(separated_labels.astype(np.int32), transform=transform):
        if val == 0:
            continue          # skip background
        poly = shape(geom)
        if poly.area >= min_area_m2:
            field_shapes.append(poly)

    print(f'\nWatershed: {len(field_shapes)} fields after noise filter (>{min_area_px}px)')

    # ── Step 3: Douglas-Peucker smoothing ────────────────────────────────────
    field_shapes = [s.simplify(simplify_tolerance, preserve_topology=True)
                    for s in field_shapes]

    # Reproject EPSG:3035 → WGS84 for standard GeoJSON
    transformer = pyproj.Transformer.from_crs(
        'EPSG:3035', 'EPSG:4326', always_xy=True
    )

    def reproject_ring(ring):
        """Reproject a single LinearRing (exterior or hole) to WGS84."""
        coords = np.array(ring.coords)
        lons, lats = transformer.transform(coords[:, 0], coords[:, 1])
        return list(zip(lons, lats))

    def reproject_shape(geom):
        """Reproject a Polygon including any interior holes."""
        exterior = reproject_ring(geom.exterior)
        holes    = [reproject_ring(h) for h in geom.interiors]
        return Polygon(exterior, holes)

    # Filter to valid Polygons only and zip with matching field_shapes for area lookup
    valid_pairs = [
        (reproject_shape(s), s)
        for s in field_shapes
        if s.geom_type == 'Polygon' and not s.is_empty
    ]
    reprojected   = [pair[0] for pair in valid_pairs]
    shapes_for_area = [pair[1] for pair in valid_pairs]

    # Build GeoJSON FeatureCollection
    features = []
    for i, (geom, src) in enumerate(zip(reprojected, shapes_for_area)):
        features.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {
                "field_id": i + 1,
                "area_m2": round(src.area, 2),    # measured in EPSG:3035 metres
                "area_ha": round(src.area / 10000, 4),
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