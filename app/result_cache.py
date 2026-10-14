"""S3 cache for completed village field-boundary predictions and satellite imagery."""

import hashlib
import io
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.transform import from_bounds
from shapely.geometry import shape

from app.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN,
    BASE_DIR,
    RESULT_CACHE_VERSION,
    S3_BUCKET,
    S3_PREFIX,
    S3_REGION,
    S3_STORE_IMAGERY,
)

logger = logging.getLogger(__name__)
_s3_client = None
_LOCAL_CACHE_DIR = Path(BASE_DIR) / 'data'


def _client():
    """Return a singleton boto3 S3 client initialized with configured credentials."""
    global _s3_client
    if _s3_client is None:
        import boto3

        client_kwargs = {'region_name': S3_REGION}
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            client_kwargs['aws_access_key_id'] = AWS_ACCESS_KEY_ID
            client_kwargs['aws_secret_access_key'] = AWS_SECRET_ACCESS_KEY
            if AWS_SESSION_TOKEN:
                client_kwargs['aws_session_token'] = AWS_SESSION_TOKEN

        _s3_client = boto3.client('s3', **client_kwargs)
    return _s3_client


def geometry_cache_digest(geom_or_feature) -> str:
    """
    Compute a stable SHA-256 digest (64-char hex) for a village geometry.
    Accepts:
    - GeoJSON Feature dict
    - GeoJSON geometry dict
    - Shapely geometry object
    """
    if geom_or_feature is None:
        raise ValueError('Geometry is required for cache key generation.')

    if hasattr(geom_or_feature, 'wkb'):
        return hashlib.sha256(geom_or_feature.wkb).hexdigest()

    if isinstance(geom_or_feature, dict):
        if geom_or_feature.get('type') == 'Feature':
            geom = geom_or_feature.get('geometry')
        else:
            geom = geom_or_feature

        try:
            s_geom = shape(geom)
            return hashlib.sha256(s_geom.wkb).hexdigest()
        except Exception:
            canonical = json.dumps(geom, sort_keys=True, separators=(',', ':'), allow_nan=False)
            return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    raise ValueError(f'Unsupported geometry type for cache key calculation: {type(geom_or_feature)}')


def _boundary_keys(digest: str) -> list[str]:
    """Return list of candidate S3 keys for boundary predictions (new and legacy)."""
    keys = []
    if S3_PREFIX:
        keys.append(f'{S3_PREFIX}/boundaries/{RESULT_CACHE_VERSION}/{digest}.geojson')
        keys.append(f'{S3_PREFIX}/{RESULT_CACHE_VERSION}/{digest}.json')
    else:
        keys.append(f'boundaries/{RESULT_CACHE_VERSION}/{digest}.geojson')
        keys.append(f'{RESULT_CACHE_VERSION}/{digest}.json')
    return keys


def _imagery_npz_key(digest: str) -> str:
    """Return S3 object key for cached imagery npz."""
    return f'{S3_PREFIX}/imagery/{digest}.npz' if S3_PREFIX else f'imagery/{digest}.npz'


def _imagery_jpg_key(digest: str) -> str:
    """Return S3 object key for cached imagery preview jpg."""
    return f'{S3_PREFIX}/imagery/{digest}.jpg' if S3_PREFIX else f'imagery/{digest}.jpg'


# ── Prediction GeoJSON cache ──────────────────────────────────────────────────

def get_cached_fields(village_feature: dict) -> dict | None:
    """
    Check S3 for pre-computed field boundaries for this village.
    Returns GeoJSON FeatureCollection dict, or None if unavailable/missing.
    """
    if not S3_BUCKET:
        return None

    try:
        digest = geometry_cache_digest(village_feature)
    except Exception as exc:
        logger.warning('Could not compute cache key for village: %s', exc)
        return None

    client = _client()
    for key in _boundary_keys(digest):
        try:
            response = client.get_object(Bucket=S3_BUCKET, Key=key)
            raw = response['Body'].read()
            result = json.loads(raw.decode('utf-8'))
            if result.get('type') != 'FeatureCollection' or not isinstance(result.get('features'), list):
                raise ValueError('Cached prediction is not a valid GeoJSON FeatureCollection.')

            props = village_feature.get('properties') or {}
            vname = props.get('name') or props.get('Name') or 'unnamed'
            logger.info(
                'Found cached field boundaries in S3 (bucket=%s, key=%s) for village %s.',
                S3_BUCKET, key, vname,
            )
            return result
        except Exception as exc:
            err_code = getattr(exc, 'response', {}).get('Error', {}).get('Code')
            if err_code in ('NoSuchKey', '404', 'NotFound'):
                continue  # try next key
            logger.warning('Error reading boundary cache from S3 (key=%s): %s', key, exc)
            return None

    return None


def store_cached_fields(village_feature: dict, result: dict) -> str | None:
    """
    Persist completed field boundaries to S3.
    Failures do not affect the API response.
    """
    if not S3_BUCKET:
        return None

    try:
        digest = geometry_cache_digest(village_feature)
        target_key = _boundary_keys(digest)[0]
        body = json.dumps(result, ensure_ascii=False, allow_nan=False).encode('utf-8')

        props = village_feature.get('properties') or {}
        vname = props.get('name') or props.get('Name') or 'unnamed'

        _client().put_object(
            Bucket=S3_BUCKET,
            Key=target_key,
            Body=body,
            ContentType='application/geo+json',
            Metadata={
                'village-name': str(vname),
                'feature-count': str(len(result.get('features', []))),
                'cache-version': str(RESULT_CACHE_VERSION),
            },
        )
        logger.info(
            'Stored field-boundary result in S3 (bucket=%s, key=%s) for village %s.',
            S3_BUCKET, target_key, vname,
        )
        return target_key
    except Exception as exc:
        logger.warning('Unable to store prediction cache in S3: %s', exc)
        return None


# ── Satellite Imagery cache ───────────────────────────────────────────────────

def get_cached_imagery(digest: str, minx: float, miny: float, maxx: float, maxy: float) -> tuple[np.ndarray, object] | None:
    """
    Load cached ESRI imagery array from S3 if available.
    Also caches to local disk for fast reuse in the same environment.
    """
    if not S3_BUCKET:
        return None

    key = _imagery_npz_key(digest)
    try:
        response = _client().get_object(Bucket=S3_BUCKET, Key=key)
        raw = response['Body'].read()
        bio = io.BytesIO(raw)
        payload = np.load(bio)
        mosaic = payload['mosaic']
        if mosaic.ndim != 3 or mosaic.shape[0] != 3:
            raise ValueError('Cached imagery array has invalid shape.')

        transform = from_bounds(minx, miny, maxx, maxy, mosaic.shape[2], mosaic.shape[1])
        logger.info('Loaded cached ESRI imagery from S3 (bucket=%s, key=%s, shape=%s).', S3_BUCKET, key, mosaic.shape)

        # Write to local disk cache so subsequent requests avoid S3 download
        try:
            _LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            local_path = _LOCAL_CACHE_DIR / f'esri_{digest[:32]}.npz'
            if not local_path.exists():
                with open(local_path, 'wb') as lf:
                    lf.write(raw)
                logger.info('Saved S3 imagery to local cache: %s', local_path)
        except Exception as local_err:
            logger.debug('Could not write local cache from S3 imagery: %s', local_err)

        return mosaic, transform
    except Exception as exc:
        err_code = getattr(exc, 'response', {}).get('Error', {}).get('Code')
        if err_code not in ('NoSuchKey', '404', 'NotFound'):
            logger.warning('Could not read imagery cache from S3 (key=%s): %s', key, exc)
        return None


def store_cached_imagery(
    digest: str,
    mosaic: np.ndarray,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    utm_crs: str = '',
) -> None:
    """
    Persist downloaded satellite imagery to S3:
    1. Compressed .npz file (array + bounding box coordinates)
    2. Viewable RGB .jpg file for easy inspection/downloads
    """
    if not S3_BUCKET or not S3_STORE_IMAGERY:
        return

    npz_key = _imagery_npz_key(digest)
    jpg_key = _imagery_jpg_key(digest)
    client = _client()

    try:
        # 1. Upload .npz archive
        bio = io.BytesIO()
        np.savez_compressed(
            bio,
            mosaic=mosaic,
            minx=np.float64(minx),
            miny=np.float64(miny),
            maxx=np.float64(maxx),
            maxy=np.float64(maxy),
        )
        bio.seek(0)
        client.put_object(
            Bucket=S3_BUCKET,
            Key=npz_key,
            Body=bio.getvalue(),
            ContentType='application/octet-stream',
            Metadata={
                'utm-crs': str(utm_crs),
                'shape': f'{mosaic.shape[0]}x{mosaic.shape[1]}x{mosaic.shape[2]}',
            },
        )
        logger.info('Stored imagery array in S3: s3://%s/%s (%d bytes)', S3_BUCKET, npz_key, bio.getbuffer().nbytes)

        # 2. Upload RGB preview JPEG
        try:
            # mosaic shape is (3, H, W) uint8 -> transpose to (H, W, 3)
            img_rgb = Image.fromarray(mosaic.transpose(1, 2, 0))
            jpg_bio = io.BytesIO()
            img_rgb.save(jpg_bio, format='JPEG', quality=85)
            jpg_bio.seek(0)
            client.put_object(
                Bucket=S3_BUCKET,
                Key=jpg_key,
                Body=jpg_bio.getvalue(),
                ContentType='image/jpeg',
            )
            logger.info('Stored imagery preview in S3: s3://%s/%s', S3_BUCKET, jpg_key)
        except Exception as img_err:
            logger.debug('Could not generate imagery preview JPEG for S3: %s', img_err)

    except Exception as exc:
        logger.warning('Failed to persist imagery to S3 (key=%s): %s', npz_key, exc)


def store_cached_imagery_async(
    digest: str,
    mosaic: np.ndarray,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    utm_crs: str = '',
):
    """
    Dispatch S3 imagery persistence to a background daemon thread so that
    model inference begins immediately without network I/O lag.
    """
    import threading

    thread = threading.Thread(
        target=store_cached_imagery,
        args=(digest, mosaic.copy(), minx, miny, maxx, maxy, utm_crs),
        daemon=True,
        name=f"s3-upload-img-{digest[:8]}",
    )
    thread.start()
    return thread


# ── Auxiliary road and urban mask cache ────────────────────────────────────────

def _aux_npz_key(digest: str) -> str:
    """Return S3 object key for cached auxiliary masks npz."""
    return f'{S3_PREFIX}/auxiliary/{digest}.npz' if S3_PREFIX else f'auxiliary/{digest}.npz'


def get_cached_auxiliary(digest: str, expected_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    """Check S3 for cached road and urban masks."""
    if not S3_BUCKET:
        return None

    client = _client()
    key = _aux_npz_key(digest)
    try:
        response = client.get_object(Bucket=S3_BUCKET, Key=key)
        raw = response['Body'].read()
        payload = np.load(io.BytesIO(raw))
        road_mask = payload['road_mask']
        urban_mask = payload['urban_mask']
        if road_mask.shape == expected_shape and urban_mask.shape == expected_shape:
            logger.info('Loaded auxiliary masks from S3: s3://%s/%s', S3_BUCKET, key)
            return road_mask.astype(bool), urban_mask.astype(bool)
    except Exception as exc:
        err_code = getattr(exc, 'response', {}).get('Error', {}).get('Code')
        if err_code not in ('NoSuchKey', '404', 'NotFound'):
            logger.warning('Error reading auxiliary cache from S3 (key=%s): %s', key, exc)
    return None


def store_cached_auxiliary(digest: str, road_mask: np.ndarray, urban_mask: np.ndarray):
    """Store road and urban masks in S3 as compressed npz."""
    if not S3_BUCKET:
        return

    client = _client()
    key = _aux_npz_key(digest)
    try:
        bio = io.BytesIO()
        np.savez_compressed(
            bio,
            road_mask=road_mask.astype(bool),
            urban_mask=urban_mask.astype(bool),
        )
        bio.seek(0)
        client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=bio.getvalue(),
            ContentType='application/octet-stream',
            Metadata={
                'version': RESULT_CACHE_VERSION,
                'shape': f'{road_mask.shape[0]}x{road_mask.shape[1]}',
            },
        )
        logger.info('Stored auxiliary masks in S3: s3://%s/%s (%d bytes)', S3_BUCKET, key, bio.getbuffer().nbytes)
    except Exception as exc:
        logger.warning('Failed to persist auxiliary masks to S3 (key=%s): %s', key, exc)


def store_cached_auxiliary_async(digest: str, road_mask: np.ndarray, urban_mask: np.ndarray):
    """Background upload of auxiliary masks to S3."""
    import threading

    thread = threading.Thread(
        target=store_cached_auxiliary,
        args=(digest, road_mask.copy(), urban_mask.copy()),
        daemon=True,
        name=f"s3-upload-aux-{digest[:8]}",
    )
    thread.start()
    return thread