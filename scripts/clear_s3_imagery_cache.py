"""
Script to clear cached ESRI imagery from S3 (and optionally local disk).
Uses the credentials and bucket configuration from your .env file.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BASE_DIR, S3_BUCKET, S3_PREFIX
from app.result_cache import _client

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("clear_cache")


def clear_s3_imagery(dry_run: bool = False) -> int:
    if not S3_BUCKET:
        logger.error("S3_BUCKET is not configured in .env or environment.")
        return 0

    s3 = _client()
    prefix = f"{S3_PREFIX}/imagery/" if S3_PREFIX else "imagery/"
    paginator = s3.get_paginator("list_objects_v2")

    deleted_count = 0
    total_bytes = 0

    objects_to_delete = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects_to_delete.append({"Key": obj["Key"]})
            total_bytes += obj["Size"]

    if not objects_to_delete:
        logger.info("No cached imagery found in s3://%s/%s", S3_BUCKET, prefix)
        return 0

    logger.info(
        "Found %d imagery objects (%.2f MB) in s3://%s/%s",
        len(objects_to_delete),
        total_bytes / (1024 * 1024),
        S3_BUCKET,
        prefix,
    )

    if dry_run:
        logger.info("[DRY RUN] Would delete %d objects. No changes made.", len(objects_to_delete))
        return len(objects_to_delete)

    # Delete in batches of 1000 (S3 API limit)
    for i in range(0, len(objects_to_delete), 1000):
        batch = objects_to_delete[i : i + 1000]
        resp = s3.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": batch})
        deleted = len(resp.get("Deleted", []))
        deleted_count += deleted
        logger.info("Deleted batch: %d objects", deleted)

    logger.info("Successfully deleted %d imagery cache objects from S3.", deleted_count)
    return deleted_count


def clear_local_disk_imagery(dry_run: bool = False) -> int:
    data_dir = Path(BASE_DIR) / "data"
    files = list(data_dir.glob("esri_*.npz"))
    if not files:
        logger.info("No local imagery files (data/esri_*.npz) found.")
        return 0

    total_bytes = sum(f.stat().st_size for f in files)
    logger.info("Found %d local disk imagery files (%.2f MB).", len(files), total_bytes / (1024 * 1024))

    if dry_run:
        logger.info("[DRY RUN] Would delete %d local files. No changes made.", len(files))
        return len(files)

    for f in files:
        f.unlink()
    logger.info("Deleted %d local disk imagery files.", len(files))
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear S3 and local ESRI imagery cache.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be deleted without actually deleting them.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Also clear local disk imagery cache in data/esri_*.npz.",
    )
    args = parser.parse_args()

    clear_s3_imagery(dry_run=args.dry_run)
    if args.local:
        clear_local_disk_imagery(dry_run=args.dry_run)
