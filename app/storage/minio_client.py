"""
MinIO evidence upload client.
Uploads violation snapshots and clips to the 'ppe-evidences' bucket.
Secret key is NEVER logged.
"""

import io
import time
import logging
from minio import Minio
from minio.error import S3Error
from app import config
from app.constants import MINIO_EVIDENCE_PREFIX

logger = logging.getLogger(__name__)

_client: Minio | None = None


def get_client() -> Minio:
    global _client
    if _client is None:
        _client = Minio(
            config.MINIO_ENDPOINT,
            access_key=config.MINIO_ACCESS_KEY,
            secret_key=config.MINIO_SECRET_KEY,   # not logged
            secure=config.MINIO_SECURE,
        )
        ensure_bucket_exists(_client, config.MINIO_BUCKET_NAME)
    return _client


def ensure_bucket_exists(client: Minio, bucket_name: str) -> None:
    """Ensure the target bucket exists with secure private access by default."""
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info("Created MinIO bucket: %s", bucket_name)
    except Exception as err:
        logger.debug("Bucket check/creation info: %s", err)


def upload_snapshot(image_bytes: bytes, site_code: str, camera_code: str, track_id: int) -> str:
    """
    Upload a JPEG snapshot to MinIO.
    Returns the public HTTP URL to include in snapshotUrl field.
    """
    ts = int(time.time())
    object_name = f"{MINIO_EVIDENCE_PREFIX}/{site_code}/{camera_code}/{ts}_track{track_id}.jpg"
    try:
        client = get_client()
        client.put_object(
            bucket_name=config.MINIO_BUCKET_NAME,
            object_name=object_name,
            data=io.BytesIO(image_bytes),
            length=len(image_bytes),
            content_type="image/jpeg",
        )
        scheme = "https" if getattr(config, "MINIO_SECURE", False) else "http"
        return f"{scheme}://{config.MINIO_ENDPOINT}/{config.MINIO_BUCKET_NAME}/{object_name}"
    except S3Error as exc:
        logger.error("MinIO upload failed for track %d: %s", track_id, exc)
        raise


def upload_clip(video_bytes: bytes, site_code: str, camera_code: str, track_id: int) -> str:
    """
    Upload a short MP4 clip (~10‑15 s) to MinIO.
    Returns the public HTTP URL for clipUrl field.
    """
    ts = int(time.time())
    object_name = f"{MINIO_EVIDENCE_PREFIX}/{site_code}/{camera_code}/{ts}_track{track_id}.mp4"
    try:
        client = get_client()
        client.put_object(
            bucket_name=config.MINIO_BUCKET_NAME,
            object_name=object_name,
            data=io.BytesIO(video_bytes),
            length=len(video_bytes),
            content_type="video/mp4",
        )
        scheme = "https" if getattr(config, "MINIO_SECURE", False) else "http"
        return f"{scheme}://{config.MINIO_ENDPOINT}/{config.MINIO_BUCKET_NAME}/{object_name}"
    except S3Error as exc:
        logger.error("MinIO clip upload failed for track %d: %s", track_id, exc)
        raise
