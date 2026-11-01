"""
HTTP publisher — gửi sự kiện vi phạm qua REST API của Backend.
Dùng khi Kafka không khả dụng hoặc cần mode đồng bộ (?sync=true).
Hỗ trợ cả async (cho async service) và sync (cho CV processing loop).
"""

import logging
import httpx
from app import config
from app.constants import HTTP_EVENTS_ENDPOINT, HTTP_BATCH_ENDPOINT, OFFLINE_BUFFER_BATCH_SIZE
from app.schemas.violation_event import ViolationEvent

logger = logging.getLogger(__name__)

# SEC-17: Persistent connection pools to prevent ephemeral port and socket exhaustion
_sync_client: httpx.Client | None = None
_async_client: httpx.AsyncClient | None = None


def get_sync_client() -> httpx.Client:
    global _sync_client
    if _sync_client is None or _sync_client.is_closed:
        _sync_client = httpx.Client(
            timeout=15.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )
    return _sync_client


def get_async_client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is None or _async_client.is_closed:
        _async_client = httpx.AsyncClient(
            timeout=15.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )
    return _async_client


async def publish_event(event: ViolationEvent, sync: bool = False) -> dict:
    """
    POST /api/v1/integration/ai-vision/events (Async)
    Returns the response body from Backend.
    """
    url = f"{config.BACKEND_BASE_URL}{HTTP_EVENTS_ENDPOINT}"
    params = {"sync": "true"} if sync else {}
    payload = event.model_dump(by_alias=True)

    client = get_async_client()
    resp = await client.post(url, json=payload, params=params)
    resp.raise_for_status()
    logger.info("HTTP event published — status %d", resp.status_code)
    return resp.json()


def publish_event_sync(event: ViolationEvent, sync: bool = False) -> dict:
    """
    POST /api/v1/integration/ai-vision/events (Synchronous)
    Returns the response body from Backend.
    """
    url = f"{config.BACKEND_BASE_URL}{HTTP_EVENTS_ENDPOINT}"
    params = {"sync": "true"} if sync else {}
    payload = event.model_dump(by_alias=True)

    client = get_sync_client()
    resp = client.post(url, json=payload, params=params)
    resp.raise_for_status()
    logger.info("HTTP event published (sync) — status %d", resp.status_code)
    return resp.json()


async def publish_batch(events: list[ViolationEvent]) -> dict:
    """
    POST /api/v1/integration/ai-vision/events/batch (Async)
    Batch size must not exceed OFFLINE_BUFFER_BATCH_SIZE (50).
    """
    if len(events) > OFFLINE_BUFFER_BATCH_SIZE:
        raise ValueError(f"Batch size {len(events)} exceeds limit {OFFLINE_BUFFER_BATCH_SIZE}")

    url = f"{config.BACKEND_BASE_URL}{HTTP_BATCH_ENDPOINT}"
    body = {"events": [e.model_dump(by_alias=True) for e in events]}

    client = get_async_client()
    resp = await client.post(url, json=body, timeout=30.0)
    resp.raise_for_status()
    logger.info("Batch of %d events published", len(events))
    return resp.json()


def publish_batch_sync(events: list[ViolationEvent]) -> dict:
    """
    POST /api/v1/integration/ai-vision/events/batch (Synchronous)
    Batch size must not exceed OFFLINE_BUFFER_BATCH_SIZE (50).
    """
    if len(events) > OFFLINE_BUFFER_BATCH_SIZE:
        raise ValueError(f"Batch size {len(events)} exceeds limit {OFFLINE_BUFFER_BATCH_SIZE}")

    url = f"{config.BACKEND_BASE_URL}{HTTP_BATCH_ENDPOINT}"
    body = {"events": [e.model_dump(by_alias=True) for e in events]}

    client = get_sync_client()
    resp = client.post(url, json=body, timeout=30.0)
    resp.raise_for_status()
    logger.info("Batch of %d events published (sync)", len(events))
    return resp.json()
