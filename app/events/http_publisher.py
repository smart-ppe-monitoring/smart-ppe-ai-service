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


async def publish_event(event: ViolationEvent, sync: bool = False) -> dict:
    """
    POST /api/v1/integration/ai-vision/events (Async)
    Returns the response body from Backend.
    """
    url = f"{config.BACKEND_BASE_URL}{HTTP_EVENTS_ENDPOINT}"
    params = {"sync": "true"} if sync else {}
    payload = event.model_dump(by_alias=True)

    async with httpx.AsyncClient(timeout=10.0) as client:
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

    with httpx.Client(timeout=10.0) as client:
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

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=body)
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

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=body)
        resp.raise_for_status()
        logger.info("Batch of %d events published (sync)", len(events))
        return resp.json()
