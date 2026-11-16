"""
Event Dispatching Subsystem — Quản lý phát sự kiện vi phạm PPE.
Tích hợp:
1. Kafka Producer: đẩy message vào topic `ppe.violation.detected`.
2. HTTP Publisher: fallback khi Kafka lỗi hoặc cần mode sync (?sync=true).
3. SQLite Offline Buffer: lưu trữ cục bộ khi mất kết nối Backend/Kafka và tự động flush khi có mạng.
"""

import logging
from typing import Optional
import httpx

from app.schemas.violation_event import ViolationEvent
from app.events.kafka_publisher import publish_violation, get_producer, flush_producer
from app.events.http_publisher import (
    publish_event,
    publish_event_sync,
    publish_batch,
    publish_batch_sync,
)
from app.events.offline_buffer import (
    save_event,
    load_batch,
    delete_events,
    pending_count,
    increment_retries,
)

logger = logging.getLogger(__name__)


def dispatch_event(event: ViolationEvent, sync: bool = False) -> str:
    """
    Điều phối sự kiện vi phạm theo cơ chế 3 tầng khả dụng cao:
    Kafka -> HTTP Fallback -> SQLite Offline Buffer.

    Trả về phương thức đã gửi thành công: 'kafka' | 'http' | 'offline_buffer'.
    Không bao giờ ném exception ra ngoài — luôn đảm bảo an toàn cho video processing loop.
    """
    # 1. Thử gửi qua Kafka trước (nếu không yêu cầu sync mode)
    if not sync:
        try:
            publish_violation(event)
            logger.info("Dispatched violation event for camera %s via Kafka", event.cameraCode)
            return "kafka"
        except Exception as k_err:
            logger.warning("Kafka dispatch failed (%s), falling back to HTTP", k_err)

    # 2. Fallback sang HTTP
    try:
        publish_event_sync(event, sync=sync)
        logger.info("Dispatched violation event for camera %s via HTTP fallback", event.cameraCode)
        return "http"
    except Exception as h_err:
        logger.warning("HTTP fallback failed (%s), saving to offline SQLite buffer", h_err)

    # 3. Fallback sang SQLite buffer
    try:
        save_event(event)
        logger.info("Buffered violation event for camera %s in offline storage", event.cameraCode)
        return "offline_buffer"
    except Exception as s_err:
        logger.critical("Failed to save event to offline buffer: %s", s_err)
        return "failed"


def _is_transient_publish_error(err: Exception) -> bool:
    """Phân biệt lỗi tạm thời (mạng / Backend 5xx / timeout) với lỗi dữ liệu poison-pill (4xx)."""
    if isinstance(
        err,
        (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
            ConnectionError,
            TimeoutError,
        ),
    ):
        return True
    if isinstance(err, httpx.HTTPStatusError):
        status = err.response.status_code
        return status >= 500 or status in (408, 425, 429)
    return False


def flush_offline_buffer(sync: bool = True) -> int:
    """
    Kiểm tra và đẩy toàn bộ các sự kiện đang tồn đọng trong SQLite offline buffer lên Backend.
    Trả về số lượng sự kiện đã flush thành công.
    - Lỗi tạm thời (Backend đang khởi động lại / 5xx / mất mạng): dừng vòng lặp, KHÔNG đốt retry.
    - Lỗi payload/4xx (poison-pill): tăng retry counter để tránh nghẽn hàng đợi vĩnh viễn.
    """
    total_flushed = 0
    while pending_count() > 0:
        ids, events = load_batch()
        if not events:
            break
        try:
            if sync:
                publish_batch_sync(events)
            delete_events(ids)
            total_flushed += len(events)
            logger.info("Successfully flushed batch of %d offline events", len(events))
        except Exception as err:
            if _is_transient_publish_error(err):
                logger.warning(
                    "Offline buffer flush paused due to transient error (%s); preserving %d queued events",
                    err,
                    len(ids),
                )
            else:
                logger.warning(
                    "Offline buffer flush halted due to error (%s). Incrementing retries for batch %s",
                    err,
                    ids,
                )
                increment_retries(ids)
            break
    return total_flushed


__all__ = [
    "dispatch_event",
    "flush_offline_buffer",
    "publish_violation",
    "get_producer",
    "flush_producer",
    "publish_event",
    "publish_event_sync",
    "publish_batch",
    "publish_batch_sync",
    "save_event",
    "load_batch",
    "delete_events",
    "pending_count",
    "increment_retries",
]
