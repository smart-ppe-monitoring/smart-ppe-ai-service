"""Security regression tests for Vulnerability Cluster:
- SEC-16: Temporal Violation Accumulator Delta Capping & Gap Reset
- SEC-17: Persistent Connection Pooling for HTTP Publisher
- SEC-18: TTL Eviction of Stale Tracking State in ViolationDeduplicator
- SEC-19: Dynamic HTTPS Scheme for MinIO Evidence URLs & Non-Root Dockerfile User
"""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import httpx

from app.rules.confirmation import ViolationConfirmation
from app.rules.safety import VIOLATION_CANDIDATE, VIOLATION_NO_HELMET, PersonState
from app.events.http_publisher import get_sync_client, get_async_client
from app.rules.deduplication import ViolationDeduplicator
from app.detection.adapter import Detection
from app.storage.minio_client import upload_snapshot, upload_clip
from app import config


def test_sec_16_temporal_delta_capping_and_gap_reset():
    """SEC-16: Verify that a network gap (> max_gap_s) resets accumulation, preventing instant confirmation."""
    conf = ViolationConfirmation(duration_s=3.0, max_gap_s=2.0)
    cand_state = [PersonState(person_track_id=1, status=VIOLATION_CANDIDATE, violations=[VIOLATION_NO_HELMET])]

    # Frame 1: Candidate starts at t=0.0
    evs = conf.update(0.0, cand_state)
    assert len(evs) == 0

    # Frame 2: 5.0 seconds later after camera reconnect (gap dt = 5.0s > max_gap_s 2.0s)
    evs = conf.update(5.0, cand_state)
    # Must NOT instantly confirm despite 5 seconds passing
    assert len(evs) == 0, "Instant confirmation must be prevented when a network gap occurs"

    # Subsequent continuous frames (dt = 0.5s <= 2.0s)
    # t = 5.5, 6.0, 6.5, 7.0, 7.5, 8.0 (3.0 continuous seconds from t=5.0)
    for t_step in (5.5, 6.0, 6.5, 7.0, 7.5):
        assert len(conf.update(t_step, cand_state)) == 0

    evs_confirmed = conf.update(8.0, cand_state)
    assert len(evs_confirmed) == 1
    assert evs_confirmed[0].confirmed_t == 8.0
    assert evs_confirmed[0].start_t == 5.0
    assert evs_confirmed[0].duration_s == 3.0


def test_sec_17_http_publisher_connection_pooling():
    """SEC-17: Verify persistent connection pooling and singleton reuse to prevent ephemeral port exhaustion."""
    sync_client_1 = get_sync_client()
    sync_client_2 = get_sync_client()
    assert isinstance(sync_client_1, httpx.Client)
    assert sync_client_1 is sync_client_2, "get_sync_client must return the same pooled singleton instance"

    async_client_1 = get_async_client()
    async_client_2 = get_async_client()
    assert isinstance(async_client_1, httpx.AsyncClient)
    assert async_client_1 is async_client_2, "get_async_client must return the same pooled singleton instance"


def test_sec_18_deduplication_ttl_eviction():
    """SEC-18: Verify that tracks not seen for longer than 2x cooldown_s are evicted from state dictionaries."""
    cooldown = 10.0
    dedup = ViolationDeduplicator(camera_id="CAM-01", cooldown_s=cooldown)

    # Track 1 appears at t=5.0
    p1 = Detection(class_name="person", confidence=0.9, bbox=(10, 10, 100, 300), track_id=1)
    dedup.filter([], [p1], 5.0)
    assert 1 in dedup._last_seen
    assert 1 in dedup._canonical
    assert 1 in dedup._last_bbox

    # Advance time to t=30.0 (> 5.0 + 20.0 = 25.0 cutoff)
    dedup.filter([], [], 30.0)

    # Track 1 state must be pruned to avoid unbounded memory growth
    assert 1 not in dedup._last_seen
    assert 1 not in dedup._canonical
    assert 1 not in dedup._last_bbox


def test_sec_19_minio_https_scheme_and_dockerfile_user():
    """SEC-19: Verify dynamic HTTPS scheme selection in MinIO client and non-root user in Dockerfile."""
    mock_minio = MagicMock()

    # 1. When MINIO_SECURE is True, URL should be https://
    with patch("app.storage.minio_client.get_client", return_value=mock_minio), \
         patch.object(config, "MINIO_SECURE", True):
        url_snap = upload_snapshot(b"image_data", "SITE-01", "CAM-01", 1)
        assert url_snap.startswith("https://")

        url_clip = upload_clip(b"video_data", "SITE-01", "CAM-01", 1)
        assert url_clip.startswith("https://")

    # 2. When MINIO_SECURE is False, URL should be http://
    with patch("app.storage.minio_client.get_client", return_value=mock_minio), \
         patch.object(config, "MINIO_SECURE", False):
        url_snap_http = upload_snapshot(b"image_data", "SITE-01", "CAM-01", 1)
        assert url_snap_http.startswith("http://")

    # 3. Verify Dockerfile contains non-root USER instruction
    dockerfile_path = Path(__file__).resolve().parent.parent / "Dockerfile"
    assert dockerfile_path.exists()
    dockerfile_content = dockerfile_path.read_text(encoding="utf-8")
    assert "USER appuser" in dockerfile_content
    assert "useradd -u 1000" in dockerfile_content
