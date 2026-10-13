"""
Tests for centralized constants module.
Ensures magic strings are not used anywhere and values match the contract.
"""

from app.constants import (
    KAFKA_TOPIC_VIOLATION,
    HTTP_EVENTS_ENDPOINT,
    HTTP_BATCH_ENDPOINT,
    MINIO_BUCKET,
    VIOLATION_NO_HELMET,
    VIOLATION_NO_VEST,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CONFIRMATION_DURATION_S,
    DEFAULT_COOLDOWN_DURATION_S,
    OFFLINE_BUFFER_BATCH_SIZE,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_FRAME_HEIGHT,
    CONTRACT_VERSION,
)


class TestConstants:
    def test_kafka_topic(self):
        assert KAFKA_TOPIC_VIOLATION == "ppe.violation.detected"

    def test_http_endpoints(self):
        assert HTTP_EVENTS_ENDPOINT == "/api/v1/integration/ai-vision/events"
        assert HTTP_BATCH_ENDPOINT == "/api/v1/integration/ai-vision/events/batch"

    def test_minio_bucket(self):
        assert MINIO_BUCKET == "ppe-evidences"

    def test_violation_types(self):
        assert VIOLATION_NO_HELMET == "NO_HELMET"
        assert VIOLATION_NO_VEST == "NO_VEST"

    def test_default_thresholds(self):
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.8
        assert DEFAULT_CONFIRMATION_DURATION_S == 2
        assert DEFAULT_COOLDOWN_DURATION_S == 30

    def test_batch_size_limit(self):
        assert OFFLINE_BUFFER_BATCH_SIZE == 50

    def test_default_resolution(self):
        assert DEFAULT_FRAME_WIDTH == 1920
        assert DEFAULT_FRAME_HEIGHT == 1080

    def test_contract_version(self):
        assert CONTRACT_VERSION == "2.1.0-PPE-STABLE"
