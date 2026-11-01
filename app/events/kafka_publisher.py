"""
Kafka publisher — đẩy sự kiện vi phạm vào topic ppe.violation.detected.
Payload phải validate qua ViolationEvent schema trước khi publish.
"""

import json
import logging
from confluent_kafka import Producer
from app import config
from app.constants import KAFKA_TOPIC_VIOLATION, KAFKA_PRODUCER_ID
from app.schemas.violation_event import ViolationEvent

logger = logging.getLogger(__name__)

_producer: Producer | None = None


def get_producer() -> Producer:
    global _producer
    if _producer is None:
        broker = config.KAFKA_BROKER
        conf = {"bootstrap.servers": broker}
        if "[::1]" in broker or "::1" in broker:
            conf["broker.address.family"] = "v6"
        _producer = Producer(conf)
    return _producer


def _delivery_report(err, msg):
    if err:
        logger.error("Kafka delivery failed: %s", err)
        # SEC-09: Fallback to local SQLite offline buffer on async Kafka delivery failure
        try:
            val = msg.value() if msg else None
            if val:
                raw_envelope = json.loads(val.decode("utf-8") if isinstance(val, bytes) else str(val))
                payload_data = raw_envelope.get("payload", {})
                from app.events.offline_buffer import save_event
                event = ViolationEvent.model_validate(payload_data)
                save_event(event)
                logger.info(
                    "Kafka async delivery failure recovered: saved event for camera %s to offline buffer",
                    event.cameraCode
                )
        except Exception as fb_err:
            logger.critical("Failed to save undelivered Kafka message to offline buffer: %s", fb_err)
    else:
        logger.debug("Message delivered to %s [%s]", msg.topic(), msg.partition())


def publish_violation(event: ViolationEvent) -> None:
    """
    Validate and publish a ViolationEvent to the Kafka topic.
    Raises ValueError if the event fails schema validation.
    """
    # Validate before publishing
    validated = ViolationEvent.model_validate(event.model_dump(by_alias=True))

    envelope = {
        "topic": KAFKA_TOPIC_VIOLATION,
        "producer": KAFKA_PRODUCER_ID,
        "payload": validated.model_dump(by_alias=True),
    }
    producer = get_producer()
    producer.produce(
        topic=KAFKA_TOPIC_VIOLATION,
        value=json.dumps(envelope, ensure_ascii=False).encode("utf-8"),
        callback=_delivery_report,
    )
    producer.poll(0)


def flush_producer(timeout: float = 5.0) -> int:
    """Flush pending messages in the producer queue."""
    global _producer
    if _producer is not None:
        remaining = _producer.flush(timeout)
        logger.info("Kafka producer flushed. Remaining messages: %d", remaining)
        return remaining
    return 0
