"""
Centralized constants for smart-ppe-ai-service.
NEVER hard-code these values elsewhere in the codebase.
"""

# ─── Kafka ───────────────────────────────────────────────────────────────────
KAFKA_TOPIC_VIOLATION = "ppe.violation.detected"
KAFKA_PRODUCER_ID = "ai-inference-engine"

# ─── HTTP Integration Endpoints ───────────────────────────────────────────────
HTTP_EVENTS_ENDPOINT = "/api/v1/integration/ai-vision/events"
HTTP_BATCH_ENDPOINT = "/api/v1/integration/ai-vision/events/batch"
HTTP_SAFETY_RULES_DEFAULT = "/api/v1/safety-rules/system-default"
HTTP_SAFETY_RULES_EFFECTIVE = "/api/v1/safety-rules/effective"

# ─── MinIO / Object Storage ───────────────────────────────────────────────────
MINIO_BUCKET = "ppe-evidences"
MINIO_EVIDENCE_PREFIX = "evidences"

# ─── Detection Labels (normalized) ───────────────────────────────────────────
LABEL_PERSON = "PERSON"
LABEL_HARD_HAT = "HARD_HAT"
LABEL_HELMET = "HELMET"
LABEL_SAFETY_VEST = "SAFETY_VEST"

# ─── Violation Types ──────────────────────────────────────────────────────────
VIOLATION_NO_HELMET = "NO_HELMET"
VIOLATION_NO_VEST = "NO_VEST"
VIOLATION_RESTRICTED_AREA = "RESTRICTED_AREA"
VIOLATION_NO_GLOVES = "NO_GLOVES"
VIOLATION_NO_BOOTS = "NO_BOOTS"
VIOLATION_NO_HARNESS = "NO_HARNESS"

# ─── Severity Levels ──────────────────────────────────────────────────────────
SEVERITY_HIGH = "HIGH"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_LOW = "LOW"

# ─── Safety Rule Defaults (fallback if API unreachable) ──────────────────────
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.8
DEFAULT_CONFIRMATION_DURATION_S: int = 2       # seconds
DEFAULT_COOLDOWN_DURATION_S: int = 30           # seconds

# ─── Violation States ─────────────────────────────────────────────────────────
STATE_COMPLIANT = "COMPLIANT"
STATE_CANDIDATE = "VIOLATION_CANDIDATE"
STATE_CONFIRMED = "CONFIRMED_VIOLATION"
STATE_UNKNOWN = "UNKNOWN"

# ─── Offline Buffer ───────────────────────────────────────────────────────────
OFFLINE_BUFFER_DB = "offline_buffer.db"
OFFLINE_BUFFER_BATCH_SIZE = 50

# ─── Frame Defaults ───────────────────────────────────────────────────────────
DEFAULT_FRAME_WIDTH = 1920
DEFAULT_FRAME_HEIGHT = 1080

# ─── Contract Version ─────────────────────────────────────────────────────────
CONTRACT_VERSION = "2.1.0-PPE-STABLE"
