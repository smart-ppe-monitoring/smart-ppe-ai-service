"""
Pydantic schema cho Violation Event — Contract 2.1.0-PPE-STABLE.
Mọi payload gửi về Backend (Kafka hoặc HTTP) phải validate qua model này.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, HttpUrl, Field, ConfigDict
from app.constants import (
    CONTRACT_VERSION,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_FRAME_HEIGHT,
    SEVERITY_HIGH,
)


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class FrameResolution(BaseModel):
    width: int = DEFAULT_FRAME_WIDTH
    height: int = DEFAULT_FRAME_HEIGHT


class Evidence(BaseModel):
    snapshotUrl: str          # Must be a valid HTTP URL accessible from MinIO
    clipUrl: Optional[str] = None
    frameResolution: FrameResolution = Field(default_factory=FrameResolution)


class Detection(BaseModel):
    """Single detected object in the frame."""
    classLabel: str = Field(alias="class")   # PERSON | HARD_HAT | SAFETY_VEST
    confidence: float
    box: BoundingBox
    trackId: int

    model_config = ConfigDict(populate_by_name=True)


class ViolationItem(BaseModel):
    violationType: str        # Use VIOLATION_* constants from constants.py
    severity: str = SEVERITY_HIGH
    description: str = ""
    confidence: float
    box: Optional[BoundingBox] = None


class EventSummary(BaseModel):
    totalPersons: int
    totalViolations: int
    highestSeverity: str


class EventMetadata(BaseModel):
    confirmedDurationMs: int
    cooldownDurationMs: int
    model: str = ""
    contractVersion: str = CONTRACT_VERSION


class ViolationEvent(BaseModel):
    """
    Root model for a violation event payload (HTTP body or Kafka message payload).
    Corresponds to Contract 2.1.0-PPE-STABLE § 4.1 / 4.2.
    """
    cameraCode: str
    timestamp: str            # ISO 8601 UTC e.g. "2026-10-13T08:30:00.000Z"
    evidence: Evidence
    detections: List[Detection]
    violations: List[ViolationItem]
    summary: Optional[EventSummary] = None
    metadata: Optional[EventMetadata] = None
