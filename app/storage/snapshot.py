"""Re-export snapshot utilities from app.evidence.snapshot."""
from app.evidence.snapshot import EvidenceResult, save_evidence, VIOLATION_LABELS

__all__ = ["EvidenceResult", "save_evidence", "VIOLATION_LABELS"]
