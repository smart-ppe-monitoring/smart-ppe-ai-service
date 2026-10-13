"""
Tests for backend_client.py (TASK-A).
Verifies retrieval of dynamic safety rules from Backend and fallback handling.
"""

import asyncio
from unittest.mock import patch, MagicMock
import httpx

from app.clients.backend_client import (
    fetch_system_default_rules,
    fetch_effective_rules,
    SafetyRulesConfig,
)
from app.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CONFIRMATION_DURATION_S,
    DEFAULT_COOLDOWN_DURATION_S,
)


def test_fetch_system_default_rules_success():
    mock_payload = {
        "success": True,
        "data": {
            "confidenceThreshold": 0.85,
            "confirmationDuration": 5,
            "cooldownDuration": 45,
            "rules": {
                "enableNoHelmet": True,
                "enableNoVest": True,
                "enableRestrictedArea": False,
            },
        },
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_payload
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        rules = asyncio.run(fetch_system_default_rules())
        assert rules.confidence == 0.85
        assert rules.confirmation_s == 5
        assert rules.cooldown_s == 45
        assert rules.enable_no_helmet is True
        assert rules.enable_no_vest is True


def test_fetch_system_default_rules_fallback_on_error():
    with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection refused")):
        rules = asyncio.run(fetch_system_default_rules())
        assert rules.confidence == DEFAULT_CONFIDENCE_THRESHOLD
        assert rules.confirmation_s == DEFAULT_CONFIRMATION_DURATION_S
        assert rules.cooldown_s == DEFAULT_COOLDOWN_DURATION_S


def test_fetch_effective_rules_success():
    mock_payload = {
        "success": True,
        "data": {
            "confidenceThreshold": 0.90,
            "confirmationDuration": 2,
            "cooldownDuration": 20,
        },
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_payload
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        rules = asyncio.run(fetch_effective_rules("CAM-01", "SITE-01"))
        assert rules.confidence == 0.90
        assert rules.confirmation_s == 2
        assert rules.cooldown_s == 20
