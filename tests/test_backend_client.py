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
    DEFAULT_CONFIRMATION_DURATION_HELMET_S,
    DEFAULT_CONFIRMATION_DURATION_VEST_S,
    DEFAULT_COOLDOWN_DURATION_S,
)


def test_fetch_system_default_rules_success():
    mock_payload = {
        "success": True,
        "data": {
            "confidenceThreshold": 0.85,
            "helmetConfirmationDuration": 3.5,
            "confirmationDuration": 3.5,
            "vestConfirmationDuration": 6.0,
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
        assert rules.helmet_confirmation_s == 3.5
        assert rules.confirmation_s == 3.5
        assert rules.vest_confirmation_s == 6.0
        assert rules.cooldown_s == 45
        assert rules.enable_no_helmet is True
        assert rules.enable_no_vest is True


def test_fetch_system_default_rules_fallback_on_error():
    with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection refused")):
        rules = asyncio.run(fetch_system_default_rules())
        assert rules.confidence == DEFAULT_CONFIDENCE_THRESHOLD
        assert rules.helmet_confirmation_s == DEFAULT_CONFIRMATION_DURATION_HELMET_S
        assert rules.confirmation_s == DEFAULT_CONFIRMATION_DURATION_S
        assert rules.vest_confirmation_s == DEFAULT_CONFIRMATION_DURATION_VEST_S
        assert rules.cooldown_s == DEFAULT_COOLDOWN_DURATION_S


def test_fetch_effective_rules_success():
    mock_payload = {
        "success": True,
        "data": {
            "confidenceThreshold": 0.90,
            "helmetConfirmationDuration": 2.0,
            "vestConfirmationDuration": 4.5,
            "cooldownDuration": 20,
        },
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_payload
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        rules = asyncio.run(fetch_effective_rules("CAM-01", "SITE-01"))
        assert rules.confidence == 0.90
        assert rules.helmet_confirmation_s == 2.0
        assert rules.confirmation_s == 2.0
        assert rules.vest_confirmation_s == 4.5
        assert rules.cooldown_s == 20


def test_fetch_effective_rules_legacy_payload_keeps_vest_default_4s():
    mock_payload = {
        "success": True,
        "data": {
            "confidenceThreshold": 0.80,
            "confirmationDuration": 2.0,
            "cooldownDuration": 30,
        },
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_payload
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        rules = asyncio.run(fetch_effective_rules("CAM-01", "SITE-01"))
        assert rules.helmet_confirmation_s == 2.0
        assert rules.confirmation_s == 2.0
        assert rules.vest_confirmation_s == DEFAULT_CONFIRMATION_DURATION_VEST_S
