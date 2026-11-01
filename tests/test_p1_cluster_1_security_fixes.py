"""Security regression tests for P1 Vulnerability Cluster:
- SEC-06: Path Traversal via job_id parameter in test-ui
- SEC-07: Unbounded Thread DoS via ThreadPoolExecutor in test-ui
- SEC-08: Stored DOM-based XSS via HTML escaping in test-ui/static/index.html
"""
import importlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Import test-ui.server dynamically due to hyphen in module name
test_ui_server = importlib.import_module("test-ui.server")


@pytest.fixture
def client():
    return TestClient(test_ui_server.app)


def test_sec_06_job_id_validation_direct():
    """SEC-06: Directly test _validate_job_id rejects traversal and invalid identifiers."""
    invalid_ids = [
        "../../etc/passwd",
        "..\\..\\windows\\win.ini",
        "nested/path",
        "job;rm -rf",
        "job*wildcard",
        "job<script>",
        "../results",
        "job..foo",
        "",
        " ",
    ]

    for bad_id in invalid_ids:
        with pytest.raises(HTTPException) as exc_info:
            test_ui_server._validate_job_id(bad_id)
        assert exc_info.value.status_code == 400
        assert "không hợp lệ" in exc_info.value.detail

    # Valid job_id
    assert test_ui_server._validate_job_id("valid_job_123") == "valid_job_123"
    assert test_ui_server._validate_job_id("JOB-ABC_456") == "JOB-ABC_456"


def test_sec_06_job_id_validation_endpoints(client):
    """SEC-06: Validate that malicious job_ids are rejected with HTTP 400 at API endpoints."""
    bad_job_ids = [
        "job;rm",
        "job*wildcard",
        "job<script>",
        "job..foo",
        "job!danger",
    ]

    for bad_id in bad_job_ids:
        # GET /api/jobs/{job_id}/status
        res = client.get(f"/api/jobs/{bad_id}/status")
        assert res.status_code == 400, f"Expected 400 for bad job_id in status, got {res.status_code}"
        assert "Mã job không hợp lệ" in res.json().get("detail", "")

        # GET /api/jobs/{job_id}/log
        res = client.get(f"/api/jobs/{bad_id}/log")
        assert res.status_code == 400, f"Expected 400 for bad job_id in log, got {res.status_code}"

        # GET /api/jobs/{job_id}/video
        res = client.get(f"/api/jobs/{bad_id}/video")
        assert res.status_code == 400, f"Expected 400 for bad job_id in video, got {res.status_code}"

        # GET /api/jobs/{job_id}/meta
        res = client.get(f"/api/jobs/{bad_id}/meta")
        assert res.status_code == 400, f"Expected 400 for bad job_id in meta, got {res.status_code}"

        # POST /api/jobs/{job_id}/start
        res = client.post(f"/api/jobs/{bad_id}/start")
        assert res.status_code == 400, f"Expected 400 for bad job_id in start, got {res.status_code}"

        # GET /api/jobs/{job_id}/evidence/{filename}
        res = client.get(f"/api/jobs/{bad_id}/evidence/0123456789ab.jpg")
        assert res.status_code == 400, f"Expected 400 for bad job_id in evidence, got {res.status_code}"

    # Valid job_id format should NOT fail validation (returns 404 for non-existent job)
    valid_id = "valid_job_123"
    res = client.get(f"/api/jobs/{valid_id}/status")
    assert res.status_code == 404, f"Expected 404 for non-existent valid job_id, got {res.status_code}"


def test_sec_07_bounded_thread_pool_executor():
    """SEC-07: Validate that video jobs are executed via bounded ThreadPoolExecutor."""
    executor = test_ui_server.JOB_EXECUTOR
    assert isinstance(executor, ThreadPoolExecutor), "JOB_EXECUTOR must be an instance of ThreadPoolExecutor"
    assert executor._max_workers == test_ui_server.MAX_CONCURRENT_JOBS
    assert executor._max_workers <= 2, f"Max workers should be capped at 2 to prevent DoS, got {executor._max_workers}"


def test_sec_08_index_html_xss_prevention():
    """SEC-08: Validate that test-ui/static/index.html includes escapeHtml and applies it to dynamic outputs."""
    html_path = Path(test_ui_server.__file__).parent / "static" / "index.html"
    assert html_path.exists(), "test-ui/static/index.html must exist"
    content = html_path.read_text(encoding="utf-8")

    # Verify escapeHtml utility function exists and handles critical characters
    assert "function escapeHtml(str)" in content
    assert "&amp;" in content
    assert "&lt;" in content
    assert "&gt;" in content
    assert "&quot;" in content
    assert "&#039;" in content

    # Verify dynamic interpolation points use escapeHtml
    assert "escapeHtml(file.name)" in content
    assert "escapeHtml(m.source" in content
    assert "escapeHtml(m.model" in content
    assert "escapeHtml(v.person)" in content
    assert "escapeHtml(v.camera_id)" in content
    assert "escapeHtml(v.event_id)" in content
    assert "escapeHtml(ev.path)" in content
    assert "escapeHtml(ev.error)" in content
    assert "escapeHtml(j.source" in content
    assert "escapeHtml(m.name" in content
