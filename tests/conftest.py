"""Cấu hình chung cho test suite.

Starlette TestClient gửi Host header mặc định là "testserver"; bổ sung vào allowlist
TrustedHostMiddleware của Test-UI trước khi module `test-ui.server` được import.
"""
import os

os.environ.setdefault("TEST_UI_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver")

