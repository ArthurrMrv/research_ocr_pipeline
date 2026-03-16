import io
import os
from unittest.mock import MagicMock

import pytest
from PIL import Image


@pytest.fixture
def mock_supabase(mocker):
    """Return a mock Supabase client with chainable query builder."""
    client = MagicMock()

    def _make_chain(data=None, error=None):
        chain = MagicMock()
        result = MagicMock()
        result.data = data or []
        chain.execute.return_value = result
        chain.select.return_value = chain
        chain.insert.return_value = chain
        chain.update.return_value = chain
        chain.upsert.return_value = chain
        chain.eq.return_value = chain
        return chain

    client.table.return_value = _make_chain()
    return client


@pytest.fixture
def fixture_pdf(tmp_path):
    """Create a minimal single-page PDF for testing."""
    try:
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test financial report content")
        pdf_path = tmp_path / "test_report.pdf"
        doc.save(str(pdf_path))
        doc.close()
        return str(pdf_path)
    except Exception:
        pytest.skip("pymupdf not available")


@pytest.fixture
def sample_image():
    """Return a simple RGB PIL Image for OCR testing."""
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    return img


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    """Set required environment variables for all tests."""
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-service-key")
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-moonshot-key")
