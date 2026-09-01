import os

import pytest

from credit_review.providers import configure_provider


def test_deepseek_maps_key_for_anthropic_sdk(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    configure_provider("deepseek")
    assert os.environ["ANTHROPIC_API_KEY"] == "test-key"


def test_deepseek_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        configure_provider("deepseek")
