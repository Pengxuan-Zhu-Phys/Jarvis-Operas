"""Keep the test process off the developer's ~/.jarvis/operas state."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_operas_home(tmp_path, monkeypatch):
    home = tmp_path / "jarvis-operas-home"
    home.mkdir()
    monkeypatch.setenv("JARVIS_OPERAS_HOME", str(home))
    monkeypatch.setenv("JARVIS_OPERAS_CURVE_CACHE_ROOT", str(home / "curve-cache"))
