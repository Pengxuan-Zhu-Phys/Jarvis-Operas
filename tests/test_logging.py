from __future__ import annotations

import pytest

from jarvis_operas.logging import get_log_mode, get_logger, set_log_mode


def test_log_mode_defaults_to_warning() -> None:
    set_log_mode("warning")
    assert get_log_mode() == "warning"


def test_set_log_mode_info_and_debug() -> None:
    set_log_mode("info")
    assert get_log_mode() == "info"

    set_log_mode("debug")
    assert get_log_mode() == "debug"

    # restore default for other tests
    set_log_mode("warning")


def test_invalid_log_mode_raises() -> None:
    with pytest.raises(ValueError):
        set_log_mode("verbose")


def test_get_logger_accepts_mode_override() -> None:
    logger = get_logger(mode="info", module="UnitTest")
    assert logger is not None
    assert get_log_mode() == "info"

    # restore default for other tests
    set_log_mode("warning")
