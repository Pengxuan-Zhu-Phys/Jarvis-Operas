from __future__ import annotations

import datetime as dt

import pytest

import jarvis_operas.logging as jl
from jarvis_operas.logging import _stream_filter, get_log_mode, get_logger, set_log_mode


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
    jl._INITIALIZED = False
    jl._SINK_ID = None
    jl._CURRENT_MODE = "warning"

    logger = get_logger(mode="info", module="UnitTest")
    assert logger is not None
    assert get_log_mode() == "info"
    assert jl._INITIALIZED is True

    # restore default for other tests
    set_log_mode("warning")


def test_stream_filter_isolated_to_operas_records() -> None:
    assert _stream_filter({"extra": {"_log_domain": "jarvis_operas"}}) is True
    assert _stream_filter({"extra": {"_log_domain": "jarvis_hep", "_jarvis_operas": True}}) is False
    assert _stream_filter({"extra": {"module": "Jarvis-Operas"}}) is True
    assert _stream_filter({"extra": {"_jarvis_operas": True}}) is True
    assert _stream_filter({"extra": {"module": "Jarvis-HEP", "to_console": True}}) is False


def test_custom_format_shows_level_only_in_debug_mode() -> None:
    record = {
        "extra": {"module": "Jarvis-Operas"},
        "time": dt.datetime.now(),
        "level": "WARNING",
    }

    set_log_mode("warning")
    fmt_warning = jl._custom_format(record)
    assert "[<level>" not in fmt_warning
    assert ">>> " in fmt_warning

    set_log_mode("debug")
    fmt_debug = jl._custom_format(record)
    assert "[<level>" in fmt_debug

    # restore default for other tests
    set_log_mode("warning")


def test_get_logger_uses_default_warning_mode_when_not_configured() -> None:
    jl._INITIALIZED = False
    jl._SINK_ID = None
    jl._CURRENT_MODE = "warning"

    logger = get_logger(module="UnitTest")
    assert logger is not None
    assert get_log_mode() == "warning"
    assert jl._INITIALIZED is True
