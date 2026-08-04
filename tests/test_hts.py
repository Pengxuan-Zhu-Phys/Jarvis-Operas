from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from jarvis_operas.api import get_global_operas_registry
from jarvis_operas.namespaces.hts.defs import core as hts


def _dataset_pair(tmp_path):
    hb = tmp_path / "hb"
    hs = tmp_path / "hs"
    hb.mkdir(parents=True)
    hs.mkdir(parents=True)
    (hb / "VERSION").write_text("hb-test\n", encoding="utf-8")
    return hb, hs


def _fake_higgs(*, allowed=False):
    class Bounds:
        created: list[str] = []

        def __init__(self, path):
            self.path = path
            self.created.append(path)

        def __call__(self, prediction):
            assert prediction == {"states": [{"mass": 125.0, "branching_ratio": 0.5}]}
            return {
                "allowed": allowed,
                "selectedLimits": [{"analysisId": "search-1", "obsRatio": 1.2, "expectedRatio": 1.0}],
            }

    class Signals:
        created: list[str] = []

        def __init__(self, path):
            self.path = path
            self.created.append(path)

        def __call__(self, prediction):
            return SimpleNamespace(chisq=3.5, loglike=-1.75)

    return SimpleNamespace(
        bounds=SimpleNamespace(Bounds=Bounds),
        signals=SimpleNamespace(Signals=Signals),
        tools=SimpleNamespace(Input=SimpleNamespace(predictionsFromDict=lambda value: value)),
    )


def test_hts_catalog_registration_and_import_are_lazy(monkeypatch) -> None:
    sys.modules.pop("Higgs", None)
    registry = get_global_operas_registry()
    assert "HTs.evaluate" in registry.list(namespace="HTs")
    assert "Higgs" not in sys.modules


def test_dataset_resolution_precedence_and_provenance(tmp_path, monkeypatch) -> None:
    explicit_hb, explicit_hs = _dataset_pair(tmp_path / "explicit")
    payload_hb, payload_hs = _dataset_pair(tmp_path / "payload")
    observable_hb, observable_hs = _dataset_pair(tmp_path / "observables")
    env_hb, env_hs = _dataset_pair(tmp_path / "env")
    default_hb, default_hs = _dataset_pair(tmp_path / "default")
    monkeypatch.setenv("HIGGSTOOLS_HBDATASET", str(env_hb))
    monkeypatch.setenv("HIGGSTOOLS_HSDATASET", str(env_hs))

    resolved = hts.resolve_dataset_resources(
        hb_dataset_path=str(explicit_hb),
        hs_dataset_path=str(explicit_hs),
        payload={"hb_dataset_path": str(payload_hb), "hs_dataset_path": str(payload_hs)},
        observables={"hb_dataset_path": str(observable_hb), "hs_dataset_path": str(observable_hs)},
        dataset_defaults={"hb_dataset_path": str(default_hb), "hs_dataset_path": str(default_hs)},
    )
    assert resolved["hb_dataset"].path == str(explicit_hb.resolve())
    assert resolved["hb_dataset"].source == "explicit"
    assert resolved["hb_dataset"].version == "hb-test"

    resolved = hts.resolve_dataset_resources(
        payload={"hb_dataset_path": str(payload_hb), "hs_dataset_path": str(payload_hs)},
        observables={"hb_dataset_path": str(observable_hb), "hs_dataset_path": str(observable_hs)},
    )
    assert resolved["hb_dataset"].path == str(payload_hb.resolve())
    assert resolved["hb_dataset"].source == "payload"


def test_missing_dataset_path_is_diagnostic() -> None:
    with pytest.raises(hts.HiggsToolsUnavailableError, match="does not exist"):
        hts.resolve_dataset_resources(hb_dataset_path="/not/a/real/dataset", hs_dataset_path="/also/missing")


def test_backend_cache_reuses_same_normalized_paths_and_isolates_different_paths(tmp_path, monkeypatch) -> None:
    first_hb, first_hs = _dataset_pair(tmp_path / "first")
    second_hb, second_hs = _dataset_pair(tmp_path / "second")
    fake = _fake_higgs()
    monkeypatch.setattr(hts, "_import_higgs", lambda: fake)
    hts.clear_higgstools_backend_cache()

    one = hts.get_higgstools_backend(str(first_hb), str(first_hs))
    two = hts.get_higgstools_backend(str(first_hb), str(first_hs))
    three = hts.get_higgstools_backend(str(second_hb), str(second_hs))
    assert one is two
    assert three is not one
    assert len(fake.bounds.Bounds.created) == 2
    hts.clear_higgstools_backend_cache()


def test_backend_initialization_failure_does_not_poison_cache(tmp_path, monkeypatch) -> None:
    hb, hs = _dataset_pair(tmp_path)

    class BrokenBounds:
        def __init__(self, path):
            raise RuntimeError("broken dataset")

    broken = SimpleNamespace(
        bounds=SimpleNamespace(Bounds=BrokenBounds),
        signals=SimpleNamespace(Signals=object),
        tools=SimpleNamespace(Input=object),
    )
    monkeypatch.setattr(hts, "_import_higgs", lambda: broken)
    hts.clear_higgstools_backend_cache()
    with pytest.raises(hts.HiggsToolsUnavailableError, match="failed to initialize"):
        hts.get_higgstools_backend(str(hb), str(hs))

    monkeypatch.setattr(hts, "_import_higgs", lambda: _fake_higgs())
    backend = hts.get_higgstools_backend(str(hb), str(hs))
    assert backend.bounds is not None
    hts.clear_higgstools_backend_cache()


def test_evaluate_returns_unavailable_when_optional_package_is_missing(tmp_path, monkeypatch) -> None:
    hb, hs = _dataset_pair(tmp_path)
    monkeypatch.setattr(
        hts,
        "_import_higgs",
        lambda: (_ for _ in ()).throw(hts.HiggsToolsUnavailableError("HiggsTools Python package is unavailable")),
    )
    hts.clear_higgstools_backend_cache()
    result = hts.evaluate_numpy(
        prediction={"states": [{"mass": 125.0}]}, hb_dataset_path=str(hb), hs_dataset_path=str(hs)
    )
    assert result["status"] == "unavailable"
    assert "unavailable" in result["errors"][0]


def test_evaluate_normalizes_success_and_exclusion_is_not_failure(tmp_path, monkeypatch) -> None:
    hb, hs = _dataset_pair(tmp_path)
    monkeypatch.setattr(hts, "_import_higgs", lambda: _fake_higgs(allowed=False))
    hts.clear_higgstools_backend_cache()
    result = hts.evaluate_numpy(
        payload={"prediction": {"states": [{"mass": 125.0, "branching_ratio": 0.5}]}},
        hb_dataset_path=str(hb),
        hs_dataset_path=str(hs),
    )
    assert result["status"] == "ok"
    assert result["ready"] is True
    assert result["direct_exclusion"]["allowed"] is False
    assert result["direct_exclusion"]["selected_limits"][0]["analysis_id"] == "search-1"
    assert result["signal_measurements"] == {"chisq": 3.5, "loglike": -1.75}
    json.dumps(result)
    hts.clear_higgstools_backend_cache()


def test_invalid_prediction_contract_is_reported() -> None:
    result = hts.evaluate_numpy(prediction={"branching_ratio": 1.5})
    assert result["status"] == "invalid_input"
    assert "branching_ratio" in result["errors"][0]


def test_invalid_runtime_observables_are_reported_as_invalid_input() -> None:
    result = hts.evaluate_numpy(prediction={"states": [{"mass": 125.0}]}, observables="bad")
    assert result["status"] == "invalid_input"
    assert "observables must be a mapping" in result["errors"][0]
