from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np

from jarvis_operas import cli as cli_mod
from jarvis_operas.core.spec import OperaFunction

_ROOT = Path(__file__).resolve().parents[1]
_OPS = _ROOT / "examples" / "unit_contract_ops.py"
_NAME = "unit_contract_ops.limit_prediction"


def _run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["JARVIS_OPERAS_PERSIST_FILE"] = str(tmp_path / "persist.json")
    return subprocess.run(
        [sys.executable, "-m", "jarvis_operas.cli", *args],
        capture_output=True, text=True, check=False, env=environment,
    )


@pytest.mark.parametrize("command", ["call", "acall"])
def test_cli_unit_projection_preserves_canonical_record(tmp_path: Path, command: str) -> None:
    process = _run_cli(
        tmp_path, command, _NAME, "--user-ops", str(_OPS), "--arg", "mass_GeV=0.2",
        "--source-units", '{"mass_GeV":"TeV"}',
        "--input-units", '{"mass_GeV":"TeV"}',
        "--output-units", '{"/value":"fb"}', "--strict-units", "--json",
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout)
    assert set(payload) == {"canonical_result", "normalized_inputs", "values", "units", "audit"}
    assert payload["normalized_inputs"]["mass_GeV"] == 200.0
    assert payload["values"]["/value"] == pytest.approx(38.332)
    assert payload["units"]["/value"] == "fb"
    assert payload["canonical_result"] == {
        "value": 0.038332,
        "unit": "pb",
        "status": "ok",
        "record_id": "demo-limit-canonical-pb",
        "provenance": {"source": "unit-contract-fixture", "mass_GeV": 200.0},
        "dependencies": ["demo-analysis"],
    }
    assert "assumed_canonical" not in json.dumps(payload["audit"])


def test_cli_default_call_keeps_raw_return_and_audit_is_opt_in(tmp_path: Path) -> None:
    args = ("call", _NAME, "--user-ops", str(_OPS), "--arg", "mass_GeV=200", "--json")
    raw = _run_cli(tmp_path, *args)
    audited = _run_cli(tmp_path, *args, "--unit-audit")
    assert raw.returncode == audited.returncode == 0, raw.stderr + audited.stderr
    assert "canonical_result" not in json.loads(raw.stdout)
    payload = json.loads(audited.stdout)
    assert payload["canonical_result"] == json.loads(raw.stdout)
    assert "assumed_canonical" in json.dumps(payload["audit"])


@pytest.mark.parametrize(
    ("options", "code"),
    [
        (("--strict-units",), "missing_unit"),
        (("--input-units", '{"mass_GeV":"pb"}'), "dimension_mismatch"),
        (("--input-units", '{"mass_GeV":"not-a-unit"}'), "unknown_unit"),
        (("--input-units", '{"mass_GeV":"GeV"}', "--source-units", '{"mass_GeV":"TeV"}'), "source_conflict"),
        (("--input-units", '{"mass_GeV":"GeV"}', "--backend", "polars"), "unsupported_backend"),
    ],
)
def test_cli_unit_errors_are_identifiable(tmp_path: Path, options: tuple[str, ...], code: str) -> None:
    process = _run_cli(
        tmp_path, "call", _NAME, "--user-ops", str(_OPS), "--arg", "mass_GeV=200", *options,
    )
    assert process.returncode == 1
    assert code in process.stderr
    assert process.stdout == ""


def test_cli_info_derives_contract_from_user_metadata(tmp_path: Path) -> None:
    process = _run_cli(tmp_path, "info", _NAME, "--user-ops", str(_OPS), "--json")
    assert process.returncode == 0, process.stderr
    contract = json.loads(process.stdout)["metadata"]["unit_contract"]
    assert contract["version"] == "1.0"
    assert contract["inputs"]["mass_GeV"]["unit"] == "GeV"
    assert contract["outputs"]["/value"]["unit"] == "pb"
    readable = _run_cli(tmp_path, "info", _NAME, "--user-ops", str(_OPS))
    assert readable.returncode == 0, readable.stderr
    assert all(fragment in readable.stdout for fragment in ("Unit contract 1.0", "mass_GeV", "/value", "GeV", "pb"))


def test_cli_rejects_invalid_user_unit_metadata(tmp_path: Path) -> None:
    path = tmp_path / "invalid_units.py"
    path.write_text(
        'from jarvis_operas import oper\n'
        '@oper("broken", unit_contract={"version":"99", "inputs":{}, "outputs":{}})\n'
        'def broken(x):\n    return x\n',
        encoding="utf-8",
    )
    process = _run_cli(tmp_path, "load", str(path), "--session-only")
    assert process.returncode == 1
    assert "invalid_contract" in process.stderr


def test_unit_map_cli_options_require_json_string_mapping() -> None:
    assert cli_mod._parse_unit_map(None, "--input-units") is None
    assert cli_mod._parse_unit_map('{"x":"GeV"}', "--input-units") == {"x": "GeV"}
    for raw in ('["GeV"]', '{"x":5}'):
        with pytest.raises(ValueError, match="--input-units"):
            cli_mod._parse_unit_map(raw, "--input-units")


def test_default_unit_display_limits_arrays_and_full_record() -> None:
    canonical = {"value": np.arange(1000), "record_id": "canonical-id", "provenance": {"full": "record"}}
    payload = {
        "canonical_result": canonical,
        "normalized_inputs": {"x": np.arange(1000)},
        "values": {"/value": np.arange(1000)},
        "units": {"/value": "fb"}, "audit": [],
    }
    compact = cli_mod._compact_unit_result(payload)
    text = json.dumps(cli_mod._to_json_safe(compact))
    assert "canonical-id" in text
    assert '"size": 1000' in text
    assert len(text) < 1000
    assert "provenance" in text and "full" not in text
    assert canonical["value"].shape == (1000,)
    assert cli_mod._to_json_safe(np.asarray(3.5)) == 3.5


def test_function_docs_derive_readable_and_machine_unit_contract(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("function_docs_test", _ROOT / "scripts" / "generate_function_docs.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = {
        "version": "1.0",
        "inputs": {"mass": {"unit": "GeV", "shape": "scalar", "nullable": False}},
        "outputs": {"": {"unit": "pb", "shape": "scalar", "nullable": False}},
    }
    declaration = OperaFunction(
        namespace="demo", name="fixture", numpy_impl=lambda mass: 0.038332,
        arity=1, return_dtype=float,
        metadata={"unit_contract": contract},
    )
    module.generate_function_docs(out_dir=tmp_path, declarations=[declaration])
    content = (tmp_path / "demo" / "fixture.md").read_text(encoding="utf-8")
    assert all(fragment in content for fragment in ("Unit Contract", "Contract version: `1.0`", "GeV", "pb", '"" (root result)'))
    catalog = json.loads((tmp_path / "functions.json").read_text(encoding="utf-8"))
    assert catalog[0]["unit_contract"] == catalog[0]["metadata"]["unit_contract"]
