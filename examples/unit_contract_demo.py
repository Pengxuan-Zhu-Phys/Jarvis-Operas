"""Run from the repository: python examples/unit_contract_demo.py."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from jarvis_operas import OperasRegistry, load_user_ops


def main() -> None:
    registry = OperasRegistry()
    load_user_ops(str(Path(__file__).with_name("unit_contract_ops.py")), registry, refresh_sympy=False)
    plan = registry.prepare(
        "unit_contract_ops.limit_prediction",
        source_units={"mass_GeV": "TeV"},
        input_units={"mass_GeV": "TeV"},
        output_units={"/value": "fb"},
        strict=True,
    )
    synchronous = plan.run(mass_GeV=0.2)
    asynchronous = asyncio.run(plan.arun(mass_GeV=0.2))
    # A host can pass point-specific source metadata separately from physical kwargs.
    dynamic = plan.run_inputs((), {"mass_GeV": 0.2}, source_units={"mass_GeV": "TeV"})
    dynamic_async = asyncio.run(plan.arun_inputs(
        (), {"mass_GeV": 0.2}, source_units={"mass_GeV": "TeV"},
    ))
    assert synchronous.normalized_inputs["mass_GeV"] == 200.0
    assert synchronous.canonical_result["value"] == 0.038332
    assert synchronous.canonical_result["unit"] == "pb"
    assert synchronous.canonical_result["record_id"] == "demo-limit-canonical-pb"
    assert abs(synchronous.values["/value"] - 38.332) < 1e-12
    assert synchronous.values == asynchronous.values
    assert synchronous.values == dynamic.values == dynamic_async.values
    print(json.dumps({
        "source_units": {"mass_GeV": "TeV"},
        "normalized_inputs": synchronous.normalized_inputs,
        "canonical_result": synchronous.canonical_result,
        "values": synchronous.values,
        "units": synchronous.units,
        "audit": synchronous.audit,
    }, indent=2))


if __name__ == "__main__":
    main()
