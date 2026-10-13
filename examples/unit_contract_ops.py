"""Dependency-free physical record fixture, loadable with --user-ops."""
from __future__ import annotations

from jarvis_operas import oper


@oper(
    "limit_prediction",
    summary="Example limit record in pb; this fixture does not evaluate HiggsTools.",
    params={"mass_GeV": "Scalar resonance mass in the contract's canonical unit."},
    unit_contract={
        "version": "1.0",
        "inputs": {
            "mass_GeV": {"unit": "GeV", "shape": "scalar", "nullable": False},
        },
        "outputs": {
            "/value": {"unit": "pb", "shape": "scalar", "nullable": True},
        },
    },
)
def limit_prediction(mass_GeV):
    """Return an illustrative record with immutable identity and provenance."""
    return {
        "value": 0.038332,
        "unit": "pb",
        "status": "ok",
        "record_id": "demo-limit-canonical-pb",
        "provenance": {"source": "unit-contract-fixture", "mass_GeV": mass_GeV},
        "dependencies": ["demo-analysis"],
    }
