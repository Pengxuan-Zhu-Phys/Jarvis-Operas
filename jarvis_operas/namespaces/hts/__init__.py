"""Optional HiggsTools evaluator namespace.

The public operator namespace is ``HTs``.  The lower-case package name follows
the layout used by the other bundled Jarvis-Operas namespaces.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...core.spec import OperaFunction
from .decls import HTS_DECLARATIONS
from .defs import (
    build_higgstools_predictions,
    clear_higgstools_backend_cache,
    configure_default_dataset_paths,
    dataset_environment_candidates,
    evaluate_bounds,
    evaluate_numpy,
    evaluate_signals,
    flatten_evaluation_result,
    get_higgstools_backend,
    import_higgstools_package,
    inspect_dataset_resource,
    make_diagnostic,
    make_evaluation_result,
    normalize_bounds_result,
    normalize_selected_limit,
    normalize_signals_result,
    parse_prediction_envelope,
    resolve_dataset_resources,
    to_json_safe,
    validate_prediction_contract,
)

DECLARATIONS: tuple[OperaFunction, ...] = HTS_DECLARATIONS


def get_declarations() -> Sequence[OperaFunction]:
    return DECLARATIONS


__all__ = [
    "DECLARATIONS", "get_declarations", "build_higgstools_predictions",
    "clear_higgstools_backend_cache", "configure_default_dataset_paths",
    "dataset_environment_candidates", "evaluate_bounds", "evaluate_numpy",
    "evaluate_signals", "flatten_evaluation_result", "get_higgstools_backend",
    "import_higgstools_package", "inspect_dataset_resource", "make_diagnostic",
    "make_evaluation_result", "normalize_bounds_result", "normalize_selected_limit",
    "normalize_signals_result", "parse_prediction_envelope", "resolve_dataset_resources",
    "to_json_safe", "validate_prediction_contract",
]
