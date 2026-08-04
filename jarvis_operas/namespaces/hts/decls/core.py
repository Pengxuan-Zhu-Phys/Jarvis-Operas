from __future__ import annotations

from ....core.spec import OperaFunction
from ..defs.core import (
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


HTS_DECLARATIONS: tuple[OperaFunction, ...] = (
    OperaFunction(
        namespace="HTs",
        name="evaluate",
        arity=None,
        return_dtype=None,
        numpy_impl=evaluate_numpy,
        polars_expr_impl=None,
        metadata={
            "category": "optional_external_evaluator",
            "summary": "Evaluate a native HiggsTools prediction with configured HB and HS datasets.",
            "params": {
                "payload": "Native prediction mapping, or {'prediction': mapping, 'metadata': mapping}.",
                "hb_dataset_path": "Optional explicit HiggsBounds dataset directory.",
                "hs_dataset_path": "Optional explicit HiggsSignals dataset directory.",
                "observables": "Optional runtime configuration mapping used for dataset resolution.",
                "dataset_defaults": "Optional default dataset paths mapping; used after environment variables.",
                "strict_validation": "Enable conservative validation of the native prediction mapping.",
            },
            "return": "JSON-safe structured evaluation result with status, direct_exclusion, signal_measurements, and provenance.",
            "backend_support": {"numpy": "native", "polars": "unsupported"},
            "examples": [
                "registry.call('HTs.evaluate', payload={'prediction': native_prediction}, "
                "hb_dataset_path='/datasets/HBDataSet', hs_dataset_path='/datasets/HSDataSet')",
            ],
            "since": "1.4.0",
            "tags": ["higgstools", "higgsbounds", "higgssignals", "optional"],
            "note": (
                "HiggsTools and its datasets are optional. Prediction construction is owned by the caller; "
                "this opera only validates, executes, and normalizes the evaluator result."
            ),
        },
    ),
    *tuple(
        OperaFunction(
            namespace="HTs",
            name=name,
            arity=None,
            return_dtype=None,
            numpy_impl=function,
            polars_expr_impl=None,
            metadata={
                "category": "higgstools_public_helper",
                "summary": summary,
                "return": "Public helper result; see jarvis_operas.namespaces.hts API documentation.",
                "backend_support": {"numpy": "native", "polars": "unsupported"},
                "examples": [f"jopera info HTs.{name}"],
                "since": "1.4.0",
                "tags": ["higgstools", "optional", "public-helper"],
            },
        )
        for name, function, summary in (
            ("configure_default_dataset_paths", configure_default_dataset_paths, "Configure process-local optional dataset defaults."),
            ("resolve_dataset_resources", resolve_dataset_resources, "Resolve local HiggsBounds and HiggsSignals datasets."),
            ("get_higgstools_backend", get_higgstools_backend, "Get a process-local cached HiggsTools backend."),
            ("clear_higgstools_backend_cache", clear_higgstools_backend_cache, "Clear the process-local HiggsTools backend cache."),
            ("parse_prediction_envelope", parse_prediction_envelope, "Parse a versioned model-independent prediction envelope."),
            ("validate_prediction_contract", validate_prediction_contract, "Validate generic native prediction input."),
            ("build_higgstools_predictions", build_higgstools_predictions, "Build native HiggsTools prediction objects."),
            ("evaluate_bounds", evaluate_bounds, "Run the HiggsBounds component."),
            ("evaluate_signals", evaluate_signals, "Run the HiggsSignals component."),
            ("normalize_selected_limit", normalize_selected_limit, "Normalize one selected direct-search limit."),
            ("normalize_bounds_result", normalize_bounds_result, "Normalize a HiggsBounds result."),
            ("normalize_signals_result", normalize_signals_result, "Normalize a HiggsSignals result."),
            ("make_evaluation_result", make_evaluation_result, "Build the versioned evaluation-result mapping."),
            ("flatten_evaluation_result", flatten_evaluation_result, "Flatten an evaluation result for scalar-observable consumers."),
            ("import_higgstools_package", import_higgstools_package, "Inspect the optional HiggsTools package."),
            ("inspect_dataset_resource", inspect_dataset_resource, "Inspect an existing local evaluator dataset."),
            ("make_diagnostic", make_diagnostic, "Build a structured diagnostic record."),
            ("to_json_safe", to_json_safe, "Convert an evaluator value to JSON-safe data."),
            ("dataset_environment_candidates", dataset_environment_candidates, "List supported dataset environment variables."),
        )
    ),
)
