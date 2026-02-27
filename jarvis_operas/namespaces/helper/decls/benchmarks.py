from __future__ import annotations

from ....core.spec import OperaFunction
from ..defs.benchmarks import (
    eggbox2d_numpy,
    eggbox2d_polars_expr,
    eggbox_numpy,
    eggbox_polars_expr,
)


HELPER_BENCHMARK_DECLARATIONS: tuple[OperaFunction, ...] = (
    OperaFunction(
        namespace="helper",
        name="eggbox",
        arity=2,
        return_dtype=None,
        numpy_impl=eggbox_numpy,
        polars_expr_impl=eggbox_polars_expr,
        metadata={
            "category": "hep_scanner_benchmark",
            "summary": "Evaluate EggBox benchmark surface z(x, y).",
            "params": {
                "x": "First axis values.",
                "y": "Second axis values.",
            },
            "return": "EggBox response array or scalar.",
            "backend_support": {
                "numpy": "native",
                "polars": "native_expr",
            },
            "examples": [
                "jopera call helper.eggbox --kwargs '{\"x\":0.5,\"y\":0.0}'",
            ],
            "since": "1.2.0",
            "tags": ["helper", "benchmark", "eggbox"],
        },
    ),
    OperaFunction(
        namespace="helper",
        name="eggbox2d",
        arity=2,
        return_dtype=None,
        numpy_impl=eggbox2d_numpy,
        polars_expr_impl=eggbox2d_polars_expr,
        metadata={
            "category": "hep_scanner_benchmark",
            "summary": "Evaluate EggBox and return mapping payload {'z': value}.",
            "params": {
                "x": "First axis values.",
                "y": "Second axis values.",
            },
            "return": "Dictionary payload with key 'z'.",
            "backend_support": {
                "numpy": "native",
                "polars": "native_expr",
            },
            "examples": [
                "jopera call helper.eggbox2d --kwargs '{\"x\":0.5,\"y\":0.0}'",
            ],
            "since": "1.2.0",
            "tags": ["helper", "benchmark", "eggbox"],
        },
    ),
)
