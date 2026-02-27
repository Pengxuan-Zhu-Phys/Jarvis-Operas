from __future__ import annotations

from ....core.spec import OperaFunction
from ..defs.basic import add_numpy, add_polars_expr, identity_numpy, identity_polars_expr


MATH_BASIC_DECLARATIONS: tuple[OperaFunction, ...] = (
    OperaFunction(
        namespace="math",
        name="add",
        arity=2,
        return_dtype=None,
        numpy_impl=add_numpy,
        polars_expr_impl=add_polars_expr,
        metadata={
            "category": "math",
            "summary": "Element-wise addition for scalar or array-like inputs.",
            "params": {
                "a": "Left operand (scalar/array-like).",
                "b": "Right operand (scalar/array-like).",
            },
            "return": "Sum of a and b with broadcast semantics from backend.",
            "backend_support": {
                "numpy": "native",
                "polars": "native_expr",
            },
            "examples": [
                "jopera call math.add --kwargs '{\"a\":1,\"b\":2}'",
            ],
            "since": "1.2.0",
            "tags": ["math", "arithmetic"],
        },
    ),
    OperaFunction(
        namespace="math",
        name="identity",
        arity=1,
        return_dtype=None,
        numpy_impl=identity_numpy,
        polars_expr_impl=identity_polars_expr,
        metadata={
            "category": "math",
            "summary": "Return the input unchanged.",
            "params": {
                "x": "Input value (scalar/array-like/expression).",
            },
            "return": "The same value as x.",
            "backend_support": {
                "numpy": "native",
                "polars": "native_expr",
            },
            "examples": [
                "jopera call math.identity --kwargs '{\"x\":3.14}'",
            ],
            "since": "1.2.0",
            "tags": ["math", "identity", "utility"],
        },
    ),
)
