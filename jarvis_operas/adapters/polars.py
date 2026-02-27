from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

import numpy as np

from ..errors import OperatorCallError

if TYPE_CHECKING:
    from ..core.spec import OperaFunction


def _require_polars(operator_name: str):
    try:
        import polars as pl  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OperatorCallError(
            operator_name,
            "Polars backend requires 'polars' package. Install it to use backend='polars'.",
        ) from exc
    return pl


def _ensure_expr_args(operator_name: str, args: tuple[Any, ...], pl: Any) -> tuple[Any, ...]:
    for index, arg in enumerate(args):
        if not isinstance(arg, pl.Expr):
            raise OperatorCallError(
                operator_name,
                "Polars backend expects positional arguments as pl.Expr. "
                f"arg[{index}] has type {type(arg).__name__}.",
            )
    return args


def _numpy_output_to_series(
    *,
    operator_name: str,
    value: Any,
    length: int,
    return_dtype: Any,
    pl: Any,
) -> Any:
    if isinstance(value, pl.Series):
        if len(value) != length:
            raise OperatorCallError(
                operator_name,
                "Polars fallback UDF returned Series with wrong length "
                f"(expected {length}, got {len(value)}).",
            )
        return value.cast(return_dtype)

    array = np.asarray(value)
    if array.ndim == 0:
        raise OperatorCallError(
            operator_name,
            "Polars fallback UDF requires vectorized numpy output; scalar output is not supported.",
        )

    flat = np.ravel(array)
    if flat.shape[0] != length:
        raise OperatorCallError(
            operator_name,
            "Polars fallback UDF returned wrong output length "
            f"(expected {length}, got {flat.shape[0]}).",
        )

    return pl.Series(flat).cast(return_dtype)


def _build_struct_fallback_expr(
    *,
    operator_name: str,
    args: tuple[Any, ...],
    numpy_impl: Any,
    return_dtype: Any,
    kwargs: dict[str, Any],
    pl: Any,
) -> Any:
    aliases = [f"__jo_arg_{index}" for index in range(len(args))]
    aliased_exprs = [expr.alias(alias) for expr, alias in zip(args, aliases)]
    struct_expr = pl.struct(aliased_exprs)

    def _udf(batch: Any) -> Any:
        if not isinstance(batch, pl.Series):
            raise OperatorCallError(
                operator_name,
                f"Unexpected batch payload type for polars fallback: {type(batch).__name__}",
            )

        np_args = []
        for alias in aliases:
            np_args.append(batch.struct.field(alias).to_numpy())

        result = numpy_impl(*np_args, **kwargs)
        return _numpy_output_to_series(
            operator_name=operator_name,
            value=result,
            length=len(batch),
            return_dtype=return_dtype,
            pl=pl,
        )

    try:
        return struct_expr.map_batches(_udf, return_dtype=return_dtype)
    except TypeError:
        # Older polars versions may require positional args.
        return struct_expr.map_batches(_udf, return_dtype)


def dispatch_polars_expr(
    *,
    operator_name: str,
    declaration: "OperaFunction",
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    pl = _require_polars(operator_name)
    expr_args = _ensure_expr_args(operator_name, args, pl)

    if declaration.polars_expr_impl is not None:
        return declaration.polars_expr_impl(*expr_args, **kwargs)

    if declaration.numpy_impl is None:
        raise OperatorCallError(
            operator_name,
            f"Operator '{operator_name}' has no numpy_impl for polars fallback.",
        )
    if declaration.return_dtype is None:
        raise OperatorCallError(
            operator_name,
            "Polars fallback requires OperaFunction.return_dtype to be set.",
        )
    if isinstance(declaration.return_dtype, str):
        raise OperatorCallError(
            operator_name,
            "Polars fallback return_dtype must be a Polars DataType object, not string.",
        )
    if not isinstance(expr_args, Sequence) or not expr_args:
        raise OperatorCallError(
            operator_name,
            "Polars backend requires at least one pl.Expr positional argument.",
        )

    return _build_struct_fallback_expr(
        operator_name=operator_name,
        args=expr_args,
        numpy_impl=declaration.numpy_impl,
        return_dtype=declaration.return_dtype,
        kwargs=kwargs,
        pl=pl,
    )
