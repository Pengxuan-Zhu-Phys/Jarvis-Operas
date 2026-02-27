from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..errors import OperatorCallError
from ..name_utils import (
    normalize_namespace,
    normalize_short_name,
)


def _normalize_arity(arity: int | None) -> int | None:
    if arity is None:
        return None
    if not isinstance(arity, int):
        raise TypeError("OperaFunction.arity must be int or None")
    if arity < 0:
        raise ValueError("OperaFunction.arity must be >= 0")
    return arity


@dataclass(frozen=True)
class OperaFunction:
    """Unified declaration class for one operas function."""

    namespace: str
    name: str
    arity: int | None
    return_dtype: Any | None
    numpy_impl: Callable[..., Any] | None
    polars_expr_impl: Callable[..., Any] | None = None
    flags: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", normalize_namespace(self.namespace))
        object.__setattr__(
            self,
            "name",
            normalize_short_name(self.name, field_label="OperaFunction.name"),
        )
        object.__setattr__(self, "arity", _normalize_arity(self.arity))

        if self.numpy_impl is not None and not callable(self.numpy_impl):
            raise TypeError("OperaFunction.numpy_impl must be callable or None")
        if self.polars_expr_impl is not None and not callable(self.polars_expr_impl):
            raise TypeError("OperaFunction.polars_expr_impl must be callable or None")
        if self.numpy_impl is None and self.polars_expr_impl is None:
            raise ValueError("OperaFunction requires at least one implementation")

        if isinstance(self.flags, frozenset):
            normalized_flags = self.flags
        else:
            normalized_flags = frozenset(str(flag) for flag in self.flags)
        object.__setattr__(self, "flags", normalized_flags)

        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))

    @property
    def full_name(self) -> str:
        return f"{self.namespace}.{self.name}"

    @property
    def supports_numpy(self) -> bool:
        return self.numpy_impl is not None

    @property
    def supports_polars_native(self) -> bool:
        return self.polars_expr_impl is not None

    @property
    def supports_polars_fallback(self) -> bool:
        return self.numpy_impl is not None and self.return_dtype is not None

    def validate_arity(self, args: tuple[Any, ...]) -> None:
        if self.arity is None:
            return
        # Keep kwargs-friendly call style: strict arity check only when positional
        # arguments are used.
        if args and len(args) != self.arity:
            raise OperatorCallError(
                self.full_name,
                f"Operator '{self.full_name}' expects arity={self.arity}, got {len(args)} positional args.",
            )

    def dispatch(
        self,
        *args: Any,
        backend: str = "numpy",
        **kwargs: Any,
    ) -> Any:
        self.validate_arity(args)

        if backend == "numpy":
            if self.numpy_impl is None:
                raise OperatorCallError(
                    self.full_name,
                    f"Operator '{self.full_name}' does not provide numpy_impl.",
                )
            return self.numpy_impl(*args, **kwargs)

        if backend == "polars":
            from ..adapters.polars import dispatch_polars_expr

            return dispatch_polars_expr(
                operator_name=self.full_name,
                declaration=self,
                args=args,
                kwargs=kwargs,
            )

        raise OperatorCallError(
            self.full_name,
            f"Unsupported backend '{backend}'. Use 'numpy' or 'polars'.",
        )

    def is_numpy_async(self) -> bool:
        if self.numpy_impl is None:
            return False
        if inspect.iscoroutinefunction(self.numpy_impl):
            return True
        call_method = getattr(self.numpy_impl, "__call__", None)
        return inspect.iscoroutinefunction(call_method)
