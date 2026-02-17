from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import Executor
from dataclasses import dataclass
from functools import partial
from threading import RLock
from typing import Any, Callable

from .errors import OperatorCallError, OperatorConflict, OperatorNotFound
from .logging import get_logger


@dataclass(frozen=True)
class OperatorRecord:
    namespace: str
    name: str
    full_name: str
    fn: Callable[..., Any]
    metadata: dict[str, Any]
    signature: str
    accepts_logger: bool
    accepts_var_kwargs: bool
    is_async: bool


def _normalize_namespace(namespace: str) -> str:
    normalized = namespace.strip()
    if not normalized:
        raise ValueError("namespace cannot be empty")
    return normalized


def _normalize_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise ValueError("operator name cannot be empty")
    if ":" in normalized:
        raise ValueError("operator name cannot contain ':'")
    return normalized


def _derive_call_traits(fn: Callable[..., Any]) -> tuple[str, bool, bool]:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return "()", False, False

    params = signature.parameters
    accepts_logger = "logger" in params
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    return str(signature), accepts_logger, accepts_var_kwargs


def _is_async_callable(fn: Callable[..., Any]) -> bool:
    if inspect.iscoroutinefunction(fn):
        return True
    call_method = getattr(fn, "__call__", None)
    return inspect.iscoroutinefunction(call_method)


class OperatorRegistry:
    """Registry for named operators, with sync and asyncio-friendly execution."""

    def __init__(self, *, logger: Any | None = None, executor: Executor | None = None) -> None:
        self._operators: dict[str, OperatorRecord] = {}
        self._logger = logger
        self._executor = executor
        self._lock = RLock()

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        namespace: str = "core",
        metadata: dict[str, Any] | None = None,
        logger: Any | None = None,
    ) -> None:
        if not callable(fn):
            raise TypeError(f"'{name}' is not callable")

        normalized_namespace = _normalize_namespace(namespace)
        normalized_name = _normalize_name(name)
        full_name = f"{normalized_namespace}:{normalized_name}"

        with self._lock:
            if full_name in self._operators:
                raise OperatorConflict(full_name, "duplicate registration in same namespace")

            core_key = f"core:{normalized_name}"
            user_key = f"user:{normalized_name}"
            if normalized_namespace == "user" and core_key in self._operators:
                raise OperatorConflict(
                    full_name,
                    f"core operator '{core_key}' cannot be overridden by user namespace",
                )
            if normalized_namespace == "core" and user_key in self._operators:
                raise OperatorConflict(
                    full_name,
                    f"user operator '{user_key}' already exists with the same name",
                )

            signature, accepts_logger, accepts_var_kwargs = _derive_call_traits(fn)
            self._operators[full_name] = OperatorRecord(
                namespace=normalized_namespace,
                name=normalized_name,
                full_name=full_name,
                fn=fn,
                metadata=dict(metadata or {}),
                signature=signature,
                accepts_logger=accepts_logger,
                accepts_var_kwargs=accepts_var_kwargs,
                is_async=_is_async_callable(fn),
            )

        local_logger = get_logger(
            logger or self._logger,
            operator=full_name,
            action="register",
        )
        local_logger.debug("registered operator")

    def get(self, full_name: str) -> Callable[..., Any]:
        return self._get_record(full_name).fn

    def call(self, full_name: str, logger: Any | None = None, **kwargs: Any) -> Any:
        record = self._get_record(full_name)
        call_logger = get_logger(
            logger or self._logger,
            operator=full_name,
            action="call",
        )
        call_kwargs = self._build_call_kwargs(record, kwargs, call_logger)

        try:
            result = record.fn(**call_kwargs)
        except OperatorCallError:
            raise
        except Exception as exc:
            raise OperatorCallError(
                full_name,
                f"Operator '{full_name}' failed during sync call: {exc}",
            ) from exc

        if inspect.isawaitable(result):
            raise OperatorCallError(
                full_name,
                f"Operator '{full_name}' returned an awaitable in sync call; use acall().",
            )

        return result

    async def acall(self, full_name: str, logger: Any | None = None, **kwargs: Any) -> Any:
        record = self._get_record(full_name)
        call_logger = get_logger(
            logger or self._logger,
            operator=full_name,
            action="acall",
        )
        call_kwargs = self._build_call_kwargs(record, kwargs, call_logger)

        try:
            if record.is_async:
                return await record.fn(**call_kwargs)

            if self._executor is None:
                return await asyncio.to_thread(record.fn, **call_kwargs)

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                partial(record.fn, **call_kwargs),
            )
        except OperatorCallError:
            raise
        except Exception as exc:
            raise OperatorCallError(
                full_name,
                f"Operator '{full_name}' failed during async call: {exc}",
            ) from exc

    def list(self, namespace: str | None = None) -> list[str]:
        with self._lock:
            names = sorted(self._operators.keys())

        if namespace is None:
            return names

        normalized_namespace = _normalize_namespace(namespace)
        prefix = f"{normalized_namespace}:"
        return [name for name in names if name.startswith(prefix)]

    def info(self, full_name: str) -> dict[str, Any]:
        record = self._get_record(full_name)
        docstring = inspect.getdoc(record.fn) or ""
        doc_summary = docstring.splitlines()[0] if docstring else ""

        return {
            "name": record.full_name,
            "namespace": record.namespace,
            "short_name": record.name,
            "metadata": dict(record.metadata),
            "signature": record.signature,
            "docstring": doc_summary,
            "module": getattr(record.fn, "__module__", ""),
            "qualname": getattr(record.fn, "__qualname__", ""),
            "is_async": record.is_async,
        }

    def _get_record(self, full_name: str) -> OperatorRecord:
        with self._lock:
            record = self._operators.get(full_name)

        if record is None:
            raise OperatorNotFound(full_name)
        return record

    @staticmethod
    def _build_call_kwargs(
        record: OperatorRecord,
        kwargs: dict[str, Any],
        logger: Any,
    ) -> dict[str, Any]:
        call_kwargs = dict(kwargs)
        if "logger" not in call_kwargs and (record.accepts_logger or record.accepts_var_kwargs):
            call_kwargs["logger"] = logger
        return call_kwargs
