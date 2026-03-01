from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor
from functools import partial
from threading import RLock
from typing import Any
from weakref import WeakKeyDictionary

from ..errors import OperatorCallError, OperatorConflict, OperatorNotFound
from ..logging import get_logger
from ..name_utils import (
    normalize_full_name as normalize_common_full_name,
    normalize_namespace as normalize_common_namespace,
    normalize_short_name as normalize_common_short_name,
)
from .spec import OperaFunction

_ID_MIN_NUMBER = 100
_ID_MAX_NUMBER = 99999
_ID_SPACE_SIZE = _ID_MAX_NUMBER - _ID_MIN_NUMBER + 1


def recommended_numpy_concurrency(*, cpu_count: int | None = None) -> int:
    """Return default numpy concurrency based on CPU count."""

    detected = cpu_count if cpu_count is not None else os.cpu_count()
    if detected is None or detected < 1:
        detected = 1
    return detected * 3


_DEFAULT_NUMPY_CONCURRENCY = recommended_numpy_concurrency()


def _normalize_namespace(namespace: str) -> str:
    return normalize_common_namespace(namespace)


def _normalize_name(name: str) -> str:
    return normalize_common_short_name(name, field_label="function name")


def _normalize_full_name(full_name: str) -> str:
    return normalize_common_full_name(full_name)


def _canonicalize_full_name_candidate(identifier: str) -> str | None:
    try:
        return _normalize_full_name(identifier)
    except ValueError:
        return None


def _namespace_id_prefix(namespace: str) -> str:
    for ch in namespace.strip().lower():
        if "a" <= ch <= "z":
            return ch
    return "o"


def _build_support_capabilities(declaration: OperaFunction) -> dict[str, bool]:
    supports_polars = declaration.supports_polars_native or declaration.supports_polars_fallback
    return {
        "call": True,
        "acall": True,
        "numpy": declaration.supports_numpy,
        "polars": supports_polars,
        "polars_native": declaration.supports_polars_native,
        "polars_fallback": declaration.supports_polars_fallback,
    }


def _supported_types_list(capabilities: Mapping[str, bool]) -> list[str]:
    ordered = ("call", "acall", "numpy", "polars")
    return [name for name in ordered if bool(capabilities.get(name))]


class OperasRegistry:
    """Registry that stores and dispatches unified OperaFunction declarations."""

    def __init__(
        self,
        *,
        logger: Any | None = None,
        log_mode: str | None = None,
        executor: Executor | None = None,
        numpy_concurrency: int | None = None,
    ) -> None:
        effective_concurrency = (
            _DEFAULT_NUMPY_CONCURRENCY if numpy_concurrency is None else numpy_concurrency
        )
        if effective_concurrency < 1:
            raise ValueError("numpy_concurrency must be >= 1")
        self._functions: dict[str, OperaFunction] = {}
        self._id_to_name: dict[str, str] = {}
        self._name_to_id: dict[str, str] = {}
        self._logger = logger
        self._log_mode = log_mode
        self._executor = executor
        self._numpy_concurrency = effective_concurrency
        self._numpy_semaphores: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            WeakKeyDictionary()
        )
        self._lock = RLock()

    def set_log_mode(self, mode: str | None) -> None:
        self._log_mode = mode

    def set_executor(self, executor: Executor | None) -> None:
        with self._lock:
            self._executor = executor

    def set_numpy_concurrency(self, numpy_concurrency: int) -> None:
        if numpy_concurrency < 1:
            raise ValueError("numpy_concurrency must be >= 1")
        with self._lock:
            self._numpy_concurrency = numpy_concurrency
            self._numpy_semaphores.clear()

    def get_numpy_concurrency(self) -> int:
        with self._lock:
            return self._numpy_concurrency

    def register(
        self,
        declaration: OperaFunction,
        *,
        replace: bool = False,
        logger: Any | None = None,
    ) -> None:
        full_name = declaration.full_name

        merged_metadata = dict(declaration.metadata)
        support_caps = _build_support_capabilities(declaration)
        merged_metadata["supports"] = support_caps
        merged_metadata["supported_types"] = _supported_types_list(support_caps)
        if declaration.namespace == "helper":
            merged_metadata.setdefault("concurrent", True)
        if merged_metadata != dict(declaration.metadata):
            declaration = OperaFunction(
                namespace=declaration.namespace,
                name=declaration.name,
                arity=declaration.arity,
                return_dtype=declaration.return_dtype,
                numpy_impl=declaration.numpy_impl,
                polars_expr_impl=declaration.polars_expr_impl,
                flags=declaration.flags,
                metadata=merged_metadata,
            )

        with self._lock:
            existing = self._functions.get(full_name)
            if existing is not None and not replace:
                raise OperatorConflict(full_name, "duplicate registration in same namespace")

            if existing is not None:
                operator_id = self._name_to_id[full_name]
            else:
                operator_id = self._allocate_operator_id_locked(declaration.namespace, full_name)

            self._functions[full_name] = declaration
            self._id_to_name[operator_id] = full_name
            self._name_to_id[full_name] = operator_id

        local_logger = get_logger(
            logger or self._logger,
            mode=self._log_mode,
            operator=full_name,
            action="register_function",
        )
        local_logger.debug("registered function declaration")

    def register_many(
        self,
        declarations: list[OperaFunction] | tuple[OperaFunction, ...],
        *,
        replace: bool = False,
        logger: Any | None = None,
    ) -> list[str]:
        registered: list[str] = []
        for declaration in declarations:
            self.register(declaration, replace=replace, logger=logger)
            registered.append(declaration.full_name)
        return sorted(registered)

    def resolve_name(self, identifier: str) -> str:
        normalized = identifier.strip()
        if not normalized:
            raise OperatorNotFound(identifier)

        with self._lock:
            if normalized in self._functions:
                return normalized

            canonical = _canonicalize_full_name_candidate(normalized)
            if canonical is not None and canonical in self._functions:
                return canonical

            by_id = self._id_to_name.get(normalized)
            if by_id is not None and by_id in self._functions:
                return by_id

        raise OperatorNotFound(identifier)

    def get(self, full_name_or_id: str) -> OperaFunction:
        resolved = self.resolve_name(full_name_or_id)
        with self._lock:
            declaration = self._functions.get(resolved)
        if declaration is None:
            raise OperatorNotFound(full_name_or_id)
        return declaration

    def get_id(self, full_name_or_id: str) -> str:
        resolved = self.resolve_name(full_name_or_id)
        with self._lock:
            operator_id = self._name_to_id.get(resolved)
        if operator_id is None:
            raise OperatorNotFound(full_name_or_id)
        return operator_id

    def list(self, namespace: str | None = None) -> list[str]:
        with self._lock:
            names = list(self._functions.keys())

        if namespace is None:
            return sorted(names)

        normalized_namespace = _normalize_namespace(namespace)
        prefix = f"{normalized_namespace}."
        return sorted(name for name in names if name.startswith(prefix))

    def info(self, full_name_or_id: str) -> dict[str, Any]:
        declaration = self.get(full_name_or_id)
        full_name = declaration.full_name
        operator_id = self.get_id(full_name)
        docstring = inspect.getdoc(declaration.numpy_impl or declaration.polars_expr_impl) or ""
        doc_summary = docstring.splitlines()[0] if docstring else ""
        metadata = dict(declaration.metadata)
        metadata_summary = metadata.get("summary")
        metadata_note = metadata.get("note")
        if isinstance(metadata_summary, str) and metadata_summary.strip():
            summary = metadata_summary.strip()
        elif isinstance(metadata_note, str) and metadata_note.strip():
            summary = metadata_note.strip()
        else:
            summary = doc_summary

        raw_caps = metadata.get("supports")
        if isinstance(raw_caps, Mapping):
            capabilities = {str(k): bool(v) for k, v in raw_caps.items()}
        else:
            capabilities = _build_support_capabilities(declaration)
        raw_supported_types = metadata.get("supported_types")
        if isinstance(raw_supported_types, list) and all(
            isinstance(item, str) and item.strip() for item in raw_supported_types
        ):
            supported_types = [item.strip() for item in raw_supported_types]
        else:
            supported_types = _supported_types_list(capabilities)

        return {
            "id": operator_id,
            "name": declaration.full_name,
            "namespace": declaration.namespace,
            "short_name": declaration.name,
            "arity": declaration.arity,
            "return_dtype": repr(declaration.return_dtype),
            "supports_numpy": declaration.supports_numpy,
            "supports_polars_native": declaration.supports_polars_native,
            "supports_polars_fallback": declaration.supports_polars_fallback,
            "metadata": metadata,
            "signature": self._build_signature(declaration),
            "docstring": summary,
            "is_async": declaration.is_numpy_async(),
            "supports_async": True,
            "supports_call": bool(capabilities.get("call", True)),
            "supports_acall": bool(capabilities.get("acall", True)),
            "capabilities": capabilities,
            "supported_types": supported_types,
        }

    def call(
        self,
        full_name_or_id: str,
        *args: Any,
        backend: str = "numpy",
        logger: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        declaration = self.get(full_name_or_id)

        try:
            result = declaration.dispatch(*args, backend=backend, **kwargs)
        except OperatorCallError:
            raise
        except Exception as exc:
            raise OperatorCallError(
                declaration.full_name,
                f"Operator '{declaration.full_name}' failed on backend='{backend}': {exc}",
            ) from exc

        if inspect.isawaitable(result):
            raise OperatorCallError(
                declaration.full_name,
                f"Operator '{declaration.full_name}' returned an awaitable in sync call; use acall().",
            )

        local_logger = get_logger(
            logger or self._logger,
            mode=self._log_mode,
            operator=declaration.full_name,
            action="call_function",
            backend=backend,
        )
        local_logger.debug("dispatched call")
        return result

    async def acall(
        self,
        full_name_or_id: str,
        *args: Any,
        backend: str = "numpy",
        logger: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        declaration = self.get(full_name_or_id)

        if backend == "polars":
            return self.call(
                declaration.full_name,
                *args,
                backend=backend,
                logger=logger,
                **kwargs,
            )

        semaphore = self._get_numpy_semaphore_for_running_loop()
        async with semaphore:
            if declaration.is_numpy_async():
                try:
                    return await declaration.dispatch(*args, backend=backend, **kwargs)
                except OperatorCallError:
                    raise
                except Exception as exc:
                    raise OperatorCallError(
                        declaration.full_name,
                        f"Operator '{declaration.full_name}' failed during async call: {exc}",
                    ) from exc

            if self._executor is None:
                return await asyncio.to_thread(
                    self.call,
                    declaration.full_name,
                    *args,
                    backend=backend,
                    logger=logger,
                    **kwargs,
                )

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                partial(
                    self.call,
                    declaration.full_name,
                    *args,
                    backend=backend,
                    logger=logger,
                    **kwargs,
                ),
            )

    async def acall_many(
        self,
        full_name_or_id: str,
        calls: Sequence[Mapping[str, Any]],
        *,
        backend: str = "numpy",
        logger: Any | None = None,
        concurrency: int | None = None,
        return_exceptions: bool = False,
    ) -> list[Any]:
        if concurrency is not None and concurrency < 1:
            raise ValueError("concurrency must be >= 1 or None")

        resolved_name = self.resolve_name(full_name_or_id)
        call_items = list(calls)
        if not call_items:
            return []

        effective_concurrency = concurrency
        if effective_concurrency is None and backend == "numpy":
            effective_concurrency = self._numpy_concurrency

        semaphore = (
            asyncio.Semaphore(effective_concurrency)
            if effective_concurrency is not None
            else None
        )

        async def _run_one(index: int, raw_kwargs: Mapping[str, Any]) -> Any:
            if not isinstance(raw_kwargs, Mapping):
                raise TypeError(f"calls[{index}] must be a mapping of kwargs")
            payload = dict(raw_kwargs)
            if semaphore is None:
                return await self.acall(
                    resolved_name,
                    backend=backend,
                    logger=logger,
                    **payload,
                )
            async with semaphore:
                return await self.acall(
                    resolved_name,
                    backend=backend,
                    logger=logger,
                    **payload,
                )

        coroutines = [_run_one(index, item) for index, item in enumerate(call_items)]
        results = await asyncio.gather(*coroutines, return_exceptions=return_exceptions)
        return list(results)

    async def acall_helper_many(
        self,
        name: str,
        calls: Sequence[Mapping[str, Any]],
        *,
        backend: str = "numpy",
        logger: Any | None = None,
        concurrency: int | None = None,
        return_exceptions: bool = False,
    ) -> list[Any]:
        normalized_name = _normalize_name(name)
        return await self.acall_many(
            f"helper.{normalized_name}",
            calls,
            backend=backend,
            logger=logger,
            concurrency=concurrency,
            return_exceptions=return_exceptions,
        )

    def delete(self, full_name_or_id: str, logger: Any | None = None) -> None:
        resolved = self.resolve_name(full_name_or_id)
        with self._lock:
            declaration = self._functions.pop(resolved, None)
            operator_id = self._name_to_id.pop(resolved, None)
            if operator_id is not None:
                self._id_to_name.pop(operator_id, None)

        if declaration is None:
            raise OperatorNotFound(full_name_or_id)

        local_logger = get_logger(
            logger or self._logger,
            mode=self._log_mode,
            operator=resolved,
            action="delete_function",
        )
        local_logger.debug("deleted function declaration")

    def rename(
        self,
        full_name_or_id: str,
        *,
        new_full_name: str | None = None,
        new_name: str | None = None,
        new_namespace: str | None = None,
        logger: Any | None = None,
    ) -> str:
        source_full = self.resolve_name(full_name_or_id)
        source_decl = self.get(source_full)
        if new_full_name is not None:
            target_full = _normalize_full_name(new_full_name)
            target_namespace, target_name = target_full.split(".", 1)
        else:
            target_namespace = (
                _normalize_namespace(new_namespace)
                if new_namespace is not None
                else source_decl.namespace
            )
            target_name = _normalize_name(new_name) if new_name is not None else source_decl.name
            target_full = f"{target_namespace}.{target_name}"

        if target_full == source_full:
            return source_full

        with self._lock:
            if target_full in self._functions and target_full != source_full:
                raise OperatorConflict(target_full, "duplicate registration in same namespace")

            operator_id = self._name_to_id.get(source_full)
            if operator_id is None:
                raise OperatorNotFound(source_full)

            target_decl = OperaFunction(
                namespace=target_namespace,
                name=target_name,
                arity=source_decl.arity,
                return_dtype=source_decl.return_dtype,
                numpy_impl=source_decl.numpy_impl,
                polars_expr_impl=source_decl.polars_expr_impl,
                flags=source_decl.flags,
                metadata=dict(source_decl.metadata),
            )
            self._functions[target_full] = target_decl
            self._functions.pop(source_full, None)
            self._name_to_id[target_full] = operator_id
            self._name_to_id.pop(source_full, None)
            self._id_to_name[operator_id] = target_full

        local_logger = get_logger(
            logger or self._logger,
            mode=self._log_mode,
            operator=source_full,
            action="rename_function",
        )
        local_logger.debug("renamed function to {}", target_full)
        return target_full

    def delete_namespace(self, namespace: str, logger: Any | None = None) -> list[str]:
        normalized_namespace = _normalize_namespace(namespace)
        prefix = f"{normalized_namespace}."

        with self._lock:
            deleted = [name for name in self._functions if name.startswith(prefix)]
            for name in deleted:
                operator_id = self._name_to_id.pop(name, None)
                if operator_id is not None:
                    self._id_to_name.pop(operator_id, None)
                self._functions.pop(name, None)

        local_logger = get_logger(
            logger or self._logger,
            mode=self._log_mode,
            namespace=normalized_namespace,
            action="delete_namespace",
        )
        local_logger.debug("deleted {} functions in namespace", len(deleted))
        return sorted(deleted)

    def rename_namespace(
        self,
        namespace: str,
        new_namespace: str,
        logger: Any | None = None,
    ) -> list[str]:
        source_namespace = _normalize_namespace(namespace)
        target_namespace = _normalize_namespace(new_namespace)
        if source_namespace == target_namespace:
            return []

        source_prefix = f"{source_namespace}."
        with self._lock:
            source_names = [
                full_name for full_name in self._functions.keys() if full_name.startswith(source_prefix)
            ]
            rename_pairs: list[tuple[str, str]] = []
            for source_full in source_names:
                decl = self._functions[source_full]
                target_full = f"{target_namespace}.{decl.name}"
                rename_pairs.append((source_full, target_full))

            for source_full, target_full in rename_pairs:
                if target_full in self._functions and target_full not in source_names:
                    raise OperatorConflict(target_full, "duplicate registration in same namespace")

            updated_names: list[str] = []
            for source_full, target_full in rename_pairs:
                decl = self._functions[source_full]
                operator_id = self._name_to_id.get(source_full)
                if operator_id is None:
                    continue
                target_decl = OperaFunction(
                    namespace=target_namespace,
                    name=decl.name,
                    arity=decl.arity,
                    return_dtype=decl.return_dtype,
                    numpy_impl=decl.numpy_impl,
                    polars_expr_impl=decl.polars_expr_impl,
                    flags=decl.flags,
                    metadata=dict(decl.metadata),
                )
                self._functions[target_full] = target_decl
                self._functions.pop(source_full, None)
                self._name_to_id[target_full] = operator_id
                self._name_to_id.pop(source_full, None)
                self._id_to_name[operator_id] = target_full
                updated_names.append(target_full)

        local_logger = get_logger(
            logger or self._logger,
            mode=self._log_mode,
            namespace=source_namespace,
            action="rename_namespace",
        )
        local_logger.debug(
            "renamed namespace '{}' -> '{}' for {} functions",
            source_namespace,
            target_namespace,
            len(updated_names),
        )
        return sorted(updated_names)

    def _get_numpy_semaphore_for_running_loop(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        with self._lock:
            semaphore = self._numpy_semaphores.get(loop)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self._numpy_concurrency)
                self._numpy_semaphores[loop] = semaphore
        return semaphore

    def _allocate_operator_id_locked(self, namespace: str, full_name: str) -> str:
        prefix = _namespace_id_prefix(namespace)
        base = int.from_bytes(
            hashlib.blake2b(full_name.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for attempt in range(_ID_SPACE_SIZE):
            number = _ID_MIN_NUMBER + ((base + attempt) % _ID_SPACE_SIZE)
            candidate = f"{prefix}{number}"
            mapped_name = self._id_to_name.get(candidate)
            if mapped_name is None or mapped_name == full_name:
                return candidate
        raise OperatorConflict(
            full_name,
            f"operator id space exhausted for prefix '{prefix}'",
        )

    @staticmethod
    def _build_signature(declaration: OperaFunction) -> str:
        fn = declaration.polars_expr_impl or declaration.numpy_impl
        if fn is None:
            return "()"
        try:
            return str(inspect.signature(fn))
        except (TypeError, ValueError):
            return "()"
