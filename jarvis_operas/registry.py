from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from functools import partial
from threading import RLock
from typing import Any, Callable

from .errors import OperatorCallError, OperatorConflict, OperatorNotFound
from .logging import get_logger

_ID_MIN_NUMBER = 100
_ID_MAX_NUMBER = 99999
_ID_SPACE_SIZE = _ID_MAX_NUMBER - _ID_MIN_NUMBER + 1
_FULL_NAME_SEPARATOR = "."
_LEGACY_FULL_NAME_SEPARATOR = ":"


@dataclass(frozen=True)
class OperatorRecord:
    namespace: str
    name: str
    full_name: str
    operator_id: str
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
    if _FULL_NAME_SEPARATOR in normalized:
        raise ValueError(f"namespace cannot contain '{_FULL_NAME_SEPARATOR}'")
    if _LEGACY_FULL_NAME_SEPARATOR in normalized:
        raise ValueError(f"namespace cannot contain '{_LEGACY_FULL_NAME_SEPARATOR}'")
    return normalized


def _normalize_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise ValueError("operator name cannot be empty")
    if _FULL_NAME_SEPARATOR in normalized:
        raise ValueError(f"operator name cannot contain '{_FULL_NAME_SEPARATOR}'")
    if _LEGACY_FULL_NAME_SEPARATOR in normalized:
        raise ValueError(f"operator name cannot contain '{_LEGACY_FULL_NAME_SEPARATOR}'")
    return normalized


def _normalize_full_name(full_name: str) -> tuple[str, str, str]:
    normalized = full_name.strip()
    namespace, name = _split_full_name(normalized)
    normalized_namespace = _normalize_namespace(namespace)
    normalized_name = _normalize_name(name)
    return (
        normalized_namespace,
        normalized_name,
        f"{normalized_namespace}{_FULL_NAME_SEPARATOR}{normalized_name}",
    )


def _split_full_name(value: str) -> tuple[str, str]:
    has_new = _FULL_NAME_SEPARATOR in value
    has_legacy = _LEGACY_FULL_NAME_SEPARATOR in value
    if has_new and has_legacy:
        raise ValueError(
            "full_name cannot mix '.' and ':' separators; use '<namespace>.<name>'"
        )
    if has_new:
        return value.split(_FULL_NAME_SEPARATOR, 1)
    if has_legacy:
        return value.split(_LEGACY_FULL_NAME_SEPARATOR, 1)
    raise ValueError("full_name must be in '<namespace>.<name>' format")


def _canonicalize_full_name_candidate(identifier: str) -> str | None:
    try:
        _, _, canonical = _normalize_full_name(identifier)
    except ValueError:
        return None
    return canonical


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

    def __init__(
        self,
        *,
        logger: Any | None = None,
        executor: Executor | None = None,
        log_mode: str | None = None,
    ) -> None:
        self._operators: dict[str, OperatorRecord] = {}
        self._id_to_name: dict[str, str] = {}
        self._logger = logger
        self._executor = executor
        self._log_mode = log_mode
        self._lock = RLock()

    def set_log_mode(self, mode: str | None) -> None:
        """Set preferred log mode for this registry when using default logger."""

        self._log_mode = mode

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
        full_name = f"{normalized_namespace}{_FULL_NAME_SEPARATOR}{normalized_name}"
        resolved_metadata = dict(metadata or {})
        if normalized_namespace == "helper":
            # Helper operators are expected to be concurrency-friendly by default.
            resolved_metadata["concurrent"] = True

        with self._lock:
            if full_name in self._operators:
                raise OperatorConflict(full_name, "duplicate registration in same namespace")

            signature, accepts_logger, accepts_var_kwargs = _derive_call_traits(fn)
            operator_id = self._allocate_operator_id_locked(
                normalized_namespace,
                full_name,
            )
            self._operators[full_name] = OperatorRecord(
                namespace=normalized_namespace,
                name=normalized_name,
                full_name=full_name,
                operator_id=operator_id,
                fn=fn,
                metadata=resolved_metadata,
                signature=signature,
                accepts_logger=accepts_logger,
                accepts_var_kwargs=accepts_var_kwargs,
                is_async=_is_async_callable(fn),
            )
            self._id_to_name[operator_id] = full_name

        local_logger = get_logger(
            logger or self._logger,
            mode=self._log_mode,
            operator=full_name,
            action="register",
        )
        local_logger.debug("registered operator")

    def resolve_name(self, identifier: str) -> str:
        normalized = identifier.strip()
        if not normalized:
            raise OperatorNotFound(identifier)

        with self._lock:
            return self._resolve_name_locked(normalized)

    def get(self, full_name: str) -> Callable[..., Any]:
        resolved_full_name = self.resolve_name(full_name)
        return self._get_record(resolved_full_name).fn

    def delete(self, full_name: str, logger: Any | None = None) -> None:
        normalized_full = self.resolve_name(full_name)
        with self._lock:
            record = self._operators.pop(normalized_full, None)
            if record is None:
                raise OperatorNotFound(normalized_full)
            self._id_to_name.pop(record.operator_id, None)

        local_logger = get_logger(
            logger or self._logger,
            mode=self._log_mode,
            operator=normalized_full,
            action="delete",
        )
        local_logger.debug("deleted operator")

    def rename(
        self,
        full_name: str,
        *,
        new_full_name: str | None = None,
        new_name: str | None = None,
        new_namespace: str | None = None,
        logger: Any | None = None,
    ) -> str:
        source_full = self.resolve_name(full_name)
        source_namespace, source_name, _ = _normalize_full_name(source_full)
        record = self._get_record(source_full)

        if new_full_name is not None:
            target_namespace, target_name, target_full = _normalize_full_name(new_full_name)
        else:
            target_namespace = (
                _normalize_namespace(new_namespace)
                if new_namespace is not None
                else source_namespace
            )
            target_name = _normalize_name(new_name) if new_name is not None else source_name
            target_full = f"{target_namespace}{_FULL_NAME_SEPARATOR}{target_name}"

        if target_full == source_full:
            return source_full

        with self._lock:
            existing = self._operators.get(target_full)
            if existing is not None:
                raise OperatorConflict(target_full, "duplicate registration in same namespace")

            self._operators[target_full] = OperatorRecord(
                namespace=target_namespace,
                name=target_name,
                full_name=target_full,
                operator_id=record.operator_id,
                fn=record.fn,
                metadata=dict(record.metadata),
                signature=record.signature,
                accepts_logger=record.accepts_logger,
                accepts_var_kwargs=record.accepts_var_kwargs,
                is_async=record.is_async,
            )
            self._operators.pop(source_full, None)
            self._id_to_name[record.operator_id] = target_full

        local_logger = get_logger(
            logger or self._logger,
            mode=self._log_mode,
            operator=source_full,
            action="rename",
        )
        local_logger.debug("renamed operator to {}", target_full)
        return target_full

    def delete_namespace(self, namespace: str, logger: Any | None = None) -> list[str]:
        normalized_namespace = _normalize_namespace(namespace)
        prefix = f"{normalized_namespace}{_FULL_NAME_SEPARATOR}"

        with self._lock:
            deleted = [name for name in self._operators if name.startswith(prefix)]
            for name in deleted:
                record = self._operators.pop(name, None)
                if record is not None:
                    self._id_to_name.pop(record.operator_id, None)

        local_logger = get_logger(
            logger or self._logger,
            mode=self._log_mode,
            namespace=normalized_namespace,
            action="delete_namespace",
        )
        local_logger.debug("deleted {} operators in namespace", len(deleted))
        return deleted

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

        source_prefix = f"{source_namespace}{_FULL_NAME_SEPARATOR}"

        with self._lock:
            source_names = [
                full_name for full_name in self._operators.keys() if full_name.startswith(source_prefix)
            ]
            rename_pairs = []
            for source_full in source_names:
                record = self._operators[source_full]
                target_full = f"{target_namespace}{_FULL_NAME_SEPARATOR}{record.name}"
                rename_pairs.append((source_full, target_full))

            for source_full, target_full in rename_pairs:
                if target_full in self._operators and target_full not in source_names:
                    raise OperatorConflict(target_full, "duplicate registration in same namespace")

            updated_names: list[str] = []
            for source_full, target_full in rename_pairs:
                record = self._operators[source_full]
                self._operators[target_full] = OperatorRecord(
                    namespace=target_namespace,
                    name=record.name,
                    full_name=target_full,
                    operator_id=record.operator_id,
                    fn=record.fn,
                    metadata=dict(record.metadata),
                    signature=record.signature,
                    accepts_logger=record.accepts_logger,
                    accepts_var_kwargs=record.accepts_var_kwargs,
                    is_async=record.is_async,
                )
                self._operators.pop(source_full, None)
                self._id_to_name[record.operator_id] = target_full
                updated_names.append(target_full)

        local_logger = get_logger(
            logger or self._logger,
            mode=self._log_mode,
            namespace=source_namespace,
            action="rename_namespace",
        )
        local_logger.debug(
            "renamed namespace '{}' -> '{}' for {} operators",
            source_namespace,
            target_namespace,
            len(updated_names),
        )
        return updated_names

    def call(self, full_name: str, logger: Any | None = None, **kwargs: Any) -> Any:
        resolved_full_name = self.resolve_name(full_name)
        record = self._get_record(resolved_full_name)
        injected_logger = logger if logger is not None else self._logger
        call_kwargs = self._build_call_kwargs(record, kwargs, injected_logger)

        try:
            result = record.fn(**call_kwargs)
        except OperatorCallError:
            raise
        except Exception as exc:
            raise OperatorCallError(
                resolved_full_name,
                f"Operator '{resolved_full_name}' failed during sync call: {exc}",
            ) from exc

        if inspect.isawaitable(result):
            raise OperatorCallError(
                resolved_full_name,
                f"Operator '{resolved_full_name}' returned an awaitable in sync call; use acall().",
            )

        return result

    async def acall(self, full_name: str, logger: Any | None = None, **kwargs: Any) -> Any:
        resolved_full_name = self.resolve_name(full_name)
        record = self._get_record(resolved_full_name)
        injected_logger = logger if logger is not None else self._logger
        call_kwargs = self._build_call_kwargs(record, kwargs, injected_logger)

        try:
            return await self._execute_async(record, call_kwargs)
        except OperatorCallError:
            raise
        except Exception as exc:
            raise OperatorCallError(
                resolved_full_name,
                f"Operator '{resolved_full_name}' failed during async call: {exc}",
            ) from exc

    async def acall_many(
        self,
        full_name: str,
        calls: Sequence[Mapping[str, Any]],
        *,
        logger: Any | None = None,
        concurrency: int | None = None,
        return_exceptions: bool = False,
    ) -> list[Any]:
        """Concurrently execute the same operator with multiple kwargs payloads."""

        if concurrency is not None and concurrency < 1:
            raise ValueError("concurrency must be >= 1 or None")

        resolved_full_name = self.resolve_name(full_name)
        record = self._get_record(resolved_full_name)
        injected_logger = logger if logger is not None else self._logger
        call_items = list(calls)
        if not call_items:
            return []

        semaphore = asyncio.Semaphore(concurrency) if concurrency is not None else None

        async def _run_one(index: int, raw_kwargs: Mapping[str, Any]) -> Any:
            if not isinstance(raw_kwargs, Mapping):
                raise TypeError(f"calls[{index}] must be a mapping of kwargs")

            call_kwargs = self._build_call_kwargs(record, dict(raw_kwargs), injected_logger)

            try:
                if semaphore is None:
                    return await self._execute_async(record, call_kwargs)
                async with semaphore:
                    return await self._execute_async(record, call_kwargs)
            except OperatorCallError:
                raise
            except Exception as exc:
                raise OperatorCallError(
                    resolved_full_name,
                    f"Operator '{resolved_full_name}' failed during async batch call index={index}: {exc}",
                ) from exc

        coroutines = [_run_one(index, item) for index, item in enumerate(call_items)]
        results = await asyncio.gather(*coroutines, return_exceptions=return_exceptions)
        return list(results)

    async def acall_helper_many(
        self,
        name: str,
        calls: Sequence[Mapping[str, Any]],
        *,
        logger: Any | None = None,
        concurrency: int | None = None,
        return_exceptions: bool = False,
    ) -> list[Any]:
        """Convenience wrapper to concurrently execute helper namespace operators."""

        normalized_name = _normalize_name(name)
        return await self.acall_many(
            f"helper{_FULL_NAME_SEPARATOR}{normalized_name}",
            calls,
            logger=logger,
            concurrency=concurrency,
            return_exceptions=return_exceptions,
        )

    def list(self, namespace: str | None = None) -> list[str]:
        with self._lock:
            names = list(self._operators.keys())

        if namespace is None:
            return names

        normalized_namespace = _normalize_namespace(namespace)
        prefix = f"{normalized_namespace}{_FULL_NAME_SEPARATOR}"
        return [name for name in names if name.startswith(prefix)]

    def info(self, full_name: str) -> dict[str, Any]:
        resolved_full_name = self.resolve_name(full_name)
        record = self._get_record(resolved_full_name)
        docstring = inspect.getdoc(record.fn) or ""
        doc_summary = docstring.splitlines()[0] if docstring else ""
        metadata = dict(record.metadata)
        metadata_summary = metadata.get("summary")
        metadata_note = metadata.get("note")

        if isinstance(metadata_summary, str) and metadata_summary.strip():
            summary = metadata_summary.strip()
        elif isinstance(metadata_note, str) and metadata_note.strip():
            summary = metadata_note.strip()
        else:
            summary = doc_summary

        return {
            "name": record.full_name,
            "id": record.operator_id,
            "namespace": record.namespace,
            "short_name": record.name,
            "metadata": metadata,
            "signature": record.signature,
            "docstring": summary,
            "module": getattr(record.fn, "__module__", ""),
            "qualname": getattr(record.fn, "__qualname__", ""),
            "is_async": record.is_async,
            "supports_async": True,
        }

    def _get_record(self, full_name: str) -> OperatorRecord:
        with self._lock:
            record = self._operators.get(full_name)

        if record is None:
            raise OperatorNotFound(full_name)
        return record

    def _resolve_name_locked(self, identifier: str) -> str:
        if identifier in self._operators:
            return identifier

        canonical = _canonicalize_full_name_candidate(identifier)
        if canonical is not None and canonical in self._operators:
            return canonical

        by_id = self._id_to_name.get(identifier)
        if by_id is not None and by_id in self._operators:
            return by_id

        raise OperatorNotFound(identifier)

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
    def _build_call_kwargs(
        record: OperatorRecord,
        kwargs: dict[str, Any],
        logger: Any | None,
    ) -> dict[str, Any]:
        call_kwargs = dict(kwargs)
        if (
            logger is not None
            and "logger" not in call_kwargs
            and (record.accepts_logger or record.accepts_var_kwargs)
        ):
            call_kwargs["logger"] = logger
        return call_kwargs

    async def _execute_async(
        self,
        record: OperatorRecord,
        call_kwargs: dict[str, Any],
    ) -> Any:
        if record.is_async:
            return await record.fn(**call_kwargs)

        if self._executor is None:
            return await asyncio.to_thread(record.fn, **call_kwargs)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(record.fn, **call_kwargs),
        )


def _namespace_id_prefix(namespace: str) -> str:
    for ch in namespace.strip().lower():
        if "a" <= ch <= "z":
            return ch
    return "o"
