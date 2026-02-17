from __future__ import annotations


class OperatorError(Exception):
    """Base class for all Jarvis-Operas errors."""


class OperatorNotFound(OperatorError):
    """Raised when an operator name cannot be resolved from the registry."""

    def __init__(self, full_name: str) -> None:
        self.full_name = full_name
        super().__init__(f"Operator '{full_name}' was not found.")


class OperatorConflict(OperatorError):
    """Raised when a registration collides with an existing operator."""

    def __init__(self, full_name: str, reason: str) -> None:
        self.full_name = full_name
        self.reason = reason
        super().__init__(f"Cannot register '{full_name}': {reason}")


class OperatorLoadError(OperatorError):
    """Raised when loading operators from file/entrypoint fails."""

    def __init__(self, source: str, message: str) -> None:
        self.source = source
        super().__init__(message)


class OperatorCallError(OperatorError):
    """Raised when operator execution fails."""

    def __init__(self, operator: str, message: str) -> None:
        self.operator = operator
        super().__init__(message)
