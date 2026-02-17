from __future__ import annotations

from .core import add, chi2_cov, identity


BUILTIN_OPERATORS = (
    ("identity", identity, {"category": "utility"}),
    ("add", add, {"category": "math"}),
    ("chi2_cov", chi2_cov, {"category": "statistics"}),
)


def register_builtins(registry) -> list[str]:
    registered = []
    for name, fn, metadata in BUILTIN_OPERATORS:
        registry.register(name=name, fn=fn, namespace="core", metadata=metadata)
        registered.append(f"core:{name}")
    return registered
