from __future__ import annotations

from .helper import eggbox, eggbox2d
from .math import add, identity
from .stat import chi2_cov


BUILTIN_OPERATORS = (
    ("math", "add", add, {"category": "math"}),
    ("stat", "chi2_cov", chi2_cov, {"category": "statistics"}),
    ("helper", "eggbox", eggbox, {"category": "hep_scanner_benchmark"}),
    ("helper", "eggbox2d", eggbox2d, {"category": "hep_scanner_benchmark"}),
    ("math", "identity", identity, {"category": "math"}),
)


def register_builtins(registry) -> list[str]:
    registered = []
    for namespace, name, fn, metadata in BUILTIN_OPERATORS:
        registry.register(name=name, fn=fn, namespace=namespace, metadata=metadata)
        registered.append(f"{namespace}:{name}")
    return registered
