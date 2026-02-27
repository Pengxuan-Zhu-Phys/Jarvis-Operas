from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from jarvis_operas import get_global_operas_registry


def _assert_close_scalar(value: float, expected: float, name: str) -> None:
    if not np.isclose(value, expected):
        raise AssertionError(f"{name}: expected {expected}, got {value}")


def _assert_close_vector(value: np.ndarray, expected: np.ndarray, name: str) -> None:
    if not np.allclose(value, expected):
        raise AssertionError(f"{name}: expected {expected}, got {value}")


async def _run_async_cases(registry, operator_id: str) -> None:
    expected_scalar = 0.5
    expected_batch = np.array([0.5, 3.0])

    # Async call by operator name, plain list input.
    async_list = await registry.acall(
        "stat.chi2_cov",
        residual=[1.0, 0.0],
        cov=[[2.0, 0.0], [0.0, 1.0]],
    )
    _assert_close_scalar(async_list, expected_scalar, "async/list/by-name")

    # Async call by operator id with numpy batch residual.
    async_batch = await registry.acall(
        operator_id,
        residual=np.array([[1.0, 0.0], [2.0, 1.0]]),
        cov=np.array([[2.0, 0.0], [0.0, 1.0]]),
    )
    _assert_close_vector(async_batch, expected_batch, "async/numpy-batch/by-id")


def main() -> None:
    registry = get_global_operas_registry()
    info = registry.info("stat.chi2_cov")
    operator_id = info["id"]
    print(f"Testing operator: {info['name']} (id={operator_id})")

    expected_scalar = 0.5
    expected_batch = np.array([0.5, 3.0])

    # 1) Sync call by operator name, list input.
    sync_list = registry.call(
        "stat.chi2_cov",
        residual=[1.0, 0.0],
        cov=[[2.0, 0.0], [0.0, 1.0]],
    )
    _assert_close_scalar(sync_list, expected_scalar, "sync/list/by-name")

    # 2) Sync call by operator name, numpy input.
    sync_numpy = registry.call(
        "stat.chi2_cov",
        residual=np.array([1.0, 0.0]),
        cov=np.array([[2.0, 0.0], [0.0, 1.0]]),
    )
    _assert_close_scalar(sync_numpy, expected_scalar, "sync/numpy/by-name")

    # 3) Sync call by operator name, pandas DataFrame input.
    sync_pandas = registry.call(
        "stat.chi2_cov",
        residual=pd.DataFrame(
            {"x": [1.0, 2.0], "y": [0.0, 1.0]},
            index=["s1", "s2"],
        ),
        cov=pd.DataFrame(
            [[2.0, 0.0], [0.0, 1.0]],
            index=["x", "y"],
            columns=["x", "y"],
        ),
    )
    if not isinstance(sync_pandas, pd.Series):
        raise AssertionError(
            f"sync/pandas/by-name: expected pandas.Series, got {type(sync_pandas)}"
        )
    _assert_close_vector(sync_pandas.to_numpy(), expected_batch, "sync/pandas/by-name")

    # 4) Sync call by operator id.
    sync_by_id = registry.call(
        operator_id,
        residual=[1.0, 0.0],
        cov=[[2.0, 0.0], [0.0, 1.0]],
    )
    _assert_close_scalar(sync_by_id, expected_scalar, "sync/list/by-id")

    # 5+) Async call variants.
    asyncio.run(_run_async_cases(registry, operator_id))

    print("All call styles and input formats passed.")
    print("Covered:")
    print("- sync: name/list, name/numpy, name/pandas, id/list")
    print("- async: name/list, id/numpy-batch")


if __name__ == "__main__":
    main()
