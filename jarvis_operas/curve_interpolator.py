from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from scipy.interpolate import interp1d


@dataclass
class Curve1DInterpolator:
    """Pickle-safe 1D interpolation callable."""

    curve_id: str
    x_values: np.ndarray
    y_values: np.ndarray
    kind: str = "cubic"
    log_x: bool = False
    log_y: bool = False
    extrapolation: str = "extrapolate"
    metadata: dict[str, Any] | None = None
    _interp: Any = field(init=False, repr=False)
    _extrapolation_mode: str = field(init=False, repr=False)
    _x_min: float = field(init=False, repr=False)
    _x_max: float = field(init=False, repr=False)
    _scalar_eval: Callable[[float], float] = field(init=False, repr=False)
    _vector_eval: Callable[[np.ndarray], np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        x_values = np.asarray(self.x_values, dtype=float)
        y_values = np.asarray(self.y_values, dtype=float)

        if x_values.ndim != 1 or y_values.ndim != 1:
            raise ValueError(f"curve '{self.curve_id}' requires 1D x/y arrays")
        if x_values.size < 2:
            raise ValueError(f"curve '{self.curve_id}' requires at least two points")
        if x_values.size != y_values.size:
            raise ValueError(f"curve '{self.curve_id}' x/y lengths do not match")

        order = np.argsort(x_values)
        x_values = x_values[order]
        y_values = y_values[order]
        if np.any(np.diff(x_values) <= 0):
            raise ValueError(f"curve '{self.curve_id}' x values must be strictly increasing")
        if self.log_x and np.any(x_values <= 0):
            raise ValueError(f"curve '{self.curve_id}' has non-positive x values under logX")
        if self.log_y and np.any(y_values <= 0):
            raise ValueError(f"curve '{self.curve_id}' has non-positive y values under logY")

        object.__setattr__(self, "x_values", x_values)
        object.__setattr__(self, "y_values", y_values)
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

        self._extrapolation_mode = self._normalize_extrapolation_mode()
        self._interp = self._build_interp()
        self._bind_fast_paths()

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, np.ndarray):
            return float(value.item())
        return float(value)

    def _normalize_extrapolation_mode(self) -> str:
        extrapolation = self.extrapolation.strip().lower()
        if extrapolation not in ("extrapolate", "clip", "error", "nan"):
            raise ValueError(
                f"curve '{self.curve_id}' has invalid extrapolation policy: {self.extrapolation}"
            )
        return extrapolation

    def _build_interp(self):
        src_x = np.log(self.x_values) if self.log_x else self.x_values
        src_y = np.log(self.y_values) if self.log_y else self.y_values

        extrapolation = self._extrapolation_mode
        if extrapolation == "extrapolate":
            kwargs = {"bounds_error": False, "fill_value": "extrapolate"}
        elif extrapolation == "clip":
            kwargs = {"bounds_error": False, "fill_value": (src_y[0], src_y[-1])}
        elif extrapolation == "nan":
            kwargs = {"bounds_error": False, "fill_value": np.nan}
        else:
            kwargs = {"bounds_error": True}

        return interp1d(
            src_x,
            src_y,
            kind=self.kind,
            assume_sorted=True,
            **kwargs,
        )

    def _bind_fast_paths(self) -> None:
        self._x_min = float(self.x_values[0])
        self._x_max = float(self.x_values[-1])
        self._vector_eval = self._build_vector_eval()
        self._scalar_eval = self._build_scalar_eval()

    def _build_scalar_eval(self) -> Callable[[float], float]:
        to_float = self._to_float
        vector_eval = self._vector_eval

        def scalar_eval(value: float) -> float:
            return to_float(vector_eval(np.asarray([value], dtype=float))[0])

        return scalar_eval

    def _build_vector_eval(self) -> Callable[[np.ndarray], np.ndarray]:
        interp = self._interp
        nan_mode = self._extrapolation_mode == "nan"
        clip_mode = self._extrapolation_mode == "clip"
        x_min = self._x_min
        x_max = self._x_max
        curve_id = self.curve_id

        def vector_eval(values: np.ndarray) -> np.ndarray:
            arr = np.asarray(values, dtype=float)
            out = np.full(arr.shape, np.nan, dtype=float)
            finite_mask = np.isfinite(arr)
            if not np.any(finite_mask):
                return out

            if nan_mode:
                finite_mask = finite_mask & (arr >= x_min) & (arr <= x_max)
                if not np.any(finite_mask):
                    return out

            eval_x = np.array(arr, copy=True)
            if clip_mode:
                eval_x[finite_mask] = np.clip(eval_x[finite_mask], x_min, x_max)

            if self.log_x:
                if clip_mode:
                    non_positive = finite_mask & (eval_x <= 0)
                else:
                    non_positive = finite_mask & (arr <= 0)

                if np.any(non_positive):
                    if nan_mode:
                        finite_mask = finite_mask & (arr > 0)
                    else:
                        raise ValueError(f"curve '{curve_id}' received non-positive x under logX")

                if not np.any(finite_mask):
                    return out
                eval_x = np.log(eval_x[finite_mask])
            else:
                eval_x = eval_x[finite_mask]

            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                evaluated = np.asarray(interp(eval_x), dtype=float)
                if self.log_y:
                    evaluated = np.exp(evaluated)

            out[finite_mask] = evaluated
            return out

        return vector_eval

    def __call__(self, x: float | np.ndarray | list[float]) -> float | np.ndarray:
        if np.isscalar(x):
            return self._scalar_eval(float(x))
        return self._vector_eval(np.asarray(x, dtype=float))

    def __getstate__(self) -> dict[str, Any]:
        return {
            "curve_id": self.curve_id,
            "x_values": np.asarray(self.x_values),
            "y_values": np.asarray(self.y_values),
            "kind": self.kind,
            "log_x": self.log_x,
            "log_y": self.log_y,
            "extrapolation": self.extrapolation,
            "metadata": dict(self.metadata or {}),
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.curve_id = str(state["curve_id"])
        self.x_values = np.asarray(state["x_values"], dtype=float)
        self.y_values = np.asarray(state["y_values"], dtype=float)
        self.kind = str(state.get("kind", "cubic"))
        self.log_x = bool(state.get("log_x", False))
        self.log_y = bool(state.get("log_y", False))
        self.extrapolation = str(state.get("extrapolation", "extrapolate"))
        self.metadata = dict(state.get("metadata", {}))
        self._extrapolation_mode = self._normalize_extrapolation_mode()
        self._interp = self._build_interp()
        self._bind_fast_paths()
