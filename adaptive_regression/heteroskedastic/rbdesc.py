"""Residual-balance descent for mean-zero heteroskedastic regression."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._validation import (
    FloatArray,
    ordinary_least_squares,
    validate_prediction_data,
    validate_regression_data,
)

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
RBDescVariant = Literal["standard", "aggressive", "hybrid"]

_CONTRACTION = 3.0 / 4.0
_UPDATE_MULTIPLIER = 8.0
_RANK_STEPS = 4
_CALIBRATION_REPETITIONS = 256
_CALIBRATION_SEED = 91_347
_LOCAL_MULTIPLIER = 2.0
_TOLERANCE_MULTIPLIER = 128.0
_VARIANTS = frozenset(("standard", "aggressive", "hybrid"))


@dataclass(frozen=True)
class RBDescPath:
    """One standard or aggressive descent trajectory."""

    coef: FloatArray
    certified: bool
    exact_residuals: int
    residual_tolerance: float
    boundary_residual: float
    iterations: int
    stages: int
    projections: int
    stop_reason: str


@dataclass(frozen=True)
class RBDescResult:
    """Selected estimate and the paths needed to audit that selection."""

    coef: FloatArray
    selected_path: Literal["standard", "aggressive"]
    selection_reason: str
    standard: RBDescPath | None
    aggressive: RBDescPath | None
    violation_threshold: float
    window_ranks: IntArray
    terminal_radius: float

    @property
    def selected(self) -> RBDescPath:
        path = self.standard if self.selected_path == "standard" else self.aggressive
        assert path is not None
        return path

    @property
    def certified(self) -> bool:
        return self.selected.certified

    @property
    def exact_residuals(self) -> int:
        return self.selected.exact_residuals

    @property
    def iterations(self) -> int:
        return sum(path.iterations for path in (self.standard, self.aggressive) if path)

    @property
    def stages(self) -> int:
        return sum(path.stages for path in (self.standard, self.aggressive) if path)


@dataclass(frozen=True)
class _Schedule:
    terminal_radius: float
    window_ranks: IntArray
    direction_bound: float
    max_stages: int
    iterations_per_stage: int


@dataclass(frozen=True)
class _Scores:
    windows: FloatArray
    counts: IntArray
    violation_ratios: FloatArray
    violated: BoolArray
    local: BoolArray
    active: BoolArray
    aggregate: FloatArray


@dataclass(frozen=True)
class _Certificate:
    certified: bool
    count: int
    tolerance: float
    boundary_residual: float


def _validate_variant(variant: str) -> RBDescVariant:
    if not isinstance(variant, str) or variant not in _VARIANTS:
        raise ValueError(f"variant must be one of: {', '.join(sorted(_VARIANTS))}")
    return cast(RBDescVariant, variant)


def _project(vector: ArrayLike, radius: float) -> FloatArray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    return value.copy() if norm <= radius or norm == 0.0 else value * radius / norm


def _geometric_ranks(n: int, d: int) -> IntArray:
    """Return ``ceil((d + 1) 2**(j/4))``, ending at ``n``."""

    first = d + 1
    levels = int(math.ceil(_RANK_STEPS * math.log2(n / first)))
    ranks = np.fromiter(
        (int(math.ceil(first * 2.0 ** (j / _RANK_STEPS))) for j in range(levels + 1)),
        dtype=np.int64,
    )
    ranks = np.unique(ranks[ranks <= n])
    return ranks if ranks[-1] == n else np.r_[ranks, n].astype(np.int64)


def _sign_threshold(X: FloatArray, counts: IntArray, confidence: float) -> float:
    """Conditionally calibrate the largest normalized score once."""

    n, d = X.shape
    uncertainty = np.sqrt(d * (counts.astype(float) + d)) + d
    rng = np.random.default_rng(_CALIBRATION_SEED)
    maxima = np.empty(_CALIBRATION_REPETITIONS)
    for repetition in range(_CALIBRATION_REPETITIONS):
        signs = rng.choice((-1.0, 1.0), size=n)
        scores = np.cumsum(X * signs[:, None], axis=0)[counts - 1]
        maxima[repetition] = np.max(np.linalg.norm(scores, axis=1) / uncertainty)
    return float(np.quantile(maxima, confidence, method="higher"))


class RBDescRegressor:
    """Bounded residual-balance descent.

    ``standard`` uses calibrated score violations. ``aggressive`` also uses
    every residual window local to the current stage. ``hybrid`` selects the
    aggressive trajectory only when it reaches the exact-residual certificate.
    """

    def __init__(
        self,
        radius: float,
        *,
        covariance_bound: float = 1.0,
        confidence_level: float = 0.95,
        variant: RBDescVariant = "standard",
    ):
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        if not np.isfinite(covariance_bound) or covariance_bound < 1:
            raise ValueError("covariance_bound must be finite and at least one")
        if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
            raise ValueError("confidence_level must lie between zero and one")
        self.radius = float(radius)
        self.covariance_bound = float(covariance_bound)
        self.confidence_level = float(confidence_level)
        self.variant = _validate_variant(variant)

    def _schedule(self, n: int, d: int) -> _Schedule:
        direction_bound = math.sqrt(self.covariance_bound * n)
        if self.radius == 0:
            terminal, stages = 0.0, 0
        else:
            terminal = min(
                self.radius,
                _UPDATE_MULTIPLIER * direction_bound * float(np.spacing(self.radius)),
            )
            terminal = self.radius if terminal <= 0 else terminal
            stages = 0 if terminal >= self.radius else math.ceil(
                math.log(self.radius / terminal) / math.log(1 / _CONTRACTION)
            )
        return _Schedule(
            terminal,
            _geometric_ranks(n, d),
            direction_bound,
            stages,
            math.ceil(_UPDATE_MULTIPLIER * direction_bound),
        )

    @staticmethod
    def _rank_grid_scores(
        sorted_residuals: FloatArray,
        cumulative_signed_X: FloatArray,
        ranks: IntArray,
        *,
        threshold: float,
        local_cutoff: float | None = None,
    ) -> _Scores:
        d = cumulative_signed_X.shape[1]
        windows = sorted_residuals[ranks - 1]
        counts = np.searchsorted(sorted_residuals, windows, side="right")
        keep = np.r_[True, counts[1:] != counts[:-1]]
        windows, counts = windows[keep], counts[keep]
        scores = cumulative_signed_X[counts - 1]
        uncertainty = np.sqrt(d * (counts.astype(float) + d)) + d
        ratios = np.linalg.norm(scores, axis=1) / uncertainty
        violated = ratios > threshold
        local = np.zeros_like(violated) if local_cutoff is None else windows <= local_cutoff
        active = violated | local
        aggregate = (
            np.sum(scores[active] / uncertainty[active, None], axis=0)
            if np.any(active)
            else np.zeros(d)
        )
        return _Scores(windows, counts, ratios, violated, local, active, aggregate)

    @classmethod
    def _scores(
        cls,
        X: FloatArray,
        y: FloatArray,
        coef: FloatArray,
        schedule: _Schedule,
        threshold: float,
        local_cutoff: float | None,
    ) -> _Scores:
        residual = y - X @ coef
        order = np.argsort(np.abs(residual), kind="stable")
        cumulative = np.cumsum(X[order] * np.sign(residual[order])[:, None], axis=0)
        return cls._rank_grid_scores(
            np.abs(residual[order]),
            cumulative,
            schedule.window_ranks,
            threshold=threshold,
            local_cutoff=local_cutoff,
        )

    @classmethod
    def _calibrate(
        cls, X: FloatArray, y: FloatArray, coef: FloatArray, schedule: _Schedule, confidence: float
    ) -> float:
        residual = y - X @ coef
        order = np.argsort(np.abs(residual), kind="stable")
        endpoints = cls._rank_grid_scores(
            np.abs(residual[order]),
            np.zeros_like(X),
            schedule.window_ranks,
            threshold=math.inf,
        )
        return _sign_threshold(X[order], endpoints.counts, confidence)

    @staticmethod
    def _certificate(X: FloatArray, y: FloatArray, coef: FloatArray) -> _Certificate:
        n, d = X.shape
        fitted = X @ coef
        scale = max(np.linalg.norm(y), np.linalg.norm(fitted)) / math.sqrt(n)
        tolerance = _TOLERANCE_MULTIPLIER * np.finfo(float).eps * scale
        residuals = np.abs(y - fitted)
        boundary = float(np.partition(residuals, d)[d])
        return _Certificate(boundary <= tolerance, int(np.sum(residuals <= tolerance)), tolerance, boundary)

    def _run(
        self,
        X: FloatArray,
        y: FloatArray,
        initial: FloatArray,
        schedule: _Schedule,
        threshold: float,
        aggressive: bool,
    ) -> RBDescPath:
        coef = initial.copy()
        certificate = self._certificate(X, y, coef)
        iterations = stages = projections = 0
        stop = "exact_residual_certificate" if certificate.certified else "completed_all_stages"

        for stage in range(schedule.max_stages):
            if certificate.certified:
                break
            stages += 1
            radius = _CONTRACTION**stage * self.radius
            step = radius / (_UPDATE_MULTIPLIER * schedule.direction_bound)
            cutoff = _LOCAL_MULTIPLIER * math.sqrt(self.covariance_bound) * radius if aggressive else None
            for _ in range(schedule.iterations_per_stage):
                scores = self._scores(X, y, coef, schedule, threshold, cutoff)
                if not np.any(scores.active):
                    stop = "no_active_scale" if aggressive else "no_violated_scale"
                    return self._path(coef, certificate, iterations, stages, projections, stop)
                norm = float(np.linalg.norm(scores.aggregate))
                if not np.isfinite(norm):
                    return self._path(coef, certificate, iterations, stages, projections, "aggregate_nonfinite")
                if norm == 0:
                    stop = "aggregate_zero_at_stage"
                    if not aggressive:
                        return self._path(coef, certificate, iterations, stages, projections, stop)
                    break
                candidate = coef + step * scores.aggregate / norm
                projections += int(np.linalg.norm(candidate) > self.radius)
                candidate = _project(candidate, self.radius)
                if np.array_equal(candidate, coef):
                    return self._path(coef, certificate, iterations, stages, projections, "numerical_resolution")
                coef = candidate
                iterations += 1
                certificate = self._certificate(X, y, coef)
                if certificate.certified:
                    stop = "exact_residual_certificate"
                    break
        if schedule.max_stages == 0 and not certificate.certified:
            stop = "radius_below_terminal_floor"
        return self._path(coef, certificate, iterations, stages, projections, stop)

    @staticmethod
    def _path(
        coef: FloatArray,
        certificate: _Certificate,
        iterations: int,
        stages: int,
        projections: int,
        stop: str,
    ) -> RBDescPath:
        return RBDescPath(
            coef.copy(), certificate.certified, certificate.count, certificate.tolerance,
            certificate.boundary_residual, iterations, stages, projections, stop,
        )

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RBDescRegressor":
        X, y = validate_regression_data(X, y, estimator_name="RB-Desc")
        n, d = X.shape
        if n <= d:
            raise ValueError("RB-Desc requires at least d + 1 observations")
        schedule = self._schedule(n, d)
        initial = _project(ordinary_least_squares(X, y), self.radius)
        threshold = self._calibrate(X, y, initial, schedule, self.confidence_level)

        standard = self._run(X, y, initial, schedule, threshold, False) if self.variant != "aggressive" else None
        aggressive = None
        if self.variant == "aggressive" or (self.variant == "hybrid" and not standard.certified):
            aggressive = self._run(X, y, initial, schedule, threshold, True)

        if self.variant == "standard":
            selected, reason = "standard", "standard_variant"
        elif self.variant == "aggressive":
            selected, reason = "aggressive", "aggressive_variant"
        elif aggressive is not None and aggressive.certified:
            selected, reason = "aggressive", "aggressive_certificate"
        else:
            selected = "standard"
            reason = "standard_already_certified" if standard.certified else "aggressive_uncertified"
        chosen = standard if selected == "standard" else aggressive
        assert chosen is not None
        self.result_ = RBDescResult(
            chosen.coef.copy(), selected, reason, standard, aggressive,
            threshold, schedule.window_ranks.copy(), schedule.terminal_radius,
        )
        self.coef_, self.n_features_in_ = self.result_.coef.copy(), d
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before predict")
        return validate_prediction_data(X, n_features=self.n_features_in_) @ self.coef_


__all__ = ["RBDescPath", "RBDescRegressor", "RBDescResult", "RBDescVariant"]
