"""Adaptive selection among Lq regression estimates."""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from ._validation import FloatArray, validate_regression_data

_MAX_ITER = 200
_TOLERANCE = 1e-8
_RADIUS_QUANTILE = 0.75


@dataclass(frozen=True)
class LqRegressionFit:
    coef: FloatArray
    q: float
    objective: float
    converged: bool


@dataclass(frozen=True)
class AdaptiveLqRegressionResult:
    coef: FloatArray
    selected_q: int
    selected_candidate: FloatArray
    selection_radius: float
    block_count: int
    block_size: int
    q_grid: tuple[int, ...]
    candidate_radii: FloatArray
    candidate_estimates: FloatArray
    optimization_converged: bool

    @property
    def n_selection_samples(self) -> int:
        return self.block_count * self.block_size


def _problem(X: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
    return validate_regression_data(
        X, y,
        estimator_name="Lq regression",
        require_more_samples_than_features=True,
        require_full_column_rank=True,
    )


def _log_lq_objective(residual: FloatArray, q: int) -> float:
    scale = float(np.max(np.abs(residual)))
    if scale == 0:
        return -math.inf
    return math.log(scale) + math.log(float(np.mean((np.abs(residual) / scale) ** q))) / q


def _scale(X: FloatArray, y: FloatArray):
    columns = np.sqrt(np.mean(X**2, axis=0))
    response = max(np.linalg.norm(y) / math.sqrt(len(y)), np.finfo(float).eps)
    return X / columns, y / response, columns, response


def _solve_clarabel(problem, *, max_iter, tolerance, warm_start=False):
    problem.solve(
        solver=cp.CLARABEL,
        max_iter=max_iter,
        tol_gap_abs=tolerance,
        tol_gap_rel=tolerance,
        tol_feas=tolerance,
        warm_start=warm_start,
    )


def fit_lad_regression(
    X: ArrayLike,
    y: ArrayLike,
    *,
    tie_break: bool = False,
    max_iter: int = _MAX_ITER,
    tolerance: float = _TOLERANCE,
) -> LqRegressionFit:
    """Fit LAD, optionally selecting the minimum-L2-residual LAD solution."""

    X, y = _problem(X, y)
    if max_iter < 1 or tolerance <= 0:
        raise ValueError("invalid optimizer controls")
    design, response, columns, response_scale = _scale(X, y)
    theta = cp.Variable(X.shape[1])
    residual = response - design @ theta
    primary = cp.Problem(cp.Minimize(cp.norm1(residual)))
    _solve_clarabel(primary, max_iter=max_iter, tolerance=tolerance)
    if theta.value is None:
        raise RuntimeError(f"LAD primary optimization status: {primary.status}")

    converged = primary.status == cp.OPTIMAL
    if tie_break:
        primary_theta = np.asarray(theta.value).copy()
        primary_loss = float(np.sum(np.abs(response - design @ theta.value)))
        loss_scale = max(1.0, primary_loss)
        face_tolerance = 10 * tolerance * loss_scale
        loss_limit = cp.Parameter(nonneg=True, value=primary_loss + face_tolerance)
        secondary = cp.Problem(
            cp.Minimize(cp.sum_squares(residual)),
            [cp.norm1(residual) <= loss_limit],
        )
        try:
            _solve_clarabel(
                secondary,
                max_iter=max_iter,
                tolerance=tolerance,
                warm_start=True,
            )
        except cp.SolverError:
            retry_tolerance = max(10 * tolerance, 1e-7)
            face_tolerance = max(100 * tolerance, 1e-6) * loss_scale
            loss_limit.value = primary_loss + face_tolerance
            try:
                _solve_clarabel(
                    secondary,
                    max_iter=max_iter,
                    tolerance=retry_tolerance,
                    warm_start=True,
                )
            except cp.SolverError:
                theta.value = primary_theta
                converged = False
                warnings.warn(
                    "LAD tie-break failed twice; using the primary LAD fit",
                    RuntimeWarning,
                )
        if theta.value is None:
            theta.value = primary_theta
            converged = False
        else:
            converged &= secondary.status == cp.OPTIMAL

    coef = np.asarray(theta.value) * response_scale / columns
    final_residual = y - X @ coef
    if tie_break:
        scaled_loss = float(np.sum(np.abs(final_residual / response_scale)))
        converged &= scaled_loss <= primary_loss + 2 * face_tolerance
    return LqRegressionFit(coef.copy(), 1, _log_lq_objective(final_residual, 1), converged)


def _value_and_gradient(theta, X, y, q):
    residual = y - X @ theta
    scale = float(np.max(np.abs(residual)))
    if scale == 0:
        return -math.inf, np.zeros_like(theta)
    ratio = np.abs(residual) / scale
    powers = ratio**q
    total = float(np.sum(powers))
    value = math.log(scale) + math.log(total / len(residual)) / q
    gradient = -(X.T @ (ratio ** (q - 2) * residual / scale)) / (scale * total)
    return value, gradient


def fit_lq_regression(
    X: ArrayLike,
    y: ArrayLike,
    q: int,
    *,
    initial: ArrayLike | None = None,
    max_iter: int = _MAX_ITER,
    tolerance: float = _TOLERANCE,
) -> LqRegressionFit:
    """Fit integer-order Lq regression with CVXPY, NumPy, or SciPy."""

    X, y = _problem(X, y)
    if isinstance(q, bool) or int(q) != q or q < 1:
        raise ValueError("q must be a positive integer")
    if max_iter < 1 or tolerance <= 0:
        raise ValueError("invalid optimizer controls")
    q = int(q)
    if q == 1:
        return fit_lad_regression(
            X, y, tie_break=True, max_iter=max_iter, tolerance=tolerance
        )
    if q == 2:
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        return LqRegressionFit(coef, q, _log_lq_objective(y - X @ coef, q), True)

    design, response, columns, response_scale = _scale(X, y)
    if initial is None:
        theta = np.linalg.lstsq(design, response, rcond=None)[0]
    else:
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (X.shape[1],) or not np.isfinite(initial).all():
            raise ValueError("initial must be a finite vector of length d")
        theta = initial * columns / response_scale
    optimized = minimize(
        _value_and_gradient,
        theta,
        args=(design, response, q),
        method="BFGS",
        jac=True,
        options={"gtol": tolerance, "maxiter": max_iter},
    )
    coef = np.asarray(optimized.x) * response_scale / columns
    residual = y - X @ coef
    converged = optimized.success or np.linalg.norm(optimized.jac, ord=np.inf) <= 10 * tolerance
    return LqRegressionFit(coef, q, _log_lq_objective(residual, q), bool(converged))


def _fit_path(X, y, grid):
    fits, initial = [], None
    for q in grid:
        fit = fit_lq_regression(X, y, q, initial=initial)
        fits.append(fit)
        initial = fit.coef
    return fits


class AdaptiveLqRegression:
    """Choose q by concentration among randomly allocated block estimates."""

    def __init__(
        self,
        *,
        delta: float = 0.05,
        q_max: int | None = None,
        refit_full_sample: bool = True,
        random_state: int | None = None,
    ) -> None:
        if not 0 < delta < 1:
            raise ValueError("delta must lie strictly between zero and one")
        if q_max is not None and (
            isinstance(q_max, bool) or int(q_max) != q_max or q_max < 2
        ):
            raise ValueError("q_max must be at least two or None")
        self.delta = float(delta)
        self.q_max = None if q_max is None else int(q_max)
        self.refit_full_sample = bool(refit_full_sample)
        self.random_state = random_state

    def _block_parameters(self, n: int, d: int) -> tuple[int, int]:
        available = n // (d + 1)
        if available < 4:
            raise ValueError("need at least 4(d+1) observations for four blocks")
        count = min(max(4, math.ceil(math.log(64 * n / self.delta))), available)
        return count, n // count

    def _q_grid(self, n: int) -> tuple[int, ...]:
        upper = 1 << (math.ceil(math.log2(n)) - 1)
        upper = min(upper, self.q_max) if self.q_max is not None else upper
        return (1,) + tuple(1 << j for j in range(1, math.floor(math.log2(upper)) + 1))

    def fit(self, X: ArrayLike, y: ArrayLike) -> "AdaptiveLqRegression":
        X, y = _problem(X, y)
        n, d = X.shape
        blocks, size = self._block_parameters(n, d)
        grid = self._q_grid(n)
        order = np.random.default_rng(self.random_state).permutation(n)[: blocks * size]
        estimates = np.empty((len(grid), blocks, d))
        convergence = []
        for block in range(blocks):
            indices = order[block * size : (block + 1) * size]
            fits = _fit_path(X[indices], y[indices], grid)
            estimates[:, block] = [fit.coef for fit in fits]
            convergence.extend(fit.converged for fit in fits)

        distances = np.linalg.norm(
            estimates[:, :, None] - estimates[:, None, :], axis=3
        )
        off_diagonal = ~np.eye(blocks, dtype=bool)
        radii = np.quantile(
            distances[:, off_diagonal].reshape(len(grid), blocks, blocks - 1),
            _RADIUS_QUANTILE,
            axis=2,
            method="higher",
        )
        q_index, block = np.unravel_index(np.argmin(radii), radii.shape)
        selected_q = grid[q_index]
        candidate = estimates[q_index, block].copy()
        coef = candidate
        if self.refit_full_sample:
            refit_grid = (1,) if selected_q == 1 else tuple(q for q in grid if 2 <= q <= selected_q)
            refit = _fit_path(X, y, refit_grid)
            coef = refit[-1].coef.copy()
            convergence.extend(fit.converged for fit in refit)
        result = AdaptiveLqRegressionResult(
            coef, selected_q, candidate, float(radii[q_index, block]), blocks,
            size, grid, radii, estimates, all(convergence),
        )
        self.result_ = result
        self.coef_ = result.coef.copy()
        self.n_features_in_ = d
        return self


__all__ = [
    "AdaptiveLqRegression",
    "AdaptiveLqRegressionResult",
    "LqRegressionFit",
    "fit_lad_regression",
    "fit_lq_regression",
]
