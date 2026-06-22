"""Patrol-allocation objectives and policies.

``water_fill`` maximises expected intercepted events under a fixed budget,
``sum_i z_i * (1 - exp(-lambda*tau_i))``.

``cost_sensitive_allocate`` may leave budget unused while minimising

``miss_cost * sum_i z_i exp(-lambda*tau_i) + sum_i c_i tau_i``.

The upper-interval policy is a predictive upper bound, not distributionally
robust optimisation (no ambiguity set over distributions).
"""
from __future__ import annotations

import numpy as np


def _validate_inputs(
    z: np.ndarray,
    budget: float,
    lam: float,
    tau_max: float | None,
) -> tuple[np.ndarray, float]:
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or len(z) == 0:
        raise ValueError("z must be a non-empty 1D array")
    if not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError("z must contain finite non-negative values")
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and non-negative")
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be finite and positive")
    if tau_max is None:
        tau_max = max(float(budget), 1.0)
    if not np.isfinite(tau_max) or tau_max <= 0:
        raise ValueError("tau_max must be finite and positive")
    if budget > len(z) * tau_max + 1e-9:
        raise ValueError("budget exceeds total per-cell capacity")
    return z, float(tau_max)


def _distribute_residual(
    allocation: np.ndarray,
    residual: float,
    tau_max: float,
    order: np.ndarray,
) -> np.ndarray:
    """Numerically close a tiny budget residual without violating bounds."""
    if residual > 0:
        for index in order:
            room = tau_max - allocation[index]
            if room <= 0:
                continue
            addition = min(room, residual)
            allocation[index] += addition
            residual -= addition
            if residual <= 1e-12:
                break
    elif residual < 0:
        residual = -residual
        for index in order[::-1]:
            removable = allocation[index]
            if removable <= 0:
                continue
            reduction = min(removable, residual)
            allocation[index] -= reduction
            residual -= reduction
            if residual <= 1e-12:
                break
    return allocation


def water_fill(
    z: np.ndarray,
    budget: float,
    lam: float = 1.0,
    tau_max: float | None = None,
) -> np.ndarray:
    """Maximise intercepted-event utility under an exact fixed budget.

    Parameters
    ----------
    z:
        Non-negative planning intensity for each cell.
    budget:
        Total patrol hours that must be allocated.
    lam:
        Detection-efficiency parameter.
    tau_max:
        Maximum patrol hours in any one cell.
    """
    z, tau_max = _validate_inputs(z, budget, lam, tau_max)
    n_cells = len(z)
    if budget == 0:
        return np.zeros(n_cells, dtype=float)

    positive = z > 0
    allocation = np.zeros(n_cells, dtype=float)
    positive_capacity = int(positive.sum()) * tau_max
    target_positive = min(float(budget), positive_capacity)

    if target_positive > 0:
        z_positive = z[positive]

        def allocate_positive(dual: float) -> np.ndarray:
            with np.errstate(divide="ignore"):
                candidate = np.log(lam * z_positive / dual) / lam
            return np.clip(candidate, 0.0, tau_max)

        dual_high = lam * float(z_positive.max()) * (1.0 + 1e-12)
        dual_low = max(
            np.finfo(float).tiny,
            lam * float(z_positive.min()) * np.exp(-lam * tau_max) * 1e-6,
        )
        for _ in range(250):
            dual_mid = np.sqrt(dual_low * dual_high)
            total = float(allocate_positive(dual_mid).sum())
            if total > target_positive:
                dual_low = dual_mid
            else:
                dual_high = dual_mid
        positive_allocation = allocate_positive(np.sqrt(dual_low * dual_high))
        positive_order = np.argsort(-z_positive)
        positive_allocation = _distribute_residual(
            positive_allocation,
            target_positive - float(positive_allocation.sum()),
            tau_max,
            positive_order,
        )
        allocation[positive] = positive_allocation

    # If every useful cell is saturated, the remaining fixed budget has zero
    # marginal utility.  Allocate it deterministically among zero-intensity cells.
    remaining = float(budget - allocation.sum())
    if remaining > 1e-10:
        zero_indices = np.flatnonzero(~positive)
        if len(zero_indices) == 0:
            zero_indices = np.arange(n_cells)
        allocation = _distribute_residual(
            allocation,
            remaining,
            tau_max,
            zero_indices,
        )

    if not np.isclose(allocation.sum(), budget, atol=1e-8):
        raise RuntimeError("water-filling solver failed to satisfy the budget")
    if np.any(allocation < -1e-10) or np.any(allocation > tau_max + 1e-10):
        raise RuntimeError("water-filling solver violated allocation bounds")
    return np.clip(allocation, 0.0, tau_max)


def utility(tau: np.ndarray, z: np.ndarray, lam: float = 1.0) -> float:
    """Expected intercepted events under the true intensity ``z``."""
    tau = np.asarray(tau, dtype=float)
    z = np.asarray(z, dtype=float)
    if tau.shape != z.shape:
        raise ValueError("tau and z must have the same shape")
    return float(np.sum(z * (1.0 - np.exp(-lam * tau))))


def missed_events(tau: np.ndarray, z: np.ndarray, lam: float = 1.0) -> float:
    """Expected events not intercepted: ``sum_i z_i exp(-lambda*tau_i)``."""
    tau = np.asarray(tau, dtype=float)
    z = np.asarray(z, dtype=float)
    if tau.shape != z.shape:
        raise ValueError("tau and z must have the same shape")
    return float(np.sum(z * np.exp(-lam * tau)))


def operational_loss(
    tau: np.ndarray,
    z: np.ndarray,
    *,
    lam: float,
    miss_cost: float,
    patrol_cost: float | np.ndarray,
) -> float:
    """Cost of missed events plus patrol-hour expenditure."""
    tau = np.asarray(tau, dtype=float)
    z = np.asarray(z, dtype=float)
    costs = np.broadcast_to(np.asarray(patrol_cost, dtype=float), tau.shape)
    if miss_cost <= 0 or np.any(costs <= 0):
        raise ValueError("miss_cost and patrol_cost must be positive")
    return miss_cost * missed_events(tau, z, lam) + float(np.sum(costs * tau))


def cost_sensitive_allocate(
    z: np.ndarray,
    max_budget: float,
    *,
    lam: float = 1.0,
    tau_max: float | None = None,
    miss_cost: float = 1.0,
    patrol_cost: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Minimise operational loss subject to ``sum(tau) <= max_budget``.

    The first-order solution is

    ``tau_i = clip(log(miss_cost*lam*z_i / (c_i + nu))/lam, 0, tau_max)``,

    where ``nu`` is zero if the maximum budget is slack and otherwise is found
    by bisection.  Unlike the fixed-budget utility objective, this formulation
    makes the false-negative/patrol-cost trade-off explicit and keeps units
    consistent.
    """
    z, tau_max = _validate_inputs(z, max_budget, lam, tau_max)
    costs = np.broadcast_to(np.asarray(patrol_cost, dtype=float), z.shape).copy()
    if not np.isfinite(miss_cost) or miss_cost <= 0:
        raise ValueError("miss_cost must be finite and positive")
    if not np.all(np.isfinite(costs)) or np.any(costs <= 0):
        raise ValueError("patrol_cost must contain finite positive values")
    if max_budget == 0:
        return np.zeros_like(z)

    numerator = miss_cost * lam * z

    def allocation_for_dual(dual: float) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            candidate = np.log(numerator / (costs + dual)) / lam
        candidate = np.where(numerator > 0, candidate, 0.0)
        return np.clip(candidate, 0.0, tau_max)

    allocation = allocation_for_dual(0.0)
    if allocation.sum() <= max_budget + 1e-10:
        return allocation

    dual_low = 0.0
    dual_high = max(float(numerator.max()), 1.0)
    while allocation_for_dual(dual_high).sum() > max_budget:
        dual_high *= 2.0

    for _ in range(250):
        dual_mid = 0.5 * (dual_low + dual_high)
        if allocation_for_dual(dual_mid).sum() > max_budget:
            dual_low = dual_mid
        else:
            dual_high = dual_mid
    allocation = allocation_for_dual(0.5 * (dual_low + dual_high))
    residual = max_budget - float(allocation.sum())
    if 0 < residual < 1e-8:
        allocation = _distribute_residual(
            allocation,
            residual,
            tau_max,
            np.argsort(-z),
        )
    return allocation


# ---------------------------------------------------------------------------
# Fixed-budget policy wrappers
# ---------------------------------------------------------------------------

def policy_oracle(z_true: np.ndarray, T: float, lam: float, tau_max: float) -> np.ndarray:
    return water_fill(z_true, T, lam, tau_max)


def policy_naive(z_hat_raw: np.ndarray, T: float, lam: float, tau_max: float) -> np.ndarray:
    return water_fill(z_hat_raw, T, lam, tau_max)


def policy_temperature_scaled(
    z_hat_temperature: np.ndarray,
    T: float,
    lam: float,
    tau_max: float,
) -> np.ndarray:
    return water_fill(z_hat_temperature, T, lam, tau_max)


def policy_calibrated(
    z_hat_calibrated: np.ndarray,
    T: float,
    lam: float,
    tau_max: float,
) -> np.ndarray:
    """Use the task-specific calibrated point estimate."""
    return water_fill(z_hat_calibrated, T, lam, tau_max)


def policy_predictive_upper(
    z_lower: np.ndarray,
    z_upper: np.ndarray,
    T: float,
    lam: float,
    tau_max: float,
) -> np.ndarray:
    """Plan against the upper endpoint of a predictive count interval."""
    del z_lower  # retained in the signature to make the interval input explicit
    return water_fill(z_upper, T, lam, tau_max)


def policy_dro_upper(
    z_lo: np.ndarray,
    z_hi: np.ndarray,
    T: float,
    lam: float,
    tau_max: float,
) -> np.ndarray:
    """Backward-compatible alias; the policy is not distributionally robust."""
    return policy_predictive_upper(z_lo, z_hi, T, lam, tau_max)


def regret(
    tau: np.ndarray,
    tau_oracle: np.ndarray,
    z_true: np.ndarray,
    lam: float = 1.0,
) -> float:
    """Oracle-relative utility regret evaluated on ``z_true``."""
    return utility(tau_oracle, z_true, lam) - utility(tau, z_true, lam)
