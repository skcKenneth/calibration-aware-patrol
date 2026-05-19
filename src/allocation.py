"""
Patrol allocation policies, taking a per-cell threat intensity (point estimate
or interval) and a total ranger budget, returning a per-cell allocation.

The defender objective is
    U(tau; z) = sum_i z_i (1 - exp(-lambda_det tau_i))
subject to sum_i tau_i = T,  0 <= tau_i <= tau_max.

This is concave separable in tau; the optimum has the water-filling form
    tau_i*(z) = (1 / lambda_det) * max(0, log(z_i) - log nu)
where nu is the dual price tied to the budget constraint. We solve by
bisection on nu (faster and more transparent than SLSQP for this structure).

For DRO under an interval z_i in [z_i^L, z_i^U], the inner minimisation gives
z_i = z_i^L for every cell (since U is increasing in z_i). The outer max then
becomes the same water-filling problem applied to z^L -- which under-allocates.
We therefore use a chance-constrained / "robust upper" variant: plan against
the (1 - alpha) upper bound, then evaluate on the truth. This rewards the
defender for hedging against under-estimation -- which is the asymmetric
regime that matters for security (false negatives are costlier than false
positives).
"""
from __future__ import annotations

import numpy as np


def water_fill(z: np.ndarray, T: float, lam: float = 1.0,
               tau_max: float | None = None) -> np.ndarray:
    """Allocate T units across N cells to maximise
        sum_i z_i (1 - exp(-lam tau_i)),  sum tau = T, 0 <= tau <= tau_max.

    Uses bisection on the dual variable nu. z_i must be non-negative.
    """
    z = np.asarray(z, dtype=float)
    N = len(z)
    z_pos = np.maximum(z, 1e-12)
    if tau_max is None:
        tau_max = T  # effectively unconstrained

    def alloc(nu):
        # tau_i = (1/lam) log(lam z_i / nu) clamped to [0, tau_max]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (1.0 / lam) * np.log(lam * z_pos / max(nu, 1e-300))
        return np.clip(t, 0.0, tau_max)

    # Bracket nu so total allocation crosses T
    nu_hi = lam * z_pos.max() * 1.01           # tau == 0 everywhere
    nu_lo = lam * z_pos.min() * np.exp(-lam * tau_max) / max(N, 1)
    nu_lo = max(nu_lo, 1e-12)

    # Ensure nu_lo gives total > T and nu_hi gives total < T
    for _ in range(80):
        if alloc(nu_lo).sum() > T:
            break
        nu_lo *= 0.5
    for _ in range(80):
        if alloc(nu_hi).sum() < T:
            break
        nu_hi *= 2.0

    for _ in range(120):
        nu_mid = np.sqrt(nu_lo * nu_hi)
        total = alloc(nu_mid).sum()
        if abs(total - T) < 1e-6 * max(T, 1.0):
            break
        if total > T:
            nu_lo = nu_mid
        else:
            nu_hi = nu_mid

    t = alloc(np.sqrt(nu_lo * nu_hi))
    # Final renormalisation to hit the budget exactly under tau_max clamp
    s = t.sum()
    if s > 0:
        t = t * (T / s)
    t = np.clip(t, 0.0, tau_max)
    # Distribute residual onto the largest-z unclamped cells
    residual = T - t.sum()
    if abs(residual) > 1e-6:
        order = np.argsort(-z_pos)
        room = tau_max - t
        for idx in order:
            if residual <= 1e-9:
                break
            add = min(room[idx], residual)
            t[idx] += add
            residual -= add
    return t


def utility(tau: np.ndarray, z: np.ndarray, lam: float = 1.0) -> float:
    """True utility evaluated against truth z."""
    return float(np.sum(z * (1.0 - np.exp(-lam * tau))))


# ---------------------------------------------------------------------------
# Concrete policies. Each returns an allocation vector tau.
# ---------------------------------------------------------------------------

def policy_oracle(z_true: np.ndarray, T: float, lam: float, tau_max: float):
    return water_fill(z_true, T, lam, tau_max)


def policy_naive(z_hat_raw: np.ndarray, T: float, lam: float, tau_max: float):
    """Use uncalibrated point estimate of threat intensity."""
    return water_fill(z_hat_raw, T, lam, tau_max)


def policy_calibrated(z_hat_cal: np.ndarray, T: float, lam: float,
                      tau_max: float):
    """Use temperature-scaled (calibrated) point estimate of threat intensity."""
    return water_fill(z_hat_cal, T, lam, tau_max)


def policy_dro_upper(z_lo: np.ndarray, z_hi: np.ndarray,
                     T: float, lam: float, tau_max: float):
    """Plan against the (1 - alpha) conformal upper bound -- 'pessimistic
    about how much threat is hidden in cells the model thinks are quiet'.
    Risk-aware for false-negative-costly settings (anti-poaching)."""
    return water_fill(z_hi, T, lam, tau_max)


# ---------------------------------------------------------------------------
# Regret = U(oracle) - U(policy), evaluated on truth
# ---------------------------------------------------------------------------

def regret(tau: np.ndarray, tau_oracle: np.ndarray, z_true: np.ndarray,
           lam: float = 1.0) -> float:
    return utility(tau_oracle, z_true, lam) - utility(tau, z_true, lam)
