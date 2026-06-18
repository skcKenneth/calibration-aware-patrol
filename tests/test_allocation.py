from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from src.allocation import (
    cost_sensitive_allocate,
    operational_loss,
    utility,
    water_fill,
)


def test_water_fill_satisfies_budget_and_bounds() -> None:
    z = np.array([0.0, 0.5, 2.0, 8.0, 1.0])
    allocation = water_fill(z, budget=4.0, lam=0.4, tau_max=1.2)
    assert np.isclose(allocation.sum(), 4.0, atol=1e-8)
    assert np.all(allocation >= 0.0)
    assert np.all(allocation <= 1.2 + 1e-10)


def test_water_fill_matches_generic_optimizer() -> None:
    rng = np.random.default_rng(8)
    z = rng.uniform(0.1, 5.0, size=7)
    budget = 4.5
    lam = 0.6
    tau_max = 1.1
    closed_form = water_fill(z, budget, lam, tau_max)

    result = minimize(
        lambda tau: -utility(tau, z, lam),
        x0=np.full(len(z), budget / len(z)),
        method="SLSQP",
        bounds=[(0.0, tau_max)] * len(z),
        constraints={"type": "eq", "fun": lambda tau: tau.sum() - budget},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    assert result.success
    assert utility(closed_form, z, lam) >= utility(result.x, z, lam) - 1e-7


def test_cost_sensitive_allocator_respects_maximum_budget() -> None:
    z = np.array([0.2, 1.0, 3.0, 8.0])
    allocation = cost_sensitive_allocate(
        z,
        max_budget=3.0,
        lam=0.4,
        tau_max=1.5,
        miss_cost=2.0,
        patrol_cost=np.array([1.0, 1.1, 0.9, 1.2]),
    )
    assert allocation.sum() <= 3.0 + 1e-8
    assert np.all((allocation >= 0.0) & (allocation <= 1.5 + 1e-10))
    zero = np.zeros_like(z)
    assert operational_loss(
        allocation,
        z,
        lam=0.4,
        miss_cost=2.0,
        patrol_cost=np.array([1.0, 1.1, 0.9, 1.2]),
    ) <= operational_loss(
        zero,
        z,
        lam=0.4,
        miss_cost=2.0,
        patrol_cost=np.array([1.0, 1.1, 0.9, 1.2]),
    )
