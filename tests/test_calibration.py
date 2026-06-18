from __future__ import annotations

import numpy as np

from src.calibration import (
    apply_binary_platt,
    apply_temperature,
    calibrate_interval_expansion,
    expand_predictive_intervals,
    fit_binary_platt,
    fit_temperature,
    negative_log_likelihood,
    poisson_binomial_interval,
    poisson_binomial_pmf,
)
from src.synthetic import WorldConfig, make_world
from src.calibration import get_flat_batch


def test_temperature_scaling_reduces_fit_nll() -> None:
    world = make_world(WorldConfig(L=8, seed=2))
    fit = get_flat_batch(world, "temperature_fit")
    temperature = fit_temperature(fit.pi, fit.y)
    scaled = apply_temperature(fit.pi, temperature)
    assert temperature > 0
    assert negative_log_likelihood(scaled, fit.y) <= negative_log_likelihood(fit.pi, fit.y)


def test_binary_platt_is_monotone_and_bounded() -> None:
    probability = np.array([0.01, 0.10, 0.25, 0.50, 0.80, 0.95])
    target = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    model = fit_binary_platt(probability, target)
    calibrated = apply_binary_platt(probability, model)
    assert model.slope > 0
    assert np.all((calibrated > 0) & (calibrated < 1))
    assert np.all(np.diff(calibrated) >= 0)


def test_poisson_binomial_known_distribution() -> None:
    pmf = poisson_binomial_pmf(np.array([0.5, 0.5]))
    assert np.allclose(pmf, np.array([0.25, 0.50, 0.25]))
    assert poisson_binomial_interval(np.array([0.5, 0.5]), alpha=0.5) == (0, 1)


def test_interval_expansion_uses_held_out_miss_distance() -> None:
    lower = np.array([0.0, 1.0, 2.0])
    upper = np.array([1.0, 2.0, 3.0])
    realised = np.array([3.0, 2.0, 2.0])
    margin = calibrate_interval_expansion(lower, upper, realised, alpha=0.25)
    assert margin == 2
    expanded_lower, expanded_upper = expand_predictive_intervals(
        lower, upper, np.array([4.0, 4.0, 4.0]), margin
    )
    assert np.array_equal(expanded_lower, np.array([0.0, 0.0, 0.0]))
    assert np.array_equal(expanded_upper, np.array([3.0, 4.0, 4.0]))


def test_binary_platt_handles_single_class_subset() -> None:
    probability = np.array([0.1, 0.2, 0.3])
    target = np.zeros(3)
    model = fit_binary_platt(probability, target)
    calibrated = apply_binary_platt(probability, model)
    assert np.all(np.isfinite(calibrated))
    assert np.all((calibrated > 0) & (calibrated < 1))
