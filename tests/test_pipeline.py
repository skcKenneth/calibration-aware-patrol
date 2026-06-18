from __future__ import annotations

import numpy as np

from experiments.run_main import pipeline
from src.synthetic import WorldConfig, make_world


def test_data_roles_are_disjoint_objects() -> None:
    world = make_world(WorldConfig(L=6, seed=7))
    fit = world.batches["temperature_fit"]
    evaluation = world.batches["evaluation"]
    deployment = world.batches["deployment"]
    assert fit is not evaluation and evaluation is not deployment and fit is not deployment
    assert not np.shares_memory(fit.M, evaluation.M)
    assert not np.shares_memory(evaluation.M, deployment.M)


def test_fit_size_changes_only_fit_subset() -> None:
    world = make_world(WorldConfig(L=8, seed=9))
    small = pipeline(world=world, fit_fraction=0.20, fit_subsample_seed=4, budget=15.0)
    large = pipeline(world=world, fit_fraction=1.00, fit_subsample_seed=4, budget=15.0)
    assert len(small["fit"].y) < len(large["fit"].y)
    assert np.array_equal(small["deployment"].y, large["deployment"].y)
    assert np.array_equal(small["deployment"].cell, large["deployment"].cell)
    assert np.allclose(small["z_expected"], large["z_expected"])


def test_pipeline_reports_realised_count_coverage() -> None:
    out = pipeline(
        WorldConfig(L=10, seed=11, mu_temperature_fit=20, mu_evaluation=15),
        budget=20.0,
        tau_max=2.0,
    )
    assert 0.0 <= out["predictive_coverage"] <= 1.0
    assert np.all(out["z_realised"] >= 0)
    assert np.all(out["z_lower"] <= out["z_upper"])
    assert np.isclose(out["tau_oracle"].sum(), 20.0, atol=1e-8)
