"""Run the calibration-aware patrol-allocation experiments.

Uses independent batches for fitting, evaluation, interval calibration, and
deployment. Reports risk-class calibration, Poisson-binomial count intervals,
and an upper-bound policy (not DRO). Includes calibration-size sweeps and an
asymmetric-cost study.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.allocation import (  # noqa: E402
    cost_sensitive_allocate,
    operational_loss,
    policy_calibrated,
    policy_naive,
    policy_oracle,
    policy_predictive_upper,
    policy_temperature_scaled,
    regret,
    utility,
)
from src.calibration import (  # noqa: E402
    apply_binary_platt,
    apply_temperature,
    binary_brier,
    binary_ece,
    binary_reliability_curve,
    calibrate_interval_expansion,
    empirical_coverage,
    expand_predictive_intervals,
    expected_count_from_probability,
    fit_binary_platt,
    fit_temperature,
    get_flat_batch,
    multiclass_brier,
    negative_log_likelihood,
    per_cell_predictive_intervals,
    realised_count,
    subsample_flat_batch,
    top_label_ece,
    top_label_reliability_curve,
)
from src.synthetic import World, WorldConfig, make_world  # noqa: E402
from src.figure_style import (  # noqa: E402
    COLORS,
    DOUBLE_COLUMN,
    PATROL_CMAP,
    SINGLE_COLUMN,
    apply_editorial_style,
    clean_axis,
    compact_legend,
    panel_label,
    save_figure,
    shade_uncertainty,
)


FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)
apply_editorial_style()


def pipeline(
    cfg: WorldConfig | None = None,
    *,
    world: World | None = None,
    fit_fraction: float = 1.0,
    fit_subsample_seed: int = 0,
    alpha: float = 0.10,
    budget: float = 60.0,
    lam: float = 0.25,
    tau_max: float = 3.0,
) -> dict[str, Any]:
    """Run one end-to-end experiment with disjoint data roles."""
    if world is None:
        world = make_world(cfg or WorldConfig())
    cfg = world.cfg
    n_cells = world.N
    threat_class = world.threat_class

    fit_pool = get_flat_batch(world, "temperature_fit")
    fit = subsample_flat_batch(
        fit_pool,
        fit_fraction,
        seed=fit_subsample_seed,
    )
    evaluation = get_flat_batch(world, "evaluation")
    interval_calibration = get_flat_batch(world, "interval_calibration")
    deployment = get_flat_batch(world, "deployment")

    temperature = fit_temperature(fit.pi, fit.y)
    platt = fit_binary_platt(
        fit.pi[:, threat_class],
        (fit.y == threat_class).astype(float),
    )

    evaluation_temperature = apply_temperature(evaluation.pi, temperature)
    evaluation_threat_raw = evaluation.pi[:, threat_class]
    evaluation_threat_temperature = evaluation_temperature[:, threat_class]
    evaluation_threat_platt = apply_binary_platt(evaluation_threat_raw, platt)
    evaluation_target = (evaluation.y == threat_class).astype(float)

    interval_threat_platt = apply_binary_platt(
        interval_calibration.pi[:, threat_class],
        platt,
    )

    deployment_temperature = apply_temperature(deployment.pi, temperature)
    deployment_threat_raw = deployment.pi[:, threat_class]
    deployment_threat_temperature = deployment_temperature[:, threat_class]
    deployment_threat_platt = apply_binary_platt(deployment_threat_raw, platt)

    z_raw, n_deployment = expected_count_from_probability(
        deployment_threat_raw,
        deployment.cell,
        n_cells,
    )
    z_temperature, _ = expected_count_from_probability(
        deployment_threat_temperature,
        deployment.cell,
        n_cells,
    )
    _, interval_lower, interval_upper, _ = per_cell_predictive_intervals(
        interval_threat_platt,
        interval_calibration.cell,
        n_cells,
        alpha=alpha,
    )
    interval_realised = realised_count(
        interval_calibration.y,
        interval_calibration.cell,
        n_cells,
        threat_class,
    )
    interval_expansion = calibrate_interval_expansion(
        interval_lower,
        interval_upper,
        interval_realised,
        alpha=alpha,
    )

    z_calibrated, z_lower_base, z_upper_base, _ = per_cell_predictive_intervals(
        deployment_threat_platt,
        deployment.cell,
        n_cells,
        alpha=alpha,
    )
    z_lower, z_upper = expand_predictive_intervals(
        z_lower_base,
        z_upper_base,
        n_deployment,
        interval_expansion,
    )
    z_expected = n_deployment * world.p_true[:, threat_class]
    z_realised = realised_count(
        deployment.y,
        deployment.cell,
        n_cells,
        threat_class,
    )
    base_predictive_coverage = empirical_coverage(
        z_lower_base,
        z_upper_base,
        z_realised,
    )
    predictive_coverage = empirical_coverage(z_lower, z_upper, z_realised)

    tau_oracle = policy_oracle(z_expected, budget, lam, tau_max)
    tau_naive = policy_naive(z_raw, budget, lam, tau_max)
    tau_temperature = policy_temperature_scaled(
        z_temperature,
        budget,
        lam,
        tau_max,
    )
    tau_calibrated = policy_calibrated(
        z_calibrated,
        budget,
        lam,
        tau_max,
    )
    tau_upper = policy_predictive_upper(
        z_lower,
        z_upper,
        budget,
        lam,
        tau_max,
    )

    utilities = {
        "oracle": utility(tau_oracle, z_expected, lam),
        "naive": utility(tau_naive, z_expected, lam),
        "temperature": utility(tau_temperature, z_expected, lam),
        "calibrated": utility(tau_calibrated, z_expected, lam),
        "upper": utility(tau_upper, z_expected, lam),
    }
    regrets = {
        name: regret(tau, tau_oracle, z_expected, lam)
        for name, tau in {
            "naive": tau_naive,
            "temperature": tau_temperature,
            "calibrated": tau_calibrated,
            "upper": tau_upper,
        }.items()
    }

    metrics = {
        "top_ece_raw": top_label_ece(evaluation.pi, evaluation.y),
        "top_ece_temperature": top_label_ece(
            evaluation_temperature,
            evaluation.y,
        ),
        "multiclass_brier_raw": multiclass_brier(evaluation.pi, evaluation.y),
        "multiclass_brier_temperature": multiclass_brier(
            evaluation_temperature,
            evaluation.y,
        ),
        "multiclass_nll_raw": negative_log_likelihood(evaluation.pi, evaluation.y),
        "multiclass_nll_temperature": negative_log_likelihood(
            evaluation_temperature,
            evaluation.y,
        ),
        "threat_ece_raw": binary_ece(evaluation_threat_raw, evaluation_target),
        "threat_ece_temperature": binary_ece(
            evaluation_threat_temperature,
            evaluation_target,
        ),
        "threat_ece_platt": binary_ece(
            evaluation_threat_platt,
            evaluation_target,
        ),
        "threat_brier_raw": binary_brier(evaluation_threat_raw, evaluation_target),
        "threat_brier_temperature": binary_brier(
            evaluation_threat_temperature,
            evaluation_target,
        ),
        "threat_brier_platt": binary_brier(
            evaluation_threat_platt,
            evaluation_target,
        ),
    }

    return {
        "world": world,
        "cfg": cfg,
        "fit": fit,
        "evaluation": evaluation,
        "interval_calibration": interval_calibration,
        "deployment": deployment,
        "temperature": temperature,
        "platt": platt,
        "evaluation_temperature": evaluation_temperature,
        "evaluation_threat_raw": evaluation_threat_raw,
        "evaluation_threat_temperature": evaluation_threat_temperature,
        "evaluation_threat_platt": evaluation_threat_platt,
        "evaluation_target": evaluation_target,
        "deployment_threat_raw": deployment_threat_raw,
        "deployment_threat_temperature": deployment_threat_temperature,
        "interval_threat_platt": interval_threat_platt,
        "deployment_threat_platt": deployment_threat_platt,
        "n_deployment": n_deployment,
        "z_expected": z_expected,
        "z_realised": z_realised,
        "z_raw": z_raw,
        "z_temperature": z_temperature,
        "z_calibrated": z_calibrated,
        "z_lower_base": z_lower_base,
        "z_upper_base": z_upper_base,
        "z_lower": z_lower,
        "z_upper": z_upper,
        "interval_expansion": interval_expansion,
        "base_predictive_coverage": base_predictive_coverage,
        "predictive_coverage": predictive_coverage,
        "tau_oracle": tau_oracle,
        "tau_naive": tau_naive,
        "tau_temperature": tau_temperature,
        "tau_calibrated": tau_calibrated,
        "tau_upper": tau_upper,
        "utilities": utilities,
        "regrets": regrets,
        "metrics": metrics,
        "alpha": alpha,
        "budget": budget,
        "lam": lam,
        "tau_max": tau_max,
        "fit_fraction": fit_fraction,
    }


def _headline_row(out: dict[str, Any]) -> dict[str, float]:
    oracle_utility = out["utilities"]["oracle"]
    naive_regret = out["regrets"]["naive"]
    result: dict[str, float] = {
        "seed": float(out["cfg"].seed),
        "T_true": float(out["cfg"].T_true),
        "T_hat": float(out["temperature"]),
        "platt_slope": float(out["platt"].slope),
        "platt_intercept": float(out["platt"].intercept),
        "fit_images": float(len(out["fit"].y)),
        "evaluation_images": float(len(out["evaluation"].y)),
        "interval_calibration_images": float(len(out["interval_calibration"].y)),
        "deployment_images": float(len(out["deployment"].y)),
        "interval_expansion": float(out["interval_expansion"]),
        "base_predictive_coverage": float(out["base_predictive_coverage"]),
        "predictive_coverage": float(out["predictive_coverage"]),
        "nominal_predictive_coverage": float(1.0 - out["alpha"]),
    }
    result.update({key: float(value) for key, value in out["metrics"].items()})
    for policy, value in out["utilities"].items():
        result[f"utility_{policy}"] = float(value)
    for policy, value in out["regrets"].items():
        result[f"regret_{policy}"] = float(value)
        result[f"regret_{policy}_pct"] = float(100.0 * value / oracle_utility)
    for policy in ("temperature", "calibrated", "upper"):
        closed = 0.0 if naive_regret <= 1e-12 else 100.0 * (
            naive_regret - out["regrets"][policy]
        ) / naive_regret
        result[f"pct_naive_regret_closed_{policy}"] = float(closed)
    return result


def _mean_sd(rows: list[dict[str, float]], keys: list[str]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "sd": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
        }
        for key in keys
    }


def run_single(*, make_figures: bool = True) -> tuple[dict[str, Any], dict[str, float]]:
    out = pipeline(WorldConfig(seed=1234))
    row = _headline_row(out)
    if make_figures:
        make_single_figures(out)
    return out, row


def make_single_figures(out: dict[str, Any]) -> None:
    """Write the main figure set to FIGDIR."""
    world = out["world"]
    cfg = out["cfg"]
    L = cfg.L
    threat = world.threat_class

    # Latent field, image density, expected counts.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(DOUBLE_COLUMN, 2.80),
        layout="constrained",
    )
    panels = [
        (
            world.p_true[:, threat].reshape(L, L),
            "magma",
            "Latent risk probability",
            "Probability",
            0.0,
            1.0,
        ),
        (
            out["n_deployment"].reshape(L, L),
            "cividis",
            "Deployment image density",
            "Images per cell",
            None,
            None,
        ),
        (
            out["z_expected"].reshape(L, L),
            "magma",
            "Expected risk count",
            "Expected count",
            0.0,
            None,
        ),
    ]
    for index, (axis, panel) in enumerate(zip(axes, panels)):
        data, cmap, title, cbar_label, vmin, vmax = panel
        image = axis.imshow(
            data,
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title, loc="left", pad=3)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.45)
            spine.set_color(COLORS["light_grey"])
        panel_label(axis, chr(ord("a") + index), x=-0.08, y=1.09)
        cbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
        cbar.ax.tick_params(length=2, width=0.5)
        cbar.outline.set_linewidth(0.45)
        cbar.set_label(cbar_label, labelpad=2)
    save_figure(fig, FIGDIR, "fig_world")
    plt.close(fig)

    # Reliability diagrams; marker size scales with bin count.
    evaluation = out["evaluation"]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(DOUBLE_COLUMN, 2.95),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    top_panels = [
        (
            axes[0],
            evaluation.pi,
            "Raw top-label",
            out["metrics"]["top_ece_raw"],
            COLORS["vermillion"],
        ),
        (
            axes[1],
            out["evaluation_temperature"],
            "Temperature-scaled",
            out["metrics"]["top_ece_temperature"],
            COLORS["blue"],
        ),
    ]
    for index, (axis, pi, title, ece, color) in enumerate(top_panels):
        _, mean_probability, event_rate, counts = top_label_reliability_curve(
            pi,
            evaluation.y,
            n_bins=10,
        )
        valid = counts > 0
        marker_size = 8.0 + 42.0 * counts[valid] / max(counts[valid].max(), 1)
        axis.plot([0, 1], [0, 1], color=COLORS["grey"], ls="--", lw=0.75, zorder=0)
        axis.plot(
            mean_probability[valid],
            event_rate[valid],
            color=color,
            lw=1.05,
            zorder=2,
        )
        axis.scatter(
            mean_probability[valid],
            event_rate[valid],
            s=marker_size,
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        axis.set_title(title, loc="left", pad=3)
        axis.text(
            0.04,
            0.94,
            f"ECE = {ece:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
        )
        panel_label(axis, chr(ord("a") + index))
        clean_axis(axis)

    axis = axes[2]
    threat_curves = [
        (
            out["evaluation_threat_raw"],
            "Raw",
            COLORS["vermillion"],
            "o",
            out["metrics"]["threat_ece_raw"],
        ),
        (
            out["evaluation_threat_temperature"],
            "Temperature",
            COLORS["blue"],
            "s",
            out["metrics"]["threat_ece_temperature"],
        ),
        (
            out["evaluation_threat_platt"],
            "Task-calibrated",
            COLORS["teal"],
            "^",
            out["metrics"]["threat_ece_platt"],
        ),
    ]
    axis.plot([0, 1], [0, 1], color=COLORS["grey"], ls="--", lw=0.75, zorder=0)
    for probability, label, color, marker, ece in threat_curves:
        _, mean_probability, event_rate, counts = binary_reliability_curve(
            probability,
            out["evaluation_target"],
            n_bins=10,
        )
        valid = counts > 0
        axis.plot(
            mean_probability[valid],
            event_rate[valid],
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=0.75,
            lw=1.05,
            ms=3.6,
            label=f"{label} ({ece:.3f})",
        )
    axis.set_title("Risk-proxy class", loc="left", pad=3)
    panel_label(axis, "c")
    axis.text(
        0.04,
        0.96,
        "ECE",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color=COLORS["grey"],
    )
    for line_index, (_, label, color, _, ece) in enumerate(threat_curves):
        axis.text(
            0.04,
            0.86 - 0.095 * line_index,
            f"{label}: {ece:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color=color,
        )
    clean_axis(axis)

    for axis in axes:
        # Pad x limits so edge markers are not clipped on export.
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xticks([0, 0.5, 1.0])
        axis.set_yticks([0, 0.5, 1.0])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed event rate")
    save_figure(fig, FIGDIR, "fig_reliability")
    plt.close(fig)

    # Count estimates and predictive intervals.
    z_true = out["z_expected"]
    z_max = max(
        z_true.max(),
        out["z_raw"].max(),
        out["z_temperature"].max(),
        out["z_calibrated"].max(),
    )
    z_limit = float(np.ceil(z_max / 5.0) * 5.0)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COLUMN, 5.25),
        layout="constrained",
    )
    flat_axes = axes.ravel()
    estimate_panels = [
        (out["z_raw"], "Raw", COLORS["vermillion"]),
        (out["z_temperature"], "Temperature", COLORS["blue"]),
        (out["z_calibrated"], "Task-calibrated", COLORS["teal"]),
    ]
    for index, (axis, (estimate, label, color)) in enumerate(
        zip(flat_axes[:3], estimate_panels)
    ):
        axis.scatter(
            z_true,
            estimate,
            s=7.5,
            alpha=0.42,
            color=color,
            edgecolor="none",
            rasterized=True,
        )
        axis.plot([0, z_limit], [0, z_limit], color=COLORS["grey"], ls="--", lw=0.7)
        slope = float((z_true @ estimate) / (z_true @ z_true + 1e-12))
        axis.set_title(label, loc="left", pad=3)
        axis.text(
            0.05,
            0.92,
            f"Slope = {slope:.2f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
        )
        margin = 0.02 * z_limit
        axis.set_xlim(-margin, z_limit + margin)
        axis.set_ylim(-margin, z_limit + margin)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Expected count")
        panel_label(axis, chr(ord("a") + index))
        clean_axis(axis)

    order = np.argsort(z_true)
    x = np.arange(len(order))
    axis = flat_axes[3]
    axis.fill_between(
        x,
        out["z_lower"][order],
        out["z_upper"][order],
        color=COLORS["sky"],
        alpha=0.24,
        linewidth=0,
        label=f"{100 * (1 - out['alpha']):.0f}% interval",
    )
    axis.plot(
        x,
        z_true[order],
        color=COLORS["blue"],
        lw=1.0,
        label="Expected count",
    )
    axis.scatter(
        x,
        out["z_realised"][order],
        s=3.0,
        color=COLORS["ink"],
        alpha=0.60,
        edgecolor="none",
        rasterized=True,
        label="Realised count",
    )
    axis.set_title("Predictive interval", loc="left", pad=3)
    axis.text(
        0.04,
        0.94,
        f"Coverage = {out['predictive_coverage']:.3f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
    )
    axis.set_xlabel("Cells ranked by expected count")
    panel_label(axis, "d")
    clean_axis(axis)
    compact_legend(axis, loc="upper left", bbox_to_anchor=(0.0, 0.82))
    axes[0, 0].set_ylabel("Estimated count")
    axes[1, 0].set_ylabel("Estimated count")
    save_figure(fig, FIGDIR, "fig_per_cell_estimates")
    plt.close(fig)

    # Allocation maps (2×3 grid + colour bar).
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(DOUBLE_COLUMN, 4.20),
        layout="constrained",
    )
    flat_axes = axes.ravel()
    allocations = [
        (out["tau_oracle"], "Oracle"),
        (out["tau_naive"], "Naive"),
        (out["tau_temperature"], "Temperature"),
        (out["tau_calibrated"], "Task-calibrated"),
        (out["tau_upper"], "Predictive upper"),
    ]
    maximum = max(allocation.max() for allocation, _ in allocations)
    image = None
    for index, (axis, (allocation, title)) in enumerate(zip(flat_axes[:5], allocations)):
        image = axis.imshow(
            allocation.reshape(L, L),
            cmap=PATROL_CMAP,
            origin="lower",
            interpolation="nearest",
            vmin=0,
            vmax=maximum,
        )
        axis.set_title(title, loc="left", pad=2)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.40)
            spine.set_color(COLORS["light_grey"])
        panel_label(axis, chr(ord("a") + index), x=-0.10, y=1.10)

    cbar_axis = flat_axes[5]
    cbar_axis.set_axis_off()
    assert image is not None
    inset = cbar_axis.inset_axes([0.36, 0.16, 0.16, 0.68])
    cbar = fig.colorbar(image, cax=inset)
    cbar.set_label("Patrol hours per cell", labelpad=3)
    cbar.ax.tick_params(length=2, width=0.5)
    cbar.outline.set_linewidth(0.45)
    save_figure(fig, FIGDIR, "fig_allocations")
    plt.close(fig)


def run_multiseed(n_seeds: int) -> tuple[list[dict[str, float]], dict[str, dict[str, float]]]:
    rows = [
        _headline_row(pipeline(WorldConfig(seed=1000 + seed)))
        for seed in range(n_seeds)
    ]
    keys = [
        "top_ece_raw",
        "top_ece_temperature",
        "threat_ece_raw",
        "threat_ece_temperature",
        "threat_ece_platt",
        "regret_naive_pct",
        "regret_temperature_pct",
        "regret_calibrated_pct",
        "regret_upper_pct",
        "pct_naive_regret_closed_temperature",
        "pct_naive_regret_closed_calibrated",
        "base_predictive_coverage",
        "predictive_coverage",
        "interval_expansion",
    ]
    return rows, _mean_sd(rows, keys)


def run_temperature_sweep(n_seeds: int) -> list[dict[str, float]]:
    temperatures = [0.30, 0.40, 0.50, 0.70, 0.90, 1.00, 1.20, 1.60, 2.00, 2.50]
    rows: list[dict[str, float]] = []
    for temperature in temperatures:
        seed_rows = []
        for seed in range(n_seeds):
            out = pipeline(
                WorldConfig(T_true=temperature, seed=2000 + seed),
                fit_subsample_seed=seed,
            )
            row = _headline_row(out)
            seed_rows.append(row)
        summary = {"T_true": float(temperature)}
        for key in (
            "regret_naive_pct",
            "regret_temperature_pct",
            "regret_calibrated_pct",
            "regret_upper_pct",
            "predictive_coverage",
        ):
            summary[f"{key}_mean"] = float(np.mean([row[key] for row in seed_rows]))
            summary[f"{key}_sd"] = float(np.std([row[key] for row in seed_rows], ddof=1))
        rows.append(summary)

    x = np.array([row["T_true"] for row in rows])
    fig, axis = plt.subplots(
        figsize=(SINGLE_COLUMN, 2.90),
        layout="constrained",
    )
    series = [
        ("regret_naive_pct", "Naive", "o", COLORS["vermillion"]),
        ("regret_temperature_pct", "Temperature", "s", COLORS["blue"]),
        ("regret_calibrated_pct", "Task-calibrated", "^", COLORS["teal"]),
        ("regret_upper_pct", "Predictive upper", "D", COLORS["gold"]),
    ]
    for base_key, label, marker, color in series:
        mean = np.array([row[f"{base_key}_mean"] for row in rows])
        sd = np.array([row[f"{base_key}_sd"] for row in rows])
        shade_uncertainty(axis, x, mean, sd, color=color)
        axis.plot(
            x,
            mean,
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=0.75,
            label=label,
            zorder=3,
        )
    axis.axvline(1.0, color=COLORS["grey"], ls="--", lw=0.75, zorder=0)
    axis.text(
        1.02,
        0.97,
        "Calibrated reference",
        transform=axis.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=6.2,
        color=COLORS["grey"],
    )
    axis.set_xlabel(r"Synthetic classifier temperature $T_{\mathrm{true}}$")
    axis.set_ylabel("Regret (% of oracle utility)")
    x_padding = 0.025 * (x.max() - x.min())
    axis.set_xlim(x.min() - x_padding, x.max() + x_padding)
    axis.set_ylim(bottom=0)
    clean_axis(axis)
    compact_legend(axis, loc="upper right", ncol=1)
    save_figure(fig, FIGDIR, "fig_regret_vs_T")
    plt.close(fig)
    return rows


def run_fit_size_sweep(n_seeds: int) -> list[dict[str, float]]:
    fractions = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
    by_fraction: dict[float, list[dict[str, float]]] = {fraction: [] for fraction in fractions}
    for seed in range(n_seeds):
        # The same world, evaluation batch, and deployment batch are reused for
        # every fit fraction.  Only the fixed temperature-fit pool is subsampled.
        world = make_world(WorldConfig(seed=3000 + seed))
        for fraction in fractions:
            out = pipeline(
                world=world,
                fit_fraction=fraction,
                fit_subsample_seed=17 + seed,
            )
            by_fraction[fraction].append(_headline_row(out))

    rows: list[dict[str, float]] = []
    for fraction in fractions:
        seed_rows = by_fraction[fraction]
        row = {
            "fit_fraction": float(fraction),
            "fit_images_mean": float(np.mean([item["fit_images"] for item in seed_rows])),
        }
        for key in (
            "regret_naive_pct",
            "regret_temperature_pct",
            "regret_calibrated_pct",
            "regret_upper_pct",
            "threat_ece_platt",
            "predictive_coverage",
        ):
            row[f"{key}_mean"] = float(np.mean([item[key] for item in seed_rows]))
            row[f"{key}_sd"] = float(np.std([item[key] for item in seed_rows], ddof=1))
        rows.append(row)

    x = np.array([row["fit_images_mean"] for row in rows])
    fig, axis = plt.subplots(
        figsize=(SINGLE_COLUMN, 2.90),
        layout="constrained",
    )
    series = [
        ("regret_naive_pct", "Naive", "o", COLORS["vermillion"]),
        ("regret_temperature_pct", "Temperature", "s", COLORS["blue"]),
        ("regret_calibrated_pct", "Task-calibrated", "^", COLORS["teal"]),
        ("regret_upper_pct", "Predictive upper", "D", COLORS["gold"]),
    ]
    for base_key, label, marker, color in series:
        mean = np.array([row[f"{base_key}_mean"] for row in rows])
        sd = np.array([row[f"{base_key}_sd"] for row in rows])
        shade_uncertainty(axis, x, mean, sd, color=color)
        axis.plot(
            x,
            mean,
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=0.75,
            label=label,
            zorder=3,
        )
    axis.set_xlabel("Labelled images used to fit calibrators")
    axis.set_ylabel("Regret (% of oracle utility)")
    x_padding = 0.025 * (x.max() - x.min())
    axis.set_xlim(x.min() - x_padding, x.max() + x_padding)
    axis.set_ylim(bottom=0)
    clean_axis(axis)
    compact_legend(axis, loc="upper right")
    save_figure(fig, FIGDIR, "fig_regret_vs_calsize")
    plt.close(fig)

    fig, axis = plt.subplots(
        figsize=(SINGLE_COLUMN, 2.75),
        layout="constrained",
    )
    coverage_mean = np.array([row["predictive_coverage_mean"] for row in rows])
    coverage_sd = np.array([row["predictive_coverage_sd"] for row in rows])
    shade_uncertainty(axis, x, coverage_mean, coverage_sd, color=COLORS["blue"])
    axis.plot(
        x,
        coverage_mean,
        color=COLORS["blue"],
        marker="o",
        markerfacecolor="white",
        markeredgewidth=0.75,
        label="Empirical coverage",
    )
    axis.axhline(
        0.90,
        color=COLORS["grey"],
        ls="--",
        lw=0.75,
        label="Nominal 90%",
    )
    axis.set_xlabel("Labelled images used to fit calibrators")
    axis.set_ylabel("Realised-count coverage")
    x_padding = 0.025 * (x.max() - x.min())
    axis.set_xlim(x.min() - x_padding, x.max() + x_padding)
    axis.set_ylim(0.78, 1.005)
    clean_axis(axis)
    compact_legend(axis, loc="lower right")
    save_figure(fig, FIGDIR, "fig_coverage")
    plt.close(fig)
    return rows


def run_asymmetric_cost_experiment(n_seeds: int) -> list[dict[str, float]]:
    """Compare point and upper-bound policies under asymmetric patrol cost.

    Point planning uses the calibrated mean count; upper-bound planning uses the
    predictive upper count. Loss is measured against the field-specific oracle.
    """
    miss_costs = [0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0]
    raw_rows: dict[float, list[dict[str, float]]] = {
        miss_cost: [] for miss_cost in miss_costs
    }

    for seed in range(n_seeds):
        out = pipeline(WorldConfig(seed=4000 + seed))
        L = out["cfg"].L
        coordinates = np.indices((L, L)).reshape(2, -1).T
        centre = np.array([(L - 1) / 2.0, (L - 1) / 2.0])
        distance = np.linalg.norm(coordinates - centre, axis=1)
        patrol_cost = 0.8 + 0.4 * distance / distance.max()

        for miss_cost in miss_costs:
            common = dict(
                max_budget=out["budget"],
                lam=out["lam"],
                tau_max=out["tau_max"],
                miss_cost=miss_cost,
                patrol_cost=patrol_cost,
            )
            point_allocation = cost_sensitive_allocate(out["z_calibrated"], **common)
            upper_allocation = cost_sensitive_allocate(out["z_upper"], **common)
            nominal_oracle = cost_sensitive_allocate(out["z_expected"], **common)
            stress_oracle = cost_sensitive_allocate(out["z_upper"], **common)

            def loss(allocation: np.ndarray, field: np.ndarray) -> float:
                return operational_loss(
                    allocation,
                    field,
                    lam=out["lam"],
                    miss_cost=miss_cost,
                    patrol_cost=patrol_cost,
                )

            nominal_optimum = loss(nominal_oracle, out["z_expected"])
            stress_optimum = loss(stress_oracle, out["z_upper"])
            nominal_point_excess = loss(point_allocation, out["z_expected"]) - nominal_optimum
            nominal_upper_excess = loss(upper_allocation, out["z_expected"]) - nominal_optimum
            stress_point_excess = loss(point_allocation, out["z_upper"]) - stress_optimum
            stress_upper_excess = loss(upper_allocation, out["z_upper"]) - stress_optimum

            raw_rows[miss_cost].append(
                {
                    "point_hours": float(point_allocation.sum()),
                    "upper_hours": float(upper_allocation.sum()),
                    "nominal_point_excess": float(max(nominal_point_excess, 0.0)),
                    "nominal_upper_excess": float(max(nominal_upper_excess, 0.0)),
                    "stress_point_excess": float(max(stress_point_excess, 0.0)),
                    "stress_upper_excess": float(max(stress_upper_excess, 0.0)),
                }
            )

    rows: list[dict[str, float]] = []
    for miss_cost in miss_costs:
        seed_rows = raw_rows[miss_cost]
        row: dict[str, float] = {"miss_cost": float(miss_cost)}
        for key in seed_rows[0]:
            values = [item[key] for item in seed_rows]
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        point_nominal = row["nominal_point_excess_mean"]
        upper_nominal = row["nominal_upper_excess_mean"]
        point_stress = row["stress_point_excess_mean"]
        upper_stress = row["stress_upper_excess_mean"]
        row["nominal_excess_reduction_pct"] = (
            0.0 if point_nominal <= 1e-12
            else 100.0 * (point_nominal - upper_nominal) / point_nominal
        )
        row["stress_excess_reduction_pct"] = (
            0.0 if point_stress <= 1e-12
            else 100.0 * (point_stress - upper_stress) / point_stress
        )
        rows.append(row)

    x = np.array([row["miss_cost"] for row in rows])
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN, 2.95),
        sharex=True,
        layout="constrained",
    )
    field_panels = [
        (
            axes[0],
            "nominal_point_excess",
            "nominal_upper_excess",
            "Nominal expected field",
        ),
        (
            axes[1],
            "stress_point_excess",
            "stress_upper_excess",
            "Upper-bound stress field",
        ),
    ]
    for index, (axis, point_key, upper_key, title) in enumerate(field_panels):
        for base_key, label, marker, color in [
            (point_key, "Task-calibrated point", "o", COLORS["blue"]),
            (upper_key, "Predictive upper", "s", COLORS["gold"]),
        ]:
            mean = np.array([row[f"{base_key}_mean"] for row in rows])
            sd = np.array([row[f"{base_key}_sd"] for row in rows])
            shade_uncertainty(axis, x, mean, sd, color=color)
            axis.plot(
                x,
                mean,
                color=color,
                marker=marker,
                markerfacecolor="white",
                markeredgewidth=0.75,
                label=label,
                zorder=3,
                clip_on=False,
            )
        axis.set_xscale("log")
        axis.set_xlim(x.min() / 1.08, x.max() * 1.08)
        axis.set_ylim(bottom=0)
        axis.set_title(title, loc="left", pad=3)
        axis.set_xlabel("Missed-event / patrol-hour cost")
        panel_label(axis, chr(ord("a") + index), x=-0.13, y=1.08)
        clean_axis(axis)
    axes[0].set_ylabel("Excess operational loss")
    compact_legend(axes[0], loc="upper left")
    save_figure(fig, FIGDIR, "fig_asymmetric_cost")
    plt.close(fig)
    return rows

def _write_seed_csv(rows: list[dict[str, float]]) -> None:
    path = ROOT / "results_seed_level.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use fewer random seeds for a fast validation run",
    )
    args = parser.parse_args()
    multiseed_n = 5 if args.quick else 20
    sweep_n = 3 if args.quick else 8

    out, single = run_single(make_figures=True)
    seed_rows, multiseed = run_multiseed(multiseed_n)
    temperature_sweep = run_temperature_sweep(sweep_n)
    fit_size_sweep = run_fit_size_sweep(sweep_n)
    asymmetric_cost = run_asymmetric_cost_experiment(sweep_n)

    results = {
        "schema_version": 2,
        "method_notes": {
            "data_roles": "disjoint temperature-fit, evaluation, and deployment batches",
            "decision_calibrator": "binary Platt scaling for the human-presence risk proxy",
            "count_interval": "Poisson-binomial realised-count interval with width calibrated on a separate deployment-sized labelled batch",
            "upper_policy": "predictive upper-bound / interval-robust; not distributionally robust optimisation",
        },
        "single": single,
        "multiseed_summary": multiseed,
        "temperature_sweep": temperature_sweep,
        "fit_size_sweep": fit_size_sweep,
        "asymmetric_cost": asymmetric_cost,
    }
    with (ROOT / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    _write_seed_csv(seed_rows)

    print("---- Corrected headline run ----")
    for key in (
        "deployment_images",
        "T_hat",
        "top_ece_raw",
        "top_ece_temperature",
        "threat_ece_raw",
        "threat_ece_temperature",
        "threat_ece_platt",
        "predictive_coverage",
        "regret_naive_pct",
        "regret_temperature_pct",
        "regret_calibrated_pct",
        "regret_upper_pct",
        "pct_naive_regret_closed_calibrated",
    ):
        print(f"  {key:42s} {single[key]:10.4f}")
    print(f"\nResults: {ROOT / 'results.json'}")
    print(f"Seed-level results: {ROOT / 'results_seed_level.csv'}")
    print(f"Figures: {FIGDIR}")


if __name__ == "__main__":
    main()
