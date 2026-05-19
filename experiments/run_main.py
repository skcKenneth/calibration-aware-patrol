"""
Main experiment: end-to-end calibration-aware patrol allocation pipeline.

Outputs (saved to ./figures/):
  fig_world.png             -- truth threat field + camera-trap network
  fig_reliability.png       -- reliability diagram before/after calibration
  fig_per_cell_estimates.png -- naive vs calibrated per-cell threat estimates
                                + conformal interval coverage panel
  fig_allocations.png        -- oracle / naive / calibrated / DRO allocations
  fig_regret_vs_T.png        -- regret vs miscalibration severity (T_true)
  fig_regret_vs_calsize.png  -- regret vs calibration set size
  fig_coverage.png           -- conformal interval empirical coverage

Numbers (printed and also saved to ./results.json):
  ECE / Brier before / after calibration
  Recovered temperature T_hat
  Per-policy utility, regret, regret-closed-percent
  Empirical conformal coverage
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.synthetic import WorldConfig, make_world, split_calibration
from src.calibration import (
    gather_calibration, fit_temperature, apply_temperature, ece,
    reliability_curve, expected_count, conformal_residuals,
    conformal_quantile, per_cell_intervals, empirical_coverage,
)
from src.allocation import (
    policy_oracle, policy_naive, policy_calibrated, policy_dro_upper,
    utility, regret,
)


FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)
RESULTS = {}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "font.family": "DejaVu Sans",
})


def brier(pi: np.ndarray, y: np.ndarray) -> float:
    K = pi.shape[1]
    onehot = np.zeros_like(pi)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((pi - onehot) ** 2).sum(axis=1).mean())


# ---------------------------------------------------------------------------
# Single-world pipeline
# ---------------------------------------------------------------------------
def pipeline(cfg: WorldConfig, frac_cal: float, alpha: float,
             budget: float, lam: float, tau_max: float, cal_seed: int = 0):
    world = make_world(cfg)
    N = cfg.L ** 2
    threat = cfg.K - 1
    is_cal = split_calibration(world, frac_cal=frac_cal, seed=cal_seed)
    b = gather_calibration(world, is_cal)
    pi_cal, y_cal = b["pi_cal"], b["y_cal"]
    pi_dep = b["pi_dep"]
    cell_cal, cell_dep = b["cell_cal"], b["cell_dep"]

    T_hat = fit_temperature(pi_cal, y_cal)
    pi_cal_T = apply_temperature(pi_cal, T_hat)
    pi_dep_T = apply_temperature(pi_dep, T_hat)

    z_naive, n_dep = expected_count(pi_dep, cell_dep, N, threat)
    z_cal, _       = expected_count(pi_dep_T, cell_dep, N, threat)
    z_true_dep     = n_dep * world.p_true[:, threat]

    residuals, _ = conformal_residuals(pi_cal_T, y_cal, cell_cal, N, threat,
                                       n_min=3)
    q = conformal_quantile(residuals, alpha=alpha)
    z_hat_cf, z_lo, z_hi, _ = per_cell_intervals(pi_dep_T, cell_dep, N,
                                                  threat, q)
    cov = empirical_coverage(z_lo, z_hi, z_true_dep)

    tau_or = policy_oracle(z_true_dep, budget, lam, tau_max)
    tau_nv = policy_naive(z_naive, budget, lam, tau_max)
    tau_cb = policy_calibrated(z_cal, budget, lam, tau_max)
    tau_dr = policy_dro_upper(z_lo, z_hi, budget, lam, tau_max)

    return dict(
        world=world, cfg=cfg, N=N, threat=threat, n_dep=n_dep,
        pi_cal=pi_cal, pi_cal_T=pi_cal_T, y_cal=y_cal,
        z_true_dep=z_true_dep, z_naive=z_naive, z_cal=z_cal,
        z_lo=z_lo, z_hi=z_hi, q_cf=q, cov=cov,
        tau_or=tau_or, tau_nv=tau_nv, tau_cb=tau_cb, tau_dr=tau_dr,
        u_or=utility(tau_or, z_true_dep, lam),
        u_nv=utility(tau_nv, z_true_dep, lam),
        u_cb=utility(tau_cb, z_true_dep, lam),
        u_dr=utility(tau_dr, z_true_dep, lam),
        T_hat=T_hat,
    )


# ---------------------------------------------------------------------------
def run_single():
    cfg = WorldConfig(L=20, K=4, seed=1234)
    out = pipeline(cfg, frac_cal=0.20, alpha=0.10, budget=60.0,
                   lam=0.25, tau_max=3.0)

    eb = ece(out["pi_cal"], out["y_cal"])
    ea = ece(out["pi_cal_T"], out["y_cal"])
    bb = brier(out["pi_cal"], out["y_cal"])
    ba = brier(out["pi_cal_T"], out["y_cal"])

    r_nv = regret(out["tau_nv"], out["tau_or"], out["z_true_dep"], 0.25)
    r_cb = regret(out["tau_cb"], out["tau_or"], out["z_true_dep"], 0.25)
    r_dr = regret(out["tau_dr"], out["tau_or"], out["z_true_dep"], 0.25)

    head = dict(
        T_true=cfg.T_true, T_hat=out["T_hat"],
        ece_before=eb, ece_after=ea,
        brier_before=bb, brier_after=ba,
        utility_oracle=out["u_or"], utility_naive=out["u_nv"],
        utility_calibrated=out["u_cb"], utility_dro=out["u_dr"],
        regret_naive=r_nv, regret_calibrated=r_cb, regret_dro=r_dr,
        regret_naive_pct=100 * r_nv / out["u_or"],
        regret_calibrated_pct=100 * r_cb / out["u_or"],
        regret_dro_pct=100 * r_dr / out["u_or"],
        pct_regret_closed_calibration=(100 * (r_nv - r_cb) / r_nv if r_nv > 1e-9 else 0.0),
        pct_regret_closed_dro=(100 * (r_nv - r_dr) / r_nv if r_nv > 1e-9 else 0.0),
        utility_uplift_calibration_pct=100 * (out["u_cb"] - out["u_nv"]) / out["u_nv"],
        utility_uplift_dro_pct=100 * (out["u_dr"] - out["u_nv"]) / out["u_nv"],
        conformal_coverage=out["cov"], nominal_coverage=0.90,
    )
    RESULTS["single"] = {k: float(v) for k, v in head.items()}
    _make_single_figures(out)
    return head


def _make_single_figures(out):
    cfg, L = out["cfg"], out["cfg"].L
    threat = out["threat"]
    world = out["world"]

    # World
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    im0 = axes[0].imshow(world.p_true[:, threat].reshape(L, L),
                         cmap="magma", origin="lower")
    axes[0].set_title(r"True $P(\mathrm{threat})$ per cell")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(world.M.reshape(L, L), cmap="viridis", origin="lower")
    axes[1].set_title(f"Camera-trap images per cell  ($\\mu={cfg.mu_images:.0f}$)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(out["z_true_dep"].reshape(L, L),
                         cmap="inferno", origin="lower")
    axes[2].set_title(r"True expected threat count $z_i^{\star}$")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_world.png"); plt.close(fig)

    # Reliability
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True)
    eb = ece(out["pi_cal"], out["y_cal"])
    ea = ece(out["pi_cal_T"], out["y_cal"])
    for ax, pi_set, title in [
        (axes[0], out["pi_cal"],   f"Uncalibrated (ECE = {eb:.3f})"),
        (axes[1], out["pi_cal_T"], f"Temperature-scaled, $\\hat T={out['T_hat']:.2f}$ (ECE = {ea:.3f})"),
    ]:
        c, mc, ma, cnt = reliability_curve(pi_set, out["y_cal"], n_bins=12)
        widths = 1.0 / 12.0
        ax.bar(c, ma, width=widths * 0.9, color="#4C72B0", alpha=0.45,
               label="Accuracy")
        ax.plot(c, mc, "o-", color="#C44E52", lw=1.4, ms=4.5,
                label="Mean confidence")
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="gray")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    axes[0].set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_reliability.png"); plt.close(fig)

    # Per-cell estimates + coverage
    z_true = out["z_true_dep"]
    z_max = max(z_true.max(), out["z_naive"].max(), out["z_cal"].max())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, zh, lab, col in [
        (axes[0], out["z_naive"], "Naive (uncalibrated)", "#C44E52"),
        (axes[1], out["z_cal"],   "Calibrated", "#4C72B0"),
    ]:
        ax.scatter(z_true, zh, s=12, alpha=0.55, color=col, edgecolor="none")
        ax.plot([0, z_max], [0, z_max], "k--", lw=0.7)
        slope = (z_true @ zh) / (z_true @ z_true + 1e-12)
        ax.set_title(f"{lab} (slope vs truth = {slope:.2f})")
        ax.set_xlabel(r"True $z_i$ on deployment")
        ax.set_xlim(0, z_max); ax.set_ylim(0, z_max)
    ax = axes[2]
    order = np.argsort(z_true)
    xs = np.arange(len(order))
    ax.fill_between(xs, out["z_lo"][order], out["z_hi"][order],
                    color="#55A868", alpha=0.30,
                    label=f"90% conformal (coverage = {out['cov']:.2f})")
    ax.plot(xs, z_true[order], "k.", ms=2, label="Truth")
    ax.set_xlabel("Cells sorted by true $z_i$")
    ax.set_title("Conformal interval coverage")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel(r"Estimated $\hat z_i$")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_per_cell_estimates.png"); plt.close(fig)

    # Allocations
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.0))
    vmax = max(out["tau_or"].max(), out["tau_nv"].max(),
               out["tau_cb"].max(), out["tau_dr"].max())
    for ax, tau, lab in [
        (axes[0], out["tau_or"], "Oracle"),
        (axes[1], out["tau_nv"], "Naive"),
        (axes[2], out["tau_cb"], "Calibrated"),
        (axes[3], out["tau_dr"], "DRO (conformal upper)"),
    ]:
        im = ax.imshow(tau.reshape(L, L), cmap="Reds", origin="lower",
                       vmin=0, vmax=vmax)
        ax.set_title(lab); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.022,
                 label=r"Rangers per cell $\tau_i$")
    fig.savefig(FIGDIR / "fig_allocations.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def run_sweep_temperature():
    T_grid = [0.30, 0.40, 0.50, 0.70, 0.90, 1.00, 1.20, 1.60, 2.00, 2.50]
    rows = []
    for T_true in T_grid:
        u_or_l, u_nv_l, u_cb_l, u_dr_l, cov_l = [], [], [], [], []
        for seed in range(4):
            cfg = WorldConfig(L=20, K=4, T_true=T_true, seed=20 * seed + 7)
            out = pipeline(cfg, frac_cal=0.20, alpha=0.10, budget=60.0,
                           lam=0.25, tau_max=3.0, cal_seed=seed)
            u_or_l.append(out["u_or"]); u_nv_l.append(out["u_nv"])
            u_cb_l.append(out["u_cb"]); u_dr_l.append(out["u_dr"])
            cov_l.append(out["cov"])
        rows.append(dict(T_true=T_true,
            u_or=float(np.mean(u_or_l)), u_nv=float(np.mean(u_nv_l)),
            u_cb=float(np.mean(u_cb_l)), u_dr=float(np.mean(u_dr_l)),
            coverage=float(np.mean(cov_l))))

    Ts = np.array([r["T_true"] for r in rows])
    u_or = np.array([r["u_or"] for r in rows])
    u_nv = np.array([r["u_nv"] for r in rows])
    u_cb = np.array([r["u_cb"] for r in rows])
    u_dr = np.array([r["u_dr"] for r in rows])

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(Ts, 100 * (u_or - u_nv) / u_or, "o-", color="#C44E52",
            label="Naive (uncalibrated)")
    ax.plot(Ts, 100 * (u_or - u_cb) / u_or, "s-", color="#4C72B0",
            label="Calibrated point estimate")
    ax.plot(Ts, 100 * (u_or - u_dr) / u_or, "^-", color="#55A868",
            label="DRO (conformal upper)")
    ax.axvline(1.0, ls="--", color="gray", lw=0.7)
    ax.set_xlabel(r"True inverse temperature $T_{\mathrm{true}}$  ($<1$: over-conf., $>1$: under-conf.)")
    ax.set_ylabel("Regret vs oracle (% of oracle utility)")
    ax.set_title("Decision-stage cost of miscalibration")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_regret_vs_T.png"); plt.close(fig)
    RESULTS["sweep_T"] = rows


def run_sweep_calsize():
    fracs = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60]
    rows = []
    for frac in fracs:
        u_or_l, u_nv_l, u_cb_l, u_dr_l, cov_l = [], [], [], [], []
        for seed in range(5):
            cfg = WorldConfig(L=20, K=4, seed=20 * seed + 11)
            out = pipeline(cfg, frac_cal=frac, alpha=0.10, budget=60.0,
                           lam=0.25, tau_max=3.0, cal_seed=seed)
            u_or_l.append(out["u_or"]); u_nv_l.append(out["u_nv"])
            u_cb_l.append(out["u_cb"]); u_dr_l.append(out["u_dr"])
            cov_l.append(out["cov"])
        rows.append(dict(frac=frac,
            u_or=float(np.mean(u_or_l)), u_nv=float(np.mean(u_nv_l)),
            u_cb=float(np.mean(u_cb_l)), u_dr=float(np.mean(u_dr_l)),
            coverage=float(np.mean(cov_l)),
            u_cb_std=float(np.std(u_cb_l)),
            u_dr_std=float(np.std(u_dr_l))))

    xs = np.array([r["frac"] for r in rows])
    u_or = np.array([r["u_or"] for r in rows])
    u_nv = np.array([r["u_nv"] for r in rows])
    u_cb = np.array([r["u_cb"] for r in rows])
    u_dr = np.array([r["u_dr"] for r in rows])

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(xs, 100 * (u_or - u_nv) / u_or, "o-", color="#C44E52", label="Naive")
    ax.plot(xs, 100 * (u_or - u_cb) / u_or, "s-", color="#4C72B0", label="Calibrated")
    ax.plot(xs, 100 * (u_or - u_dr) / u_or, "^-", color="#55A868", label="DRO")
    ax.set_xlabel("Fraction of images held out for calibration")
    ax.set_ylabel("Regret vs oracle (% of oracle utility)")
    ax.set_title("Regret as calibration sample size varies")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_regret_vs_calsize.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(xs, [r["coverage"] for r in rows], "o-", color="#55A868")
    ax.axhline(0.90, ls="--", color="gray", lw=0.8, label="Nominal 90%")
    ax.set_xlabel("Calibration set fraction")
    ax.set_ylabel("Empirical interval coverage")
    ax.set_title("Conformal interval validity")
    ax.set_ylim(0.7, 1.02); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_coverage.png"); plt.close(fig)

    RESULTS["sweep_calsize"] = rows


if __name__ == "__main__":
    head = run_single()
    print("---- Headline numbers ----")
    for k, v in head.items():
        print(f"  {k:38s} {v:9.4f}")
    print("\n---- Sweep over T_true ----")
    run_sweep_temperature(); print("done")
    print("\n---- Sweep over calibration set size ----")
    run_sweep_calsize(); print("done")

    with open(ROOT / "results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\nFigures saved in {FIGDIR}")
    print(f"Results JSON saved at {ROOT / 'results.json'}")
