"""Synthetic camera-trap network with disjoint data roles.

The generator is designed for decision-stage calibration experiments.  A
spatially correlated latent class-probability field is sampled once, then four
*independent* image batches are generated from it:

``temperature_fit``
    Labelled images used only to fit post-hoc calibrators.
``evaluation``
    Labelled images used only to report calibration metrics.
``interval_calibration``
    A deployment-sized labelled batch used only to calibrate predictive-interval
    width under residual model misspecification.
``deployment``
    Images whose probabilities are aggregated for patrol allocation.  Their
    labels are retained by the simulator only for interval-coverage checks.

Keeping these roles disjoint prevents the optimistic reuse of calibration
labels for evaluation.  The final class is a human-presence risk proxy; it
must not be interpreted as a direct poacher label in a real deployment.
"""
from __future__ import annotations

import dataclasses as dc
from typing import Mapping

import numpy as np


@dc.dataclass(frozen=True)
class WorldConfig:
    """Parameters of the latent landscape and synthetic classifier."""

    L: int = 20
    K: int = 4
    rho: float = 3.0
    threat_logit_mean: float = -1.5
    threat_logit_amp: float = 3.0
    T_true: float = 0.45
    bias_threat: float = 0.6
    sigma_obs: float = 0.8
    mu_temperature_fit: float = 16.0
    mu_evaluation: float = 12.0
    mu_interval_calibration: float = 40.0
    mu_deployment: float = 40.0
    min_images_per_batch: int = 5
    seed: int = 1234

    def __post_init__(self) -> None:
        if self.L <= 0 or self.K < 2:
            raise ValueError("L must be positive and K must be at least 2")
        if self.rho <= 0 or self.T_true <= 0 or self.sigma_obs < 0:
            raise ValueError("rho and T_true must be positive; sigma_obs cannot be negative")
        for value in (
            self.mu_temperature_fit,
            self.mu_evaluation,
            self.mu_interval_calibration,
            self.mu_deployment,
        ):
            if value <= 0:
                raise ValueError("all batch mean image counts must be positive")
        if self.min_images_per_batch < 1:
            raise ValueError("min_images_per_batch must be at least 1")


@dc.dataclass(frozen=True)
class CameraBatch:
    """One independent image batch sampled from the latent landscape."""

    name: str
    M: np.ndarray
    y: tuple[np.ndarray, ...]
    pi_obs: tuple[np.ndarray, ...]


@dc.dataclass(frozen=True)
class World:
    """Latent landscape and independent image batches."""

    cfg: WorldConfig
    p_true: np.ndarray
    batches: Mapping[str, CameraBatch]

    @property
    def N(self) -> int:
        return self.cfg.L * self.cfg.L

    @property
    def threat_class(self) -> int:
        return self.cfg.K - 1

    @property
    def deployment(self) -> CameraBatch:
        return self.batches["deployment"]

    @property
    def z_true(self) -> np.ndarray:
        """Expected deployment-batch risk-event count per cell."""
        return self.deployment.M * self.p_true[:, self.threat_class]

    # Backwards-compatible deployment aliases used by early repository users.
    @property
    def M(self) -> np.ndarray:
        return self.deployment.M

    @property
    def y(self) -> tuple[np.ndarray, ...]:
        return self.deployment.y

    @property
    def pi_obs(self) -> tuple[np.ndarray, ...]:
        return self.deployment.pi_obs


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def _grf(L: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Return an ``L x L`` Gaussian random field with a Gaussian spectrum."""
    fx = np.fft.fftfreq(L)[:, None]
    fy = np.fft.fftfreq(L)[None, :]
    radius_sq = fx**2 + fy**2
    spectrum = np.exp(-0.5 * (2.0 * np.pi * rho) ** 2 * radius_sq)
    white = rng.standard_normal((L, L))
    field = np.real(np.fft.ifft2(np.fft.fft2(white) * np.sqrt(spectrum)))
    return (field - field.mean()) / (field.std() + 1e-12)


def _make_batch(
    *,
    name: str,
    mean_images: float,
    p_true: np.ndarray,
    cfg: WorldConfig,
    rng: np.random.Generator,
) -> CameraBatch:
    """Generate one independent batch conditional on ``p_true``."""
    N, K = p_true.shape
    M = rng.poisson(mean_images, size=N).clip(min=cfg.min_images_per_batch)
    bias = np.zeros(K, dtype=float)
    bias[-1] = cfg.bias_threat

    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for cell in range(N):
        n_images = int(M[cell])
        y_cell = rng.choice(K, size=n_images, p=p_true[cell])
        base_logits = np.log(p_true[cell] + 1e-12) + bias
        noisy_logits = base_logits[None, :] + rng.normal(
            0.0, cfg.sigma_obs, size=(n_images, K)
        )
        pi_cell = _softmax(noisy_logits / cfg.T_true, axis=1)
        labels.append(y_cell.astype(int, copy=False))
        probabilities.append(pi_cell)

    return CameraBatch(
        name=name,
        M=M.astype(int, copy=False),
        y=tuple(labels),
        pi_obs=tuple(probabilities),
    )


def make_world(cfg: WorldConfig | None = None) -> World:
    """Create a latent world and disjoint fit/evaluation/deployment batches."""
    cfg = cfg or WorldConfig()
    seed_sequence = np.random.SeedSequence(cfg.seed)
    latent_seed, fit_seed, eval_seed, interval_seed, dep_seed = seed_sequence.spawn(5)
    latent_rng = np.random.default_rng(latent_seed)

    N, K = cfg.L * cfg.L, cfg.K
    latent_logits = latent_rng.standard_normal((N, K)) * 0.3
    threat_field = _grf(cfg.L, cfg.rho, latent_rng).reshape(N)
    latent_logits[:, -1] = (
        cfg.threat_logit_mean + cfg.threat_logit_amp * threat_field
    )
    p_true = _softmax(latent_logits, axis=1)

    batch_specs = (
        ("temperature_fit", cfg.mu_temperature_fit, fit_seed),
        ("evaluation", cfg.mu_evaluation, eval_seed),
        ("interval_calibration", cfg.mu_interval_calibration, interval_seed),
        ("deployment", cfg.mu_deployment, dep_seed),
    )
    batches = {
        name: _make_batch(
            name=name,
            mean_images=mean_images,
            p_true=p_true,
            cfg=cfg,
            rng=np.random.default_rng(batch_seed),
        )
        for name, mean_images, batch_seed in batch_specs
    }
    return World(cfg=cfg, p_true=p_true, batches=batches)


if __name__ == "__main__":
    world = make_world()
    print(f"World: N={world.N} cells, K={world.cfg.K} classes")
    for name, batch in world.batches.items():
        print(f"  {name:16s}: {batch.M.sum():6d} images")
    print(
        "Deployment expected risk-count range: "
        f"[{world.z_true.min():.2f}, {world.z_true.max():.2f}]"
    )
