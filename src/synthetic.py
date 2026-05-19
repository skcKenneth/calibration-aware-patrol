"""
Synthetic camera-trap network with controllable classifier miscalibration.

We need ground truth to evaluate allocation regret. Public camera-trap datasets
have image-level labels but not per-site true threat intensity, so synthetic
data lets us close the loop. The classifier model is intentionally simple but
captures the structural properties that matter for decision-stage propagation:
(i) softmax outputs, (ii) class-conditional miscalibration, (iii) temperature
distortion. Real-data validation (e.g. SpeciesNet ECE on Snapshot Serengeti)
is handled separately and is consistent with the published literature.

The world model:
  - N cells on an L x L grid
  - K classes; class K-1 is the "threat" (e.g. human/poacher signature)
  - Per-cell true class probabilities p_i in Delta^{K-1} drawn from a
    spatially-correlated Gaussian random field passed through softmax
  - Each cell receives M_i ~ Poisson(mu) camera-trap images
  - Each image has a true label y ~ Categorical(p_i)
  - Classifier emits logits l = log p_i + bias_class + sigma_obs * eps
    then is observed with global inverse temperature T_true:
      pi_hat = softmax(l / T_true)
"""
from __future__ import annotations

import dataclasses as dc
import numpy as np


@dc.dataclass
class WorldConfig:
    L: int = 20                 # grid side length -> N = L * L cells
    K: int = 4                  # number of classes (last class is "threat")
    rho: float = 3.0            # spatial correlation length (cells)
    mu_images: float = 40.0     # mean images per cell (Poisson)
    threat_logit_mean: float = -1.5   # baseline threat propensity
    threat_logit_amp: float = 3.0     # threat spatial amplitude
    T_true: float = 0.45        # < 1 -> over-confident (typical for DL classifiers)
    bias_threat: float = 0.6    # systematic over-prediction of threat class
    sigma_obs: float = 0.8      # per-image classifier noise (in logit units)
    seed: int = 1234


@dc.dataclass
class World:
    cfg: WorldConfig
    p_true: np.ndarray          # (N, K) per-cell true class probabilities
    z_true: np.ndarray          # (N,)  per-cell true threat intensity = M_i * p_true[:, -1]
    M: np.ndarray               # (N,) images per cell
    y: list                     # per-cell array of true labels (length M_i)
    pi_obs: list                # per-cell array of (M_i, K) raw classifier softmax outputs


# ---------------------------------------------------------------------------
# Spatially correlated Gaussian random field via FFT (one realisation per call)
# ---------------------------------------------------------------------------

def _grf(L: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Return an L x L Gaussian random field with isotropic Gaussian kernel."""
    # Spectral filter with characteristic length rho
    fx = np.fft.fftfreq(L)[:, None]
    fy = np.fft.fftfreq(L)[None, :]
    r2 = fx ** 2 + fy ** 2
    spectrum = np.exp(-0.5 * (2 * np.pi * rho) ** 2 * r2)
    white = rng.standard_normal((L, L))
    field = np.real(np.fft.ifft2(np.fft.fft2(white) * np.sqrt(spectrum)))
    # Standardise
    field = (field - field.mean()) / (field.std() + 1e-12)
    return field


# ---------------------------------------------------------------------------
# World construction
# ---------------------------------------------------------------------------

def make_world(cfg: WorldConfig | None = None) -> World:
    cfg = cfg or WorldConfig()
    rng = np.random.default_rng(cfg.seed)

    L, K, N = cfg.L, cfg.K, cfg.L * cfg.L

    # Per-cell logits per class. Threat class (last) is spatially structured;
    # other classes are uniform noise around zero.
    logits = rng.standard_normal((N, K)) * 0.3
    threat_field = _grf(L, cfg.rho, rng).reshape(N)
    logits[:, -1] = cfg.threat_logit_mean + cfg.threat_logit_amp * threat_field

    p_true = _softmax(logits, axis=1)

    # Number of images per cell
    M = rng.poisson(cfg.mu_images, size=N).clip(min=5)

    y = [None] * N
    pi_obs = [None] * N

    for i in range(N):
        # True image labels
        yi = rng.choice(K, size=M[i], p=p_true[i])
        # Classifier logits = true log-prob + per-class bias + per-image noise
        bias = np.zeros(K)
        bias[-1] = cfg.bias_threat
        true_logits = np.log(p_true[i] + 1e-12) + bias
        # Per-image i.i.d. additive noise in logit space
        l_img = true_logits[None, :] + rng.normal(0, cfg.sigma_obs, size=(M[i], K))
        # Temperature distortion (this is what we want to recover)
        pi = _softmax(l_img / cfg.T_true, axis=1)
        y[i] = yi
        pi_obs[i] = pi

    z_true = M * p_true[:, -1]

    return World(cfg=cfg, p_true=p_true, z_true=z_true, M=M, y=y, pi_obs=pi_obs)


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Train / calibration / deployment split at the image level
# ---------------------------------------------------------------------------

def split_calibration(world: World, frac_cal: float = 0.2, seed: int = 0):
    """Randomly mark a fraction of every cell's images as the calibration set
    (i.e. labels are revealed). The remaining images are 'deployment' images
    on which only raw classifier outputs are available.
    """
    rng = np.random.default_rng(seed)
    N = world.cfg.L ** 2
    is_cal = []
    for i in range(N):
        m = world.M[i]
        mask = rng.random(m) < frac_cal
        # Guarantee at least one calibration image per cell
        if mask.sum() == 0:
            mask[rng.integers(0, m)] = True
        is_cal.append(mask)
    return is_cal


if __name__ == "__main__":
    w = make_world()
    print(f"World: N={w.cfg.L**2} cells, K={w.cfg.K} classes")
    print(f"Mean true threat intensity per cell: {w.z_true.mean():.2f}")
    print(f"True threat field range: [{w.z_true.min():.2f}, {w.z_true.max():.2f}]")
    is_cal = split_calibration(w, frac_cal=0.2)
    cal_per_cell = np.array([m.sum() for m in is_cal])
    print(f"Calibration images per cell: mean={cal_per_cell.mean():.1f}")
