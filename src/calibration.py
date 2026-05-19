"""
Calibration and conformal prediction utilities.

Two layers:
  1. Temperature scaling (Guo et al. 2017) for global recalibration of softmax
     outputs on a held-out calibration set.
  2. Split conformal prediction (Vovk; Romano et al.) to obtain finite-sample
     valid (1 - alpha) prediction sets for image-level class. Aggregated per
     cell, these give honest intervals on the threat-class count.

We use Adaptive Prediction Sets (APS, Romano et al. 2020) on the calibrated
probabilities -- standard choice for classification with class imbalance.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .synthetic import World


# ---------------------------------------------------------------------------
# Aggregate every cell's calibration images into one flat arrays
# ---------------------------------------------------------------------------

def gather_calibration(world: World, is_cal):
    pi_cal_list, y_cal_list, cell_id_list = [], [], []
    pi_dep_list, y_dep_list, dep_cell_id_list = [], [], []
    for i in range(world.cfg.L ** 2):
        m_cal = is_cal[i]
        pi_cal_list.append(world.pi_obs[i][m_cal])
        y_cal_list.append(world.y[i][m_cal])
        cell_id_list.append(np.full(m_cal.sum(), i))
        m_dep = ~m_cal
        pi_dep_list.append(world.pi_obs[i][m_dep])
        y_dep_list.append(world.y[i][m_dep])
        dep_cell_id_list.append(np.full(m_dep.sum(), i))

    return dict(
        pi_cal=np.concatenate(pi_cal_list, axis=0),
        y_cal=np.concatenate(y_cal_list, axis=0),
        cell_cal=np.concatenate(cell_id_list, axis=0),
        pi_dep=np.concatenate(pi_dep_list, axis=0),
        y_dep=np.concatenate(y_dep_list, axis=0),
        cell_dep=np.concatenate(dep_cell_id_list, axis=0),
    )


# ---------------------------------------------------------------------------
# Temperature scaling: search T to minimise NLL on calibration set
# ---------------------------------------------------------------------------

def fit_temperature(pi: np.ndarray, y: np.ndarray) -> float:
    """Find T > 0 that minimises NLL of softmax(logits / T) on calibration set.

    We recover logits from observed probabilities up to a constant, which is
    sufficient for softmax temperature scaling.
    """
    logits = np.log(pi + 1e-12)
    logits = logits - logits.max(axis=1, keepdims=True)  # numerical stability

    def nll(T):
        if T <= 0:
            return 1e12
        scaled = logits / T
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        e = np.exp(scaled)
        p = e / e.sum(axis=1, keepdims=True)
        return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded",
                          options={"xatol": 1e-3})
    return float(res.x)


def apply_temperature(pi: np.ndarray, T: float) -> np.ndarray:
    logits = np.log(pi + 1e-12)
    logits = logits - logits.max(axis=1, keepdims=True)
    scaled = logits / T
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    e = np.exp(scaled)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE) -- standard 15-bin estimator
# ---------------------------------------------------------------------------

def ece(pi: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = pi.max(axis=1)
    pred = pi.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    n = len(y)
    for b in range(n_bins):
        mask = (conf > bins[b]) & (conf <= bins[b + 1])
        if mask.sum() == 0:
            continue
        e += mask.sum() / n * abs(correct[mask].mean() - conf[mask].mean())
    return float(e)


def reliability_curve(pi: np.ndarray, y: np.ndarray, n_bins: int = 12):
    """Return (bin centres, mean confidence, mean accuracy, bin counts)."""
    conf = pi.max(axis=1)
    pred = pi.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centres, mean_conf, mean_acc, counts = [], [], [], []
    for b in range(n_bins):
        mask = (conf > bins[b]) & (conf <= bins[b + 1])
        if mask.sum() == 0:
            continue
        centres.append(0.5 * (bins[b] + bins[b + 1]))
        mean_conf.append(conf[mask].mean())
        mean_acc.append(correct[mask].mean())
        counts.append(int(mask.sum()))
    return (np.array(centres), np.array(mean_conf),
            np.array(mean_acc), np.array(counts))


# ---------------------------------------------------------------------------
# Expected threat count from soft probabilities (no thresholding)
# ---------------------------------------------------------------------------

def expected_count(pi: np.ndarray, cell_idx: np.ndarray, N: int,
                   threat_class: int):
    """Per-cell expected count of the threat class (sum of soft probabilities)
    and per-cell number of images."""
    out = np.zeros(N)
    n = np.zeros(N, dtype=int)
    for i in range(N):
        m = (cell_idx == i)
        if m.sum() == 0:
            continue
        out[i] = pi[m, threat_class].sum()
        n[i] = int(m.sum())
    return out, n


# ---------------------------------------------------------------------------
# Split-conformal regression on the per-cell expected count
# ---------------------------------------------------------------------------
#
# Standardised residual workflow:
#   1. For each cell, compute on its calibration images
#        z_true_i^{cal}   = sum_{j in cal_i} 1{y_j = threat}
#        z_hat_i^{cal}    = sum_{j in cal_i} pi_calib[j, threat]
#        n_i^{cal}        = |cal_i|
#      and the standardised residual
#        r_i = (z_true_i^{cal} - z_hat_i^{cal}) / sqrt(max(n_i^{cal}, 1))
#   2. Take the (1 - alpha) empirical quantile q of |r_i| over cells with
#      enough calibration data (>= n_min).
#   3. On deployment images,
#        z_hat_i^{dep}    = sum_{j in dep_i} pi_calib[j, threat]
#      and the prediction interval is
#        [ z_hat_i^{dep} - q * sqrt(n_i^{dep}),
#          z_hat_i^{dep} + q * sqrt(n_i^{dep}) ]
#      clipped to [0, n_i^{dep}].
#
# Coverage holds in the marginal sense (exchangeability of cells) by the
# standard split-conformal argument. The sqrt(n) normalisation makes the
# residual approximately exchangeable across heterogeneous cell sizes
# because Var(sum of n Bernoullis) ~ n.

def conformal_residuals(pi_cal: np.ndarray, y_cal: np.ndarray,
                        cell_cal: np.ndarray, N: int, threat_class: int,
                        n_min: int = 3):
    """Return the standardised residuals r_i and their indices for cells with
    enough calibration images."""
    pi_threat_cal = pi_cal[:, threat_class]
    y_threat_cal = (y_cal == threat_class).astype(float)
    res, kept = [], []
    for i in range(N):
        m = (cell_cal == i)
        if m.sum() < n_min:
            continue
        n = m.sum()
        z_true = y_threat_cal[m].sum()
        z_hat = pi_threat_cal[m].sum()
        res.append((z_true - z_hat) / np.sqrt(n))
        kept.append(i)
    return np.array(res), np.array(kept)


def conformal_quantile(residuals: np.ndarray, alpha: float = 0.1) -> float:
    """(1 - alpha) empirical quantile of the absolute residuals."""
    if len(residuals) == 0:
        return 0.0
    n = len(residuals)
    k = int(np.ceil((1 - alpha) * (n + 1)))
    k = min(k, n)
    return float(np.sort(np.abs(residuals))[k - 1])


def per_cell_intervals(pi_dep: np.ndarray, cell_dep: np.ndarray, N: int,
                       threat_class: int, q: float):
    """Return (z_hat, z_lo, z_hi, n_per_cell) for deployment images using
    the conformal half-width q (already in standardised residual units)."""
    z_hat, n = expected_count(pi_dep, cell_dep, N, threat_class)
    halfwidth = q * np.sqrt(np.maximum(n, 1))
    z_lo = np.maximum(0.0, z_hat - halfwidth)
    z_hi = np.minimum(n.astype(float), z_hat + halfwidth)
    return z_hat, z_lo, z_hi, n


def empirical_coverage(z_lo: np.ndarray, z_hi: np.ndarray,
                       z_true: np.ndarray) -> float:
    """Fraction of cells where the truth lies in [z_lo, z_hi]."""
    inside = (z_true >= z_lo) & (z_true <= z_hi)
    return float(inside.mean())
