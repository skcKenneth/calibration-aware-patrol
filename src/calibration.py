"""Post-hoc calibration, diagnostics, and count-prediction utilities.

Fitting, metric evaluation, interval tuning, and deployment each use a
different image batch from ``synthetic``. Patrol decisions use the final-class
(human-presence risk-proxy) probability, so the module reports multiclass
temperature-scaling metrics and a binary Platt calibrator for that class.

Cell-level count intervals come from the exact Poisson-binomial distribution of
calibrated image probabilities. A held-out batch picks an integer expansion margin
when the model is slightly misspecified. Intervals target realised event counts,
not the latent expectation; no distribution-free conformal guarantee is claimed.
"""
from __future__ import annotations

import dataclasses as dc

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit

from .synthetic import CameraBatch, World


@dc.dataclass(frozen=True)
class FlatBatch:
    """Flattened image probabilities, labels, and cell identifiers."""

    pi: np.ndarray
    y: np.ndarray
    cell: np.ndarray

    def __post_init__(self) -> None:
        if self.pi.ndim != 2:
            raise ValueError("pi must be a 2D probability array")
        if len(self.y) != len(self.pi) or len(self.cell) != len(self.pi):
            raise ValueError("pi, y, and cell must contain the same number of rows")


@dc.dataclass(frozen=True)
class BinaryPlattModel:
    """Monotone logistic recalibration model ``sigmoid(a*logit(p)+b)``."""

    slope: float
    intercept: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.slope) or self.slope <= 0:
            raise ValueError("slope must be finite and positive")
        if not np.isfinite(self.intercept):
            raise ValueError("intercept must be finite")


def flatten_batch(batch: CameraBatch) -> FlatBatch:
    """Flatten one per-cell :class:`CameraBatch`."""
    pi_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    cell_parts: list[np.ndarray] = []
    for cell, (pi_cell, y_cell) in enumerate(zip(batch.pi_obs, batch.y)):
        pi_parts.append(pi_cell)
        y_parts.append(y_cell)
        cell_parts.append(np.full(len(y_cell), cell, dtype=int))
    return FlatBatch(
        pi=np.concatenate(pi_parts, axis=0),
        y=np.concatenate(y_parts, axis=0),
        cell=np.concatenate(cell_parts, axis=0),
    )


def get_flat_batch(world: World, name: str) -> FlatBatch:
    """Return a named world batch in flattened form."""
    try:
        return flatten_batch(world.batches[name])
    except KeyError as exc:
        raise KeyError(f"unknown batch {name!r}; available: {tuple(world.batches)}") from exc


def subsample_flat_batch(
    batch: FlatBatch,
    fraction: float,
    *,
    seed: int = 0,
) -> FlatBatch:
    """Take a deterministic random subset of a fixed calibration pool.

    Using the same ``seed`` across fractions produces nested subsets because
    every row receives one fixed pseudo-random score and rows with
    ``score < fraction`` are retained.  Evaluation and deployment batches are
    unaffected, removing the sample-size confounding present in the original
    repository.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    if fraction == 1:
        return batch
    rng = np.random.default_rng(seed)
    mask = rng.random(len(batch.y)) < fraction
    if not mask.any():
        mask[rng.integers(0, len(mask))] = True
    return FlatBatch(pi=batch.pi[mask], y=batch.y[mask], cell=batch.cell[mask])


# ---------------------------------------------------------------------------
# Multiclass temperature scaling
# ---------------------------------------------------------------------------

def _validate_probabilities(pi: np.ndarray, y: np.ndarray | None = None) -> None:
    if pi.ndim != 2 or len(pi) == 0:
        raise ValueError("pi must be a non-empty 2D array")
    if not np.all(np.isfinite(pi)) or np.any(pi < 0):
        raise ValueError("pi must contain finite non-negative values")
    if not np.allclose(pi.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each probability row must sum to one")
    if y is not None:
        if y.ndim != 1 or len(y) != len(pi):
            raise ValueError("y must be a 1D array aligned with pi")
        if np.any((y < 0) | (y >= pi.shape[1])):
            raise ValueError("y contains an invalid class index")


def negative_log_likelihood(pi: np.ndarray, y: np.ndarray) -> float:
    _validate_probabilities(pi, y)
    return float(-np.mean(np.log(np.clip(pi[np.arange(len(y)), y], 1e-12, 1.0))))


def fit_temperature(pi: np.ndarray, y: np.ndarray) -> float:
    """Fit a positive scalar temperature by minimising multiclass NLL."""
    _validate_probabilities(pi, y)
    logits = np.log(np.clip(pi, 1e-12, 1.0))
    logits -= logits.max(axis=1, keepdims=True)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        scaled = logits / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        log_normaliser = np.log(np.exp(scaled).sum(axis=1))
        return float(np.mean(log_normaliser - scaled[np.arange(len(y)), y]))

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"temperature optimisation failed: {result.message}")
    return float(np.exp(result.x))


def apply_temperature(pi: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling without changing class ranking."""
    _validate_probabilities(pi)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    logits = np.log(np.clip(pi, 1e-12, 1.0))
    logits -= logits.max(axis=1, keepdims=True)
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Task-specific binary Platt calibration
# ---------------------------------------------------------------------------

def fit_binary_platt(probability: np.ndarray, target: np.ndarray) -> BinaryPlattModel:
    """Fit monotone binary Platt scaling for one operationally relevant class."""
    probability = np.asarray(probability, dtype=float)
    target = np.asarray(target, dtype=float)
    if probability.ndim != 1 or target.ndim != 1 or len(probability) != len(target):
        raise ValueError("probability and target must be aligned 1D arrays")
    if len(target) == 0 or not np.all(np.isin(target, (0.0, 1.0))):
        raise ValueError("target must be a non-empty binary array")

    clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
    logit_probability = np.log(clipped / (1.0 - clipped))
    if np.unique(target).size < 2:
        # A very small monotone slope plus a smoothed-prevalence intercept is a
        # stable fallback for extremely small calibration subsets.
        prevalence = (target.sum() + 0.5) / (len(target) + 1.0)
        return BinaryPlattModel(
            slope=1e-6,
            intercept=float(np.log(prevalence / (1.0 - prevalence))),
        )

    # Optimise log(slope) so the mapping remains monotone increasing.
    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        log_slope, intercept = theta
        slope = np.exp(log_slope)
        linear = slope * logit_probability + intercept
        regularisation = 1e-6
        loss = (
            np.mean(np.logaddexp(0.0, linear) - target * linear)
            + 0.5 * regularisation * (log_slope**2 + intercept**2)
        )
        fitted = expit(linear)
        common = fitted - target
        gradient = np.array(
            [
                np.mean(common * slope * logit_probability) + regularisation * log_slope,
                np.mean(common) + regularisation * intercept,
            ]
        )
        return float(loss), gradient

    result = minimize(
        fun=lambda theta: objective(theta)[0],
        x0=np.array([0.0, 0.0]),
        jac=lambda theta: objective(theta)[1],
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 1000},
    )
    if not result.success and np.linalg.norm(result.jac) > 1e-5:
        raise RuntimeError(f"binary Platt optimisation failed: {result.message}")
    return BinaryPlattModel(
        slope=float(np.exp(result.x[0])),
        intercept=float(result.x[1]),
    )


def apply_binary_platt(
    probability: np.ndarray,
    model: BinaryPlattModel,
) -> np.ndarray:
    """Apply a fitted binary Platt model."""
    probability = np.asarray(probability, dtype=float)
    clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
    logit_probability = np.log(clipped / (1.0 - clipped))
    calibrated = expit(model.slope * logit_probability + model.intercept)
    return np.clip(calibrated, 1e-12, 1.0 - 1e-12)


# ---------------------------------------------------------------------------
# Calibration metrics and reliability curves
# ---------------------------------------------------------------------------

def top_label_ece(pi: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error for maximum confidence vs correctness."""
    _validate_probabilities(pi, y)
    confidence = pi.max(axis=1)
    correctness = (pi.argmax(axis=1) == y).astype(float)
    return _binary_ece_from_scores(confidence, correctness, n_bins=n_bins)


def binary_ece(probability: np.ndarray, target: np.ndarray, n_bins: int = 15) -> float:
    """ECE for a named class probability against its binary indicator."""
    probability = np.asarray(probability, dtype=float)
    target = np.asarray(target, dtype=float)
    return _binary_ece_from_scores(probability, target, n_bins=n_bins)


def _binary_ece_from_scores(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    n_bins: int,
) -> float:
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if probability.ndim != 1 or target.ndim != 1 or len(probability) != len(target):
        raise ValueError("probability and target must be aligned 1D arrays")
    if len(target) == 0 or np.any((probability < 0) | (probability > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(target)
    error = 0.0
    for index in range(n_bins):
        if index == 0:
            mask = (probability >= bins[index]) & (probability <= bins[index + 1])
        else:
            mask = (probability > bins[index]) & (probability <= bins[index + 1])
        count = int(mask.sum())
        if count:
            error += count / total * abs(target[mask].mean() - probability[mask].mean())
    return float(error)


def multiclass_brier(pi: np.ndarray, y: np.ndarray) -> float:
    _validate_probabilities(pi, y)
    one_hot = np.zeros_like(pi)
    one_hot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((pi - one_hot) ** 2, axis=1)))


def binary_brier(probability: np.ndarray, target: np.ndarray) -> float:
    probability = np.asarray(probability, dtype=float)
    target = np.asarray(target, dtype=float)
    if probability.shape != target.shape:
        raise ValueError("probability and target must have the same shape")
    return float(np.mean((probability - target) ** 2))


def top_label_reliability_curve(
    pi: np.ndarray,
    y: np.ndarray,
    n_bins: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _validate_probabilities(pi, y)
    confidence = pi.max(axis=1)
    correctness = (pi.argmax(axis=1) == y).astype(float)
    return binary_reliability_curve(confidence, correctness, n_bins=n_bins)


def binary_reliability_curve(
    probability: np.ndarray,
    target: np.ndarray,
    n_bins: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return bin centres, mean probability, event rate, and bin counts."""
    probability = np.asarray(probability, dtype=float)
    target = np.asarray(target, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centres: list[float] = []
    means: list[float] = []
    rates: list[float] = []
    counts: list[int] = []
    for index in range(n_bins):
        if index == 0:
            mask = (probability >= bins[index]) & (probability <= bins[index + 1])
        else:
            mask = (probability > bins[index]) & (probability <= bins[index + 1])
        if mask.any():
            centres.append(0.5 * (bins[index] + bins[index + 1]))
            means.append(float(probability[mask].mean()))
            rates.append(float(target[mask].mean()))
            counts.append(int(mask.sum()))
    return (
        np.asarray(centres),
        np.asarray(means),
        np.asarray(rates),
        np.asarray(counts, dtype=int),
    )


# Backwards-compatible names from the first release.
ece = top_label_ece
reliability_curve = top_label_reliability_curve


# ---------------------------------------------------------------------------
# Per-cell estimates and Poisson-binomial predictive intervals
# ---------------------------------------------------------------------------

def expected_count_from_probability(
    probability: np.ndarray,
    cell_index: np.ndarray,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum image-level event probabilities within each cell."""
    probability = np.asarray(probability, dtype=float)
    cell_index = np.asarray(cell_index, dtype=int)
    if probability.ndim != 1 or cell_index.ndim != 1 or len(probability) != len(cell_index):
        raise ValueError("probability and cell_index must be aligned 1D arrays")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("probability must lie in [0, 1]")
    if np.any((cell_index < 0) | (cell_index >= n_cells)):
        raise ValueError("cell_index contains an invalid cell")
    count = np.bincount(cell_index, minlength=n_cells).astype(int)
    expected = np.bincount(
        cell_index,
        weights=probability,
        minlength=n_cells,
    ).astype(float)
    return expected, count


def expected_count(
    pi: np.ndarray,
    cell_idx: np.ndarray,
    N: int,
    threat_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper accepting a full multiclass probability matrix."""
    return expected_count_from_probability(pi[:, threat_class], cell_idx, N)


def realised_count(
    y: np.ndarray,
    cell_index: np.ndarray,
    n_cells: int,
    event_class: int,
) -> np.ndarray:
    """Observed event count per cell."""
    return np.bincount(
        cell_index,
        weights=(np.asarray(y) == event_class).astype(float),
        minlength=n_cells,
    ).astype(int)


def poisson_binomial_pmf(probability: np.ndarray) -> np.ndarray:
    """Exact PMF of a sum of independent non-identical Bernoulli variables."""
    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 1 or np.any((probability < 0) | (probability > 1)):
        raise ValueError("probability must be a 1D array with values in [0, 1]")
    pmf = np.zeros(len(probability) + 1, dtype=float)
    pmf[0] = 1.0
    active = 1
    for value in probability:
        previous = pmf[:active].copy()
        pmf[1 : active + 1] = pmf[1 : active + 1] * (1.0 - value) + previous * value
        pmf[0] *= 1.0 - value
        active += 1
    pmf = np.maximum(pmf, 0.0)
    return pmf / pmf.sum()


def poisson_binomial_interval(
    probability: np.ndarray,
    alpha: float = 0.10,
) -> tuple[int, int]:
    """Equal-tailed predictive interval for a Poisson-binomial count."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    pmf = poisson_binomial_pmf(probability)
    cdf = np.cumsum(pmf)
    lower = int(np.searchsorted(cdf, alpha / 2.0, side="left"))
    upper = int(np.searchsorted(cdf, 1.0 - alpha / 2.0, side="left"))
    upper = min(upper, len(probability))
    return lower, upper


def per_cell_predictive_intervals(
    probability: np.ndarray,
    cell_index: np.ndarray,
    n_cells: int,
    alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expected counts and exact predictive intervals for every cell."""
    expected, count = expected_count_from_probability(probability, cell_index, n_cells)
    lower = np.zeros(n_cells, dtype=float)
    upper = np.zeros(n_cells, dtype=float)
    for cell in range(n_cells):
        cell_probability = probability[cell_index == cell]
        lo, hi = poisson_binomial_interval(cell_probability, alpha=alpha)
        lower[cell] = lo
        upper[cell] = hi
    return expected, lower, upper, count


def calibrate_interval_expansion(
    lower: np.ndarray,
    upper: np.ndarray,
    realised: np.ndarray,
    alpha: float = 0.10,
) -> int:
    """Pick a global integer margin from held-out count violations.

    Scores how far each realised count falls outside its base Poisson-binomial
    interval, then takes an empirical quantile. Cells are not assumed
    exchangeable, so no distribution-free conformal guarantee is claimed.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    realised = np.asarray(realised, dtype=float)
    if lower.shape != upper.shape or lower.shape != realised.shape:
        raise ValueError("lower, upper, and realised must have the same shape")
    score = np.maximum.reduce((lower - realised, realised - upper, np.zeros_like(realised)))
    n = len(score)
    if n == 0:
        return 0
    rank = min(int(np.ceil((1.0 - alpha) * (n + 1))), n)
    return int(np.ceil(np.sort(score)[rank - 1]))


def expand_predictive_intervals(
    lower: np.ndarray,
    upper: np.ndarray,
    maximum_count: np.ndarray,
    margin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand count intervals by an integer margin and clip to support."""
    if margin < 0:
        raise ValueError("margin must be non-negative")
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    maximum_count = np.asarray(maximum_count, dtype=float)
    if lower.shape != upper.shape or lower.shape != maximum_count.shape:
        raise ValueError("interval arrays and maximum_count must have the same shape")
    return (
        np.maximum(0.0, lower - margin),
        np.minimum(maximum_count, upper + margin),
    )


def empirical_coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    realised: np.ndarray,
) -> float:
    """Fraction of realised cell counts lying in their predictive intervals."""
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    realised = np.asarray(realised)
    if lower.shape != upper.shape or lower.shape != realised.shape:
        raise ValueError("lower, upper, and realised must have the same shape")
    return float(np.mean((realised >= lower) & (realised <= upper)))
