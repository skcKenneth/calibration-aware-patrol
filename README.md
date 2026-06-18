# Calibration-Aware Patrol Allocation

A reproducible synthetic pipeline for studying how camera-trap probability
calibration affects spatial patrol decisions. The final class is treated as a
**human-presence risk proxy**, not as a direct poacher label.

This revision corrects the main methodological issues in the first release:
calibrator fitting, metric evaluation, interval-width calibration, and
operational deployment now use separate image batches; calibration is measured
for the decision-relevant class; and the former conformal/DRO terminology has
been replaced by claims supported by the implementation.

## Corrected pipeline

1. **Synthetic landscape and independent batches**
   (`src/synthetic.py`)
   - A spatially correlated latent class-probability field is generated over a
     20 × 20 grid.
   - Four conditionally independent image batches are sampled:
     `temperature_fit`, `evaluation`, `interval_calibration`, and `deployment`.
   - The deployment batch contains about 16,000 images in the default setting.

2. **Multiclass temperature scaling**
   (`src/calibration.py`)
   - One positive temperature is fitted by minimising multiclass NLL on the
     `temperature_fit` batch.
   - Top-label ECE, Brier score, and NLL are reported only on the independent
     `evaluation` batch.

3. **Decision-specific threat calibration**
   - A monotone binary Platt model is fitted for the risk-proxy class.
   - Risk-class ECE, Brier score, and reliability curves are reported alongside
     the conventional top-label metrics.

4. **Realised-count predictive intervals**
   - Calibrated image probabilities induce an exact Poisson-binomial count
     distribution in each cell.
   - A separate, deployment-sized labelled batch selects an integer interval
     expansion margin to account for residual model misspecification.
   - Final coverage is evaluated against **realised deployment counts**, not the
     unknown latent expected count.
   - No distribution-free conformal guarantee is claimed.

5. **Patrol allocation**
   (`src/allocation.py`)
   - `water_fill()` solves the fixed-budget concave allocation problem.
   - The evaluated policies are oracle, naive, temperature-scaled,
     task-calibrated point estimate, and predictive upper-bound.
   - The upper policy is an interval-based risk policy; it is **not** described
     as distributionally robust optimisation.

6. **Asymmetric operational-cost analysis**
   - A second objective combines missed-event cost with patrol-hour cost:

     `miss_cost × Σ z_i exp(-λτ_i) + Σ c_i τ_i`.

   - The experiment compares point and upper-bound planning on both the nominal
     expected field and an upper-bound stress field.

## Default results

The full default run uses 20 random seeds for the main summary and 8 seeds per
sensitivity setting. Selected multi-seed results are:

| Metric | Mean ± SD |
|---|---:|
| Top-label ECE, raw | 0.341 ± 0.010 |
| Top-label ECE, temperature-scaled | 0.048 ± 0.010 |
| Risk-class ECE, raw | 0.111 ± 0.013 |
| Risk-class ECE, temperature-scaled | 0.057 ± 0.005 |
| Risk-class ECE, Platt-calibrated | 0.017 ± 0.005 |
| Naive regret (% oracle utility) | 2.64 ± 0.51% |
| Temperature-scaled regret | 0.54 ± 0.21% |
| Task-calibrated regret | 0.33 ± 0.08% |
| Predictive upper-bound regret | 0.34 ± 0.15% |
| Naive regret closed by task calibration | 87.46 ± 2.20% |
| Calibrated predictive-interval coverage | 0.932 ± 0.020 |

The single seed used for the main figures has 0.900 base Poisson-binomial
coverage; the held-out interval calibration chooses a one-count expansion and
yields 0.953 deployment coverage. Exact values are stored in `results.json`.

## Repository layout

```text
calibration-aware-patrol/
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── results.json
├── results_seed_level.csv
├── experiments/
│   └── run_main.py
├── figures/
├── src/
│   ├── allocation.py
│   ├── calibration.py
│   ├── figure_style.py
│   └── synthetic.py
└── tests/
```

## Installation and reproduction

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python experiments/run_main.py
```

A faster validation run is available:

```bash
python experiments/run_main.py --quick
```

Outputs:

- `results.json`: headline, multi-seed, sensitivity, and asymmetric-cost results
- `results_seed_level.csv`: raw seed-level metrics
- `figures/`: regenerated 600-dpi PNG and vector PDF/SVG figures

The full default experiment takes roughly 30 seconds on the current CPU-only
test environment. No model download or GPU is required.

### Figure style

The plotting code uses a compact editorial style suitable for two-column
scientific manuscripts: restrained colour-blind-safe accents, lowercase panel
labels, thin axes, outward ticks, and uncertainty bands for multi-seed results.
The style is inspired by common Nature/Science figure conventions rather than
being an exact copy of either journal's production template.  Shared settings
are defined in `src/figure_style.py`.

## Tests

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

The tests check:

- exact budget and per-cell cap satisfaction;
- agreement of water-filling with a generic constrained optimiser;
- temperature-scaling NLL improvement;
- monotonic binary calibration;
- a known Poisson-binomial distribution;
- fixed deployment data during calibration-size sweeps; and
- basic end-to-end interval and allocation invariants.

## Interpretation and deployment cautions

- A person detection is not equivalent to a poacher detection. Real systems
  should combine human-presence alerts with authorised-person records, patrol
  logs, access routes, time, vehicle detections, and local operational context.
- Predictive intervals rely on the calibrated probability model and the
  representativeness of the interval-calibration batch. Seasonal or hardware
  shifts require re-evaluation and possibly recalibration.
- The synthetic experiments establish a controlled methodological result, not
  evidence of field effectiveness.

## Licence and citation

The code is released under the MIT License. Citation metadata is provided in
`CITATION.cff`. Generated figures and the accompanying paper may be distributed
under their separately stated licence.
