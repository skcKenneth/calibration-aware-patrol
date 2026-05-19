# From Pixels to Patrols: Calibration-Aware Anti-Poaching

End-to-end pipeline that propagates camera-trap classifier
uncertainty through to spatial patrol allocation, comprising
**temperature scaling**, **split-conformal regression on the
per-cell expected threat count**, and four allocation policies
(*oracle*, *naive*, *calibrated point estimate*, *distributionally
robust upper*) on a budget-constrained water-filling objective.

Submission for the **AI4Nature Workshop at AVSS 2026** (Lecce,
Italy). The paper is in `paper/main.tex`; this README documents the
code.

## Gap addressed

Camera-trap classifiers in current ecological deployments are
*systematically miscalibrated* (Dussert et al., 2025). Existing
calibration work covers inferential downstream tasks (occupancy,
interaction inference, alerts) but **not** decision-stage tasks like
patrol allocation. We close that gap: we measure how much the
miscalibration costs at the decision stage, and how cheaply
standard post-hoc calibration plus a conformal regression layer
recovers it.

## Headline results

On a synthetic camera-trap network of 400 cells and ~16,000
images, calibrated to the over-confidence regime reported in the
literature (`T_true = 0.45`, classifier ECE = 0.34):

| Metric                                           | Value   |
| ------------------------------------------------ | ------- |
| ECE before / after temperature scaling           | 0.344 / 0.052 |
| Brier score before / after                       | 0.852 / 0.660 |
| Recovered temperature $\hat T$                   | 4.27     |
| Naive policy regret (% of oracle utility)        | 2.99%    |
| Calibrated policy regret                         | 0.63%    |
| DRO (conformal upper) regret                     | 0.77%    |
| **% of naive regret closed by calibration**      | **78.96%** |
| **% of naive regret closed by DRO**              | **74.25%** |
| Empirical conformal coverage (nominal 90%)       | 0.845    |

Sensitivity sweeps in `experiments/run_main.py` confirm the gain is
monotone in miscalibration severity and that empirical conformal
coverage approaches its nominal value as the calibration set grows.

## Repository layout

```
calibration-aware-patrol/
├── README.md
├── requirements.txt
├── results.json                  # numerical results from running run_main.py
├── src/
│   ├── synthetic.py              # synthetic camera-trap world + classifier
│   ├── calibration.py            # temperature scaling + split conformal
│   └── allocation.py             # water-filling allocator + four policies
├── experiments/
│   └── run_main.py               # main driver: single run + two sweeps
├── figures/                      # output of run_main.py
└── paper/
    ├── main.tex                  # IEEE-format AI4Nature submission
    ├── ref.bib
    └── figs/                     # paper figures (= figures/)
```

## Reproducing the results

```bash
python -m pip install -r requirements.txt
python experiments/run_main.py
```

This runs the headline single-world pipeline, the $T_{\mathrm{true}}$
sweep, and the calibration-fraction sweep. Outputs land in
`figures/` (PNG plots) and `results.json` (all numerical values
quoted in the paper).

Total runtime is under 30 seconds on a 2020-era laptop CPU; no GPU
or model downloads are required.

## Method walkthrough

1. **Synthetic world** (`src/synthetic.py`).
   `make_world()` returns ground-truth per-cell class probabilities
   drawn from a Fourier-filtered Gaussian random field, the realised
   image counts per cell, and a classifier that emits softmax
   outputs with controllable miscalibration `T_true` and per-class
   bias `bias_threat`. Setting `T_true < 1` produces over-confidence
   in the style reported for real DL camera-trap classifiers.

2. **Calibration** (`src/calibration.py`).
   `fit_temperature()` minimises NLL of `softmax(logits/T)` on the
   calibration set via `scipy.optimize.minimize_scalar`. ECE and
   reliability diagrams are 15-bin estimators on
   maximum-confidence/correctness pairs.

3. **Split-conformal regression on per-cell expected count.**
   `conformal_residuals()` computes the standardised residual
   `r_i = (z_true_cal_i - z_hat_cal_i) / sqrt(n_cal_i)` per cell,
   `conformal_quantile()` returns the $(1-\alpha)$ empirical
   quantile of `|r_i|`, and `per_cell_intervals()` produces a
   prediction interval $\hat z_i \pm q \sqrt{n^{\mathrm{dep}}_i}$ on
   the deployment set. The $\sqrt{n}$ normalisation makes residuals
   approximately exchangeable across heterogeneous cell sizes.

4. **Allocation** (`src/allocation.py`).
   `water_fill()` solves
   $\max\sum_i z_i (1 - e^{-\lambda \tau_i})$ subject to a budget
   and per-cell cap by bisection on the KKT dual variable. Four
   policy wrappers (`policy_oracle`, `policy_naive`,
   `policy_calibrated`, `policy_dro_upper`) plug different planning
   values $z^{\mathrm{plan}}$ into the same allocator.

5. **Evaluation.** Regret = $U(\boldsymbol{\tau}_{\mathrm{oracle}};
   \mathbf{z}^{\star}) - U(\boldsymbol{\tau}_{\mathrm{policy}};
   \mathbf{z}^{\star})$, evaluated on the true threat field, and
   reported as a percentage of oracle utility.

## Real-data extension

The pipeline is module-local in the classifier. Replacing
`src/synthetic.py` with a real classifier (e.g.
[MegaDetector](https://github.com/agentmorris/MegaDetector) +
[SpeciesNet](https://github.com/google/cameratrapai) on
[iWildCam](https://github.com/visipedia/iwildcam_comp) or Snapshot
Serengeti) gives a real-data instantiation; no other module
changes. The calibration and conformal layers operate on softmax
outputs and image-level labels alone.

## License

MIT for the code; figures and paper text CC-BY-4.0.

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{cdsj5_2026_pixels_patrols,
  title     = {From Pixels to Patrols: A Calibration-Aware Sensor
               Fusion Pipeline for Camera-Trap-Driven Anti-Poaching
               Resource Allocation},
  booktitle = {AI4Nature Workshop at AVSS},
  year      = {2026}
}
```
