# HC-MoE

Empirical-null fitting utilities and scripts for analyzing logits and visualizing QQ plots.

This repo includes two complementary approaches for estimating an empirical null and computing p-values for model outputs (e.g., logits):
- Efron-style central matching (polynomial fit to log-density) — `empirical_null.py`
- Lindsey-inspired approaches (GLM with Laplace/Gaussian shapes or Student's t) — `empirical_null_lindsey.py`

Both scripts produce QQ plots of −log(p) against Exp(1) theoretical quantiles to assess calibration.

## Quick Start

- Python 3.9+ recommended
- Install dependencies:

```
pip install -r requirements.txt
```

## Data

- The scripts expect a Parquet file at `measurements/layer_1.parquet` containing columns like `logit_<i>`.
- Example CSVs may also be present in `measurements/`, but the primary scripts read Parquet.

## Usage

Efron central matching approach (plots shown interactively):

```
python empirical_null.py
```

- Randomly selects several `logit_<i>` blocks from `layer_1.parquet`.
- Left panel: histogram with fitted empirical null (polynomial in log-density of standardized data).
- Right panel: QQ plot of −log(p) vs Exp(1).

Lindsey-inspired methods (saves figures):

```
python empirical_null_lindsey.py
```

- Fits one of: `laplace`, `gaussian` (via Poisson GLM) or `student_t` (non-linear fit). The script currently uses `student_t`.
- Saves figures to `figures/qq_plot_logit_<i>.png`.

## Repo Structure

- `empirical_null.py` — central matching implementation and QQ plotting.
- `empirical_null_lindsey.py` — GLM / Student's t variants and QQ plotting.
- `measurements/` — example measurement data (Parquet and CSVs).
- `figures/` — generated images (gitignored by default).
- `requirements.txt` — Python dependencies.
- `.gitignore` — ignores envs, caches, figures, OS files.

## Notes

- If you prefer deterministic runs, set a NumPy seed at the top of the scripts before sampling blocks.
- Adjust bin counts, quantile ranges, and fit options inside the scripts to match your dataset.

## Contributing

Issues and PRs are welcome. Please keep changes minimal and focused.
