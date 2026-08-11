# API Reference

The `reflection-method` package provides a small, functional API. All core
functions operate on plain NumPy arrays in **linear time units** (JD, HJD,
MJD, phase, minutes, ...) — the library performs no I/O and no time
conversion.

## Package layout

| Module | Contents |
|--------|----------|
| `reflection_method` | Public API (`find_minimum`, `MinimumResult`, low-level core functions) |
| `reflection_method.core` | Core algorithm: splines, reflection, scan, bootstrap |
| `reflection_method.result` | `MinimumResult` container |
| `reflection_method.plot` | Optional matplotlib plotting (needs `[plot]` extra) |
| `reflection_method.cli` | `click`-based command-line interface (needs `[cli]` extra) |

## `reflection_method` — public API

```python
from reflection_method import (
    find_minimum,
    find_x0,
    fit_spline,
    spline_variance,
    combine,
    bootstrap_x0,
    MinimumResult,
)
```

Plotting functions are exported automatically when the `[plot]` extra is
installed:

```python
from reflection_method import plot_all, plot_scan, plot_original_spline
```

## `MinimumResult`

A `typing.NamedTuple` returned by `find_minimum`.

| Field | Type | Description |
|-------|------|-------------|
| `x0` | `float` | Optimal reflection point (same units as input `x`) |
| `x0_std` | `float` | Bootstrap standard error (sample std, ddof=1) |
| `x0_lo` | `float` | 16th percentile of bootstrap distribution (lower 68% CI) |
| `x0_hi` | `float` | 84th percentile of bootstrap distribution (upper 68% CI) |
| `sigma_min` | `float` | Minimum residual standard deviation σ₂ |
| `n_points` | `int` | Number of data points used |
| `n_bootstrap` | `int` | Number of bootstrap iterations performed |

The 68% confidence interval `[x0_lo, x0_hi]` is approximately ±1σ for a
Gaussian bootstrap distribution; for asymmetric distributions it is not
centered on `x0`. All `x0*` values share the units of the input `x` array.

## Core algorithm

### `fit_spline(x, y, pts_per_knot=10, degree=3, w=None)`

Fit a least-squares B-spline (`scipy.interpolate.LSQUnivariateSpline`) with
automatically placed knots. The number of interior knots is derived from
`pts_per_knot` (points per knot) and adaptively halved until the fit
succeeds. Uses `ext=3`, so evaluation outside the knot range raises.

- **Parameters**: `x`, `y` — 1-D arrays of equal length; `pts_per_knot` —
  target points per knot (default 10); `degree` — spline degree 1–5
  (default 3); `w` — optional weights (typically `1 / sigma`).
- **Returns**: `LSQUnivariateSpline`.
- **Raises**: `ValueError` on mismatched lengths, too few points, or fit
  failure.

### `spline_variance(spl, x, y)`

Mean squared residual `mean((y - spl(x))**2)` of a spline against data.

- **Parameters**: `spl` — fitted spline; `x`, `y` — evaluation arrays.
- **Returns**: `float`.

### `combine(x, y, x0, w=None)`

Combine original points with points reflected about a trial minimum `x0`.
The reflected point of `(x_i, y_i)` is `(2*x0 - x_i, y_i)`. Result is
sorted by abscissa.

- **Parameters**: `x`, `y` — original data; `x0` — trial reflection point;
  `w` — optional weights (duplicated for reflected points).
- **Returns**: `(x_all, y_all, w_all)` tuple of sorted arrays; `w_all` is
  `None` if `w` is `None`.

### `find_x0(x, y, pts_per_knot=10, degree=3, w=None, n_scan=200, x0_window=0.1, x0_initial_guess=None)`

Scan trial reflection points to locate the minimum of σ₂(x₀). The σ₂ curve
is evaluated on a grid centered on an initial guess (by default the minimum
of a coarse spline fit to the data) and spanning
`x0_window * (xmax - xmin)`. The minimum is refined with a cubic spline
interpolation of σ₂(x₀) minimized by bounded Brent's method.

- **Parameters**: `n_scan` — grid resolution (default 200);
  `x0_window` — scan window as a fraction of the x-range (default 0.1);
  `x0_initial_guess` — optional explicit initial guess (default None).
- **Returns**: `(x0_opt, x0_grid, sigma2_grid)` — refined minimum, scan
  grid, and σ₂ values at each grid point.

### `bootstrap_x0(x, y, x0_opt, spl1, pts_per_knot, degree, w, n_bootstrap=60, n_scan_boot=80, x0_window=0.1, rng=None)`

Estimate the uncertainty of `x0` via **residual bootstrap**: residuals from
the initial spline fit are resampled with replacement and added to the fit;
the full scan is repeated per iteration at reduced resolution.

- **Parameters**: `spl1` — initial spline fit to the original data;
  `n_bootstrap` — iterations (default 60; use 100–200 for tighter CIs);
  `n_scan_boot` — per-iteration scan resolution (default 80);
  `rng` — `numpy.random.Generator` (default: new PCG64 generator; pass a
  seeded generator for reproducibility).
- **Returns**: `(x0_std, x0_lo, x0_hi)` — bootstrap standard error (ddof=1)
  and the 16th/84th percentiles.

### `find_minimum(x, y, pts_per_knot=10, degree=3, w=None, n_scan=200, x0_window=0.1, n_bootstrap=60, n_scan_boot=80, rng=None)`

Full pipeline: runs `find_x0`, then `bootstrap_x0`, and packages the result
in a `MinimumResult`.

- **Parameters**: as for the components above (see `find_x0` and
  `bootstrap_x0`).
- **Returns**: `MinimumResult`.
- **Raises**: `ValueError` for invalid input arrays or failed fits/scans.

**Example**

```python
import numpy as np
from reflection_method import find_minimum

rng = np.random.default_rng(42)
x = np.sort(rng.uniform(0, 1, 300))
y = 1 - 0.5 * np.exp(-((x - 0.5) ** 2) / 0.001) + rng.normal(0, 0.01, 300)

result = find_minimum(x, y, n_bootstrap=100, rng=np.random.default_rng(42))
print(result.x0, "±", result.x0_std)          # ~0.5000 ± ~0.0002
print([result.x0_lo, result.x0_hi])           # 68% confidence interval
```

## Plotting (optional `[plot]` extra)

All plotting functions accept an optional `ax` and return the axes (or the
figure for `plot_all`). They require matplotlib.

### `plot_original_spline(x, y, spl1, x0_opt, x0_std, xlabel="Time", ylabel="relative magnitude", x_unit="", ax=None)`

Plots the original data, the initial spline fit, and a vertical line at
`x0_opt ± x0_std`.

### `plot_scan(x0_grid, sigma1, sigma2, spl_sigma, x0_opt, x0_grid_idx_min, xlabel="x₀", ylabel="residual σ", x_unit="", ax=None)`

Plots σ₁ and σ₂ over the scan grid plus the interpolating spline for σ₂,
with vertical lines at the grid minimum and the refined `x0_opt`.

### `plot_all(x, y, spl1, xr, spl2, x0_opt, x0_std, x0_grid, sigma2, spl_sigma, x0_boot, xlabel="Time", ylabel="relative magnitude", x_unit="", **kwargs)`

Standard 2×2 diagnostic figure:

1. **Light curve** — original + reflected points, original and refitted
   splines, `x0` line with uncertainty.
2. **σ₂ scan** — σ₂(x₀) markers and smooth spline curve, x-axis limited to
   `x0 ± 3σ`.
3. **Bootstrap histogram** — distribution of `x0` estimates, sharing the
   same x-limits as the σ₂ panel.

Returns a `matplotlib.figure.Figure`.

```python
fig = plot_all(x, y, spl1, xr, spl2, x0_opt, x0_std,
               x0_grid, sigma2, spl_sigma, x0_boot,
               xlabel="Time", ylabel="relative magnitude", x_unit="min")
fig.savefig("output.png", dpi=150, bbox_inches="tight")
```

## CLI (optional `[cli]` extra)

```bash
reflection-method find DATA.csv \
    --x-col DATE-OBS --y-col MAG --w-col MAG_ERR --invert-mag \
    --time-format iso \
    --pts-per-knot 10 --degree 3 --n-scan 200 --n-bootstrap 60 --seed 42 \
    --output result.json --plot diagnostic.png
```

| Option | Default | Description |
|--------|---------|-------------|
| `INPUT_FILE` | — | CSV file (AAVSO extended format supported) |
| `--x-col` | `DATE-OBS` | Time column name |
| `--y-col` | `MAG` | Magnitude/flux column name |
| `--w-col` | (none) | Uncertainty column name |
| `--time-format` | `iso` | `iso`, `jd`, `hjd`, `mjd`, `minutes` |
| `--invert-mag/--no-invert-mag` | on | Convert MAG to relative flux |
| `--pts-per-knot` | `10` | Points per spline knot |
| `--degree` | `3` | Spline degree (1–5) |
| `--n-scan` | `200` | Scan resolution for x₀ |
| `--x0-window` | `0.1` | Scan window as fraction of x-range |
| `--n-bootstrap` | `60` | Bootstrap iterations |
| `--n-scan-boot` | `80` | Bootstrap scan resolution |
| `--seed` | (none) | Random seed for reproducibility |
| `--output`, `-o` | stdout | Output JSON file |
| `--plot` | (none) | Save 4-panel plot to file |
| `--plot-format` | `png` | `png`, `pdf`, `svg` |

For `--time-format iso`, the earliest timestamp is used as the origin, `x0`
is reported in minutes from that origin, and the JSON output additionally
contains `utc_time` (ISO-8601, trailing `Z`) and `utc_uncertainty_s`.

## Reproducibility

Pass a seeded generator to `find_minimum` or `bootstrap_x0`:

```python
from numpy.random import default_rng
rng = default_rng(42)
result = find_minimum(x, y, rng=rng)
```

The CLI `--seed` option provides the equivalent guarantee from the shell.
