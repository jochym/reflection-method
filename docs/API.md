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
    refine_x0_minimum,
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

### `find_x0(x, y, pts_per_knot=10, degree=3, w=None, n_scan=200, x0_window=0.1, x0_initial_guess=None, find_peak=False, parabola_window=0.05)`

Scan trial reflection points to locate the minimum of σ₂(x₀). The σ₂ curve
is evaluated on a grid centered on an initial guess (by default the minimum
— or the maximum when `find_peak=True` — of a coarse spline fit to the data)
and spanning `x0_window * (xmax - xmin)`. The minimum is refined with a
local polynomial fit to σ₂(x₀) (see `refine_x0_minimum` below).

- **Parameters**: `n_scan` — grid resolution (default 200);
  `x0_window` — scan window as a fraction of the x-range (default 0.1);
  `x0_initial_guess` — optional explicit initial guess (default None);
  `find_peak` — if True, the feature is a *maximum* of `y` (e.g. an eclipse
  on a magnitude scale) and the initial guess uses the spline maximum
  instead of the minimum (default False); `parabola_window` — fit window for
  the polynomial refinement as a fraction of the scan range (default 0.05).
- **Returns**: `(x0_opt, x0_grid, sigma2_grid)` — refined minimum, scan
  grid, and σ₂ values at each grid point.

### `refine_x0_minimum(x0_grid, sigma2, parabola_window=0.05)`

Refine the σ₂(x₀) minimum with a local polynomial fit. A quartic polynomial
is fitted to the grid points inside `parabola_window` around the grid
minimum — a parabola when fewer than five points fall in the window, and the
grid minimum itself when fewer than three do. The minimum of the polynomial
is located analytically (roots of the derivative, or the vertex formula for
a parabola) and clamped to the scan bounds. The local fit is more stable
than interpolating the σ₂ curve with a spline, which can produce spurious
minima when the curve is flat near the minimum.

- **Parameters**: `x0_grid` — scan grid; `sigma2` — σ₂ values at each grid
  point; `parabola_window` — fit window fraction of the scan range
  (default 0.05).
- **Returns**: `(x0_opt, poly, window_lo, window_hi)` — refined minimum, the
  fitted polynomial (`numpy.poly1d`), and the bounds of the fitted window in
  the units of `x0_grid`. `poly` is a constant equal to the median σ₂ when
  the fit is not possible. Pass `poly`, `window_lo` and `window_hi` to
  `plot_all` to draw the fitted curve over the points it was fitted to.

### `bootstrap_x0(x, y, x0_opt, spl1, pts_per_knot, degree, w, n_bootstrap=60, n_scan_boot=80, x0_window=0.1, rng=None, return_samples=False, find_peak=False, parabola_window=0.05)`

Estimate the uncertainty of `x0` via **residual bootstrap**: residuals from
the initial spline fit are resampled with replacement and added to the fit;
the full scan is repeated per iteration at reduced resolution. Each
iteration's minimum is refined with the same local polynomial fit as
`find_x0`.

- **Parameters**: `spl1` — initial spline fit to the original data;
  `n_bootstrap` — iterations (default 60; use 100–200 for tighter CIs);
  `n_scan_boot` — per-iteration scan resolution (default 80);
  `rng` — `numpy.random.Generator` (default: new PCG64 generator; pass a
  seeded generator for reproducibility); `return_samples` — when True,
  append the array of individual bootstrap estimates to the return value;
  `find_peak` — if True, the per-iteration initial guess uses the spline
  maximum (eclipses on a magnitude scale); must match the value used in
  `find_x0` (default False); `parabola_window` — fit window fraction for the
  polynomial refinement (default 0.05).
- **Returns**: `(x0_std, x0_lo, x0_hi)` — bootstrap standard error (ddof=1)
  and the 16th/84th percentiles. With `return_samples=True` a fourth
  element `x0_boot` (the raw estimates, length `n_bootstrap`) is appended.

### `find_minimum(x, y, pts_per_knot=10, degree=3, w=None, n_scan=200, x0_window=0.1, n_bootstrap=60, n_scan_boot=80, rng=None, return_samples=False, find_peak=False, parabola_window=0.05)`

Full pipeline: runs `find_x0`, then `bootstrap_x0`, and packages the result
in a `MinimumResult`.

- **Parameters**: as for the components above (see `find_x0` and
  `bootstrap_x0`); `return_samples` — when True, also return the raw
  bootstrap estimates; `find_peak` — if True, the eclipsing feature is a
  *maximum* of `y` (the eclipse is the largest magnitude value), so the
  initial-guess step looks for a peak instead of a dip. Use this when
  working directly on magnitudes (default False); `parabola_window` — fit
  window fraction for the polynomial refinement (default 0.05).
- **Returns**: `MinimumResult`. With `return_samples=True`, a tuple
  `(result, x0_boot)` is returned instead, where `x0_boot` is the array of
  `n_bootstrap` individual estimates (useful for plotting the bootstrap
  distribution).
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

### `plot_original_spline(x, y, spl1, x0_opt, x0_std, xlabel="Time", ylabel="magnitude", x_unit="", invert_y=False, ax=None)`

Plots the original data, the initial spline fit, and a vertical line at
`x0_opt ± x0_std`.

### `plot_scan(x0_grid, sigma1, sigma2, spl_sigma, x0_opt, x0_grid_idx_min, xlabel="x₀", ylabel="residual σ", x_unit="", ax=None)`

Plots σ₁ and σ₂ over the scan grid plus the polynomial fit to σ₂ (pass the
`poly` returned by `refine_x0_minimum`), with vertical lines at the grid
minimum and the refined `x0_opt`.

### `plot_all(x, y, spl1, xr, spl2, x0_opt, x0_std, x0_grid, sigma2, spl_sigma, x0_boot, xlabel="Time", ylabel="magnitude", x_unit="", invert_y=False, utc0=None, fit_window_lo=None, fit_window_hi=None, **kwargs)`

Standard 2×2 diagnostic figure:

1. **Light curve** — original + reflected points, original and refitted
   splines, `x0` line with uncertainty.
2. **σ₂ scan** — σ₂(x₀) markers and the polynomial fit curve, drawn only
   over the window used for the fit (`fit_window_lo` … `fit_window_hi`),
   x-axis limited to `x0 ± 3σ` (same as the histogram).
3. **Bootstrap histogram** — distribution of `x0` estimates, sharing the
   same x-limits as the σ₂ panel.

`invert_y=True` draws the magnitude axis with brighter stars on top (the
standard convention); the σ₂ and histogram panels are unaffected.

Returns a `matplotlib.figure.Figure`.

```python
fig = plot_all(x, y, spl1, xr, spl2, x0_opt, x0_std,
               x0_grid, sigma2, spl_sigma, x0_boot,
               xlabel="Time", ylabel="magnitude", x_unit="min", invert_y=True)
fig.savefig("output.png", dpi=150, bbox_inches="tight")
```

## CLI (optional `[cli]` extra)

```bash
reflection-method DATA.csv \
    -x DATE-OBS -y MAG -w MAG_ERR \
    -t iso -k 10 -d 3 -n 200 -b 60 -s 42 \
    -o result.json -p diagnostic.png
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `INPUT_FILE` | — | — | CSV file (AAVSO extended format supported) |
| `--x-col` | `-x` | `DATE-OBS` | Time column name |
| `--y-col` | `-y` | `MAG` | Magnitude column name |
| `--w-col` | `-w` | (none) | Uncertainty column name |
| `--time-format` | `-t` | `iso` | `iso`, `jd`, `hjd`, `mjd`, `minutes` |
| `--pts-per-knot` | `-k` | `10` | Points per spline knot |
| `--degree` | `-d` | `3` | Spline degree (1–5) |
| `--n-scan` | `-n` | `200` | Scan resolution for x₀ |
| `--x0-window` | `-W` | `0.1` | Scan window as fraction of x-range |
| `--parabola-window` | `-P` | `0.05` | Fit window for polynomial refinement, as fraction of scan range |
| `--n-bootstrap` | `-b` | `60` | Bootstrap iterations |
| `--n-scan-boot` | `-B` | `80` | Bootstrap scan resolution |
| `--seed` | `-s` | (none) | Random seed for reproducibility |
| `--output`, `-o` | — | stdout | Output JSON file |
| `--plot` | `-p` | (none) | Save 4-panel plot to file |
| `--plot-format` | `-F` | `png` | `png`, `pdf`, `svg` |

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
