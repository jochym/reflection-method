# reflection-method

Eclipsing binary minimum finder using the **reflection method** (symmetry about the eclipse minimum).

## Installation

```bash
# Core library only (numpy, scipy)
pip install reflection-method

# With plotting (matplotlib)
pip install "reflection-method[plot]"

# With CLI (click)
pip install "reflection-method[cli]"

# Full
pip install "reflection-method[plot,cli]"
```

## Quickstart

```python
import numpy as np
from reflection_method import find_minimum

# x, y in any linear time units (JD, HJD, MJD, phase, minutes...)
x = np.array([...])  # time
y = np.array([...])  # flux or magnitude

result = find_minimum(x, y, pts_per_knot=10, degree=3, n_bootstrap=60)

print(f"Minimum at x0 = {result.x0:.4f} ± {result.x0_std:.4f}")
print(f"68% CI: [{result.x0_lo:.4f}, {result.x0_hi:.4f}]")
```

Result is a `MinimumResult` NamedTuple with all values in the **same units as input `x`**. The library performs no time conversion.

## Algorithm

1. Fit a low-order spline to the original data
2. Scan reflection point `x0` — for each candidate, reflect all points about `x0`, combine with original, refit spline, compute residual standard deviation `σ₂`
3. The `x0` minimizing `σ₂` is the eclipse minimum
4. Bootstrap uncertainty: resample residuals, repeat scan, compute percentiles

## CLI

```bash
reflection-method find data.csv \
    --x-col DATE-OBS --y-col MAG --w-col MAG_ERR --invert-mag \
    --time-format jd \
    --pts-per-knot 10 --degree 3 --n-scan 200 --n-bootstrap 60 \
    --output result.json --plot output.png
```

Options:
- `--time-format`: `jd` | `hjd` | `mjd` | `iso` (DATE-OBS) | `minutes` (relative)
- `--invert-mag`: Convert magnitudes to relative flux

## API Reference

### Core functions
```python
from reflection_method import (
    fit_spline,       # Fit LSQ spline to data
    find_x0,          # Scan x0 to minimize σ₂
    bootstrap_x0,     # Residual bootstrap uncertainty
    find_minimum,     # Full pipeline
    MinimumResult,    # Result container
)
```

### Result type
```python
MinimumResult(
    x0: float,           # reflection point (input units)
    x0_std: float,       # bootstrap std (input units)
    x0_lo: float,        # 16th percentile (input units)
    x0_hi: float,        # 84th percentile (input units)
    sigma_min: float,    # minimum σ₂
    n_points: int,       # data points used
    n_bootstrap: int,    # bootstrap iterations
)
```

### Reproducibility
Pass `rng=np.random.default_rng(seed)` to `find_minimum` or `bootstrap_x0`.

## Requirements

- Python ≥ 3.10
- numpy ≥ 1.24
- scipy ≥ 1.10

Optional:
- matplotlib ≥ 3.7 (`[plot]` extra)
- click ≥ 8.1 (`[cli]` extra)

## License

MIT