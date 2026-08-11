"""Core reflection-method algorithm.

All functions operate on linear time units (JD, HJD, MJD, phase, minutes, etc.).
The library performs no time conversion or I/O.
"""

from typing import Optional
import numpy as np
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline
from scipy.optimize import minimize_scalar

from .result import MinimumResult


def fit_spline(
    x: np.ndarray,
    y: np.ndarray,
    pts_per_knot: int = 10,
    degree: int = 3,
    w: Optional[np.ndarray] = None,
) -> LSQUnivariateSpline:
    """Fit a least-squares spline to the data.

    Args:
        x: Independent variable (must be sorted, but will be sorted internally)
        y: Dependent variable
        pts_per_knot: Target number of data points per spline knot
        degree: Spline degree (1-5)
        w: Optional weights (1/sigma)

    Returns:
        Fitted LSQUnivariateSpline with ext=3 (raises on extrapolation)

    Raises:
        ValueError: If too few points for the given degree or fit fails
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("x and y must be numpy arrays")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n < degree + 2:
        raise ValueError(f"Too few points ({n}) for spline degree {degree}.")

    # Ensure x is sorted (spline requires monotonic x)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    w_sorted = w[order] if w is not None else None

    xmin, xmax = float(x_sorted.min()), float(x_sorted.max())
    n_inner = max(1, int(n / pts_per_knot))
    n_inner = min(n_inner, max(1, n - degree - 1))

    while n_inner >= 1:
        knots = np.linspace(xmin, xmax, n_inner + 2)[1:-1]
        try:
            return LSQUnivariateSpline(x_sorted, y_sorted, knots, k=degree, w=w_sorted, ext=3)
        except ValueError:
            n_inner //= 2

    raise ValueError("Failed to fit the spline — too few points.")


def spline_variance(spl: LSQUnivariateSpline, x: np.ndarray, y: np.ndarray) -> float:
    """Mean squared residual of spline fit."""
    return float(np.mean((y - spl(x)) ** 2))


def combine(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    w: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Combine original and reflected points about x0.

    Args:
        x: Original x values
        y: Original y values
        x0: Reflection point
        w: Optional weights

    Returns:
        (x_all, y_all, w_all) sorted by x
    """
    x_all = np.concatenate([x, 2 * x0 - x])
    y_all = np.concatenate([y, y])
    if w is not None:
        w_all = np.concatenate([w, w])
    else:
        w_all = None

    order = np.argsort(x_all)
    return x_all[order], y_all[order], (w_all[order] if w_all is not None else None)


def find_x0(
    x: np.ndarray,
    y: np.ndarray,
    pts_per_knot: int = 10,
    degree: int = 3,
    w: Optional[np.ndarray] = None,
    n_scan: int = 200,
    x0_window: float = 0.1,
    x0_initial_guess: Optional[float] = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Scan x0 to find the minimum of σ₂ (residual std of refitted spline).

    Args:
        x, y: Data arrays (x must be sorted or will be sorted)
        pts_per_knot, degree, w: Spline parameters
        n_scan: Number of x0 values to evaluate
        x0_window: Fraction of x-range to scan around initial guess
        x0_initial_guess: Starting guess; if None, uses minimum of initial spline

    Returns:
        (x0_opt, x0_grid, sigma2_grid) where x0_opt minimizes σ₂
    """
    # Initial spline on original data
    spl1 = fit_spline(x, y, pts_per_knot, degree, w)

    xmin, xmax = float(x.min()), float(x.max())
    if x0_initial_guess is None:
        x_fine = np.linspace(xmin, xmax, 2001)
        x0_initial_guess = float(x_fine[int(np.argmin(spl1(x_fine)))])

    window_width = x0_window * (xmax - xmin)
    lo = max(xmin, x0_initial_guess - window_width / 2)
    hi = min(xmax, x0_initial_guess + window_width / 2)

    x0_grid = np.linspace(lo, hi, n_scan)
    sigma2 = np.empty(n_scan)

    for i, x0 in enumerate(x0_grid):
        xs, ys, ws = combine(x, y, x0, w)
        spl_refit = fit_spline(xs, ys, 2 * pts_per_knot, degree, ws)
        sigma2[i] = np.sqrt(spline_variance(spl_refit, xs, ys))

    sigma2 = np.maximum(sigma2, 1e-6)

    # Refine minimum with spline interpolation
    spl_sigma = UnivariateSpline(x0_grid, sigma2, k=3, s=0)
    res = minimize_scalar(spl_sigma, bounds=(x0_grid[0], x0_grid[-1]), method="bounded")
    x0_opt = float(res.x)

    return x0_opt, x0_grid, sigma2


def bootstrap_x0(
    x: np.ndarray,
    y: np.ndarray,
    x0_opt: float,
    spl1: LSQUnivariateSpline,
    pts_per_knot: int,
    degree: int,
    w: Optional[np.ndarray],
    n_bootstrap: int = 60,
    n_scan_boot: int = 80,
    x0_window: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, float]:
    """Estimate x0 uncertainty via residual bootstrap.

    Args:
        x, y: Original data
        x0_opt: Optimal x0 from find_x0
        spl1: Initial spline fit to original data
        pts_per_knot, degree, w: Spline parameters
        n_bootstrap: Number of bootstrap iterations
        n_scan_boot: Grid resolution for each bootstrap iteration
        x0_window: Fraction of range for scan window
        rng: Random number generator (default: new PCG64)

    Returns:
        (x0_std, x0_lo, x0_hi) — std, 16th and 84th percentiles
    """
    if rng is None:
        rng = np.random.default_rng()

    residuals = y - spl1(x)
    n = len(x)
    x0_boot = np.empty(n_bootstrap)

    xmin, xmax = float(x.min()), float(x.max())
    window_width = x0_window * (xmax - xmin)

    for k in range(n_bootstrap):
        y_boot = spl1(x) + residuals[rng.integers(0, n, n)]
        spl_boot = fit_spline(x, y_boot, pts_per_knot, degree, w)

        # Coarse scan for this bootstrap sample
        x_fine = np.linspace(xmin, xmax, 1001)
        center = float(x_fine[int(np.argmin(spl_boot(x_fine)))])
        lo = max(xmin, center - window_width / 2)
        hi = min(xmax, center + window_width / 2)

        grid = np.linspace(lo, hi, n_scan_boot)
        sig = np.empty(len(grid))

        for i, x0 in enumerate(grid):
            xs, ys, ws = combine(x, y_boot, x0, w)
            sp = fit_spline(xs, ys, 2 * pts_per_knot, degree, ws)
            sig[i] = np.sqrt(spline_variance(sp, xs, ys))

        sig = np.maximum(sig, 1e-6)
        spl_sig = UnivariateSpline(grid, sig, k=3, s=0)
        x0_boot[k] = minimize_scalar(spl_sig, bounds=(grid[0], grid[-1]), method="bounded").x

    x0_std = float(x0_boot.std(ddof=1))
    x0_lo = float(np.percentile(x0_boot, 16))
    x0_hi = float(np.percentile(x0_boot, 84))

    return x0_std, x0_lo, x0_hi


def find_minimum(
    x: np.ndarray,
    y: np.ndarray,
    pts_per_knot: int = 10,
    degree: int = 3,
    w: Optional[np.ndarray] = None,
    n_scan: int = 200,
    x0_window: float = 0.1,
    n_bootstrap: int = 60,
    n_scan_boot: int = 80,
    rng: Optional[np.random.Generator] = None,
) -> MinimumResult:
    """Full pipeline: find x0 and its uncertainty.

    Args:
        x, y: Data arrays (x will be sorted internally)
        pts_per_knot: Points per knot for spline
        degree: Spline degree (1-5)
        w: Optional weights (1/sigma)
        n_scan: Scan resolution for main search
        x0_window: Fraction of x-range for scan window
        n_bootstrap: Bootstrap iterations
        n_scan_boot: Grid resolution per bootstrap iteration
        rng: Random generator for bootstrap

    Returns:
        MinimumResult with x0, uncertainties, and diagnostics
    """
    if rng is None:
        rng = np.random.default_rng()

    # Main scan
    x0_opt, x0_grid, sigma2 = find_x0(
        x, y, pts_per_knot, degree, w, n_scan, x0_window
    )

    # Initial spline for bootstrap
    spl1 = fit_spline(x, y, pts_per_knot, degree, w)

    # Bootstrap
    x0_std, x0_lo, x0_hi = bootstrap_x0(
        x, y, x0_opt, spl1, pts_per_knot, degree, w,
        n_bootstrap, n_scan_boot, x0_window, rng
    )

    sigma_min = float(np.min(sigma2))

    return MinimumResult(
        x0=x0_opt,
        x0_std=x0_std,
        x0_lo=x0_lo,
        x0_hi=x0_hi,
        sigma_min=sigma_min,
        n_points=len(x),
        n_bootstrap=n_bootstrap,
    )