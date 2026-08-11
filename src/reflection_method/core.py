"""Core reflection-method algorithm.

All functions operate on linear time units (JD, HJD, MJD, phase, minutes, etc.).
The library performs no time conversion or I/O.

This module implements the reflection method for finding minima of eclipsing
binary light curves. The method exploits the symmetry of the light curve about
the minimum: reflecting all points about a trial minimum and measuring the
residual variance of a spline fit to the combined (original + reflected) data.
The trial point that minimizes this variance is the estimated minimum.

References
----------
- Kwee, K. K., & van Woerden, H. (1956). Bull. Astron. Inst. Netherlands, 12, 327.
- Andronov, I. L. (2003). Odessa Astron. Publ., 16, 55.
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
    """Fit a least-squares B-spline to the data.

    Fits a B-spline using `scipy.interpolate.LSQUnivariateSpline` with
    automatically placed knots. The number of knots is determined by
    ``pts_per_knot`` (target number of data points per knot). The spline
    is fitted with `ext=3`, meaning evaluation outside the knot range
    raises a `ValueError` rather than extrapolating.

    Parameters
    ----------
    x : np.ndarray
        Independent variable (abscissae). Must be 1-D array. Will be sorted
        internally if not already monotonic.
    y : np.ndarray
        Dependent variable (ordinates). Must be 1-D array of same length as ``x``.
    pts_per_knot : int, optional
        Target number of data points per interior knot. The actual number of
        knots is ``max(1, len(x) // pts_per_knot)``, capped at
        ``len(x) - degree - 1``. Default is 10.
    degree : int, optional
        Degree of the spline (1 to 5). Cubic (3) is default and recommended
        for most light curves. Use lower degree for noisy/sparse data.
    w : np.ndarray or None, optional
        Weights for weighted least squares. If provided, must be same length
        as ``x``. Typically ``1 / sigma`` where ``sigma`` are measurement
        uncertainties. Default is None (equal weights).

    Returns
    -------
    LSQUnivariateSpline
        Fitted spline object. Can be called like a function to evaluate at
        arbitrary ``x`` values within the knot range.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different lengths, or too few points for the
        given degree, or if the fit fails (e.g., due to degenerate knots).

    Notes
    -----
    The number of interior knots is chosen adaptively: starting from the
    target based on ``pts_per_knot``, it is halved until the fit succeeds.
    This handles cases where the data is too sparse for the requested knot
    density.

    The spline uses ``ext=3`` (raise on extrapolation) to prevent silent
    extrapolation beyond the data range. This is a safety feature; the caller
    should ensure evaluation points are within the data range.

    Examples
    --------
    >>> import numpy as np
    >>> from reflection_method import fit_spline
    >>> x = np.linspace(0, 1, 50)
    >>> y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, 50)
    >>> spl = fit_spline(x, y, pts_per_knot=10, degree=3)
    >>> y_fit = spl(np.linspace(0, 1, 100))
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
    """Compute mean squared residual of spline fit.

    Parameters
    ----------
    spl : LSQUnivariateSpline
        Fitted spline object.
    x : np.ndarray
        Abscissae at which to evaluate.
    y : np.ndarray
        Observed ordinates.

    Returns
    -------
    float
        Mean squared residual: ``mean((y - spl(x))**2)``.
    """
    return float(np.mean((y - spl(x)) ** 2))


def combine(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    w: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Combine original and reflected points about a trial minimum ``x0``.

    For each point ``(x_i, y_i)``, the reflected point is ``(2*x0 - x_i, y_i)``.
    The combined dataset is sorted by the abscissa.

    Parameters
    ----------
    x : np.ndarray
        Original abscissae.
    y : np.ndarray
        Original ordinates.
    x0 : float
        Trial reflection point (abscissa of symmetry axis).
    w : np.ndarray or None, optional
        Weights for original points. If provided, weights are duplicated for
        reflected points. Default is None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray or None]
        ``(x_all, y_all, w_all)`` where all arrays are sorted by ``x_all``.
        ``w_all`` is None if ``w`` is None.

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([1, 2, 1])
    >>> x_all, y_all, _ = combine(x, y, x0=2.0)
    >>> print(x_all)
    [1. 2. 3. 1. 2. 3.]  # Actually: [1, 3, 1, 3, ...] after sort
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
    """Scan trial reflection points to find the minimum of σ₂(x₀).

    The function evaluates the residual standard deviation σ₂ of a spline
    fitted to the combined (original + reflected) data for a grid of trial
    reflection points ``x0``. The grid is centered on an initial guess and
    spans a window of width ``x0_window * (xmax - xmin)``. The minimum is
    refined by fitting a cubic spline to the ``σ₂(x₀)`` curve and minimizing
    it with Brent's method.

    Parameters
    ----------
    x : np.ndarray
        Abscissae (time, phase, etc.). 1-D array.
    y : np.ndarray
        Ordinates (flux, magnitude, etc.). 1-D array, same length as ``x``.
    pts_per_knot : int, optional
        Points per knot for spline fits. Default 10.
    degree : int, optional
        Spline degree (1-5). Default 3.
    w : np.ndarray or None, optional
        Weights for weighted least squares. Default None.
    n_scan : int, optional
        Number of trial ``x0`` values in the scan grid. Default 200.
    x0_window : float, optional
        Fraction of the total ``x`` range to scan around the initial guess.
        The scan window is ``x0_window * (xmax - xmin)``. Default 0.1 (10%).
    x0_initial_guess : float or None, optional
        Initial guess for the minimum location. If None (default), the minimum
        of a coarse spline fit to the original data is used (evaluated on
        a 2001-point grid).

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        ``(x0_opt, x0_grid, sigma2_grid)`` where:
        - ``x0_opt`` : float, the refined minimum location
        - ``x0_grid`` : np.ndarray, the scan grid points
        - ``sigma2_grid`` : np.ndarray, σ₂ values at each grid point

    Raises
    ------
    ValueError
        If the scan fails (e.g., insufficient data, spline fit fails).

    Notes
    -----
    The scan window is clamped to the data range ``[xmin, xmax]``. The
    ``σ₂`` values are computed as the square root of the mean squared
    residual of the spline fitted to the combined (original + reflected)
    data at each trial ``x0``. A floor of 1e-6 is applied to avoid numerical
    issues.

    The refinement uses an interpolating cubic spline (``UnivariateSpline``
    with ``s=0``) through the ``(x0_grid, σ₂)`` points, minimized with
    SciPy's ``minimize_scalar`` (bounded Brent method).

    Examples
    --------
    >>> import numpy as np
    >>> from reflection_method import find_x0
    >>> phase = np.linspace(0, 1, 200)
    >>> flux = 1 - 0.5 * np.exp(-((phase - 0.5)**2) / 0.001)
    >>> x0_opt, x0_grid, sigma2 = find_x0(phase, flux, pts_per_knot=10, n_scan=200)
    >>> print(f"Minimum at x0 = {x0_opt:.4f}")
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
    return_samples: bool = False,
) -> tuple[float, float, float] | tuple[float, float, float, np.ndarray]:
    """Estimate uncertainty of x0 via residual bootstrap.

    Performs a residual bootstrap to estimate the sampling distribution of
    the minimum location ``x0``. Residuals from the initial spline fit are
    resampled with replacement to generate bootstrap samples. For each
    bootstrap sample, the full ``find_x0`` scan is repeated (at reduced
    resolution for speed). The empirical distribution of bootstrap estimates
    provides standard error and confidence intervals.

    Parameters
    ----------
    x : np.ndarray
        Original abscissae.
    y : np.ndarray
        Original ordinates.
    x0_opt : float
        Optimal ``x0`` from ``find_x0``.
    spl1 : LSQUnivariateSpline
        Initial spline fit to the original data (without reflection).
    pts_per_knot : int
        Points per knot for spline fits.
    degree : int
        Spline degree.
    w : np.ndarray or None
        Weights (same as passed to ``find_x0``).
    n_bootstrap : int, optional
        Number of bootstrap iterations. Default 60. Use larger values (100-200)
        for more precise confidence intervals.
    n_scan_boot : int, optional
        Grid resolution for each bootstrap iteration's scan. Default 80
        (coarser than main scan for speed).
    x0_window : float, optional
        Scan window fraction, same as in ``find_x0``. Default 0.1.
    rng : np.random.Generator or None, optional
        Random number generator. If None, a new PCG64 generator is created.
        Pass a seeded generator for reproducible results.
    return_samples : bool, optional
        If True, additionally return the raw bootstrap estimates ``x0_boot``
        (e.g. for plotting the bootstrap distribution). Default False.

    Returns
    -------
    tuple[float, float, float] or tuple[float, float, float, np.ndarray]
        ``(x0_std, x0_lo, x0_hi)`` where:
        - ``x0_std`` : float, bootstrap standard error (sample std, ddof=1)
        - ``x0_lo`` : float, 16th percentile (lower bound of 68% CI)
        - ``x0_hi`` : float, 84th percentile (upper bound of 68% CI)
        If ``return_samples`` is True, a fourth element ``x0_boot`` (the array
        of ``n_bootstrap`` individual estimates) is appended.

    Notes
    -----
    The bootstrap uses **residual resampling**: residuals from the initial
    spline fit are resampled with replacement and added back to the fitted
    values. This preserves the noise structure while generating new samples
    under the null hypothesis that the spline model is correct.

    The per-iteration scan uses a coarser grid (``n_scan_boot``) and the same
    window fraction ``x0_window``. The initial guess for each bootstrap
    iteration is the minimum of a coarse spline fit to the bootstrap sample.

    For reproducibility, pass a seeded generator:
    ``rng = np.random.default_rng(42)``.

    References
    ----------
    Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
    Chapman & Hall.
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

    if return_samples:
        return x0_std, x0_lo, x0_hi, x0_boot
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
    return_samples: bool = False,
) -> MinimumResult | tuple[MinimumResult, np.ndarray]:
    """Full pipeline: find the light-curve minimum and its uncertainty.

    High-level function that runs the complete reflection-method pipeline:
    1. Scans trial ``x0`` values to find the minimum of σ₂(x₀) (``find_x0``).
    2. Estimates uncertainty via residual bootstrap (``bootstrap_x0``).
    3. Returns a ``MinimumResult`` containing the minimum location,
       uncertainties, and diagnostics.

    Parameters
    ----------
    x : np.ndarray
        Abscissae (time in any linear units: JD, HJD, MJD, phase, minutes,
        etc.). 1-D array.
    y : np.ndarray
        Ordinates (flux, magnitude, etc.). 1-D array, same length as ``x``.
    pts_per_knot : int, optional
        Points per knot for spline fits. Default 10. Typical range: 5-20.
    degree : int, optional
        Spline degree (1-5). Default 3 (cubic).
    w : np.ndarray or None, optional
        Weights (typically ``1 / sigma``). If None, equal weights. Default None.
    n_scan : int, optional
        Scan resolution for main search. Default 200.
    x0_window : float, optional
        Scan window as fraction of ``x`` range. Default 0.1 (10%).
    n_bootstrap : int, optional
        Number of bootstrap iterations. Default 60. Increase for more
        precise confidence intervals.
    n_scan_boot : int, optional
        Grid resolution per bootstrap iteration. Default 80.
    rng : np.random.Generator or None, optional
        Random number generator for bootstrap. Pass a seeded generator for
        reproducibility. Default is a new PCG64 generator.
    return_samples : bool, optional
        If True, additionally return the raw bootstrap estimates ``x0_boot``
        (e.g. for plotting the bootstrap distribution). Default False.

    Returns
    -------
    MinimumResult or tuple[MinimumResult, np.ndarray]
        ``MinimumResult`` with fields:
        - ``x0`` : float, optimal reflection point (same units as input ``x``)
        - ``x0_std`` : float, bootstrap standard error
        - ``x0_lo`` : float, 16th percentile (lower 68% CI)
        - ``x0_hi`` : float, 84th percentile (upper 68% CI)
        - ``sigma_min`` : float, minimum σ₂ value
        - ``n_points`` : int, number of data points
        - ``n_bootstrap`` : int, bootstrap iterations used
        If ``return_samples`` is True, a tuple ``(result, x0_boot)`` is
        returned instead, where ``x0_boot`` is the array of ``n_bootstrap``
        individual bootstrap estimates.

    Raises
    ------
    ValueError
        If input arrays are invalid, spline fitting fails, or scan fails.

    Examples
    --------
    Basic usage with synthetic data:

    >>> import numpy as np
    >>> from reflection_method import find_minimum
    >>> rng = np.random.default_rng(42)
    >>> phase = np.sort(rng.uniform(0, 1, 300))
    >>> flux = 1 - 0.5 * np.exp(-((phase - 0.5)**2) / 0.001) + rng.normal(0, 0.01, 300)
    >>> result = find_minimum(phase, flux, n_bootstrap=100, rng=np.random.default_rng(42))
    >>> print(f"Minimum at x0 = {result.x0:.4f} ± {result.x0_std:.4f}")
    Minimum at x0 = 0.5000 ± 0.0002

    With weights and reproducibility:

    >>> w = 1 / np.full_like(flux, 0.01)  # constant uncertainty
    >>> result = find_minimum(phase, flux, w=w, n_bootstrap=200,
    ...                       rng=np.random.default_rng(42))
    >>> print(f"68% CI: [{result.x0_lo:.4f}, {result.x0_hi:.4f}]")
    68% CI: [0.4997, 0.5003]
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
    x0_std, x0_lo, x0_hi, x0_boot = bootstrap_x0(
        x, y, x0_opt, spl1, pts_per_knot, degree, w,
        n_bootstrap, n_scan_boot, x0_window, rng, return_samples=True
    )

    sigma_min = float(np.min(sigma2))

    result = MinimumResult(
        x0=x0_opt,
        x0_std=x0_std,
        x0_lo=x0_lo,
        x0_hi=x0_hi,
        sigma_min=sigma_min,
        n_points=len(x),
        n_bootstrap=n_bootstrap,
    )
    if return_samples:
        return result, x0_boot
    return result