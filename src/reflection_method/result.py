"""Result container for the reflection method."""

from typing import NamedTuple
import numpy as np


class MinimumResult(NamedTuple):
    """Result of the reflection-method minimum finding.

    All x-values are in the **same linear time units as the input `x` array**
    (e.g., JD, HJD, MJD, phase, minutes from start). The library performs
    **no time conversion** — this is the responsibility of the caller
    (or the CLI layer).

    Attributes
    ----------
    x0 : float
        Optimal reflection point (abscissa of the symmetry axis / minimum).
        Units: same as input ``x``.
    x0_std : float
        Bootstrap standard error (sample standard deviation of bootstrap
        distribution, ddof=1). Units: same as input ``x``.
    x0_lo : float
        16th percentile of the bootstrap distribution (lower bound of the
        68% confidence interval). Units: same as input ``x``.
    x0_hi : float
        84th percentile of the bootstrap distribution (upper bound of the
        68% confidence interval). Units: same as input ``x``.
    sigma_min : float
        Minimum value of the residual standard deviation ``σ₂`` achieved
        at ``x0``. Dimensionless (same units as input ``y``).
    n_points : int
        Number of data points used in the fit.
    n_bootstrap : int
        Number of bootstrap iterations performed.

    Notes
    -----
    - The 68% confidence interval ``[x0_lo, x0_hi]`` corresponds to the
      central 68% of the bootstrap distribution, approximately equivalent
      to ±1σ for a Gaussian distribution.
    - For asymmetric bootstrap distributions, the interval is not centered
      on ``x0``; report both bounds.
    - ``x0_std`` is the sample standard deviation (ddof=1) of the bootstrap
      estimates. It may differ from ``(x0_hi - x0_lo) / 2`` for non-Gaussian
      bootstrap distributions.

    Examples
    --------
    >>> from reflection_method import MinimumResult
    >>> result = MinimumResult(
    ...     x0=2459000.5, x0_std=0.001, x0_lo=2459000.498, x0_hi=2459000.502,
    ...     sigma_min=0.005, n_points=150, n_bootstrap=100
    ... )
    >>> print(f"Minimum at JD {result.x0:.3f} ± {result.x0_std:.3f}")
    Minimum at JD 2459000.500 ± 0.001
    >>> print(f"68% CI: [{result.x0_lo:.3f}, {result.x0_hi:.3f}]")
    68% CI: [2459000.498, 2459000.502]
    """
    x0: float
    x0_std: float
    x0_lo: float
    x0_hi: float
    sigma_min: float
    n_points: int
    n_bootstrap: int