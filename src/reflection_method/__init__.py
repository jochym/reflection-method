"""reflection-method: Eclipsing binary minimum finder using the reflection method.

This library provides a functional API for finding the minimum of an
eclipsing star's light curve using the reflection method (symmetry about
the eclipse minimum).

All functions operate on linear time units (JD, HJD, MJD, phase, minutes, etc.).
The library performs no I/O or time conversion.
"""

from .core import (
    fit_spline,
    spline_variance,
    combine,
    find_x0,
    bootstrap_x0,
    find_minimum,
)
from .result import MinimumResult

try:
    from .plot import (
        plot_original_spline,
        plot_scan,
        plot_composite,
        plot_all,
    )
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False

__version__ = "0.1.0"

__all__ = [
    "MinimumResult",
    "fit_spline",
    "spline_variance",
    "combine",
    "find_x0",
    "bootstrap_x0",
    "find_minimum",
]

if _PLOT_AVAILABLE:
    __all__.extend([
        "plot_original_spline",
        "plot_scan",
        "plot_composite",
        "plot_all",
    ])