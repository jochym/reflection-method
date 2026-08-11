from typing import NamedTuple, Optional
import numpy as np


class MinimumResult(NamedTuple):
    """Result of the reflection-method minimum finding.

    All x-values are in the same linear time units as the input x array
    (e.g., JD, HJD, MJD, phase, minutes from start). No time conversion
    is performed by the library.
    """
    x0: float                   # Reflection point (same units as input x)
    x0_std: float               # Bootstrap standard deviation (same units as input x)
    x0_lo: float                # 16th percentile (same units as input x)
    x0_hi: float                # 84th percentile (same units as input x)
    sigma_min: float            # Minimum residual standard deviation
    n_points: int               # Number of data points
    n_bootstrap: int            # Bootstrap iterations used