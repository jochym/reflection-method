#!/usr/bin/env python3
"""Quickstart example for reflection-method.

Run with: python examples/quickstart.py
"""

import numpy as np
from reflection_method import find_minimum

# Generate synthetic eclipsing binary light curve
# (or load your data: x = time in any linear units, y = flux/magnitude)
rng = np.random.default_rng(42)
n = 400
phase = np.sort(rng.uniform(0.0, 1.0, n))

# Primary minimum at phase 0.5
primary = 0.7 * np.exp(-((phase - 0.5) ** 2) / (2 * 0.02**2))
# Secondary minima
secondary = 0.15 * (
    np.exp(-((phase - 0.0) ** 2) / (2 * 0.03**2))
    + np.exp(-((phase - 1.0) ** 2) / (2 * 0.03**2))
)
flux = 1.0 - primary - secondary + rng.normal(0.0, 0.01, n)

# Find minimum
result = find_minimum(
    phase, flux,
    pts_per_knot=10,
    degree=3,
    n_scan=200,
    n_bootstrap=60,
    n_scan_boot=80,
    rng=np.random.default_rng(123)
)

print(f"Minimum at phase x0 = {result.x0:.4f} ± {result.x0_std:.4f}")
print(f"68% CI: [{result.x0_lo:.4f}, {result.x0_hi:.4f}]")
print(f"Sigma min: {result.sigma_min:.4f}")
print(f"N points: {result.n_points}, Bootstrap iterations: {result.n_bootstrap}")

# With real AAVSO data, use the CLI (magnitudes handled directly):
# reflection-method find data.csv -x DATE-OBS -y MAG -w MAG_ERR -t iso -k 10 -d 3 -b 60 -s 42 -p output.png