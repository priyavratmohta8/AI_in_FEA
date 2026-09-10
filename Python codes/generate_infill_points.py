"""
LOCAL INFILL SAMPLING: generate 20 new design points concentrated around
the design that failed verification (predicted FOS=1.5, real FOS=0.83).

Instead of sampling the whole global range again, this samples a SMALL
box around that specific failing point, so the new FEA runs directly
teach the model the shape of the region it got wrong -- rather than
spreading 20 points thinly across the whole space again.
"""

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube

np.random.seed(42)

N = 20  # number of new infill points

# ---- GLOBAL bounds (same as your original full sweep) ----
GLOBAL_BOUNDS = {
    "length":    (0.02,  0.06),
    "width":     (0.02,  0.05),
    "thickness": (0.002, 0.005),
    "height":    (0.02,  0.05),
}

# ---- CENTER: the exact failing design from your GA run ----
# <-- FILL THESE IN with the real values your GA printed out -->
CENTER = {
    "length":    0.02,
    "width":     0.02,
    "thickness": 0.003805,
    "height":    0.02,
}

# ---- LOCAL WINDOW: how far around the center to sample ----
# Expressed as a fraction of each parameter's GLOBAL range.
# 0.15 means +/- 15% of the full range, centered on CENTER, then clipped
# so it never goes outside the global bounds.
WINDOW_FRACTION = 0.15

param_names = list(GLOBAL_BOUNDS.keys())

# build local bounds per parameter: center +/- window, clipped to global range
local_lows = []
local_highs = []
for name in param_names:
    g_low, g_high = GLOBAL_BOUNDS[name]
    span = g_high - g_low
    window = span * WINDOW_FRACTION

    c = CENTER[name]
    low = max(g_low, c - window)
    high = min(g_high, c + window)

    local_lows.append(low)
    local_highs.append(high)

local_lows = np.array(local_lows)
local_highs = np.array(local_highs)

print("Local sampling box for this infill batch:")
for name, low, high in zip(param_names, local_lows, local_highs):
    print(f"  {name:10s}: {low:.6f} to {high:.6f}")

# ---- generate N LHS points inside this local box ----
sampler = LatinHypercube(d=len(param_names), seed=42)
unit_samples = sampler.random(n=N)
scaled_samples = local_lows + unit_samples * (local_highs - local_lows)

df = pd.DataFrame(scaled_samples, columns=param_names)
df.insert(0, "Name", [f"INFILL {i}" for i in range(len(df))])

print(f"\nGenerated {len(df)} infill points.")
print(df.head())

# ---- export in the same ANSYS-ready format as before ----
# Matches your project's actual parameter mapping: P1-length, P2-width,
# P3-thickness, P4-height (confirmed from your earlier ANSYS export).
P_ORDER = ["length", "width", "thickness", "height"]
P_NAMES = ["P1", "P2", "P3", "P4"]

df_export = df[["Name"] + P_ORDER].copy()
df_export.columns = ["Name"] + P_NAMES

output_path = "infill_design_points.csv"
param_map_line = "#           " + " ".join(f"{p} - {l}" for p, l in zip(P_NAMES, P_ORDER))

with open(output_path, "w", newline="") as f:
    f.write("#\n")
    f.write("# Local infill LHS points around the failed GA verification point\n")
    f.write("# The parameters defined in the project are:\n")
    f.write(param_map_line + "\n")
    f.write("#\n")
    f.write("# The following header line defines the name of the columns by reference to the parameters.\n")
    df_export.to_csv(f, index=False, lineterminator="\n")

print(f"\nSaved to {output_path}")