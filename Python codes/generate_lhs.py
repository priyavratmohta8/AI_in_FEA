"""
True Latin Hypercube Sampling: N=100 sample points, jointly space-filling
across all 4 parameters. Each parameter's range is divided into 100 equal
bins (matching N), one random value drawn per bin, then the 100 values
per parameter are randomly shuffled and paired across parameters to form
100 complete design points.

Uses scipy's LHS implementation (same underlying method, more robust /
better joint space-filling than a naive per-parameter shuffle).
"""

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube

np.random.seed(42)  # remove or change this if you want a different draw each run

N = 870  # total number of design points / samples

# ---- define your parameters here, IN THE PROJECT'S P1..P4 ORDER ----
# order confirmed from the ANSYS export: P1-length, P2-width, P3-thickness, P4-height
parameters = [
    ("P1", "length",    0.02,  0.06),
    ("P2", "width",     0.02,  0.05),
    ("P3", "thickness", 0.0025, 0.005),
    ("P4", "height",    0.02,  0.05),
]

pnames  = [p[0] for p in parameters]   # P1, P2, P3, P4 -> used as actual column headers
labels  = [p[1] for p in parameters]   # friendly names -> used only for printing/checking
lows    = np.array([p[2] for p in parameters])
highs   = np.array([p[3] for p in parameters])

# generate N samples in [0,1]^4, space-filling across all dimensions jointly
sampler = LatinHypercube(d=len(parameters), seed=42)
unit_samples = sampler.random(n=N)

# scale each column from [0,1] to its real parameter range
scaled_samples = lows + unit_samples * (highs - lows)

df = pd.DataFrame(scaled_samples, columns=pnames)
df.insert(0, "Name", [f"DP {i}" for i in range(len(df))])  # note the space, matches ANSYS's "DP 0" style

print(f"Total design points: {len(df)}")
print(df.head())
print("\nPer-parameter ranges achieved:")
for pname, label in zip(pnames, labels):
    print(f"  {pname} ({label}): min={df[pname].min():.6f}, max={df[pname].max():.6f}")

# ---- export CSV, matching ANSYS's native comment + header format ----
output_path = "lhs_design_points_1k.csv"
param_map_line = "#           " + " ".join(f"{p} - {l}" for p, l in zip(pnames, labels))

with open(output_path, "w", newline="") as f:
    f.write("#\n")
    f.write("# LHS design points generated in Python\n")
    f.write("# The parameters defined in the project are:\n")
    f.write(param_map_line + "\n")
    f.write("#\n")
    f.write("# The following header line defines the name of the columns by reference to the parameters.\n")
    df.to_csv(f, index=False, lineterminator="\n")

print(f"\nSaved to {output_path}")