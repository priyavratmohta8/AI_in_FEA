"""
DIAGNOSTIC: visually check if length, width, height genuinely have
little effect on FOS, or if the low feature importance is hiding a
real trend that sparse sampling just didn't capture well.

Run this on its own, look at the saved PNG, and decide whether to
trust fixing width/height based on what you see.
"""

import pandas as pd
import matplotlib.pyplot as plt

FILENAME = "CSV FIles/v3/training input file.csv"   # <-- same file as always, change if needed

df = pd.read_csv(FILENAME, comment="#")

fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, param in zip(axes, ["length", "width", "height", "thickness"]):
    ax.scatter(df[param], df["FOS"])
    ax.set_xlabel(param)
    ax.set_ylabel("FOS")
    ax.set_title(f"FOS vs {param}")
plt.tight_layout()
plt.savefig("importance_check.png")
plt.show()

print("Saved importance_check.png -- look for a flat, trendless cloud")
print("of points on width and height (confirms low importance), versus")
print("any visible upward/downward trend (would suggest the importance")
print("score is misleading due to sparse sampling in that direction).")
