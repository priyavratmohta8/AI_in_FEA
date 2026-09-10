"""
5-FOLD CROSS-VALIDATION
=========================
Instead of one random 80/20 split (which gives one noisy R^2 number that
depends on which specific points happened to land in the test set), this
splits the data into 5 roughly equal chunks ("folds"). It then does 5
separate train/test rounds -- each time holding out a DIFFERENT one of
the 5 chunks as the test set and training on the other 4 -- and averages
the 5 resulting R^2 scores.

This gives a much more stable read on how well the model actually
generalizes, since every single data point gets used as a test point
exactly once across the 5 rounds, rather than the result depending on
one lucky/unlucky split.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

FILENAME = "CSV FIles/v5/training input file.csv"   # <-- your CSV with all 121 points (101 original + 20 infill)

df = pd.read_csv(FILENAME, comment="#")

X = df[["length", "width", "thickness", "height"]].values
y = df["FOS"].values

# KFold with shuffle=True so the 5 chunks are randomly assigned, not just
# split in the order rows appear in the file (important since your 20
# infill points are all sitting at the bottom of the dataframe -- without
# shuffling, one fold could end up being almost entirely infill points).
kf = KFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestRegressor(n_estimators=500, random_state=42)

# cross_val_score does the whole loop for you: trains 5 times, each on a
# different 4/5 of the data, tests on the held-out 1/5, returns the 5
# R^2 scores as an array.
scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

print("R² for each of the 5 folds:")
for i, s in enumerate(scores):
    print(f"  Fold {i+1}: {s:.4f}")

print(f"\nMean R² across all 5 folds: {scores.mean():.4f}")
print(f"Standard deviation:         {scores.std():.4f}")
print("\nThe mean is a more stable estimate of true generalization than a")
print("single train/test split. The standard deviation tells you how much")
print("that estimate might vary depending on which points end up where --")
print("a small std means the model is consistently good; a large std means")
print("the R^2 you'd get from any ONE split is fairly unreliable on its own.")
