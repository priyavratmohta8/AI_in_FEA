"""
STEP 2: TRAIN AND TEST THE SURROGATE MODEL
============================================
Picks up after step1_load_and_check.py confirmed the data is clean.
This script: splits data into train/test, trains a model, checks how
good it actually is.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

FILENAME = "CSV FIles/v4/training input file.csv"   # <-- same file as before, change if needed

# -----------------------------------------------------------------------
# STEP 4: Load data and split it into "train" and "test" sets.
# -----------------------------------------------------------------------
# WHY split at all? If you train the model on ALL 101 points and then
# check its accuracy on those SAME 101 points, you're not testing
# whether it learned the underlying pattern -- you're testing whether
# it memorized the answers. A model can memorize perfectly and still
# be useless on any new design it hasn't seen.
#
# So: we hide 20% of the data (test set) from the model during training.
# After training, we ask the model to predict FOS for those hidden
# points and compare its guesses to the real ANSYS answers. That's an
# honest test, because the model never saw those points while learning.

df = pd.read_csv(FILENAME, comment="#")

# X = the inputs (the 4 dimensions). y = the output we want to predict (FOS).
X = df[["length", "width", "thickness", "height"]].values
y = df["FOS"].values

# test_size=0.2 means 20% of rows go into the test set, 80% into training.
# random_state=42 just makes the split reproducible -- rerun this script
# and you'll get the SAME split every time, rather than a different
# random split each run. 42 is an arbitrary number, not special.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} points, testing on {len(X_test)} points.")


# -----------------------------------------------------------------------
# STEP 5: Train the model.
# -----------------------------------------------------------------------
# RandomForestRegressor builds many decision trees (n_estimators=200 of
# them), each trained on a slightly different random subset of the
# training data, and averages their predictions. This tends to work
# well "out of the box" with small-to-medium tabular datasets like
# yours, without much tuning needed.
#
# .fit(X_train, y_train) is the actual "learning" step: the model looks
# at the 80 training examples (4 numbers in, 1 number out) and adjusts
# itself to capture the pattern between them.

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Model trained.")


# -----------------------------------------------------------------------
# STEP 6: Test the model -- how good are its guesses on data it never saw?
# -----------------------------------------------------------------------
# .predict(X_test) asks the trained model to guess FOS for the 21 hidden
# test points, using ONLY their 4 input dimensions (it never sees the
# real y_test answers during this step).

pred = model.predict(X_test)

# R^2 (R-squared): a score from -infinity to 1.0 that tells you how much
# of the variation in the real FOS values the model's predictions
# explain. 1.0 = perfect predictions. 0.0 = the model is no better than
# just guessing the average FOS every time. Negative = worse than that.
# Rule of thumb for a decent surrogate: aim for > 0.8, ideally > 0.9.
r2 = r2_score(y_test, pred)

# MAE (Mean Absolute Error): the average size of the model's mistakes,
# in the same units as FOS itself. E.g. MAE of 0.1 means the model's
# guesses are, on average, off by about 0.1 FOS units from the truth.
# This is more intuitive than R^2 -- ask yourself "is being off by
# this much acceptable for what I'm using the surrogate for?"
mae = mean_absolute_error(y_test, pred)

print(f"\nR^2 score: {r2:.4f}")
print(f"MAE: {mae:.4f}")

# Print the actual vs predicted values side by side so you can eyeball
# individual predictions, not just the summary numbers.
comparison = pd.DataFrame({"Actual FOS": y_test, "Predicted FOS": pred})
comparison["Error"] = comparison["Predicted FOS"] - comparison["Actual FOS"]
print("\nActual vs predicted on the test set:")
print(comparison.to_string(index=False))


# -----------------------------------------------------------------------
# STEP 7: Plot predicted vs actual -- the fastest visual sanity check.
# -----------------------------------------------------------------------
# If every point sits close to the diagonal red dashed line, the model's
# predictions closely match reality. Points far from the line are cases
# where the model got it wrong -- look at whether those are scattered
# randomly (probably fine, just noise) or systematically off in one
# direction (a sign of a real problem, e.g. model underestimates FOS
# at low values, or your data has an outlier/error).

plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, color="steelblue", label="Test predictions")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    label="Perfect prediction line",
)
plt.xlabel("Actual FOS (from ANSYS)")
plt.ylabel("Predicted FOS (from surrogate)")
plt.title(f"Surrogate model check (R² = {r2:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig("prediction_check.png")  # saves a PNG you can open and look at
plt.show()

print("\nPlot saved as prediction_check.png -- open it and look at how")
print("close the blue dots sit to the red dashed line.")
