"""
GAUSSIAN PROCESS REGRESSION SURROGATE
========================================
Trains a GP on the same data as the random forest, evaluates it the same
way (train/test split + 5-fold CV, matching step2/cross_validation.py),
and produces the one figure that actually matters for this project:
does the GP's predicted uncertainty correlate with its real error --
specifically at the corner that broke the random forest.

Run standalone. Needs your combined CSV (all points, including infill).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import r2_score, mean_absolute_error

FILENAME = "CSV FIles/v5/training input file.csv"   # <-- your full dataset (original + infill points)

df = pd.read_csv(FILENAME, comment="#")

X = df[["length", "width", "thickness", "height"]].values
y = df["FOS"].values

# -----------------------------------------------------------------------
# STEP 1: Scale the inputs. THIS IS NOT OPTIONAL FOR A GP.
# -----------------------------------------------------------------------
# A GP measures distance between points to decide how correlated two
# predictions should be. Your 4 parameters live on wildly different
# scales (thickness ~0.003, length ~0.04) -- without scaling, thickness
# barely registers as "far" from any other thickness value, and the GP
# effectively ignores it. A random forest never needed this because it
# splits on thresholds per-feature, never on distance.
#
# StandardScaler transforms each column to mean=0, std=1, independently.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------------------
# STEP 2: Define the kernel -- this IS the GP's model of the world.
# -----------------------------------------------------------------------
# ConstantKernel: an overall scale factor for the output variance.
# RBF: "points close together in input space should have similar output."
#      length_scale is learned PER FEATURE (one value per parameter) --
#      a large learned length_scale for a feature means the GP thinks
#      that feature barely affects the output. This is the GP's own
#      version of feature_importances_.
# WhiteKernel: models measurement/simulation noise, so the GP doesn't
#      treat every FEA result as if it were perfectly noise-free.
n_features = X_scaled.shape[1]
kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))
    * RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e3))
    + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
)

# n_restarts_optimizer: the kernel's hyperparameters (length scales, noise
# level) are FIT to the data by maximizing likelihood, similar in spirit
# to how a random forest's splits are fit -- but this optimization can
# get stuck in a poor local solution, so restart it from several random
# starting points and keep the best.
gp_template = GaussianProcessRegressor(
    kernel=kernel, normalize_y=True, n_restarts_optimizer=15, random_state=42
)

# -----------------------------------------------------------------------
# STEP 3: Train/test split -- same discipline as the random forest script.
# -----------------------------------------------------------------------
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, df.index, test_size=0.2, random_state=42
)

gp = GaussianProcessRegressor(
    kernel=kernel, normalize_y=True, n_restarts_optimizer=15, random_state=42
)
gp.fit(X_train, y_train)

# return_std=True is the whole point of using a GP: get a prediction AND
# a confidence measure at every point, not just a prediction.
mean_pred, std_pred = gp.predict(X_test, return_std=True)

r2 = r2_score(y_test, mean_pred)
mae = mean_absolute_error(y_test, mean_pred)
print(f"GP  single-split  R²: {r2:.4f}   MAE: {mae:.4f}")

# Show what the GP actually learned about each parameter's relevance.
# A SHORT length_scale means the GP is sensitive to that feature (small
# changes in it cause big changes in prediction). A LONG length_scale
# means the GP barely reacts to it -- the GP's analogue of low feature
# importance.
learned_kernel = gp.kernel_
print("\nLearned kernel:", learned_kernel)

# -----------------------------------------------------------------------
# STEP 4: 5-fold cross-validation -- the stable estimate, same as before.
# -----------------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(gp_template, X_scaled, y, cv=kf, scoring="r2")
print(f"\nGP  5-fold CV  R²: {cv_scores.mean():.4f}  (+/- {cv_scores.std():.4f})")

# -----------------------------------------------------------------------
# STEP 5: predicted-vs-actual, same style plot as the random forest check.
# -----------------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.errorbar(
    y_test, mean_pred, yerr=std_pred, fmt="o", alpha=0.6,
    ecolor="lightsteelblue", capsize=3, label="GP prediction ± std",
)
lims = [min(y_test.min(), mean_pred.min()), max(y_test.max(), mean_pred.max())]
plt.plot(lims, lims, "r--", label="Perfect prediction line")
plt.xlabel("Actual FOS (from ANSYS)")
plt.ylabel("Predicted FOS (GP)")
plt.title(f"GP surrogate check (R² = {r2:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig("gp_prediction_check.png")

# -----------------------------------------------------------------------
# STEP 6: THE KEY DIAGNOSTIC -- does uncertainty predict actual error?
# -----------------------------------------------------------------------
# This is the plot that actually matters for this project. If points
# with high predicted std also tend to have high real error, the GP
# genuinely knows where it's unreliable -- exactly the information the
# random forest could never give you, and exactly what would have
# flagged your original failing corner (predicted FOS=1.5, actual=0.83)
# as untrustworthy BEFORE spending an ANSYS run on it.
errors = np.abs(mean_pred - y_test)

plt.figure(figsize=(6, 5))
plt.scatter(std_pred, errors, alpha=0.7)
plt.xlabel("GP predicted uncertainty (std)")
plt.ylabel("Actual prediction error")
plt.title("Does GP uncertainty predict real error?")
plt.tight_layout()
plt.savefig("gp_uncertainty_vs_error.png")

correlation = np.corrcoef(std_pred, errors)[0, 1]
print(f"\nCorrelation between predicted uncertainty and actual error: {correlation:.3f}")
print("A positive correlation means the GP's uncertainty is meaningful --")
print("high-uncertainty predictions really do tend to be less accurate.")

# -----------------------------------------------------------------------
# STEP 7: check the GP specifically at your known failing corner.
# -----------------------------------------------------------------------
# The design that broke the random forest: length=0.02, width=0.02,
# thickness=0.003805, height=0.02, real FOS=0.83.
failing_design_raw = np.array([[0.02, 0.02, 0.003805, 0.02]])
failing_design_scaled = scaler.transform(failing_design_raw)

fail_mean, fail_std = gp.predict(failing_design_scaled, return_std=True)
print(f"\nAt the known failing corner:")
print(f"  GP prediction: {fail_mean[0]:.4f} +/- {fail_std[0]:.4f}")
print(f"  Real ANSYS value: 0.83")
print(f"  (Random forest, before infill, predicted 1.50 here with no")
print(f"   warning -- compare that confident wrongness to the GP's own")
print(f"   uncertainty at this exact point.)")

# -----------------------------------------------------------------------
# STEP 8: retrain on ALL data and save, same pattern as step3.
# -----------------------------------------------------------------------
gp_final = GaussianProcessRegressor(
    kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=42
)
gp_final.fit(X_scaled, y)

joblib.dump({"model": gp_final, "scaler": scaler}, "fos_surrogate_gp.pkl")
print("\nSaved final GP (with its scaler) to fos_surrogate_gp.pkl")
print("NOTE: unlike the random forest, this GP needs the SAME scaler")
print("applied to any new input before calling .predict() -- save and")
print("load them together, as done here, or predictions will be wrong.")
