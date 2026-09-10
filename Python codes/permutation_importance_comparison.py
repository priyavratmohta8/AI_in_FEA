"""
FAIR COMPARISON: permutation importance on BOTH the random forest and the
GP, using the identical method for each.

Why this is needed: RF's feature_importances_ and the GP's kernel length
scales are computed completely differently -- comparing them directly
(as we did) isn't quite apples-to-apples. Permutation importance applies
the SAME test to any fitted model: shuffle one feature's values (breaking
its relationship with the target), see how much predictive performance
drops. A feature that matters a lot causes a big drop when shuffled; an
irrelevant feature causes almost no drop. This works identically for a
random forest, a GP, or anything else with a .predict() method.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

FILENAME = "CSV FIles/v2/training input file.csv"          # <-- your full dataset
RF_MODEL_PATH = "Python Codes/fos_surrogate.pkl"           # <-- your saved random forest
GP_BUNDLE_PATH = "Python Codes/fos_surrogate_gp.pkl"       # <-- saved GP + scaler together

df = pd.read_csv(FILENAME, comment="#")
param_names = ["length", "width", "thickness", "height"]

X = df[param_names].values
y = df["FOS"].values

# use the SAME held-out test set for both, so the comparison is apples-to-apples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------------------------------
# Random forest: works directly on raw (unscaled) features.
# -----------------------------------------------------------------------
rf = joblib.load(RF_MODEL_PATH)

rf_result = permutation_importance(
    rf, X_test, y_test, n_repeats=30, random_state=42, scoring="r2"
)

print("Random Forest -- permutation importance (drop in R^2 when shuffled):")
for name, mean, std in zip(param_names, rf_result.importances_mean, rf_result.importances_std):
    print(f"  {name:10s}: {mean:.4f} +/- {std:.4f}")

# -----------------------------------------------------------------------
# GP: needs the SAME scaler used during training, applied to X_test first.
# permutation_importance shuffles columns of whatever X you give it, so
# scale X_test BEFORE calling it, and pass the already-scaled array.
# -----------------------------------------------------------------------
bundle = joblib.load(GP_BUNDLE_PATH)
gp = bundle["model"]
scaler = bundle["scaler"]

X_test_scaled = scaler.transform(X_test)

gp_result = permutation_importance(
    gp, X_test_scaled, y_test, n_repeats=30, random_state=42, scoring="r2"
)

print("\nGaussian Process -- permutation importance (drop in R^2 when shuffled):")
for name, mean, std in zip(param_names, gp_result.importances_mean, gp_result.importances_std):
    print(f"  {name:10s}: {mean:.4f} +/- {std:.4f}")

# -----------------------------------------------------------------------
# Side-by-side table -- now genuinely comparable, same method, same test set.
# -----------------------------------------------------------------------
comparison = pd.DataFrame({
    "parameter": param_names,
    "RF importance": rf_result.importances_mean,
    "GP importance": gp_result.importances_mean,
})
comparison = comparison.sort_values("RF importance", ascending=False)
print("\nSide-by-side comparison:")
print(comparison.to_string(index=False))

print("\nIf both models now roughly agree on ranking, the earlier mismatch")
print("was mostly an artifact of comparing two differently-defined metrics.")
print("If they STILL disagree under this fair test, that's a real finding")
print("worth investigating further (e.g. checking for a nonlinear effect")
print("the RF's impurity-based importance underweights).")
