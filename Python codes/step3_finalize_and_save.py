"""
STEP 3: RETRAIN ON ALL DATA AND SAVE THE MODEL
=================================================
Now that step2_train_test.py proved the model works well (R^2 = 0.92
on unseen test data), we don't need to hold back 20% anymore -- that
held-back set already did its job (proving trustworthiness). For the
model we'll actually USE (in the GA), we retrain on every point we
have, since more data only helps the final model.

This script also SAVES the trained model to a file, so the GA script
can load it later without retraining from scratch every time.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

FILENAME = "CSV FIles/v4/training input file.csv"   # <-- same file as before, change if needed

# -----------------------------------------------------------------------
# Load ALL the data -- no train/test split this time.
# -----------------------------------------------------------------------
df = pd.read_csv(FILENAME, comment="#")

X = df[["length", "width", "thickness", "height"]].values
y = df["FOS"].values

print(f"Training final model on all {len(X)} data points.")


# -----------------------------------------------------------------------
# Train (same settings as before, for consistency).
# -----------------------------------------------------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

print("Model trained on full dataset.")

# -----------------------------------------------------------------------
# Check which parameter actually matters most to the model's predictions.
# -----------------------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=["length", "width", "thickness", "height"])
print("\nFeature importances (higher = more influence on predicted FOS):")
print(importances.sort_values(ascending=False))


# -----------------------------------------------------------------------
# Save the trained model to a file.
# -----------------------------------------------------------------------
# joblib.dump() writes the entire trained model (all 200 tree structures,
# everything it learned) to a single file on disk. This means any other
# script -- like your future GA script -- can load this exact trained
# model with one line, instead of retraining every time it runs.

MODEL_PATH = "fos_surrogate.pkl"
joblib.dump(model, MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")
print("\nTo use this model in another script later, load it with:")
print('    import joblib')
print(f'    model = joblib.load("{MODEL_PATH}")')
print("    model.predict(some_array_of_designs)")