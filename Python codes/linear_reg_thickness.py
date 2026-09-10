"""
DIAGNOSTIC: compare the random forest's behavior along thickness against
a dead-simple linear fit, as a sanity check that the forest hasn't
learned anything strange (overfit noise) in this dominant parameter.

Run standalone. Needs your saved model (fos_surrogate.pkl) and your
original CSV.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

FILENAME = "CSV FIles/training input file.csv"          # <-- same file as always
MODEL_PATH = "fos_surrogate.pkl"    # <-- your saved random forest

df = pd.read_csv(FILENAME, comment="#")
model = joblib.load(MODEL_PATH)

# -----------------------------------------------------------------------
# Fit a simple straight line: FOS = a * thickness + b
# -----------------------------------------------------------------------
linear_model = LinearRegression()
linear_model.fit(df[["thickness"]], df["FOS"])

# -----------------------------------------------------------------------
# Generate predictions from both models across the thickness range,
# holding length/width/height fixed at their median values for the
# random forest (it needs all 4 inputs; the linear model only needs
# thickness).
# -----------------------------------------------------------------------
thickness_range = np.linspace(df["thickness"].min(), df["thickness"].max(), 50).reshape(-1, 1)

linear_pred = linear_model.predict(thickness_range)

fixed_length = df["length"].median()
fixed_width  = df["width"].median()
fixed_height = df["height"].median()

# NOTE: column order here MUST match the order your model was trained on
# (length, width, thickness, height) -- adjust if yours differs.
rf_input = np.column_stack([
    np.full(50, fixed_length),
    np.full(50, fixed_width),
    thickness_range.flatten(),
    np.full(50, fixed_height),
])
rf_pred = model.predict(rf_input)

# -----------------------------------------------------------------------
# Plot both models together against the real data.
# -----------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(df["thickness"], df["FOS"], alpha=0.3, label="Actual data")
plt.plot(thickness_range, linear_pred, label="Simple linear fit", linewidth=2)
plt.plot(thickness_range, rf_pred, label="Random forest (others at median)", linewidth=2)
plt.xlabel("thickness")
plt.ylabel("FOS")
plt.title("Random forest vs. simple linear fit, along thickness")
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

print("Saved model_comparison.png")
print("If the two lines track closely, that's reassurance the forest")
print("learned a sensible trend. If they diverge somewhere, that's a")
print("region worth investigating further before trusting the GA there.")