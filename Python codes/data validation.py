"""
STEP-BY-STEP SURROGATE MODEL TRAINING SCRIPT
==============================================
Goal: teach a model to predict FOS (Factor of Safety) from the 4 bracket
dimensions, using the 100 ANSYS design points as training examples.

Run this section by section (or all at once) and read the print()
outputs at each step before moving to the next -- that's the whole point
of doing this in stages rather than blindly running everything.
"""

import pandas as pd

# -----------------------------------------------------------------------
# STEP 1: Load the CSV and LOOK AT IT before doing anything else.
# -----------------------------------------------------------------------
# comment="#" tells pandas to skip any lines starting with # (the header
# comment block ANSYS puts at the top of exported tables, like we saw
# in your earlier screenshot).

FILENAME = "CSV FIles/v4/training input file.csv"   # <-- CHANGE THIS to your actual exported filename

df = pd.read_csv(FILENAME, comment="#")

print("Shape of data (rows, columns):", df.shape)
print("\nColumn names:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head())

# STOP AND CHECK: does df.shape say (100, 5) or (100, 6)?
# (5 if Name wasn't exported, 6 if it was -- either is fine)
# Do the column names look like Name, P1, P2, P3, P4, P5 (or similar)?
# If the column names look wrong or there's garbage in the first rows,
# the "comment" skipping didn't work right -- open the CSV in a text
# editor and check where the real header row actually starts.


# -----------------------------------------------------------------------
# STEP 2: Check for missing or broken values.
# -----------------------------------------------------------------------
# isna() marks every blank/missing cell as True. .sum() counts them
# per column. If any design points failed to solve in ANSYS, their P5
# (FOS) value will show up as missing here.

print("\nMissing values per column:")
print(df.isna().sum())

# STOP AND CHECK: if any column shows a number greater than 0 here,
# those rows are incomplete. Decide whether to drop them:
# df = df.dropna()
# Uncomment the line above ONLY if Step 2 showed missing values.


# -----------------------------------------------------------------------
# STEP 3: Look at the actual range of values -- sanity check the physics.
# -----------------------------------------------------------------------
# .describe() gives min, max, mean, etc. for every numeric column.
# This is where you'd catch a units mistake (e.g. FOS of 50000, which
# would mean something is scaled wrong) or a parameter that never
# actually varied (min == max, meaning the LHS/import didn't work).

print("\nSummary statistics:")
print(df.describe())

# STOP AND CHECK against what you expect:
# - P1 (length) should range roughly 0.02 to 0.06 (your original bounds)
# - P2 (width)     0.02 to 0.05
# - P3 (thickness) 0.002 to 0.005
# - P4 (height)    0.02 to 0.05
# - P5 (FOS) should be a "reasonable" number for your material/loads --
#   you know your target range better than I do, but if you see
#   negative values or something like 1e+12, stop here and investigate
#   before training anything.