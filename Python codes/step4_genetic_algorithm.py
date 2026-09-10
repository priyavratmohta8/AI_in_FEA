"""
STEP 4: GENETIC ALGORITHM - FIND THE LIGHTEST DESIGN THAT MEETS THE FOS TARGET
=================================================================================
Uses the surrogate model saved by step3_finalize_and_save.py to search
for the combination of (length, width, thickness, height) that minimizes
mass while keeping predicted FOS above a target value.

Nothing here calls ANSYS -- every "evaluation" is just a call to the
saved model's .predict(), which is why this can run thousands of times
in a couple of seconds.
"""

import numpy as np
import joblib

# -----------------------------------------------------------------------
# SETTINGS -- change these as needed
# -----------------------------------------------------------------------
MODEL_PATH = "Python Codes/fos_surrogate.pkl"

FOS_TARGET = 1.5         # minimum acceptable factor of safety
PENALTY_WEIGHT = 1e4      # how harshly to punish designs below FOS_TARGET
                           # (large on purpose -- see note below)

# search bounds -- MUST match (or stay within) the ranges you originally
# sampled with LHS. The surrogate has never seen data outside these
# ranges, so letting the GA wander outside them risks nonsense answers
# (see the "extrapolation" discussion from earlier in this conversation).
BOUNDS = {
    "length":    (0.02,  0.06),
    "width":     (0.02,  0.05),
    "thickness": (0.002, 0.005),
    "height":    (0.02,  0.05),
}
PARAM_NAMES = list(BOUNDS.keys())  # ["length", "width", "thickness", "height"]

DENSITY = 7850  # kg/m^3, steel

POP_SIZE = 100
GENERATIONS = 150
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.05
MUTATION_SIGMA = 0.05  # mutation step size, as a fraction of each param's range

RANDOM_SEED = 42


# -----------------------------------------------------------------------
# Load the trained surrogate model.
# -----------------------------------------------------------------------
model = joblib.load(MODEL_PATH)
print(f"Loaded surrogate model from {MODEL_PATH}")


# -----------------------------------------------------------------------
# Mass formula -- your L-bracket's volume, times density.
# -----------------------------------------------------------------------
def compute_mass(pop):
    """
    pop: array of shape (N, 4) -- columns are [length, width, thickness, height]
    returns: array of shape (N,) -- mass in kg for each row
    """
    length    = pop[:, 0]
    width     = pop[:, 1]
    thickness = pop[:, 2]
    height    = pop[:, 3]

    volume = (
        (width + height) * length * thickness
        + (0.003**2 * (4 - np.pi) / 4) * length
        - (0.005**2 * np.pi) * thickness
    )
    return volume * DENSITY


# -----------------------------------------------------------------------
# Fitness function -- what the GA is actually trying to minimize.
# -----------------------------------------------------------------------
def fitness(pop):
    """
    pop: array of shape (N, 4)
    returns: array of shape (N,) -- lower is better
    """
    fos_pred = model.predict(pop)          # surrogate's guess at FOS for each design
    mass = compute_mass(pop)               # exact, computed directly (no model needed)

    # if predicted FOS is BELOW target, this is positive (bad).
    # if predicted FOS is AT or ABOVE target, this is exactly 0 (no penalty).
    shortfall = np.maximum(0, FOS_TARGET - fos_pred)

    return mass + PENALTY_WEIGHT * shortfall


# -----------------------------------------------------------------------
# The genetic algorithm itself (same structure explained earlier in chat).
# -----------------------------------------------------------------------
def genetic_algorithm():
    rng = np.random.default_rng(RANDOM_SEED)

    lows  = np.array([BOUNDS[p][0] for p in PARAM_NAMES])
    highs = np.array([BOUNDS[p][1] for p in PARAM_NAMES])
    n_genes = len(PARAM_NAMES)

    # 1. Initialize: random population within bounds.
    pop = lows + rng.random((POP_SIZE, n_genes)) * (highs - lows)

    best_ever = None
    best_fitness = np.inf
    history = []  # track best fitness per generation, useful to plot convergence

    for gen in range(GENERATIONS):
        # 2. Evaluate.
        fit = fitness(pop)

        gen_best_idx = np.argmin(fit)
        if fit[gen_best_idx] < best_fitness:
            best_fitness = fit[gen_best_idx]
            best_ever = pop[gen_best_idx].copy()

        history.append(best_fitness)

        # 3d (elitism first, so the best solution is never lost).
        new_pop = [best_ever.copy()]

        while len(new_pop) < POP_SIZE:
            # 3a. Tournament selection.
            def tournament():
                idx = rng.choice(POP_SIZE, TOURNAMENT_K, replace=False)
                return pop[idx[np.argmin(fit[idx])]]

            parent_a, parent_b = tournament(), tournament()

            # 3b. Crossover.
            if rng.random() < CROSSOVER_RATE:
                r = rng.random(n_genes)
                child = parent_a + r * (parent_b - parent_a)
            else:
                child = parent_a.copy()

            # 3c. Mutation.
            mutate_mask = rng.random(n_genes) < MUTATION_RATE
            noise = rng.normal(0, MUTATION_SIGMA, n_genes) * (highs - lows)
            child[mutate_mask] += noise[mutate_mask]
            child = np.clip(child, lows, highs)  # stay within your sampled bounds

            new_pop.append(child)

        pop = np.array(new_pop)

        if gen % 20 == 0 or gen == GENERATIONS - 1:
            print(f"Generation {gen:4d} | best fitness so far: {best_fitness:.5f}")

    return best_ever, best_fitness, history


# -----------------------------------------------------------------------
# Run it.
# -----------------------------------------------------------------------
if __name__ == "__main__":
    best_design, best_score, history = genetic_algorithm()

    best_design_2d = best_design.reshape(1, -1)
    predicted_fos = model.predict(best_design_2d)[0]
    predicted_mass = compute_mass(best_design_2d)[0]

    print("\n" + "=" * 50)
    print("BEST DESIGN FOUND")
    print("=" * 50)
    for name, value in zip(PARAM_NAMES, best_design):
        print(f"  {name:10s}: {value:.6f} m")
    print(f"\n  Predicted FOS:  {predicted_fos:.4f}  (target: {FOS_TARGET})")
    print(f"  Predicted mass: {predicted_mass:.4f} kg")
    print("\nNEXT STEP: verify this design with a real ANSYS run before trusting it.")
    print("The surrogate is an approximation -- if this design's real FOS")
    print("comes out noticeably different from the prediction above, add")
    print("this point to your training data and retrain.")
