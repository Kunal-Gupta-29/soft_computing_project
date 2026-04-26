"""
ga_optimizer.py
---------------
Genetic Algorithm (GA) for optimizing CNN hyperparameters.

This module implements a GA that evolves the best combination of:
    * learning_rate  - Adam optimizer step size
    * batch_size     - mini-batch size during training
    * dropout_conv   - dropout rate after each conv block
    * dropout_dense1 - dropout rate in the first FC layer
    * dropout_dense2 - dropout rate in the second FC layer
    * dense_units    - size of the first Dense layer
    * l2_reg         - L2 weight-decay regularization coefficient

WHY Genetic Algorithm instead of manual tuning?
  - The hyperparameter search space is exponentially large (7 genes x multiple values)
  - Manual tuning is subjective and misses non-obvious interactions (e.g., high
    dropout works best with high L2 regularization simultaneously)
  - Grid search is computationally infeasible (7 genes x avg 5 values = 78,125 combos)
  - GA intelligently explores the space using evolution principles:
      -> Selection ensures good genes survive
      -> Crossover combines good traits from two parents
      -> Mutation prevents premature convergence to local optima
  - GA is a classic SOFT COMPUTING technique (biologically inspired optimization)

GA Flow
-------
1. Initialize a random population of N individuals (chromosomes)
2. Evaluate fitness of each individual (val_accuracy after GA_TRIAL_EPOCHS)
3. Select parents via tournament selection
4. Apply single-point crossover to produce offspring
5. Apply random mutation with probability GA_MUTATION_RATE
6. Elitism: carry the top GA_ELITE_SIZE individuals unchanged
7. Repeat for GA_GENERATIONS generations
8. Return best individual and compare accuracy vs baseline config

Usage
-----
    # Standalone
    python ga_optimizer.py

    # Quick test (fewer generations)
    python ga_optimizer.py --gen 2 --pop 4 --trial-epochs 3

    # From train.py (via --ga flag)
    from ga_optimizer import run_ga_optimization
    best_params, baseline_acc, ga_acc = run_ga_optimization()
"""

import os
import csv
import json
import random
import copy
import math
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# -- Project imports -----------------------------------------------------------
from config import (
    # GA settings
    GA_POPULATION_SIZE, GA_GENERATIONS, GA_MUTATION_RATE,
    GA_CROSSOVER_POINT, GA_TRIAL_EPOCHS, GA_ELITE_SIZE,
    GA_TOURNAMENT_SIZE, GA_BEST_PARAMS_PATH, GA_FITNESS_LOG_PATH,
    GA_COMPARISON_PLOT, GA_SUMMARY_PATH,
    # CNN defaults (used as baseline)
    LEARNING_RATE, BATCH_SIZE, NUM_CLASSES, IMG_SIZE, CHANNELS,
    OUTPUT_DIR,
)


# --- Search Space -------------------------------------------------------------

SEARCH_SPACE = {
    "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2],
    "batch_size":    [16, 32, 64, 128],
    "dropout_conv":  [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "dropout_dense1":[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    "dropout_dense2":[0.20, 0.25, 0.30, 0.35, 0.40],
    "dense_units":   [128, 256, 512],
    "l2_reg":        [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
}

GENE_KEYS = list(SEARCH_SPACE.keys())   # Fixed gene order for crossover


# --- Chromosome ---------------------------------------------------------------

@dataclass
class Individual:
    """
    One candidate solution (chromosome) in the GA population.

    Each field maps to a hyperparameter gene.
    fitness is the validation accuracy achieved after GA_TRIAL_EPOCHS.

    GENE ENCODING:
      Each Individual is a vector of hyperparameter values (its "chromosome").
      The GA treats each hyperparameter as a discrete "gene" that can be:
        - Inherited from a parent (via crossover)
        - Randomly changed (via mutation)
      This is analogous to biological evolution where traits are inherited
      and modified by random mutations.
    """
    learning_rate : float = LEARNING_RATE
    batch_size    : int   = BATCH_SIZE
    dropout_conv  : float = 0.25
    dropout_dense1: float = 0.50
    dropout_dense2: float = 0.30
    dense_units   : int   = 256
    l2_reg        : float = 1e-4
    fitness       : float = 0.0          # filled after evaluation

    def to_dict(self) -> dict:
        """Return hyperparameter dict (excludes fitness)."""
        return {k: v for k, v in asdict(self).items() if k != "fitness"}

    def genes(self) -> list:
        """Return gene list in GENE_KEYS order."""
        d = self.to_dict()
        return [d[k] for k in GENE_KEYS]

    @classmethod
    def from_genes(cls, genes: list) -> "Individual":
        """Build an Individual from a gene list."""
        kwargs = {k: genes[i] for i, k in enumerate(GENE_KEYS)}
        return cls(**kwargs)

    def __repr__(self) -> str:
        return (
            f"Individual(lr={self.learning_rate:.5f}, bs={self.batch_size}, "
            f"dc={self.dropout_conv:.2f}, dd1={self.dropout_dense1:.2f}, "
            f"dd2={self.dropout_dense2:.2f}, du={self.dense_units}, "
            f"l2={self.l2_reg:.0e}, fit={self.fitness:.4f})"
        )


# --- CNN Builder (GA-parameterized) -------------------------------------------

def build_cnn_from_individual(ind: Individual):
    """
    Build and compile a CNN using hyperparameters from the Individual.
    Uses the custom CNN (48x48) for fast GA trial evaluations.
    """
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import (
        Conv2D, BatchNormalization, Activation,
        MaxPooling2D, Dropout, Flatten, Dense,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2 as keras_l2

    def conv_block(x, filters):
        x = Conv2D(filters, 3, padding="same",
                   kernel_regularizer=keras_l2(ind.l2_reg))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, 3, padding="same",
                   kernel_regularizer=keras_l2(ind.l2_reg))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D()(x)
        x = Dropout(ind.dropout_conv)(x)
        return x

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS), name="face_input")
    x = conv_block(inputs, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = Flatten()(x)

    x = Dense(ind.dense_units, kernel_regularizer=keras_l2(ind.l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(ind.dropout_dense1)(x)

    x = Dense(ind.dense_units // 2, kernel_regularizer=keras_l2(ind.l2_reg))(x)
    x = Activation("relu")(x)
    x = Dropout(ind.dropout_dense2)(x)

    outputs = Dense(NUM_CLASSES, activation="softmax", name="emotion_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="EmotionCNN_GA")
    model.compile(
        optimizer=Adam(learning_rate=ind.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --- Data Loader (shared across fitness evaluations) -------------------------

_data_cache = {}     # module-level cache so we load once


def _get_training_data():
    """
    Load and cache the dataset for GA fitness evaluations.
    Returns cached generators/arrays to avoid re-loading between individuals.
    """
    global _data_cache
    if _data_cache:
        return _data_cache

    from preprocess import _detect_format, get_folder_generators, load_fer2013, get_data_generators

    fmt = _detect_format()
    if fmt == "folder":
        train_gen, val_gen, _, _ = get_folder_generators()
        steps = max(1, train_gen.samples // 64)
        val_steps = max(1, val_gen.samples // 64)
        _data_cache = dict(fmt="folder", train=train_gen, val=val_gen,
                           steps=steps, val_steps=val_steps)
    else:
        (X_tr, y_tr), (X_v, y_v), _ = load_fer2013()
        train_gen, val_gen = get_data_generators(X_tr, y_tr, X_v, y_v)
        steps = max(1, len(X_tr) // 64)
        val_steps = max(1, len(X_v) // 64)
        _data_cache = dict(fmt="csv", train=train_gen, val=val_gen,
                           steps=steps, val_steps=val_steps,
                           X_val=X_v, y_val=y_v)

    return _data_cache


# --- Genetic Algorithm --------------------------------------------------------

class GeneticAlgorithm:
    """
    Standard GA for optimizing CNN hyperparameters.

    SOFT COMPUTING ALIGNMENT:
      - Unlike hard computing (exact algorithms), GA uses probabilistic
        operations (selection, crossover, mutation) inspired by biology
      - It is robust to noisy fitness functions (training accuracy varies
        due to random weight initialization)
      - It explores the global search space rather than doing local gradient descent
      - This is exactly what "soft computing" means: approximate, heuristic,
        nature-inspired computation for complex optimization problems

    Parameters
    ----------
    population_size : int   - individuals per generation
    generations     : int   - number of evolution cycles
    mutation_rate   : float - probability of mutating a gene
    crossover_point : int   - split index for single-point crossover
    elite_size      : int   - top individuals carried unchanged
    tournament_size : int   - candidates compared in each tournament
    trial_epochs    : int   - training epochs per fitness evaluation
    verbose         : bool  - print progress
    """

    def __init__(
        self,
        population_size : int   = GA_POPULATION_SIZE,
        generations     : int   = GA_GENERATIONS,
        mutation_rate   : float = GA_MUTATION_RATE,
        crossover_point : int   = GA_CROSSOVER_POINT,
        elite_size      : int   = GA_ELITE_SIZE,
        tournament_size : int   = GA_TOURNAMENT_SIZE,
        trial_epochs    : int   = GA_TRIAL_EPOCHS,
        verbose         : bool  = True,
    ):
        self.population_size  = population_size
        self.generations      = generations
        self.mutation_rate    = mutation_rate
        self.crossover_point  = crossover_point
        self.elite_size       = elite_size
        self.tournament_size  = tournament_size
        self.trial_epochs     = trial_epochs
        self.verbose          = verbose

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.fitness_history: List[dict] = []   # per-generation stats

    # -- Initialization --------------------------------------------------------

    def _random_individual(self) -> Individual:
        """Create one random Individual by sampling each gene from SEARCH_SPACE."""
        genes = [random.choice(SEARCH_SPACE[k]) for k in GENE_KEYS]
        return Individual.from_genes(genes)

    def initialize_population(self) -> None:
        """
        Create the initial random population.
        Seed one individual with the baseline config (so GA starts informed).
        """
        baseline = Individual()   # uses config defaults (informed start)
        self.population = [baseline] + [
            self._random_individual()
            for _ in range(self.population_size - 1)
        ]
        if self.verbose:
            print(f"\n[GA] Initial population: {self.population_size} individuals")
            print(f"[GA] Generations: {self.generations}  |  "
                  f"Trial epochs per individual: {self.trial_epochs}")
            print(f"[GA] Mutation rate: {self.mutation_rate}  |  "
                  f"Elite size: {self.elite_size}")
            print(f"[GA] Search space: {len(GENE_KEYS)} genes, "
                  f"~{sum(len(v) for v in SEARCH_SPACE.values())} total values\n")

    # -- Fitness Evaluation ----------------------------------------------------

    def evaluate_fitness(self, ind: Individual) -> float:
        """
        FITNESS FUNCTION: Train the CNN for trial_epochs, return val_accuracy.

        In GA terminology:
          - The "fitness" of an individual = how well it performs
          - Higher fitness -> more likely to be selected as a parent
          - We use validation accuracy (not training accuracy) to avoid
            selecting individuals that overfit

        WHY short trial epochs?
          - Running full training (50 epochs) per individual would take days
          - 5 trial epochs gives a fast, noisy estimate of the hyperparameters'
            quality -- still enough to distinguish good from bad configurations
        """
        import tensorflow as tf

        data = _get_training_data()
        bs = ind.batch_size
        if data["fmt"] == "folder":
            steps     = max(1, data["train"].samples // bs)
            val_steps = max(1, data["val"].samples   // bs)
        else:
            steps     = max(1, data["steps"])
            val_steps = max(1, data["val_steps"])

        model = build_cnn_from_individual(ind)

        history = model.fit(
            data["train"],
            steps_per_epoch  = steps,
            epochs           = self.trial_epochs,
            validation_data  = data["val"],
            validation_steps = val_steps,
            verbose          = 0,
        )

        val_acc = max(history.history["val_accuracy"])

        del model
        import tensorflow.keras.backend as K
        K.clear_session()

        return round(float(val_acc), 5)

    def evaluate_population(self, label: str = "") -> None:
        """Evaluate fitness for every unevaluated individual."""
        n = len(self.population)
        for i, ind in enumerate(self.population):
            if ind.fitness == 0.0:
                if self.verbose:
                    print(f"  [{label}] Evaluating individual {i+1}/{n}: {ind}")
                ind.fitness = self.evaluate_fitness(ind)
                if self.verbose:
                    print(f"           -> val_accuracy = {ind.fitness:.4f}")

    # -- Selection -- Tournament ------------------------------------------------

    def tournament_select(self) -> Individual:
        """
        TOURNAMENT SELECTION: Choose the fittest individual from a random subset.

        WHY tournament selection?
          - Simpler than roulette-wheel selection (no fitness normalization needed)
          - Higher tournament_size -> stronger selection pressure
          - Acts like a natural competition -- only the "strongest" survive to reproduce
        """
        contestants = random.sample(self.population, min(self.tournament_size,
                                                         len(self.population)))
        return max(contestants, key=lambda x: x.fitness)

    # -- Crossover -------------------------------------------------------------

    def crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """
        SINGLE-POINT CROSSOVER: Combine genes from two parents.

        WHY crossover?
          - In nature, offspring inherit traits from both parents
          - Here, child1 gets genes [0:cp] from p1 and [cp:] from p2
          - Combines the "strengths" of two well-performing configurations
          - Creates genetic diversity without being completely random

        Example (cp=3):
          p1: [lr=1e-3, bs=64, dc=0.25, dd1=0.5, dd2=0.3, du=256, l2=1e-4]
          p2: [lr=3e-4, bs=32, dc=0.20, dd1=0.4, dd2=0.2, du=128, l2=5e-5]
          c1: [lr=1e-3, bs=64, dc=0.25, dd1=0.4, dd2=0.2, du=128, l2=5e-5]
          c2: [lr=3e-4, bs=32, dc=0.20, dd1=0.5, dd2=0.3, du=256, l2=1e-4]
        """
        g1 = p1.genes()
        g2 = p2.genes()
        cp = self.crossover_point

        c1_genes = g1[:cp] + g2[cp:]
        c2_genes = g2[:cp] + g1[cp:]

        child1 = Individual.from_genes(c1_genes)
        child2 = Individual.from_genes(c2_genes)
        return child1, child2

    # -- Mutation --------------------------------------------------------------

    def mutate(self, ind: Individual) -> Individual:
        """
        MUTATION: Randomly replace a gene with a new value.

        WHY mutation?
          - Without mutation, the GA can get stuck in local optima
          - Mutation introduces random "jumps" to unexplored regions
          - GA_MUTATION_RATE = 0.20 means each gene has 20% chance of mutating
          - Too high mutation -> random search (loses learned structure)
          - Too low  mutation -> premature convergence
        """
        genes   = ind.genes()
        mutated = False
        for i, key in enumerate(GENE_KEYS):
            if random.random() < self.mutation_rate:
                genes[i] = random.choice(SEARCH_SPACE[key])
                mutated   = True

        if mutated:
            new_ind         = Individual.from_genes(genes)
            new_ind.fitness = 0.0   # needs re-evaluation
            return new_ind
        return ind   # no mutation -> return as-is (fitness preserved)

    # -- Evolution Loop --------------------------------------------------------

    def evolve(self) -> Individual:
        """
        Main GA loop. Returns the best Individual found.

        Steps per generation:
          1. Evaluate fitness of all unevaluated individuals
          2. Sort by fitness (descending)
          3. Carry top elite_size individuals unchanged (ELITISM)
          4. Fill rest: tournament_select -> crossover -> mutate
          5. Repeat for all generations
        """
        self.initialize_population()

        for gen in range(1, self.generations + 1):
            banner = f"Generation {gen}/{self.generations}"
            if self.verbose:
                print("=" * 60)
                print(f"  {banner}")
                print("=" * 60)

            # Step 1: Evaluate fitness
            self.evaluate_population(label=banner)

            # Step 2: Sort by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            best    = self.population[0]
            avg_fit = sum(x.fitness for x in self.population) / len(self.population)

            if self.verbose:
                print(f"\n  [GA] Gen {gen} results:")
                print(f"       Best fitness : {best.fitness:.4f}")
                print(f"       Avg  fitness : {avg_fit:.4f}")
                print(f"       Best params  : {best}\n")

            self.fitness_history.append({
                "generation"  : gen,
                "best_fitness": best.fitness,
                "avg_fitness" : avg_fit,
            })

            if gen == self.generations:
                break   # no new generation after last evaluation

            # Step 3: Elitism
            elites   = copy.deepcopy(self.population[:self.elite_size])

            # Step 4: Selection -> Crossover -> Mutation
            offspring: List[Individual] = list(elites)
            while len(offspring) < self.population_size:
                p1 = self.tournament_select()
                p2 = self.tournament_select()
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                offspring.append(c1)
                if len(offspring) < self.population_size:
                    offspring.append(c2)

            self.population = offspring

        # Global best
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_individual = self.population[0]

        if self.verbose:
            print("\n" + "=" * 60)
            print("  GA COMPLETE")
            print(f"  Best individual: {self.best_individual}")
            print("=" * 60)

        return self.best_individual

    # -- Logging ---------------------------------------------------------------

    def save_fitness_log(self) -> None:
        """Save per-generation fitness stats to CSV."""
        with open(GA_FITNESS_LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["generation", "best_fitness", "avg_fitness"])
            writer.writeheader()
            writer.writerows(self.fitness_history)
        print(f"[GA] Fitness log saved -> {GA_FITNESS_LOG_PATH}")

    def save_best_params(self) -> None:
        """Save best Individual as JSON."""
        data = self.best_individual.to_dict()
        data["fitness_trial"] = self.best_individual.fitness
        with open(GA_BEST_PARAMS_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[GA] Best params saved -> {GA_BEST_PARAMS_PATH}")


# --- Full Training with GA Params --------------------------------------------

def train_with_params(
    ind: Individual,
    label: str = "GA-Optimized",
    epochs: int = 50,
) -> Tuple[float, float]:
    """
    Train the CNN fully (all epochs) using the best GA-found hyperparameters.
    Returns (test_loss, test_accuracy).
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    data  = _get_training_data()
    bs    = ind.batch_size
    model = build_cnn_from_individual(ind)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
    ]

    if data["fmt"] == "folder":
        steps     = max(1, data["train"].samples // bs)
        val_steps = max(1, data["val"].samples   // bs)
        model.fit(
            data["train"],
            steps_per_epoch  = steps,
            epochs           = epochs,
            validation_data  = data["val"],
            validation_steps = val_steps,
            callbacks        = callbacks,
            verbose          = 1,
        )
        loss, acc = model.evaluate(data["val"], steps=val_steps, verbose=0)
    else:
        steps     = max(1, data["steps"])
        val_steps = max(1, data["val_steps"])
        model.fit(
            data["train"],
            steps_per_epoch  = steps,
            epochs           = epochs,
            validation_data  = data["val"],
            validation_steps = val_steps,
            callbacks        = callbacks,
            verbose          = 1,
        )
        loss, acc = model.evaluate(
            data.get("X_val", data["val"]),
            data.get("y_val", None),
            verbose=0,
        )

    print(f"\n[{label}] Final val accuracy : {acc*100:.2f}%")
    print(f"[{label}] Final val loss     : {loss:.4f}")

    import tensorflow.keras.backend as K
    del model
    K.clear_session()

    return round(float(loss), 5), round(float(acc), 5)


# --- Accuracy Comparison Plot -------------------------------------------------

def plot_accuracy_comparison(
    baseline_acc : float,
    ga_acc       : float,
    ga_trial_acc : float,
    fitness_history: List[dict],
) -> None:
    """
    Generate and save a two-panel accuracy comparison figure:
        Panel 1 - Bar chart: Baseline vs GA-optimized validation accuracy
        Panel 2 - Line chart: GA fitness evolution across generations
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0f1117")

    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # -- Panel 1: Bar comparison -----------------------------------------------
    ax1 = axes[0]
    labels = ["Baseline\n(Fixed Config)", f"GA Trial\n({GA_TRIAL_EPOCHS} epochs)",
              "GA Optimized\n(Full Training)"]
    values = [baseline_acc * 100, ga_trial_acc * 100, ga_acc * 100]
    colors  = ["#5c6bc0", "#26a69a", "#ef5350"]

    bars = ax1.bar(labels, values, color=colors, width=0.5,
                   edgecolor="#ffffff22", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.2f}%",
            ha="center", va="bottom",
            color="white", fontsize=12, fontweight="bold",
        )

    ax1.set_ylim(0, max(values) * 1.15)
    ax1.set_title("Accuracy Comparison\nBaseline vs GA-Optimized",
                  color="white", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Validation Accuracy (%)", color="#aaa", fontsize=11)
    ax1.tick_params(axis="x", colors="white", labelsize=10)
    ax1.tick_params(axis="y", colors="#aaa")

    improvement = ga_acc * 100 - baseline_acc * 100
    sign = "+" if improvement >= 0 else ""
    ax1.annotate(
        f"GA Improvement: {sign}{improvement:.2f}%",
        xy=(0.5, 0.05), xycoords="axes fraction",
        ha="center", color="#ffd54f", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#2a2d3a", ec="#ffd54f44", alpha=0.9),
    )

    # -- Panel 2: GA fitness evolution -----------------------------------------
    ax2 = axes[1]
    gens       = [h["generation"]   for h in fitness_history]
    best_fits  = [h["best_fitness"] * 100 for h in fitness_history]
    avg_fits   = [h["avg_fitness"]  * 100 for h in fitness_history]

    ax2.plot(gens, best_fits, color="#26a69a", marker="o",
             linewidth=2.5, markersize=7, label="Best Fitness")
    ax2.plot(gens, avg_fits,  color="#ef5350", marker="s",
             linewidth=2, markersize=6, linestyle="--", label="Avg Fitness")
    ax2.fill_between(gens, avg_fits, best_fits, alpha=0.08, color="#26a69a")

    ax2.set_title("GA Fitness Evolution\nacross Generations",
                  color="white", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Generation", color="#aaa", fontsize=11)
    ax2.set_ylabel("Fitness (val_accuracy %)", color="#aaa", fontsize=11)
    ax2.legend(facecolor="#2a2d3a", edgecolor="#444", labelcolor="white", fontsize=10)
    ax2.xaxis.label.set_color("#aaa")
    ax2.set_xticks(gens)

    plt.suptitle(
        "Genetic Algorithm Hyperparameter Optimization\n"
        "Real-Time Emotion Recognition & Autism Detection",
        color="white", fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    plt.savefig(GA_COMPARISON_PLOT, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"[GA] Comparison plot saved -> {GA_COMPARISON_PLOT}")
    plt.close()


# --- Text Summary -------------------------------------------------------------

def save_ga_summary(
    baseline_acc : float,
    ga_trial_acc : float,
    ga_final_acc : float,
    best_ind     : Individual,
    fitness_history: List[dict],
) -> None:
    """
    Save a full text summary of the GA run to GA_SUMMARY_PATH.
    Useful for viva preparation and project reports.
    """
    improvement = (ga_final_acc - baseline_acc) * 100
    sign = "+" if improvement >= 0 else ""

    lines = [
        "=" * 62,
        "  GENETIC ALGORITHM -- CNN Hyperparameter Optimization",
        "  Real-Time Emotion Recognition & Autism Detection Project",
        "=" * 62,
        "",
        "GA SETTINGS:",
        f"  Population size  : {GA_POPULATION_SIZE}",
        f"  Generations      : {GA_GENERATIONS}",
        f"  Mutation rate    : {GA_MUTATION_RATE}",
        f"  Crossover point  : {GA_CROSSOVER_POINT}",
        f"  Trial epochs     : {GA_TRIAL_EPOCHS}",
        f"  Elite size       : {GA_ELITE_SIZE}",
        f"  Tournament size  : {GA_TOURNAMENT_SIZE}",
        "",
        "ACCURACY COMPARISON:",
        f"  Baseline (config defaults)   : {baseline_acc*100:.2f}%",
        f"  GA Best Trial Accuracy       : {ga_trial_acc*100:.2f}%",
        f"  GA Final (full training)     : {ga_final_acc*100:.2f}%",
        f"  Improvement                  : {sign}{improvement:.2f}%",
        "",
        "BEST HYPERPARAMETERS FOUND BY GA:",
    ]
    for k, v in best_ind.to_dict().items():
        lines.append(f"  {k:20s}: {v}")

    lines += [
        "",
        "FITNESS EVOLUTION (per generation):",
        "  Gen | Best Fitness | Avg Fitness",
        "  " + "-" * 38,
    ]
    for h in fitness_history:
        lines.append(
            f"  {h['generation']:3d} | "
            f"{h['best_fitness']*100:11.2f}% | "
            f"{h['avg_fitness']*100:10.2f}%"
        )

    lines += [
        "",
        "WHY GA OVER MANUAL TUNING?",
        "  - Search space: 7 genes x avg 5 values = 78,125 combinations",
        "  - GA intelligently explores using evolution principles:",
        "      Selection -> only fit individuals reproduce",
        "      Crossover -> combine strengths from two parents",
        "      Mutation  -> explore new regions, avoid local optima",
        "      Elitism   -> best solutions always survive",
        "  - Soft Computing: heuristic, probabilistic, nature-inspired",
        "=" * 62,
    ]

    with open(GA_SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"[GA] Summary saved -> {GA_SUMMARY_PATH}")


# --- Baseline Accuracy Helper -------------------------------------------------

def get_baseline_accuracy(trial_epochs: int = GA_TRIAL_EPOCHS) -> float:
    """
    Train the CNN with the default config hyperparameters for trial_epochs
    and return validation accuracy. This is the GA benchmark to beat.
    """
    import tensorflow.keras.backend as K
    baseline_ind = Individual()   # all defaults from config
    print("\n[GA] Measuring baseline accuracy (config defaults)...")
    print(f"     {baseline_ind}")

    data = _get_training_data()
    bs   = baseline_ind.batch_size
    model = build_cnn_from_individual(baseline_ind)

    if data["fmt"] == "folder":
        steps     = max(1, data["train"].samples // bs)
        val_steps = max(1, data["val"].samples   // bs)
        history = model.fit(
            data["train"],
            steps_per_epoch  = steps,
            epochs           = trial_epochs,
            validation_data  = data["val"],
            validation_steps = val_steps,
            verbose          = 1,
        )
    else:
        steps     = max(1, data["steps"])
        val_steps = max(1, data["val_steps"])
        history = model.fit(
            data["train"],
            steps_per_epoch  = steps,
            epochs           = trial_epochs,
            validation_data  = data["val"],
            validation_steps = val_steps,
            verbose          = 1,
        )

    acc = max(history.history["val_accuracy"])
    print(f"[GA] Baseline val_accuracy (over {trial_epochs} epochs): {acc*100:.2f}%")

    del model
    K.clear_session()
    return round(float(acc), 5)


# --- Main Entry Point ---------------------------------------------------------

def run_ga_optimization(full_train: bool = True) -> Tuple[dict, float, float]:
    """
    End-to-end GA optimization pipeline:
        1. Measure baseline accuracy (config defaults, trial epochs)
        2. Run GA to find best hyperparameters (trial epochs per individual)
        3. Optionally retrain full model with best params (full epochs)
        4. Compare and plot baseline vs GA accuracy
        5. Save best params JSON, fitness log CSV, comparison PNG, summary TXT

    Returns: best_params, baseline_acc, ga_final_acc
    """
    print("\n" + "+" + "=" * 60 + "+")
    print("|  GENETIC ALGORITHM -- CNN Hyperparameter Optimization     |")
    print("+" + "=" * 60 + "+\n")

    # Step 1: Baseline
    baseline_acc = get_baseline_accuracy(trial_epochs=GA_TRIAL_EPOCHS)

    # Step 2: Run GA
    ga = GeneticAlgorithm()
    best_ind = ga.evolve()
    ga_trial_acc = best_ind.fitness

    ga.save_fitness_log()
    ga.save_best_params()

    # Step 3: Full retraining
    if full_train:
        print("\n[GA] Retraining best individual for full epochs ...")
        from config import EPOCHS as FULL_EPOCHS
        _, ga_final_acc = train_with_params(best_ind, label="GA-Optimized",
                                            epochs=FULL_EPOCHS)
    else:
        ga_final_acc = ga_trial_acc

    # Step 4: Summary
    improvement = (ga_final_acc - baseline_acc) * 100
    sign = "+" if improvement >= 0 else ""

    print("\n" + "+" + "=" * 60 + "+")
    print("|           ACCURACY COMPARISON SUMMARY                    |")
    print("+" + "=" * 60 + "+")
    print(f"|  Baseline accuracy  (config defaults)  : "
          f"{baseline_acc*100:>6.2f}%        |")
    print(f"|  GA best trial acc  ({GA_TRIAL_EPOCHS} epochs / indiv): "
          f"{ga_trial_acc*100:>6.2f}%        |")
    print(f"|  GA final accuracy  (full training)    : "
          f"{ga_final_acc*100:>6.2f}%        |")
    print(f"|  Improvement                           : "
          f"{sign}{improvement:>5.2f}%        |")
    print("+" + "=" * 60 + "+")
    print(f"\n  Best hyperparameters:")
    for k, v in best_ind.to_dict().items():
        print(f"    {k:20s}: {v}")

    # Step 5: Plot + Summary text
    plot_accuracy_comparison(
        baseline_acc    = baseline_acc,
        ga_acc          = ga_final_acc,
        ga_trial_acc    = ga_trial_acc,
        fitness_history = ga.fitness_history,
    )
    save_ga_summary(baseline_acc, ga_trial_acc, ga_final_acc,
                    best_ind, ga.fitness_history)

    return best_ind.to_dict(), baseline_acc, ga_final_acc


# --- Standalone runner --------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GA Hyperparameter Optimizer for EmotionCNN")
    parser.add_argument(
        "--no-full-train", action="store_true",
        help="Skip final full training; just run GA trials and compare trial accuracies.",
    )
    parser.add_argument("--pop",  type=int, default=None, help="Override GA_POPULATION_SIZE")
    parser.add_argument("--gen",  type=int, default=None, help="Override GA_GENERATIONS")
    parser.add_argument("--trial-epochs", type=int, default=None, help="Override GA_TRIAL_EPOCHS")
    args = parser.parse_args()

    import config as _cfg
    if args.pop:          _cfg.GA_POPULATION_SIZE = args.pop
    if args.gen:          _cfg.GA_GENERATIONS      = args.gen
    if args.trial_epochs: _cfg.GA_TRIAL_EPOCHS     = args.trial_epochs

    best_params, baseline, ga_acc = run_ga_optimization(
        full_train=not args.no_full_train
    )
    print(f"\n[GA] Done! Results saved to: {OUTPUT_DIR}")
