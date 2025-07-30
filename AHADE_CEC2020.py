import os
from copy import deepcopy
import numpy as np
from opfunu.cec_based.cec2020 import *
import random

# --- Parameters based on HRCOA for fair comparison ---
PopSize = 100
DimSize = 10  # Initial value, will be updated in main
LB = [-100] * DimSize
UB = [100] * DimSize

TrialRuns = 30
MaxFEs = 0  # To be set in main
curFEs = 0

MaxIter = 0 # To be set in main
curIter = 0

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

FuncNum = 0
SuiteName = "CEC2020"

# --- DE Parameters (as provided) ---
F = 0.8  # DE mutation factor
CR = 0.9 # DE crossover rate

# --- AHA-DE Hybridization Parameters (as provided) ---
initial_DE_frequency = 10
diversity_threshold = 0.5
convergence_threshold = 1e-6

def initialize_population(func):
    """
    Initializes the population for a given function from the benchmark suite.
    """
    global Pop, FitPop, curFEs, DimSize, LB, UB
    Pop = np.zeros((PopSize, DimSize))
    LB, UB = func.lb, func.ub
    for i in range(PopSize):
        Pop[i] = LB + (UB - LB) * np.random.rand(DimSize)
        FitPop[i] = func.evaluate(Pop[i])
        curFEs += 1

# --- Artificial Hummingbird Algorithm (AHA) components ---
def update_visit_table(visit_table):
    """
    Updates the visit table. In this simplified version, it increments for all.
    """
    return visit_table + 1

def AHA_foraging(population, fitness, func, visit_table):
    """
    Performs the foraging behavior of the AHA algorithm.
    This combines guided, territorial, and migratory foraging.
    """
    global curFEs, DimSize, MaxIter, curIter

    population_size, dimension = population.shape
    new_population = deepcopy(population)
    new_fitness = deepcopy(fitness)

    best_solution_index = np.argmin(fitness)
    best_solution = population[best_solution_index]

    # Guided and Territorial Foraging
    for i in range(population_size):
        # Guided foraging
        target_hummingbird_index = np.argmin(fitness)
        target_hummingbird = population[target_hummingbird_index]

        mutant = population[i] + np.random.normal(0, 1, dimension) * (target_hummingbird - population[i])
        mutant = np.clip(mutant, LB, UB)
        mutant_fitness = func.evaluate(mutant)
        curFEs += 1

        if mutant_fitness < new_fitness[i]:
            new_population[i] = mutant
            new_fitness[i] = mutant_fitness

        # Territorial foraging
        local_target_index = i # Simplified local search
        local_target = population[local_target_index]

        mutant = population[i] + np.random.normal(0, 1, dimension) * (local_target - population[i])
        mutant = np.clip(mutant, LB, UB)
        mutant_fitness = func.evaluate(mutant)
        curFEs += 1

        if mutant_fitness < new_fitness[i]:
            new_population[i] = mutant
            new_fitness[i] = mutant_fitness

    # Migration Foraging
    if curIter % (2 * dimension) == 0:
        worst_hummingbird_index = np.argmax(new_fitness)
        new_population[worst_hummingbird_index] = LB + (UB - LB) * np.random.rand(dimension)
        new_fitness[worst_hummingbird_index] = func.evaluate(new_population[worst_hummingbird_index])
        curFEs += 1

    return new_population, new_fitness

# --- Differential Evolution (DE) components ---
def DE_mutation(population):
    """
    Performs DE mutation (rand/1).
    """
    population_size, _ = population.shape
    mutant_vectors = np.zeros_like(population)
    for i in range(population_size):
        indices = [j for j in range(population_size) if j != i]
        a, b, c = random.sample(indices, 3)
        mutant_vectors[i] = population[a] + F * (population[b] - population[c])
    return mutant_vectors

def DE_crossover(population, mutant_vectors):
    """
    Performs DE crossover (binomial).
    """
    trial_vectors = np.zeros_like(population)
    population_size, dimension = population.shape
    for i in range(population_size):
        j_rand = random.randint(0, dimension - 1)
        for j in range(dimension):
            if random.random() < CR or j == j_rand:
                trial_vectors[i, j] = mutant_vectors[i, j]
            else:
                trial_vectors[i, j] = population[i, j]
    return np.clip(trial_vectors, LB, UB)


def DE_selection(population, fitness, trial_vectors, func):
    """
    Performs DE selection.
    """
    global curFEs
    selected_vectors = deepcopy(population)
    selected_fitness = deepcopy(fitness)
    for i in range(len(population)):
        trial_fitness = func.evaluate(trial_vectors[i])
        curFEs += 1
        if trial_fitness < fitness[i]:
            selected_vectors[i] = trial_vectors[i]
            selected_fitness[i] = trial_fitness
    return selected_vectors, selected_fitness

# --- Main AHADE Algorithm ---
def AHADE(func):
    """
    Main loop for the Hybrid AHA-DE algorithm.
    """
    global Pop, FitPop, curFEs, curIter

    # --- Adaptive Strategy ---
    # Simplified for integration: DE is triggered based on a fixed frequency
    # This matches the simplicity of the HRCOA logic for fair comparison.
    use_de = curIter % initial_DE_frequency == 0

    if use_de:
        mutant_vectors = DE_mutation(Pop)
        trial_vectors = DE_crossover(Pop, mutant_vectors)
        Pop, FitPop = DE_selection(Pop, FitPop, trial_vectors, func)
    else:
        # Initialize visit_table for AHA phase
        visit_table = np.zeros(PopSize)
        Pop, FitPop = AHA_foraging(Pop, FitPop, func, visit_table)
        visit_table = update_visit_table(visit_table)

def RunAHADE(func):
    """
    Runs the AHADE algorithm for a given function over multiple trials.
    """
    global curFEs, curIter, MaxFEs, TrialRuns, Pop, FitPop, DimSize, MaxIter
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curFEs = 0
        curIter = 0

        # Set seed for reproducibility
        np.random.seed(2020 + 88 * i)
        random.seed(2020 + 88 * i)

        initialize_population(func)
        Best_list.append(min(FitPop))

        while curFEs < MaxFEs:
            AHADE(func)
            curIter += 1
            # Ensure not to exceed MaxFEs
            if curFEs >= MaxFEs:
                break
            Best_list.append(min(FitPop))

        # Handle cases where the loop finishes but we need one last value for plotting
        while len(Best_list) < MaxIter + 1:
             Best_list.append(Best_list[-1])

        All_Trial_Best.append(Best_list[:MaxIter+1])

    # Save results
    if not os.path.exists(f'./AHADE_Data/{SuiteName}'):
        os.makedirs(f'./AHADE_Data/{SuiteName}')
    np.savetxt(f"./AHADE_Data/{SuiteName}/F{FuncNum}_{DimSize}D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    """
    Main function to run the experiments for a given dimension.
    """
    global FuncNum, DimSize, Pop, MaxFEs, MaxIter, SuiteName, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, DimSize))
    MaxFEs = DimSize * 1000  # As per HRCOA paper
    MaxIter = int(MaxFEs / PopSize)

    # CEC2020 Benchmark Suite
    CEC2020_suite = [
        F12020(DimSize), F22020(DimSize), F32020(DimSize), F42020(DimSize),
        F52020(DimSize), F62020(DimSize), F72020(DimSize), F82020(DimSize),
        F92020(DimSize), F102020(DimSize)
    ]

    for i in range(len(CEC2020_suite)):
        FuncNum = i + 1
        print(f"Running Function {FuncNum} for {DimSize}D...")
        RunAHADE(CEC2020_suite[i])


if __name__ == "__main__":
    # Create main data directory
    if not os.path.exists('./AHADE_Data'):
        os.makedirs('./AHADE_Data')

    Dims = [50, 100] # As per HRCOA paper for CEC2020
    for dim in Dims:
        main(dim)