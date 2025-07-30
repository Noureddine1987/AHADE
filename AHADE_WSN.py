import os
from copy import deepcopy
import numpy as np
from WSN import WSN_fit, Draw_indi # Import from our new WSN.py
import random

# --- Parameters based on HRCOA for fair comparison (WSN specific) ---
PopSize = 30
DimSize = 0   # To be set in main
LB = []       # To be set in main
UB = []       # To be set in main

TrialRuns = 30
MaxIter = 100 # WSN uses MaxIter instead of MaxFEs
curIter = 0

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

FuncNum = "" 
SuiteName = "WSN"

# --- DE Parameters ---
F = 0.8
CR = 0.9

# --- AHA-DE Hybridization Parameters ---
initial_DE_frequency = 10

def initialize_population(func):
    """
    Initializes the population for the WSN problem.
    """
    global Pop, FitPop, DimSize, LB, UB
    Pop = np.zeros((PopSize, DimSize))
    for i in range(PopSize):
        Pop[i] = LB + (UB - LB) * np.random.rand(DimSize)
        # We negate the fitness because WSN is a maximization problem
        FitPop[i] = -func(Pop[i])

# --- Artificial Hummingbird Algorithm (AHA) components ---
def AHA_foraging(population, fitness, func):
    global curIter, DimSize, LB, UB
    
    population_size, dimension = population.shape
    new_population = deepcopy(population)
    new_fitness = deepcopy(fitness)

    best_solution_index = np.argmin(fitness)
    best_solution = population[best_solution_index]
    
    for i in range(population_size):
        target_hummingbird_index = np.argmin(fitness)
        target_hummingbird = population[target_hummingbird_index]
        
        mutant = population[i] + np.random.normal(0, 1, dimension) * (target_hummingbird - population[i])
        mutant = np.clip(mutant, LB, UB)
        mutant_fitness = -func(mutant)

        if mutant_fitness < new_fitness[i]:
            new_population[i] = mutant
            new_fitness[i] = mutant_fitness

        local_target_index = i
        local_target = population[local_target_index]
        
        mutant = population[i] + np.random.normal(0, 1, dimension) * (local_target - population[i])
        mutant = np.clip(mutant, LB, UB)
        mutant_fitness = -func(mutant)

        if mutant_fitness < new_fitness[i]:
            new_population[i] = mutant
            new_fitness[i] = mutant_fitness

    if curIter % (2 * dimension) == 0:
        worst_hummingbird_index = np.argmax(new_fitness)
        new_population[worst_hummingbird_index] = LB + (UB - LB) * np.random.rand(dimension)
        new_fitness[worst_hummingbird_index] = -func(new_population[worst_hummingbird_index])

    return new_population, new_fitness

# --- Differential Evolution (DE) components ---
def DE_mutation(population):
    population_size, _ = population.shape
    mutant_vectors = np.zeros_like(population)
    for i in range(population_size):
        indices = [j for j in range(population_size) if j != i]
        a, b, c = random.sample(indices, 3)
        mutant_vectors[i] = population[a] + F * (population[b] - population[c])
    return mutant_vectors

def DE_crossover(population, mutant_vectors):
    global LB, UB
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
    selected_vectors = deepcopy(population)
    selected_fitness = deepcopy(fitness)
    for i in range(len(population)):
        trial_fitness = -func(trial_vectors[i])
        if trial_fitness < fitness[i]:
            selected_vectors[i] = trial_vectors[i]
            selected_fitness[i] = trial_fitness
    return selected_vectors, selected_fitness

# --- Main AHADE Algorithm ---
def AHADE(func):
    global Pop, FitPop, curIter
    use_de = curIter % initial_DE_frequency == 0
    if use_de:
        mutant_vectors = DE_mutation(Pop)
        trial_vectors = DE_crossover(Pop, mutant_vectors)
        Pop, FitPop = DE_selection(Pop, FitPop, trial_vectors, func)
    else:
        Pop, FitPop = AHA_foraging(Pop, FitPop, func)

def RunAHADE(func):
    global curIter, MaxIter, TrialRuns, Pop, FitPop, DimSize
    All_Trial_Best = []
    All_Best_Solutions = []
    for i in range(TrialRuns):
        Best_list = []
        curIter = 0
        
        np.random.seed(2022 + 88 * i)
        random.seed(2022 + 88 * i)

        initialize_population(func)
        Best_list.append(min(FitPop))
        
        while curIter < MaxIter:
            AHADE(func)
            curIter += 1
            Best_list.append(min(FitPop))
        
        All_Best_Solutions.append(Pop[np.argmin(FitPop)])
        All_Trial_Best.append(np.abs(Best_list))
        
    # Save results
    sol_path = f'./AHADE_Data/{SuiteName}/Sol'
    obj_path = f'./AHADE_Data/{SuiteName}/Obj'
    if not os.path.exists(sol_path): os.makedirs(sol_path)
    if not os.path.exists(obj_path): os.makedirs(obj_path)

    num_sensors = int(DimSize / 2)
    np.savetxt(f"{sol_path}/WSN_{num_sensors}.csv", All_Best_Solutions, delimiter=",")
    np.savetxt(f"{obj_path}/WSN_{num_sensors}.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global Pop, DimSize, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, DimSize))
    # Convert lists to NumPy arrays
    LB = np.array([0] * dim)
    UB = np.array([50] * dim)
    
    print(f"Running WSN Problem for {int(dim/2)} sensors...")
    RunAHADE(WSN_fit)


if __name__ == "__main__":
    if not os.path.exists('./AHADE_Data'):
        os.makedirs('./AHADE_Data')
        
    # Dimensions correspond to num_sensors * 2 (x,y coordinates)
    # 32, 42, 54 sensors as per HRCOA paper
    Dims = [64, 84, 108] 
    for dim in Dims:
        main(dim)