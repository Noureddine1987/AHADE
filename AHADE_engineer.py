import os
from copy import deepcopy
from enoppy.paper_based.pdo_2022 import *
import numpy as np
import random

# --- Parameters based on HRCOA for fair comparison ---
PopSize = 100
DimSize = 10  # Initial value, will be updated in main
LB = [-100] * DimSize
UB = [100] * DimSize

TrialRuns = 30
MaxFEs = 20000  # As per HRCOA paper for Engineering problems
curFEs = 0

MaxIter = int(MaxFEs / PopSize)
curIter = 0

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

FuncNum = "" # Will be the problem name
SuiteName = "Engineer"

# --- DE Parameters (as provided) ---
F = 0.8  # DE mutation factor
CR = 0.9 # DE crossover rate

# --- AHA-DE Hybridization Parameters (as provided) ---
initial_DE_frequency = 10
diversity_threshold = 0.5
convergence_threshold = 1e-6

def initialize_population(func):
    """
    Initializes the population for a given engineering problem.
    """
    global Pop, FitPop, curFEs, DimSize, LB, UB
    DimSize = func.n_dims
    LB, UB = func.lb, func.ub
    Pop = np.zeros((PopSize, DimSize))
    
    for i in range(PopSize):
        Pop[i] = LB + (UB - LB) * np.random.rand(DimSize)
        FitPop[i] = func.evaluate(Pop[i])
        curFEs += 1

# --- Artificial Hummingbird Algorithm (AHA) components ---
def update_visit_table(visit_table):
    return visit_table + 1

def AHA_foraging(population, fitness, func, visit_table):
    global curFEs, DimSize, MaxIter, curIter, LB, UB
    
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
        mutant_fitness = func.evaluate(mutant)
        curFEs += 1

        if mutant_fitness < new_fitness[i]:
            new_population[i] = mutant
            new_fitness[i] = mutant_fitness

        local_target_index = i
        local_target = population[local_target_index]
        
        mutant = population[i] + np.random.normal(0, 1, dimension) * (local_target - population[i])
        mutant = np.clip(mutant, LB, UB)
        mutant_fitness = func.evaluate(mutant)
        curFEs += 1

        if mutant_fitness < new_fitness[i]:
            new_population[i] = mutant
            new_fitness[i] = mutant_fitness

    if curIter % (2 * dimension) == 0:
        worst_hummingbird_index = np.argmax(new_fitness)
        new_population[worst_hummingbird_index] = LB + (UB - LB) * np.random.rand(dimension)
        new_fitness[worst_hummingbird_index] = func.evaluate(new_population[worst_hummingbird_index])
        curFEs += 1

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
    global Pop, FitPop, curFEs, curIter
    use_de = curIter % initial_DE_frequency == 0
    if use_de:
        mutant_vectors = DE_mutation(Pop)
        trial_vectors = DE_crossover(Pop, mutant_vectors)
        Pop, FitPop = DE_selection(Pop, FitPop, trial_vectors, func)
    else:
        visit_table = np.zeros(PopSize)
        Pop, FitPop = AHA_foraging(Pop, FitPop, func, visit_table)

def RunAHADE(func):
    global curFEs, curIter, MaxFEs, TrialRuns, Pop, FitPop, DimSize, MaxIter, SuiteName
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curFEs = 0
        curIter = 0
        
        np.random.seed(2022 + 88 * i)
        random.seed(2022 + 88 * i)

        initialize_population(func)
        Best_list.append(min(FitPop))
        
        while curFEs < MaxFEs:
            AHADE(func)
            curIter += 1
            if curFEs >= MaxFEs:
                break
            Best_list.append(min(FitPop))
        
        while len(Best_list) < MaxIter + 1:
            Best_list.append(Best_list[-1])

        All_Trial_Best.append(Best_list[:MaxIter+1])
        
    if not os.path.exists(f'./AHADE_Data/{SuiteName}'):
        os.makedirs(f'./AHADE_Data/{SuiteName}')
    np.savetxt(f"./AHADE_Data/{SuiteName}/{FuncNum}.csv", All_Trial_Best, delimiter=",")


def main():
    global FuncNum, DimSize, Pop, MaxFEs, MaxIter
    
    Probs = [WBP(), PVP(), CSP(), SRD(), TBTD(), GTD(), CBD(), IBD(), TCD(), PLD(), CBHD(), RCB()]
    Names = ["WBP", "PVP", "CSP", "SRD", "TBTD", "GTD", "CBD", "IBD", "TCD", "PLD", "CBHD", "RCB"]
    
    for i in range(len(Probs)):
        FuncNum = Names[i]
        print(f"Running Engineering Problem: {FuncNum}...")
        RunAHADE(Probs[i])


if __name__ == "__main__":
    if not os.path.exists('./AHADE_Data'):
        os.makedirs('./AHADE_Data')
    main()
