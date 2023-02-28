import math
import numpy as np
import random
import sys
import tsplib95
import networkx as nx

problem = tsplib95.load('Instances/att48.txt')
print(problem.name)
graph = problem.get_graph()
print(graph)
dist_matrix = nx.to_numpy_matrix(graph)

def get_distance_matrix(tsp_file):
    problem = tsplib95.load(tsp_file)
    nodes = list(problem.get_nodes())
    distances = {}
    for i in nodes:
        distances[i] = {}
        for j in nodes:
            distances[i][j] = problem.get_weight(i, j)
    return distances

dist_matrix = get_distance_matrix('Instances/att48.txt')

# Define the distance matrix

# Define the number of iterations for each temperature level
iters_per_temp = 5

# Define the initial temperature, cooling rate, and stopping criterion
init_temp = 100
cooling_rate = 0.9
min_temp = 0.001

# Define the GRASP construction parameters
alpha = 0.5
num_candidates = 5

# Define the RVND neighborhoods
neighborhoods = [lambda x: swap(x),
                 lambda x: insert(x),
                 lambda x: two_opt(x)]


# Define the swap neighborhood function
def swap(tour):
    n = len(tour)
    i, j = random.sample(range(n), 2)
    new_tour = np.copy(tour)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


# Define the insert neighborhood function
def insert(tour):
    n = len(tour)
    i, j = random.sample(range(n), 2)
    if i > j:
        i, j = j, i
    new_tour = np.concatenate((tour[:i], tour[i + 1:j + 1], tour[i:i + 1], tour[j + 1:]))
    return new_tour


# Define the 2-opt neighborhood function
def two_opt(tour):
    n = len(tour)
    i, j = random.sample(range(n), 2)
    if i > j:
        i, j = j, i
    new_tour = np.concatenate((tour[:i], tour[i:j + 1][::-1], tour[j + 1:]))
    return new_tour


# Define the objective function (total distance)
def objective(tour):
    n = len(tour)
    return sum(dist_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))


# Define the GRASP construction algorithm
def construct(alpha, num_candidates):
    n = dist_matrix.shape[0]
    candidates = []
    for i in range(num_candidates):
        remaining = set(range(n))
        remaining = list(remaining)

        tour = [random.sample(remaining, 1)[0]]
        remaining.remove(tour[0])
        while remaining:
            probs = dist_matrix[tour[-1], list(remaining)] ** -alpha
            probs /= sum(probs)
            next_city = list(remaining)[np.random.choice(range(len(remaining)), p=probs)]
            tour.append(next_city)
            remaining.remove(next_city)
        candidates.append(tour)
    return candidates


# Define the GILS-RVND algorithm
def gils_rvnd(dist_matrix, iters_per_temp, init_temp, cooling_rate, min_temp, alpha, num_candidates, neighborhoods):
    # Initialize the current solution and its objective value
    current_sol = np.arange(dist_matrix.shape[0])
    np.random.shuffle(current_sol)
    current_obj = objective(current_sol)
    # Initialize the best solution and its objective value
    best_sol = np.copy(current_sol)
    best_obj = current_obj
    # Initialize the temperature
    temp = init_temp
    # Main loop
    while temp > min_temp:
        # Iterate for the current temperature level
        for i in range(iters_per_temp):
            # Construct a set of candidate solutions using GRASP
            candidates = construct(alpha, num_candidates)
            # Evaluate the candidate solutions using the current solution as a reference
            candidate_objs = [objective(candidate) for candidate in candidates]
            # Select the best candidate solution
            best_candidate = candidates[np.argmin(candidate_objs)]
            best_candidate_obj = candidate_objs[np.argmin(candidate_objs)]
            # Perform RVND on the best candidate solution
            current_tour = np.copy(best_candidate)
            current_obj = best_candidate_obj
            while True:
                neighborhood = random.choice(neighborhoods)
                new_tour = neighborhood(current_tour)
                new_obj = objective(new_tour)
                if new_obj < current_obj:
                    current_tour = new_tour
                    current_obj = new_obj
                    if current_obj < best_candidate_obj:
                        best_candidate = current_tour
                        best_candidate_obj = current_obj
                else:
                    break
            # Accept the best candidate solution with a certain probability
            delta_obj = best_candidate_obj - current_obj
            if delta_obj < 0 or np.exp(-delta_obj / temp) > random.random():
                current_sol = best_candidate
                current_obj = best_candidate_obj
            # Update the best solution if necessary
            if current_obj < best_obj:
                best_sol = np.copy(current_sol)
                best_obj = current_obj
        # Decrease the temperature
        temp *= cooling_rate
    # Return the best solution found
    return best_sol, best_obj


# Run the GILS-RVND algorithm
best_sol, best_obj = gils_rvnd(dist_matrix, iters_per_temp, init_temp, cooling_rate, min_temp, alpha, num_candidates,
                               neighborhoods)

print('Best solution: {}'.format(best_sol))
print('Best objective: {}'.format(best_obj))

