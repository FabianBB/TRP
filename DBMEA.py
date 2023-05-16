import random
import time

import networkx as nx
import numpy as np
import numpy.random as npr
import tsplib95
import matplotlib.pyplot as plt

# Plots of city coords
do_visualisation = True
# Save the best solution to xbest.log
save_xbest = False
# candidate list for local search
use_candidates_list = True


def run(instance):
    problem = tsplib95.load('Instances/' + instance)
    graph = problem.get_graph()
    dist_matrix = nx.to_numpy_matrix(graph)

    n = len(graph.nodes)

    # city of corrds for visualizuation
    cities_matrix = np.zeros((n, 2))
    for i in range(0, n):
        # ensure numpy matrix
        cities_matrix[i] = problem.get_display(i + 1)

    if use_candidates_list:
        # Optimal number from the paper, speed/optimality tradeoff
        n_cl = int(np.sqrt(n))
    else:
        n_cl = n

    start_time = time.time()

    xbest, _ = dbmea(n_ind=100, n_clones=int(n / 15), n_inf=40, i_seg=int(n / 20), i_trans=int(n / 20),
                     n_cl=n_cl, dist_matrix=dist_matrix, cities_matrix=cities_matrix, maxIter=1)

    #print("My program took", time.time() - start_time, "to run")

    return xbest, cost(xbest, dist_matrix), time.time() - start_time


# createpop uniformly
def createPop(n, dist_matrix):
    # use numpy
    pop = np.zeros((n, len(dist_matrix)), dtype=int)
    # make 0 indexed
    for i in range(n):
        pop[i][1:len(dist_matrix)] = np.random.permutation(len(dist_matrix) - 1) + 1

    return pop


# cost of sol
def cost(s, dist_matrix):
    cost = sum((len(s) - i - 1) * dist_matrix[s[i], s[i + 1]] for i in range(len(s) - 1))
    return cost


# works
def fitness(s, dist_matrix):
    # i love code clarity
    c = cost(s, dist_matrix)
    return 1 / c


# cool datascience visualizations
def visualise(solution, cities, plot_title="TRP", filename="temp.png"):
    x = cities[:, 0]
    y = cities[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='red')

    # Plot the solution path
    solution_x = x[solution]
    solution_y = y[solution]
    plt.plot(solution_x, solution_y, 'b-', linewidth=0.5)

    # Connect the last first cities, should not be needed as we're doing TRP instead of TSP, but it looks nicer
    plt.plot([solution_x[-1], solution_x[0]], [solution_y[-1], solution_y[0]], 'b-', linewidth=0.5)

    # Label each point with city index
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(i), color='black', ha='center', va='center')

    # Set plot title and labels
    plt.title(plot_title)

    plt.savefig(filename)
    plt.close()


# works
def coherentSeg(clones, i_seg):
    for clone in clones:
        i = np.random.randint(0, len(clone) - i_seg)
        j = i + i_seg
        if clone is clones[0]:
            clone[i:j] = clone[i:j][::-1]
        else:
            clone[i:j] = np.random.permutation(clone[i:j])
    return clones


# works
def looseSeg(clones, i_seg):
    for clone in clones:
        indices = np.random.choice(len(clone), i_seg, replace=False)
        if clone is clones[0]:
            clone[indices] = clone[indices][::-1]
        else:
            clone[indices] = np.random.permutation(clone[indices])
    return clones


# works
def bestChrom(clones, p, dist_matrix):
    best = p
    for clone in clones:
        if fitness(clone, dist_matrix) > fitness(best, dist_matrix):
            best = clone
    return best


# works
def bacMutate(pop, n_clones, i_seg, dist_matrix):
    for i in range(len(pop)):
        r = np.random.rand()
        p = pop[i]
        clones = [p] + [p.copy() for _ in range(n_clones)]
        if r <= 0.9:
            clones = coherentSeg(clones, i_seg)
        else:
            clones = looseSeg(clones, i_seg)
        pop[i] = bestChrom(clones, p, dist_matrix)
    return pop


def two_opt_swap(s, i, j):
    # Perform a 2-opt swap between two edges in the solution
    # Referring to notations in the paper, A = s[i-1], B = s[i], C = s[j], D = s[j+1]
    # After swapping, the order will become ACBD
    new_solution = np.append(s[:i], np.flip(s[i:j + 1]))
    new_solution = np.append(new_solution, s[j + 1:])

    return new_solution


def two_opt_cl(s, dist_matrix, cl, n_cl, max_iterations=1000):
    n = len(dist_matrix)  # Use another variable to store number of nodes for better code clarity
    best_solution = s
    best_cost = cost(best_solution, dist_matrix)
    improved = True
    iteration = 0

    # Keep looping while it can still be improved
    # Ideally it should not need to be be limited by a max_iterations
    while improved and iteration < max_iterations:
        improved = False
        for i in range(1, n - 1):
            for j in range(n_cl):  # Loop through the closest neighbors
                # First convert j to corresponding city index in the current best_solution
                k = best_solution.tolist().index(cl[best_solution[i - 1], j])
                # If k is less than i+1, that means it should have been check
                if k < i + 1:
                    continue
                new_solution = two_opt_swap(best_solution, i, k)
                new_cost = cost(new_solution, dist_matrix)
                if new_cost < best_cost:
                    best_solution = new_solution
                    best_cost = new_cost
                    improved = True
        iteration += 1

    if iteration == max_iterations:
        print("Potential problem with two-opt implementation as it went upto max iterations %s" % iteration)
    return best_solution


def two_opt(s, dist_matrix, max_iterations=1000):
    n = len(dist_matrix)  # Use another variable to store number of nodes for better code clarity
    best_solution = s
    best_cost = cost(best_solution, dist_matrix)
    improved = True
    iteration = 0

    # Keep looping while it can still be improved
    # Ideally it should not need to be be limited by a max_iterations
    while improved and iteration < max_iterations:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_solution = two_opt_swap(best_solution, i, j)
                new_cost = cost(new_solution, dist_matrix)
                if new_cost < best_cost:
                    best_solution = new_solution
                    best_cost = new_cost
                    improved = True
        iteration += 1

    if iteration == max_iterations:
        print("Potential problem with two-opt implementation as it went upto max iterations %s" % iteration)
    return best_solution


# Change the path of solution
def three_opt_swap(s, i, j, k, alt):
    # Using the 4 reconnecting methods as outlined in the paper
    # the points are: A, B, C, D, E, F = s[i-1], s[i], s[j-1], s[j], s[k-1], s[k]
    # All start from point A and end at point F

    # Original path: AB CD EF
    current_solution = s

    # np append is so weird, maybe use list instead
    if alt == 0:  # Alternative 1: A CB ED F
        current_solution = np.append(s[:i], np.flip(s[i:j]))
        current_solution = np.append(current_solution, np.flip(s[j:k]))
        current_solution = np.append(current_solution, s[k:])

    elif alt == 1:  # Alternative 2: A DE BC F
        current_solution = np.append(s[:i], s[j:k])
        current_solution = np.append(current_solution, s[i:j])
        current_solution = np.append(current_solution, s[k:])

    elif alt == 2:  # Alternative 3: A DE CB F
        current_solution = np.append(s[:i], s[j:k])
        current_solution = np.append(current_solution, np.flip(s[i:j]))
        current_solution = np.append(current_solution, s[k:])

    elif alt == 3:  # Alternative 4: A ED BC F
        current_solution = np.append(s[:i], np.flip(s[j:k]))
        current_solution = np.append(current_solution, s[i:j])
        current_solution = np.append(current_solution, s[k:])

    return current_solution


# Iterative improvement based on 3 exchange
def three_opt_cl(s, dist_matrix, cl, n_cl, max_iterations=1000):
    n = len(s)
    best_solution = s
    best_cost = cost(best_solution, dist_matrix)
    current_solution = s
    current_cost = best_cost

    improved = True
    iteration = 0

    # maybe remove maxiter since only 1 anyways
    while improved and iteration < max_iterations:
        improved = False
        for i in range(1, n - 4):
            for j in range(n_cl):  # loop through cl of node i
                for k in range(n_cl):  # loop through cl of node j
                    # First convert j and k to corresponding city index in the current best_solution
                    x = best_solution.tolist().index(cl[best_solution[i - 1], j])
                    y = best_solution.tolist().index(cl[best_solution[j - 1], k])
                    if x < i + 2 or y < x + 2:
                        continue
                    for alt in range(4):
                        new_solution = three_opt_swap(best_solution, i, x, y, alt=alt)
                        new_cost = cost(new_solution, dist_matrix)

                        if new_cost < current_cost:
                            current_solution = new_solution
                            current_cost = new_cost
                            improved = True
                    # Update the best sol
                    best_solution = current_solution
                    best_cost = current_cost
        iteration += 1
        # uncomment to see process, 3opt is super slow
        # print("Three opt at iteration %s best_cost %s " % (iteration, best_cost))

    if iteration == max_iterations:
        print("PROBLEM, MAX ITER REACHED")

    return best_solution


# Iterative improvement based on 3 exchange.
def three_opt(s, dist_matrix, max_iterations=1000):
    n = len(s)
    best_solution = s
    best_cost = cost(best_solution, dist_matrix)
    current_solution = s
    current_cost = best_cost

    improved = True
    iteration = 0


    while improved and iteration < max_iterations:
        improved = False
        for i in range(1, n):
            for j in range(i + 2, n):
                for k in range(j + 2, n):
                    for alt in range(4):
                        new_solution = three_opt_swap(best_solution, i, j, k, alt=alt)
                        new_cost = cost(new_solution, dist_matrix)

                        if new_cost < current_cost:
                            current_solution = new_solution
                            current_cost = new_cost
                            improved = True

                    best_solution = current_solution
                    best_cost = current_cost
        iteration += 1

        # print("Three opt at iteration %s best_cost %s " % (iteration, best_cost))

    if iteration == max_iterations:
        print("Problem, max iterations reached")

    return best_solution


def localSearch(pop, dist_matrix, cl, n_cl):
    # CL to speedup local search but sacrifices some accuracy
    if use_candidates_list:
        for i in range(len(pop)):
            # 2opt first
            pop[i] = two_opt_cl(pop[i], dist_matrix, cl, n_cl)
            # tthree_opt optimization
            # print("3opt search for pop[%s]..." % i)
            pop[i] = three_opt_cl(pop[i], dist_matrix, cl, n_cl)
    else:
        for i in range(len(pop)):
            # 2opt
            pop[i] = two_opt(pop[i], dist_matrix)
            # 3opt

            # print("Performing three-opt search for pop[%s]..." % i)
            pop[i] = three_opt(pop[i], dist_matrix)

    # print("Finish local search")
    return pop


def geneTrans(pop, n_inf, i_trans, dist_matrix):
    pop = sorted(pop, key=lambda x: fitness(x, dist_matrix), reverse=True)
    good = pop[:len(pop) // 2]
    bad = pop[len(pop) // 2:]

    for _ in range(n_inf):

        psource = random.choice(good)
        ptarget = random.choice(bad)

        a = np.random.randint(0, len(psource) - i_trans)
        b = a + i_trans

        segment = psource[a:b]
        c = np.random.randint(0, len(ptarget))

        ptarget = np.insert(ptarget, c, segment)
        ptarget = np.unique(ptarget)

    return pop


# Generate a matrix of candidate lists consisting of n_cl closest neighbors for each city
def generate_cl(dist_matrix, n_cl):
    cl = np.empty((0, n_cl), dtype=int)
    n = len(dist_matrix)
    for i in range(n):
        distances = dist_matrix[i, :]
        neighbor_indices = np.argsort(distances)
        neighbor_indices = neighbor_indices[0, 1:n_cl + 1]
        cl = np.append(cl, neighbor_indices, axis=0)

    return cl


def dbmea(n_ind, n_clones, n_inf, i_seg, i_trans, n_cl, dist_matrix, cities_matrix, maxIter=100):
    pop = createPop(n_ind, dist_matrix)
    xbest = pop[0]
    fbest = fitness(xbest, dist_matrix)
    cl = []

    if do_visualisation:
        c = cost(xbest, dist_matrix)
        ptitle = "TRP Cost=" + str(c)
        visualise(xbest, cities_matrix, ptitle, "DBMEA_before.png")

    if use_candidates_list:
        cl = generate_cl(dist_matrix, n_cl)

    for i in range(maxIter):
        pop = bacMutate(pop, n_clones, i_seg, dist_matrix)
        pop = localSearch(pop, dist_matrix, cl, n_cl)
        pop = geneTrans(pop, n_inf, i_trans, dist_matrix)
        pop = sorted(pop, key=lambda x: fitness(x, dist_matrix), reverse=True)

        if fitness(pop[0], dist_matrix) > fbest:
            xbest = pop[0]
            fbest = fitness(pop[0], dist_matrix)

        #print("cost best", cost(pop[0], dist_matrix), i)  # if i % 10 == 0 else None
        # print("cost worst", cost(pop[90], dist_matrix), i) if i % 10 == 0 else None

    if do_visualisation:
        c = cost(pop[0], dist_matrix)
        ptitle = "TRP Cost=" + str(c)
        visualise(xbest, cities_matrix, ptitle, "DBMEA_after.png")

    if save_xbest:
        with open("xbest.log", 'w') as f:
            print(xbest, file=f)

    return xbest, fbest

# Verifying the best cost obtained from one of the run
# a = [0, 21, 31, 48, 35, 34, 38, 39, 37, 4, 14, 5, 23, 47, 36, 33, 43, 45, 15, 49, 19, 22, 29, 1, 6, 41, 20, 30, 17, 2, 44, 18, 40, 7, 8, 9, 42, 3, 24, 11, 27, 26, 25, 46, 13, 12, 51, 10, 50, 28, 16, 32,]
# problem = tsplib95.load('Instances/' + instance)
# graph = problem.get_graph()
# dist_matrix = nx.to_numpy_matrix(graph)
# c = cost(a, dist_matrix)
# print("Cost ", c)
