import numpy as np
import random
from time import sleep
import numpy.random as npr
import math
import numpy as np
import random
import sys
import tsplib95
import networkx as nx
import time

from grasp import grasp

DEBUG = False



problem = tsplib95.load('Instances/berlin52.tsp')
print(problem.name)
graph = problem.get_graph()
print(graph)
dist_matrix = nx.to_numpy_matrix(graph)

# Genetic Algo

# create a population of n chromosomes, the chromosome being a list of n indices representing the cities generated
# by generatePoints
def createPop(n):
    pop = []

    # generate permutations of the indices
    for i in range(n):
        # if random val < 0.5, append a random permutation of the indices
        # else use greedy strategy
        if np.random.rand() < 0.5:
            pop.append(np.random.permutation(len(dist_matrix)))
        else:
            pop.append(grasp(0, lim=1))



    return pop


def cost(s, show=False):
    cost = 0
    for i in range(len(s) - 1):
        cost += (len(s)-i-1) * dist_matrix[s[i], s[i + 1]]
        print(cost) if show else None

    return cost


def fitness(s, show=False):
    cost = 0
    for i in range(len(s) - 1):
        cost += (len(s)-i-1) * dist_matrix[s[i], s[i + 1]]
        print(cost) if show else None

    return 1/cost


# # select N/2 parents using proportional selection
# def selectParents(pop):
#     print(pop)
#     # calculate population fitness
#     total = 0
#     for c in pop:
#         total += fitness(c)
#
#     # calc prob of each chromomsome
#     probs = []
#     for c in pop:
#         probs.append(fitness(c) / total)
#         print(probs) if DEBUG else None
#
#     print(sum(probs)) if DEBUG else None
#     print(probs) if DEBUG else None
#     print(pop) if DEBUG else None
#
#     # select N/2 parents
#     parents = random.sample(pop, k=(max(50, int(len(pop) / 2))))
#     return parents

# function above didnt work quite well so i slightly modified this one
# from https://stackoverflow.com/questions/10324015/fitness-proportionate-selection-roulette-wheel-selection-in-python
def selectParents(population):
    totalf = np.sum([fitness(c) for c in population])
    selection_probs = [(fitness(c) / totalf) for c in population]
    #print(selection_probs)
    idx = npr.choice(len(population), p=selection_probs, replace=False, size=max(50, int(len(population) / 2)))
    return [population[i] for i in idx]


# apply mutation with small chance
# swapping 2 indices is nice since it does not change the solution drastically but may perturb the solution enough to
# create better offspring in the future
# this comes from research I did on TRP
def mutate(c):
    if np.random.rand() < 0.05:
        # swap two random indices
        i = np.random.randint(0, len(c))
        j = np.random.randint(0, len(c))
        c[i], c[j] = c[j], c[i]
    return c


def cx2(p1, p2):
    # o1 = []
    # o2 = []
    #
    # # select first bit from other parent
    # o1.append(p2[0])
    # o2.append(p1[0])
    #
    # while len(o1) < len(p1):
    #     # find index of last bit in o1 in p1
    #     index = np.where(p1 == o1[-1])
    #
    #     # add bit at index in p2 to o1
    #     o1.append(p2[index])
    #     print(o1) if DEBUG else None
    #
    # while len(o2) < len(p2):
    #     index = np.where(p2 == o2[-1])
    #
    #     o2.append(p1[index])
    #     print(o2) if DEBUG else None
    #
    # return p1, p2

    # take first half from p1 and second half from p2
    # o1 = np.concatenate((p1[:int(len(p1) / 2)], p2[int(len(p2) / 2):]))
    # o2 = np.concatenate((p2[:int(len(p2) / 2)], p1[int(len(p1) / 2):]))

    o1 = p1[0:len(p1) // 2]
    o2 = p2[0:len(p2) // 2]

    for val in p2:

        if not val in o1:
            o1 = np.concatenate((o1, [val]))

    for val in p1:
        if not val in o2:
            o2 = np.concatenate((o2, [val]))

    return o1, o2


# genetic algo
def GA(pop, ngen):
    minF = 100000
    for i in range(ngen):
        # select parents
        parents = selectParents(pop)

        # create offspring
        offspring = []
        for i in range(0, len(parents) - 2, 2):
            # crossover
            o1, o2 = cx2(parents[i], parents[i + 1])

            # mutate
            o1 = mutate(o1)
            o2 = mutate(o2)

            offspring.append(o1)
            offspring.append(o2)

        # replace old population with offspring
        pop = offspring
        # duplicate every chromosome in population
        pop = pop + pop

        print(pop) if DEBUG else None
        # print average fitness of population
        print("Average fitness: ", sum([fitness(c) for c in pop]) / len(pop))
        # keep track of minimum fitness
        if min([fitness(c) for c in pop]) < minF:
            minF = min([fitness(c) for c in pop])
            maxF = max([fitness(c) for c in pop])
            print("New min: ", minF)

    #print minF
    print("Min fitness: ", 1/minF, " at ", minF)
    print("Max fitness: ", 1/maxF, " at ", maxF)
    # find best chromosome
    best = pop[0]
    for c in pop:
        if fitness(c) < fitness(best):
            best = c

    return best




# run GA
pop = createPop(100)


best = GA(pop, 500)
print("best: ", best)
print(fitness(best))
print(cost(best))