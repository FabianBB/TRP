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

problem = tsplib95.load('Instances/berlin52.tsp')
print(problem.name)
graph = problem.get_graph()
print(graph)
dist_matrix = nx.to_numpy_matrix(graph)


# pseudo code of DBMEA
# 1. generate a population of n chromosomes randomoly to ensure uniform distribution and not getting stuck in local optima
# while terminal condition is not satisfied:
# bacterial mutation(pop,Nclones,Iseg)
# local serach operation
# Gene  transfer(Pop, Ninf, Itrans)
# if f(x1) < f(xbest):
# xbest = x1
# fbest = f(x1)
# endif
# end while
# return population

# bacterial mutation
# for each chromosome in the population:
# r = rand(0,1)
# p = pop[i]
# create Nclones of p
# if(r < 0.9):
# cut p into coherent segments with Iseg length
# else
# cut p into loose segments with Iseg length
# endif
# for each segment in the chromosome:
# choose an unmutated segment
# reverse the order of the vertices in the selected segment in the first clone
# randomly change the order of the vertices in the selected segment of the clones
# choose best among clones and p
# copy the segment of the best into the clones and p
# endfor
# replace pop[i] with the best among the clones and p
# endfor
# return pop

# createpop uniformly
def createPop(n):
    pop = []
    for i in range(n):
        pop.append(np.random.permutation(len(dist_matrix)))
    return pop


# fitness function
def fitness(s, show=False):
    cost = 0
    for i in range(len(s) - 1):
        cost += (len(s) - i - 1) * dist_matrix[s[i], s[i + 1]]
        print(cost) if show else None

    return 1 / cost


# cost of sol
def cost(s, show=False):
    cost = 0
    for i in range(len(s) - 1):
        cost += (len(s) - i - 1) * dist_matrix[s[i], s[i + 1]]
        print(cost) if show else None

    return cost


def coherentSeg(clones, i_seg):
    for clone in clones:
        # pick a random segment of length i_seg
        i = np.random.randint(0, len(clone) - i_seg)
        j = i + i_seg
        # if this is the first iteration then reverse the order of the segment
        if clone is clones[0]:
            clone[i:j] = clone[i:j][::-1]
        # else randomly change the order of the segment
        else:
            clone[i:j] = np.random.permutation(clone[i:j])
    return clones


def looseSeg(clones, i_seg):
    for clone in clones:
        # pick i_seg random indices
        indices = np.random.choice(len(clone), i_seg, replace=False)
        # if this is the first iteration then reverse the order of the segment
        if clone is clones[0]:
            clone[indices] = clone[indices][::-1]
        # else randomly change the order of the segment
        else:
            clone[indices] = np.random.permutation(clone[indices])
    return clones


def bestChrom(clones, p):
    best = p
    for clone in clones:
        if fitness(clone) > fitness(best):
            best = clone
    return best


def bacMutate(pop, n_clones, i_seg):
    for i in range(len(pop)):
        r = np.random.rand()
        p = pop[i]
        clones = []
        clones.append(p)
        for j in range(n_clones):
            clones.append(p.copy())

        if r <= 0.9:
            clones = coherentSeg(clones, i_seg)
        else:
            clones = looseSeg(clones, i_seg)

        pop[i] = bestChrom(clones, p)

    return pop


def two_opt(s):
    sprime = s.copy()


    n = len(sprime)
    i = random.randrange(0, n)
    j = random.randrange(0, n)

    # stop overshoot
    exclude = set([i])
    if i == 0:
        exclude.add(n - 1)
    else:
        exclude.add(i - 1)

    if i == n - 1:
        exclude.add(0)
    else:
        exclude.add(i + 1)

    while j in exclude:
        j = random.randrange(0, n)

    # ensure i < j
    if j < i:
        i, j = j, i

    # reverse
    #print(sprime[i:j])
    #print(sprime[i:j][::-1])
    #print(reversed(sprime[i:j]) )
    sprime[i:j] = sprime[i:j][::-1]

    return sprime

def localSearch(pop):
    for p in pop:
        # perform 2-opt minimum 3 times and maximum len(p) times
        #print(p)
        for i in range(np.random.randint(3, len(p))):
            cprime = cost(p)
            s = two_opt(p)
            c = cost(s)
            if c < cprime:
                p = s

    return pop


def geneTrans(pop, n_inf, i_trans):
    # sort pop
    pop = sorted(pop, key=fitness, reverse=True)
    # divide the population into two groups split in the middle called good and bad
    good = pop[:len(pop) // 2]
    bad = pop[len(pop) // 2:]

    # from 1 to n_inf
    for i in range(1, n_inf):
        # select random good chromosome called psource
        psource = good[np.random.randint(0, len(good))]
        #print("source", cost(psource))
        # select random bad chromosome called ptarget
        ptarget = bad[np.random.randint(0, len(bad))]
        #print("target", cost(ptarget))

        # select a random segment from psource with i_trans length
        a = np.random.randint(0, len(psource) - i_trans)
        b = i + i_trans
        segment = psource[a:b]

        # insert the segment into ptarget at a random position
        c = np.random.randint(0, len(ptarget))
        ptarget = np.insert(ptarget, c, segment)
        # eliminate duplicate numbers in ptarget
        ptarget = np.unique(ptarget)

    return pop


def dbmea(n_ind, n_clones, n_inf, i_seg, i_trans, maxIter=100):
    xbest = 0
    fbest = 0

    pop = createPop(n_ind)
    i = 0
    while cost(pop[0]) > 200000:
        pop = bacMutate(pop, n_clones, i_seg)
        pop = localSearch(pop)
        pop = geneTrans(pop, n_inf, i_trans)

        # sort pop by fitness
        pop = sorted(pop, key=fitness, reverse=True)

        # get min and max fitness individuals
        minf = min(pop, key=fitness)
        maxf = max(pop, key=fitness)
        #print("min", cost(minf))
        #print("max", cost(maxf))

        # update xbest and fbest

        if fitness(pop[0]) > fbest:
            xbest = pop[0]
            fbest = fitness(pop[0])

        i += 1
        #print(cost(pop[0]), i) if i % 10 == 0 else None

    return xbest, fbest

start_time = time.time()
xbest, fbest = dbmea(100, 5, 40, 3, 3, maxIter=100)
print("My program took", time.time() - start_time, "to run")
print(xbest)
print(fbest)
print(cost(xbest))