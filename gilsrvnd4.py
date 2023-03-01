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


# cost function
# assumes a return to depot at the end of the tour TODO: make this optional
# calculate the cost of a tour
# the cost is defined as the sum of first arrival times to each vertex
# the first arrival time to a vertex is the sum of the distances from the depot to the vertex
def cost(s, show=False):
    cost = 0
    for i in range(len(s) - 1):
        cost += (len(s)-i-1) * dist_matrix[s[i], s[i + 1]]
        print(cost) if show else None

    return cost


# define a method for GILS-RVND algorithm
# GILS-RVND brings together the components of GRASP, ILS and RVND to solve TRP (Traveling Repairman Problem)
# The method performs IMax iterations (lines 3–21),
# where in each of which an initial solution is generated using a
# greedy randomized procedure. The level of greediness is controlled
# by a parameter alpha, which is randomly chosen among the values of a
# given set R. Each initial solution is then improved by means of a
# RVND procedure combined with a perturbation mechanism in
# the spirit of ILS (lines 8–15), which is run until IILS consecutive perturbations without improvements are performed.
# It is important to mention that the perturbation is always performed on
# the best current solution sprime of a given iteration (acceptance criterion). Finally,
# the heuristic returns the best solution s* among all iterations.
def gilsrvnd(IMAX, IILS, R):
    fstar = math.inf
    for i in range(IMAX):
        alpha = random.choice(R)
        s = construct(alpha)
        sprime = s
        iterILS = 0
        while iterILS < IILS:
            s = RVND(s)
            if cost(s) < cost(sprime):
                sprime = s
                iterILS = 0
            s = Perturb(sprime)
            iterILS += 1
        if cost(sprime) < fstar:
            sstar = sprime
            fstar = cost(sprime)
    return sstar


# The constructive procedure, used to generate initial solutions, is described in Algorithm 2. Firstly,
# a partial solution s is initialized with a vertex associated to the depot , whereas a Candidate List (CL) is
# initialized with the remaining vertices. In the main loop , CL is sorted in ascending order according to the
# nearest neighbor criterion with respect to the last vertex added to s . A Restricted Candidate List (RCL)  is then
# built by considering only the alpha% best candidates of CL. Next, a customer is chosen at random from RCL and added
# to s. When the set of the alpha% best candidates is of size less than one or when a = 0, the algorithm chooses the
# best candidate. The constructive procedure terminates when all customers are added to s.

def construct(alpha):
    s = [0]
    n = len(dist_matrix)
    CL = list(range(1, n))
    r = 1
    while len(CL) > 0:
        # sort CL in ascending order according to distance to r
        CL.sort(key=lambda x: dist_matrix[r, x])
        # build RCL random choice from the alpha% best candidates
        cands = int(alpha * len(CL))
        if cands < 1:
            cands = 1
        RCL = CL[:cands]
        c = random.choice(RCL)
        s.append(c)
        r = c
        CL.remove(r)
    print(s)
    return s


# Improvement procedure with efficient move evaluations
# The local search is performed by a method based on RVND. Let t
# be the number of neighborhood structures and N = {N1, N2, N3,. . . , Nt} be their corresponding set.
# Whenever a given neighborhood of the set N fails to improve the current best solution, RVND
# randomly selects another neighborhood from the same set to continue the search.
# Preliminary tests revealed that this approach is
# capable of finding better solutions as compared to those that adopt a deterministic order.
# The set N is composed of the following five well-known TSP
# neighborhood structures, whose associated solutions are explored
# in an exhaustive fashion with a best improvement strategy.
# Swap—N(1)—Two customers of the tour are interchanged.
# 2-opt—N(2)—Two non-adjacent arcs are removed and another two are inserted in order to build a new feasible tour.
# Reinsertion—N(3)—One customer is relocated to another position of the tour.
# Or-opt2—N(4)—Two adjacent customers are reallocated to
# another position of the tour.
# Or-opt3—N(5)—Three adjacent customers are reallocated to
# another position of the tour.

def RVND(s):
    NL = ["swap", "two_opt", "reinsertion", "or_opt2", "or_opt3"]
    while len(NL) > 0:
        n = random.choice(NL)
        if n == "swap":
            print("swap")
            sprime = swap(s)
        elif n == "two_opt":
            print("two_opt")
            sprime = two_opt(s)
        elif n == "reinsertion":
            print("reinsertion")
            sprime = reinsertion(s)
        elif n == "or_opt2":
            print("or_opt2")
            sprime = or_opt2(s)
        elif n == "or_opt3":
            print("or_opt3")
            sprime = or_opt3(s)

        print(s)
        print(sprime)

        print("cost(s): ", cost(s))
        print("cost(sprime): ", cost(sprime))

        if cost(sprime) < cost(s):
            s = sprime
            NL = ["swap", "two_opt", "reinsertion", "or_opt2", "or_opt3"]
        else:
            NL.remove(n)

    return s


# Swap Two customers of the tour are interchanged.
# works
def swap(s):
    sprime = s.copy()
    i = random.randint(1, len(s) - 1)
    j = random.randint(1, len(s) - 1)
    sprime[i], sprime[j] = sprime[j], sprime[i]
    return sprime


# 2-opt Two non-adjacent arcs are removed and another two are inserted in order to build a new feasible tour.
# works
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
    sprime[i:j] = reversed(sprime[i:j])

    return sprime



# Reinsertion One customer is relocated to another position of the tour.
# works
def reinsertion(s):
    sprime = s.copy()
    i = random.randint(1, len(s) - 1)
    j = random.randint(1, len(s) - 1)
    c = sprime.pop(i)
    sprime.insert(j, c)
    return sprime


# Or-opt2 Two adjacent customers are reallocated to another position of the tour.
# works
def or_opt2(s):
    sprime = s.copy()
    i = random.randint(1, len(s) - 1)
    j = random.randint(1, len(s) - 1)

    c = sprime.pop(i)
    sprime.insert(j, c)

    c = sprime.pop(i)
    sprime.insert(j, c)
    return sprime


# Or-opt3 Three adjacent customers are reallocated to another position of the tour.
# TODO: doesnt work
def or_opt3(s):
    sprime = s.copy()

    n = len(sprime)
    i = random.randrange(1, n - 2)
    j = random.randrange(1, n - 2)

    #consecutive 3
    sub = sprime[i:i + 3]

    # remove 3
    lst = sprime[:i] + sprime[i + 3:]
    #insert
    sprime = lst[:j] + sub + lst[j:]

    return sprime


# Perturbation procedure
# When the local search fails to improve a solution s, a perturbation is applied over the best current solution sprime
# of the corresponding GILS-RVND iteration. The perturbation mechanism,
# called double-bridge, was originally developed by Martin, Otto, and Felten (1991) for the TSP.
# It consists in removing and re-inserting four arcs in such a way that a feasible tour is generated.
# This mechanism can also be seen as a permutation of two disjoint segments of a tour.
# example: s = [0, 1, 2, 3, 4, 5, 6, 7, 0]
# i = 2, j = 4, k = 6, l = 8
# remove the arcs (2,3) (4,5) (6,7) (0,1)
# insert the arcs (2,7) (3,6) (4,1) (5,0
def Perturb(s):
    sprime = s.copy()
    # make four slices
    slen = int(len(sprime) / 4)
    p1 = 1 + random.randrange(0, slen)
    p2 = p1 + 1 + random.randrange(0, slen)
    p3 = p2 + 1 + random.randrange(0, slen)
    # Combine 1st and 4th slice in order
    # Combine 3rd and 2nd slice in order
    # return the combination
    return sprime


# define params for gilsrvnd
IMAX = 10
IILS = min(100, len(dist_matrix))
R = [0, 0.01, 0.02, 0.05, 0.1, 0.25]

start_time = time.time()

sol = gilsrvnd(IMAX, IILS, R)

print("My program took", time.time() - start_time, "to run")
print(sol)
print(cost(sol, False))

