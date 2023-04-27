import math
import numpy as np
import random
import sys
import tsplib95
import networkx as nx
import time


def run(instance):
    problem = tsplib95.load('Instances/' + instance)
    graph = problem.get_graph()
    dist_matrix = nx.to_numpy_matrix(graph)

    start_time = time.time()

    sol = grasp(0.1, dist_matrix)

    print("My program took", time.time() - start_time, "to run")

    return sol, cost(sol, dist_matrix), time.time() - start_time


def grasp(alpha, dist_matrix, lim=100):
    sstar = []
    coststar = math.inf
    niter = 0
    while niter < lim:
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
        # print(s)
        if (cost(s, dist_matrix) < coststar):
            sstar = s
            coststar = cost(s, dist_matrix)
        niter += 1
    return sstar


def cost(s, dist_matrix, show=False):
    cost = 0
    for i in range(len(s) - 1):
        cost += (len(s) - i - 1) * dist_matrix[s[i], s[i + 1]]

    print(cost) if show else None
    return cost
