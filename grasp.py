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

def grasp(alpha, lim=100):
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
        #print(s)
        if(cost(s) < coststar):
            sstar = s
            coststar = cost(s)
        niter += 1
    return sstar

def cost(s, show=False):
    cost = 0
    for i in range(len(s) - 1):
        cost += (len(s)-i-1) * dist_matrix[s[i], s[i + 1]]

    print(cost) if show else None
    return cost

sol = grasp(0.1)
print(sol)
print(cost(sol))