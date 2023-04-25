import gurobipy as gp
import math
import numpy as np
import random
import sys
import tsplib95
import networkx as nx
import time

import gurobipy as gp
from gurobipy import GRB
DEBUG = False

problem = tsplib95.load('Instances/dantzig42.tsp')
print(problem.name)
graph = problem.get_graph()
print(graph)
dist_matrix = nx.to_numpy_matrix(graph)
print(dist_matrix)
print("")


# Define the distance matrix C, the number of nodes n and the positions of the nodes X and Y

# Create the model
model = gp.Model("Traveling Repairman Problem")

# Define the decision variables
n = len(dist_matrix)
X = model.addVars(n, n, vtype=GRB.BINARY, name="X")
Y = model.addVars(n, n, vtype=GRB.INTEGER, name="Y")

# Define the objective function
model.setObjective(gp.quicksum(dist_matrix[i, j] * Y[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

# Define the constraints
for i in range(n):
    model.addConstr(gp.quicksum(X[i, j] for j in range(n)) == 1)
for j in range(n):
    model.addConstr(gp.quicksum(X[i, j] for i in range(n)) == 1)
for i in range(1, n):
    model.addConstr(gp.quicksum(Y[i, 1] for i in range(1, n)) == 1)
model.addConstr(gp.quicksum(Y[i, 1] for i in range(1, n)) - gp.quicksum(Y[1, j] for j in range(1, n)) == 1 - n)
for k in range(2, n):
    model.addConstr(gp.quicksum(Y[i, k] for i in range(n)) - gp.quicksum(Y[k, j] for j in range(n)) == 1)
for i in range(2, n):
    model.addConstr(Y[i, 1] <= X[i, 1])
for j in range(2, n):
    model.addConstr(Y[1, j] <= n * X[1, j])
for i in range(2, n):
    for j in range(2, n):
        model.addConstr(Y[i, j] <= (n - 1) * X[i, j])

# Optimize the model
model.optimize()

# Print the optimal solution
if model.status == GRB.OPTIMAL:
    print("Optimal objective value:", model.objVal)
    for i in range(n):
        for j in range(n):
            if X[i, j].x > 0.5:
                print("X[", i, ",", j, "] =", X[i, j].x)
            if Y[i, j].x > 0:
                print("Y[", i, ",", j, "] =", Y[i, j].x)
else:
    print("No solution found.")
