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
import DBMEA
import grasp
import gilsrvnd

problem = tsplib95.load('Instances/f10.tsp')
print(problem.name)
graph = problem.get_graph()
print(graph)
dist_matrix = nx.to_numpy_matrix(graph)

grasp_sol, grasp_cost, grasp_time = grasp.run("berlin52.tsp")
print(grasp_sol)

DBMEA_sol, DBMEA_cost, DBMEA_time = DBMEA.run("f10.tsp")
print(DBMEA_sol)


gilsrvnd_sol, gilsrvnd_cost, gilsrvnd_time = gilsrvnd.run("berlin52.tsp")
print(gilsrvnd_sol)
