import networkx as nx
import tsplib95

import DBMEA
import gilsrvnd
import grasp

problem = tsplib95.load('Instances/f10.tsp')
print(problem.name)
graph = problem.get_graph()
print(graph)
dist_matrix = nx.to_numpy_matrix(graph)

grasp_sol, grasp_cost, grasp_time = grasp.run("f10.tsp")
print("GRASP")
print(grasp_sol)
print(grasp_cost)

DBMEA_sol, DBMEA_cost, DBMEA_time = DBMEA.run("berlin52.tsp")
print("DBMEA")
print(DBMEA_sol)
print(DBMEA_cost)


gilsrvnd_sol, gilsrvnd_cost, gilsrvnd_time = gilsrvnd.run("berlin52.tsp")
print("GILSRVND")
print(gilsrvnd_sol)
print(gilsrvnd_cost)
