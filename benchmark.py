import networkx as nx
import tsplib95

import DBMEA
import gilsrvnd
import grasp

instance = "line_10_3.tsp"

problem = tsplib95.load(instance)
print(problem.name)
graph = problem.get_graph()
print(graph)
dist_matrix = nx.to_numpy_matrix(graph)

grasp_sol, grasp_cost, grasp_time = grasp.run(instance)
print("GRASP")
print(grasp_sol)
print(grasp_cost)
print(grasp_time)

DBMEA_sol, DBMEA_cost, DBMEA_time = DBMEA.run(instance)
print("DBMEA")
print(DBMEA_sol)
print(DBMEA_cost)
print(DBMEA_time)


gilsrvnd_sol, gilsrvnd_cost, gilsrvnd_time = gilsrvnd.run(instance)
print("GILSRVND")
print(gilsrvnd_sol)
print(gilsrvnd_cost)
print(gilsrvnd_time)
