import networkx as nx
import tsplib95

import DBMEA
import gilsrvnd
import grasp

instance = "mod_kroA100.tsp"

problem = tsplib95.load('Instances/' + instance)
print(problem.name)
graph = problem.get_graph()
print(graph)
dist_matrix = nx.to_numpy_matrix(graph)

grasp_sol, grasp_cost, grasp_time = grasp.run(instance)
print("GRASP")
print(grasp_sol)
print(grasp_cost)

DBMEA_sol, DBMEA_cost, DBMEA_time = DBMEA.run(instance)
print("DBMEA")
print(DBMEA_sol)
print(DBMEA_cost)


gilsrvnd_sol, gilsrvnd_cost, gilsrvnd_time = gilsrvnd.run(instance)
print("GILSRVND")
print(gilsrvnd_sol)
print(gilsrvnd_cost)
