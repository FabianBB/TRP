# Solving The Traveling Repairman Problem: Extending Existing Algorithms By Identifying And Targetting Difficult Instances.


There are multiple formulations for the traveling repairman problem (TRP), also known as minimal latency problem
(MLP). In all formulations the input is a graph with edge lengths (which can be viewed as the time to traverse the
edge), and goal is to visit all vertices of the graph such that the sum of first arrival times is minimized. The first arrival
time of a vertex is the first time that it is visited i.e. the sum of edge lengths that are traversed before the vertex is
reached for the first time. In some formulations the TRP problem is defined as the problem of finding a hamiltonian
path/circuit that minimizes waiting times, whilst others set no limit to the times a vertex can be visited.


This assignment will consider the latter, thus, given an edge-weighted graph TRP asks for a hamiltonian walk that
minimizes the sum of first arrival times for all vertices. TRP being a notorious NP-hard problem makes solving it in
"reasonable" time non-trivial. The first goal of this assignment is to compare different state of the art approaches to
solving TRP, such as different ILP formulations, exact algorithms (e.g. branch & bound) and metaheuristics (e.g.
GRASP, Discrete Bacterial Memetic Evolutionary Algorithm (DBMEA)). The second goal to rival or improve the state
of the art approaches to solving TRP on restricted instances of the problem that all existing solvers seem to struggle
on. The third goal is, time permitting, to produce extra fast exact algorithms for certain special restricted instances,
such as caterpillars, and to determine whether these can be used to speed up general algorithms for TRP (which
might encounter such special graphs frequently as subinstances).


The different approaches will be compared regarding the runtime needed to find the exact solution. If the exact
solutions are unknown an exact method such as ILP or BB will precompute (bounds on) the solution and the
runtimes will be tested after
