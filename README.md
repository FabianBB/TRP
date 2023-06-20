# Meta-Learning The Traveling Repairman Problem: Extending Existing Algorithms By Identifying And Targetting Difficult Instances.


The traveling repairman problem (TRP), also known
as the minimal latency problem, aims at finding a Hamiltonian
cycle such that the total latency is minimized. Many approaches
exist to solve the TRP, all of which have varied performance.
Here, the focus lies on two state-of-the-art meta-heuristics, as
well as an own implementation of an outdated meta-heuristic.
This paper presents a novel meta-learning approach for the
selection of algorithms based on machine learning. In essence a
decision tree is trained with instances for which the performance
of the set of algorithms is known a priori, followed by the metaalgorithm generating a prediction of which algorithm to run.
Each instance is described by meta-features that aim to capture
characteristics of the TRP. Additionally, the meta-algorithm is
used to create instances that benefit specific algorithms but not
the others. Considering that multiple algorithms may find the
optimal solution for an instance, the ties are broken based on
runtime. Results show significant performance improvement over
running a single algorithm, as well as good predictive power of
the meta-features.

Index Terms—Traveling Repairman, Minimum Latency Problem, Meta-algorithms, Meta-heuristics, Algorithm Selection
