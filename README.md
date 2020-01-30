# Single- and Multi-Objective Particle Swarm Optimization of Echo State Network architecture

> Echo State Networks ESNs are specific kind of recurrent networks providing a black box modeling of dynamic
non-linear problems. Their architecture is distinguished by a randomly recurrent hidden infra-structure called dynamic reservoir.
Coming up with an efficient reservoir structure depends mainly on selecting the right parameters including the number of neurons
and connectivity rate within it. Despite expertise and repeatedly tests, the optimal reservoir topology is hard to be determined in
advance. Topology evolving can provide a potential way to define a suitable reservoir according to the problem to be modeled. 
This last can be mono- or multi-constrained. 
Throughout this code, a mono-objective as well as a multi-objective particle swarm optimizations are applied to ESN to provide a set of
optimal reservoir architectures. Both accuracy and complexity of the network are considered as objectives to be optimized during
the evolution process.

## MOPSO Particles
Each particle in the swarm is composed of the reservoir size (the number of neurons), the reservoir connectivity rate (rate of non-zero connection weights), the input connectity rate and the feedback connectivity rate.

The Objectives to be minimized:

 (*) Mono-objective PSO: the precision (error between desired and network output).
 
 (*) Bi-objective PSO: the precision reservoir connectivity rate.
 
 (*) Tri-objective PSO: the precision + reservoir connectivity rate+ reservoir size.
 
### Getting started
The implemented code is designed for mono-, bi- and tri-objective PSO optimization of ESN structure.

Run the script with Matlab: main.m. 
The script main.m includes:

A tri-objective PSO of ESN architecture  (mopsoSize). (the obtained figure is like fig5 in the paper). 

A bi-objective case of ESN architecture (mopsoBiobj). (the obtained figure is like fig4 in the paper). 
.

A mono-objective PSO of ESN architecture (pso).