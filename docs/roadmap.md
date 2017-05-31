# Roadmap

This is an intial roadmap/wish list for pygenome.
Last update 2017/05/31.

* version 0.1.x
  + Main evolutionary paradigms: GA, ES, GP (+ STGP) and GE
  + Genotype: single and dual chromossomes (for ES)
  + Genotype: binary, integer, permutation, floats, parse trees
  + Replacement: generational, steady-state, basic elitism, (mu,lambda), (mu+lambda)
  + Selection: roulette-wheel and tournament (including negative)
  + Selection: minimization only (maximization problems must convert)
  + Crossover and mutation: only minimum required set to have main EAs working
  + Fitness: single objective
  + Examples for all EAs
  + Basic logging and population statistics
  + Basic Unit tests

* version 0.2.x
  + Fitness: multi-objective (NSGA-2, SPEA2)
  + Paradigms: NEAT, SGE, CFGGP, CMA-ES
  + GP: add bloat control methods
  + GP: add additional tree creation algorithms
  + Memetic Algorithms: allow the use of local search
  + Fitness scaling
  + Visualization of evolution statistics
  + Full Units tests
  + More documentation

* version 0.3.x
  + Master-slave distribution
  + Island models
  + Sub-populations and species
  + Add co-evolution (competitive and collaborative, single and multiple population)
  + Paradigms: DE, NES, Stack-GP (or PushGP)
  + Create documentation pages
  + Create tutorial jupyter notebooks

* version 0.4.x
  + Add other Neuroevolution methods
  + Bridge to DL toolkits
  + add check-point saving and loading
  + GP: ADFs and ADMs

* version 0.5.x
  + TBD