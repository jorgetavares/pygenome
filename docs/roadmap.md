# Roadmap

This is an intial roadmap/wish list for pygenome.
Last update 2017/04/20.

* version 0.01
  + Main evolutionary paradigms: GA, ES, GP (+ STGP) and GE
  + Genotype: single and dual chromossomes
  + Genotype: binary, integer, permutation, floats, parse trees
  + Replacement: generational, steady-state, basic elitism, mu,lambda, mu+lambda
  + Selection: roulette-wheel and tournament (including negative)
  + Selection: minimization only (maximization problems must convert)
  + Crossover and mutation: only minimum required set to have main EAs working
  + Fitness: single objective
  + Examples for all EAs (framework and library)
  + Basic logging and population statistics
  + Basic Unit tests

* version 0.02
  + Fitness: multi-objective (NSGA-2, SPEAR)
  + Paradigms: NEAT, SGE, CFGGP, CMA-ES
  + GP: add bloat control methods
  + Memetic Algorithms: allow the use of local search
  + Visualization of evolution statistics
  + Full Units tests
  + More documentation

* version 0.03
  + Master-slave distribution
  + Add co-evolution (competitive and collaborative)
  + Paradigms: DE, NES, Stack-GP (or PushGP)
  + Fitness scaling
  + Create documentation pages
  + Create tutorial jupyter notebooks

* version 0.04
  + Add other Neuroevolution methods
  + Bridge to DL toolkits

* version 0.05
  + TBD