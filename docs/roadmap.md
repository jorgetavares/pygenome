# Roadmap

This is an intial roadmap/wish list for pygenome.
Last update 2017/04/02.

* version 0.01

  + Main evolutionary paradigms: GA, ES, GP (+ STGP), GE and NEAT.
  + Representation: single chromossome only
  + Representation: binary, integer, permutation, floats, parse trees
  + Replacement: generational, steady-state, basic elitism, mu,lambda, mu+lambda
  + Selection: oulette-wheel and tournament
  + Selection: minimization only (maximization problems must convert)
  + Crossover and mutation: only minimum required set to have main EAs working
  + Fitness: single objective
  + Examples for all EAs (framework and library)
  + Basic logging and population statistics
  + Unit tests

* version 0.02

  + Fitness: multi-objective (NSGA-2, SPEAR)
  + ES: add CMA-ES, NES
  + Paradigm: DE
  + Memetic Algorithms: allow the use of local search
  + Representation: dual chromossomes (problem + parameters)
  + Grammar-based approaches: add SGE, CFGGP

* version 0.03

  + Add other Neuroevolution methods
  + Bridge to DL toolkits

* version 0.04

  + TBD