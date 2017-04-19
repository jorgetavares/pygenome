# Roadmap

This is an intial roadmap/wish list for pygenome.
Last update 2017/04/19.

* version 0.01
  + Main evolutionary paradigms: GA, ES, GP (+ STGP), GE and NEAT.
  + Representation: single chromossome only
  + Representation: binary, integer, permutation, floats, parse trees
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
  + Paradigms: Stack-GP (or PushGP), SGE, CFGGP, CMA-ES
  + Memetic Algorithms: allow the use of local search
  + Representation: dual chromossomes (problem + parameters)
  + visualization of evolution statistics
  + Full Units tests
  + More documentation

* version 0.03
  + Master-slave distribution
  + Paradigms: DE, NES
  + fitness scaling
  + Create documentation pages
  + Create tutorial jupyter notebooks

* version 0.04
  + Add other Neuroevolution methods
  + Bridge to DL toolkits

* version 0.05
  + TBD