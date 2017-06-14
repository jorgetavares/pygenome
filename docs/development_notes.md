# Development Notes
These are some notes regarding the development of pygenome in general. 
Last update 2017/06/24.

## Design Philosophy
The main design decision so far has been in how to apply object-oriented programming. To keep things more simple, classes should only be used as containers (e.g., a kind of record) with the minimum amount of methods. Ideally not methods at all. To operate on objects we use only functions. This is losely inspired in CLOS, i.e., no class definition has the methods contained in the them. However, here functions are to be seen as functions and not really methods. 
**Update:** It seems an API closer to scikit-learn would be beneficial. Other Python libs also follow a similar approach, making a library/framework more familiar to users. It also might allow easier connection between libraries. 

The motivation behind this is to try to keep the design as close as possible to a functional design, try to avoid as much as possible hidden side-effects and making the structure/api more complex. However, it's possible that this can also lead to the opposite desired effects and standard object-oriented design ends up being the best choice. With the use of the library/framework in making the examples, as well as the current development in terms of adding new features, it will help to change/improvement current decisions. As such, this is a work in progress and could be naturally change in the future.

## Library/Framework
The goal of pygenome is to be a framework and a library. By this we mean that pygenome must have high-level functions and objects that allow easy use of all the main evolutionary algorithms, configured in the most standard way and that allows easy modification of know hyper-parameters. For example, how to use a standard GA where you only provide your fitness function and pick the parameters you want. 

Still, you should be able also to have a more fine-grained control of what to do. As such, pygenome should provide all the building blocks to code whatever algorithm you want. This is the library side. The examples folder should contain code examples of how to use the framework and how the same could be achieved in a library-style development. 

The framework will definetly be more constrained and, to some extent, limited. The library side should allow more open development.

## Random Generator and Sampling
At the moment, pygenome uses numpy.random as the provider for the prng (Mersenne Twister) and sampling distributions. The main reason is that numpy is used as the basis for the individuals representation, i.e., chromossomes/genomes, and numpy.random has more distributions than random.random (see [docs](https://docs.scipy.org/doc/numpy/reference/routines.random.html)). However, it is not thread-safe, meaning that when extending pygenome for multiprocessing it will not work properly. Fortunately, the thread-safety issue is solved by manually using [numpy.random.RandomState](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState) object (see this [SO](http://stackoverflow.com/a/5837352) answer). In light of this, it's better to not use diretly numpy.random and encapsulate it in a wrapper, e.g., pygenome.random. This way we can have more conveniente methods to deal with prng/sampling and even allow the possibility to change the library behind it without the need to recode the entire project.

## Multiprocessing and distributed computing
In the future, it's a wish that the algorithms are using multiprocessing as default in order to better use the many cores a CPU has. Current this is not done. This should be done in the most transparent/easy way for the user. 

Distributed computing should also be done to allow better scalability but also more complex algorithms based on coevolution or for neuroevolution. In principle this should be achieved using MPI.

## Neuroevolution
PyGenome must be a first-class library for Neuroevolution. Besides supporting the most popular methods like NEAT, it should also allow to connect easily to a external toolkit like Keras. This will allow to do Deep Neuroevolution. The main issues here will be how to do this bridge. How much can it be abstracted? How will concurrency/distributed will play out? Right now this is just an idea/goal.

## Backends
The main library to implement the core components is numpy. However, this can introduce some potential limitations, like, does not run on GPUs, less portable because of other underlying libraries, etc. Ideally, we would have the possibility to have multiple backends to run PyGenome, like pure python, where it would allow to use PyPy or something that does not rely on C-based libraries. Or PyTorch, and that way we could run easily on GPUs, etc.
