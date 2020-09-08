"""
Reference: 
https://yarpiz.com/632/ypga191215-practical-genetic-algorithms-in-python-and-matlab

"""

import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga

# example cost function 
# sphere test function
def sphere(x):
    return sum(x**2)

# Problem definition
problem = structure()
problem.costfunc = sphere
problem.nvar = 5 #number of variables
problem.varmin = -10 #variable bounds
problem.varmax = 10
# problem.varmin = [-10, -10, -1, -5,  4] #variable bounds
# problem.varmax = [ 10,  10,  1,  5, 10]

# GA parameters
params = structure()
params.maxit = 100  # max # of iterations
params.npop = 50    # size of population
params.beta = 1     # for parents selection
params.pc = 1       # ratio of next generation/children
params.gamma = 0.1  # for crossover
params.mu = 0.01    # mutation rate
params.sigma = 0.1  # mutation step size?

# Run GA
out = ga.run(problem, params)

# Results
plt.plot(out.bestcost)
plt.semilogy(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()