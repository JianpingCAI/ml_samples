import numpy as np
from ypstruct import structure


def run(problem, params):

    # Problem information
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin # variable bounds
    varmax = problem.varmax

    # Parameters
    maxit = params.maxit
    npop = params.npop
    beta = params.beta
    pc = params.pc
    nc = int(np.round(pc*npop/2)*2) # number of children
    gamma = params.gamma # for crossover range
    mu = params.mu          # mutation rate
    sigma = params.sigma    # mutation step size

    # Empty individual template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    # Best solution ever found
    bestsol = empty_individual.deepcopy()
    bestsol.cost = np.inf

    # step1. Initialize population
    pop = empty_individual.repeat(npop)
    for i in range(npop):
        pop[i].position = np.random.uniform(varmin, varmax, nvar)
        pop[i].cost = costfunc(pop[i].position)

        if pop[i].cost < bestsol.cost:
            bestsol = pop[i].deepcopy()

    # Best cost of iterations - record the best cost of each iteration
    bestcost = np.empty(maxit)

    # step2. Main loop 
    for it in range(maxit):

        # cost of the current population - for parents selection based on cost
        # (higher probability for those with lower cost)
        costs = np.array([x.cost for x in pop])
        avg_cost = np.mean(costs)
        if avg_cost!=0:
            costs = costs/avg_cost
        probs = np.exp(-beta * costs) # probabilities of being selected as a parent

        popc = []  # children (next generation)
        for _ in range(nc//2): # generate two children per parents
            
            # step. Select parents

            # method1. random selection
            # q = np.random.permuation(npop)
            # p1 = pop[q[0]] # parent1
            # p2 = pop[q[1]] # parent2

            # method2. Perform Roulette wheel selection
            p1 =pop[roulette_wheel_selection(probs)] 
            p2 =pop[roulette_wheel_selection(probs)] 

            # step. Perform crossover --> get two children
            c1, c2 = crossover(p1, p2, gamma)

            # step. Perform muation
            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)

            # step. Apply bounds
            apply_bound(c1, varmin, varmax)
            apply_bound(c2, varmin, varmax)

            # step. Evaluate costs of new generation / offsprings
            c1.cost = costfunc(c1.position)
            c2.cost = costfunc(c2.position)
            # update best solution
            if c1.cost < bestsol.cost:
                bestsol = c1.deepcopy()
            if c2.cost < bestsol.cost:
                bestcol = c2.deepcopy()

            # Add offsprings to popc
            popc.append(c1)
            popc.append(c2)

        # step. Merge, Sort and Select
        pop += popc
        pop = sorted(pop, key=lambda x: x.cost)
        pop = pop[0:npop]

        # Store best cost
        bestcost[it] = bestsol.cost

        # Show iteration information
        print("Iteration {}: Best cost = {}".format(it, bestcost[it]))

    # Result/Output
    out = structure()
    out.pop=pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    return out          

def crossover(p1, p2, gamma):
    """
    Perform crossover of two parents
        p1, p2: parents
        gamma: for mutation range
        return: two children
    """
    c1 = p1.deepcopy()
    c2 = p1.deepcopy()
    alpha = np.random.uniform(-gamma, 1+gamma, *c1.position.shape)
    c1.position = alpha*p1.position + (1-alpha)*p2.position
    c1.position = alpha*p2.position + (1-alpha)*p1.position
    return c1, c2

def mutate(x, mu, sigma):
    """
    Perform mutation.
        x: original 
        mu: mutation rate
        sigma: 
    """
    y = x.deepcopy()
    flag = np.random.rand(*x.position.shape) <= mu
    ind = np.argwhere(flag) # indices that has True values
    y.position[ind] += sigma*np.random.randn(*ind.shape)
    return y


def apply_bound(x, varmin, varmax):
    """
    apply variable bounds
    """
    x.position = np.maximum(x.position, varmin)
    x.position = np.minimum(x.position, varmax)

def roulette_wheel_selection(p):
    """
    p: probabilities of being selected as parent
    """
    c = np.cumsum(p)
    r = sum(p) * np.random.rand()
    ind = np.argwhere(r<=c)
    return ind[0][0]


