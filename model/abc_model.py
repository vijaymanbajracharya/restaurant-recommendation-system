import numpy as np
from bee_class import Bee, OnlookerBee, EmployedBee, ScoutBee
import pdb

seed = 871623
np.random.seed(seed)

MAX_ITER = 1000

lower_bound = -2.0
upper_bound = 2.0
D = 2

# standard rosenbrock function
def f(x): 
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

"""ABC algorithm"""

def solve(f, num_bees = 50, abandonment_criteria = 0.1):
    # initialize the bees uniformly in the function space
    population = [Bee(np.random.uniform(lower_bound, upper_bound, D), f) for _ in range(num_bees)]

    # fitness of population at initialization
    for bee in population:
        bee.update_fitness()

    best_idx = np.argmin([bee.fitness for bee in population])
    best_soln = population[best_idx].position
    best_fitness = population[best_idx].fitness
    
    # optimization
    for i in range(MAX_ITER):
        # employed bees
        for i, bee in enumerate(population):
            # employ the bee population
            bee.__class__ = EmployedBee

            # idx of random neighboring candidate
            random_candidate_idx = np.random.randint(0, num_bees)
            while random_candidate_idx == i:
                random_candidate_idx = np.random.randint(0, num_bees)

            # generate new candidate solution
            # refer 'https://en.wikipedia.org/wiki/Artificial_bee_colony_algorithm' for this formula
            neighbor_bee = bee.neighbor(population[random_candidate_idx].position)
            new_fitness = neighbor_bee.fitness
            
            # compare fitness with parent
            if new_fitness < bee.fitness:
                population[i] = neighbor_bee

        # calculate probabilities
        fitness_sum = np.sum([bee.fitness for bee in population])
        prob = [bee.fitness/fitness_sum for bee in population]

        # onlooker bees
        for i, bee in enumerate(population):
            if np.random.uniform() < prob[i]:
                bee.__class__ = OnlookerBee
                # generate neighborhood source and test its fitness
                neighborhood_source = np.random.randint(0, num_bees)
                while neighborhood_source == i:
                    neighborhood_source = np.random.randint(0, num_bees)

                # refer 'https://en.wikipedia.org/wiki/Artificial_bee_colony_algorithm' for this formula
                neighbor_bee = bee.neighbor(population[neighborhood_source].position)
                new_fitness = neighbor_bee.fitness

                # recruit onlooker bees to richer sources of food               
                if new_fitness < bee.fitness:
                    population[i] = neighbor_bee

        # scout bees
        # TODO: incorporate abandonment criteria, right now random sources are abandoned based on a small probability
        # instead, it should track the number of times this source has failed to yield a positive outcome and then 
        # abandon the source if the number of attempts is too large (hyper param)
        for i, bee in enumerate(population):
            if np.random.uniform() < abandonment_criteria:
                bee.__class__ = ScoutBee
                population[i] = bee.neighbor()

        # update best solutions
        best_idx = np.argmin([bee.fitness for bee in population])
        if population[best_idx].fitness < best_fitness:
            best_soln = population[best_idx].position
            best_fitness = population[best_idx].fitness

    return best_soln, best_fitness

print(solve(f))
