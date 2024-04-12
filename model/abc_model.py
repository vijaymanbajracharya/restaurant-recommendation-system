import numpy as np

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
    population = [np.random.uniform(lower_bound, upper_bound, D) for _ in range(num_bees)]
    
    # fitness of population at initialization
    fitness = [f(bee) for bee in population]

    best_idx = np.argmin(fitness)
    best_soln = population[best_idx]
    best_fitness = fitness[best_idx]

    # optimization
    for i in range(MAX_ITER):
        # employed bees
        for i in range(num_bees):

            random_candidate_idx = np.random.randint(0, num_bees)
            while random_candidate_idx == i:
                random_candidate_idx = np.random.randint(0, num_bees)

            # generate new candidate solution
            # refer 'https://en.wikipedia.org/wiki/Artificial_bee_colony_algorithm' for this formula
            new_soln = population[i] + np.random.uniform(-1, 1, size=D) * (population[i] - population[random_candidate_idx])
            
            new_soln = np.clip(new_soln, lower_bound, upper_bound)
            new_fitness = f(new_soln)

            # compare fitness with parent
            if new_fitness < fitness[i]:
                population[i] = new_soln
                fitness[i] = new_fitness

        # calculate probabilities
        fitness_sum = np.sum(fitness)
        prob = [fitness[idx]/fitness_sum for idx in range(num_bees)]

        # onlooker bees
        for i in range(num_bees):
            if np.random.uniform() < prob[i]:
                # generate neighborhood source and test its fitness
                neighborhood_source = np.random.randint(0, num_bees)
                while neighborhood_source == i:
                    neighborhood_source = np.random.randint(0, num_bees)

                # refer 'https://en.wikipedia.org/wiki/Artificial_bee_colony_algorithm' for this formula
                new_soln = population[i] + np.random.uniform(-1, 1, size=D) * (population[i] - population[neighborhood_source])

                new_soln = np.clip(new_soln, lower_bound, upper_bound)
                new_fitness = f(new_soln)

                # recruit onlooker bees to richer sources of food
                if new_fitness < fitness[i]:
                    population[i] = new_soln
                    fitness[i] = new_fitness

        # scout bees
        # TODO: incorporate abandonment criteria, right now random sources are abandoned based on a small probability
        # instead, it should track the number of times this source has failed to yield a positive outcome and then 
        # abandon the source if the number of attempts is too large (hyper param)
        for i in range(num_bees):
            if np.random.uniform() < abandonment_criteria:
                population[i] = np.random.uniform(lower_bound, upper_bound, D)
                fitness[i] = f(population[i])

        # update best solutions
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_soln = population[best_idx]
            best_fitness = fitness[best_idx]

    return best_soln, best_fitness

print(solve(f))
