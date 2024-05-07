import numpy as np
from bee_class import Bee, OnlookerBee, EmployedBee, ScoutBee
import pdb
import math

seed = 871623
np.random.seed(seed)

MAX_ITER = 1000

lower_bound = -2.0
upper_bound = 2.0
D = 2

# standard rosenbrock function
def f(x): 
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def ackley(x): # returns a scalar
    """
    The objective function.
    Implements the Ackley algorithm.
    :param x: The 1xD array to calculate with.
    :return: The cost of the point x.
    """
    sum = 0
    leftSum = 0
    rightSum = 0
    for i in range(D):
        leftSum += pow(x[i], 2)
        rightSum += math.cos(2 * math.pi * x[i])

    sum += -20 * math.exp(-0.02 * math.sqrt((1/D) * leftSum)) - math.exp((1/D) * rightSum) + 20 + math.e

    return sum

def eggcrate(x):  # returns a scalar
    """
    The objective function.
    Implements the Eggcrate Function.
    :param x: The 1xD array to calculate with.
    :return: The cost of the point x.
    """
    return pow(x[0], 2) + pow(x[1], 2) + 25 * (pow(math.sin(x[0]), 2) + pow(math.sin(x[1]), 2))

def easom(x): # returns a scalar
    """
    The objective function.
    Implements the Easom algorithm.
    :param x: The 1xD array to calculate with.
    :return: The cost of the point x.
    """
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi)**2) - ((x[1] - np.pi)**2))

def rosenbrock(x):  # returns a scalar
    """
        The objective function.
        Implements the Rosenbrock algorithm.
        :param x: The 1xD array to calculate with.
        :return: The cost of the point x.
        """
    sum = 0
    for k in range(D - 1):
        sum += pow(x[0] - 1, 2) + 100 * pow((float(x[1]) - pow(x[0], 2)), 2)
    return sum

"""ABC algorithm"""

def solve(f, num_bees = 50, abandonment_limit=25):
    # initialize the bees uniformly in the function space
    #Need to pass primary filter and idx even though they won't be used
    population = [Bee(np.random.uniform(lower_bound, upper_bound, D), f, 0, 0, "name") for _ in range(num_bees)]

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
            else:
                bee.nonImprovementCounter += 1

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
                else:
                    bee.nonImprovementCounter += 1

        # scout bees
        for i, bee in enumerate(population):
            if bee.nonImprovementCounter >= abandonment_limit:
                bee.__class__ = ScoutBee
                population[i] = bee.neighbor()
                bee.nonImprovementCounter = 0

        # update best solutions
        best_idx = np.argmin([bee.fitness for bee in population])
        if population[best_idx].fitness < best_fitness:
            best_soln = population[best_idx].position
            best_fitness = population[best_idx].fitness

    return best_soln, best_fitness

def solve_Init(obj_func, pop_size, number_of_trials, abandoment_limit):
    best_fitnesses = [0] * number_of_trials
    for i in range(number_of_trials):
        best_solution, best_fitness = solve(obj_func, pop_size, abandoment_limit)
        best_fitnesses[i] = best_fitness
        
    mean = np.mean(best_fitnesses)
    sample_std_dev = np.std(best_fitnesses, ddof=1)
    return mean, sample_std_dev

population_sizes = [100, 30]
number_of_trials = 30
abandoment_limits = [50, 10]

with open('../continousOutput.txt', 'w') as file:
    for i in range(len(population_sizes)):
        for j in range(len(abandoment_limits)):
            mean, std = solve_Init(ackley, population_sizes[i], number_of_trials, abandoment_limits[j])
            file.write("Population Size: " + str(population_sizes[i]) + '\n')
            print("Population Size: ", population_sizes[i])
            file.write("Abandoment Limit: " + str(abandoment_limits[j]) + '\n')
            print("Abandoment Limit: ", abandoment_limits[j])
            file.write("Mean: " + str(mean) + '\n')
            print("Mean: ", mean)
            file.write("Standard Deviation: " + str(std) + '\n\n')
            print("Standard Deviation: ", std)
