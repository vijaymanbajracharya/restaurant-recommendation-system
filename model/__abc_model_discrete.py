import numpy as np
import math
import random
import pdb
from bee_class import Bee, OnlookerBee, EmployedBee, ScoutBee

# ARGS
seed = 871623
np.random.seed(seed)

MAX_ITER = 5

lower_bound = -2.0
upper_bound = 2.0
D = 2

# The number of times we can go out from the center. For example if the max_std is 3 and the std is 2
# and I am looking at price, where my current price is 50 I will accept anything between 56 and 44.
max_std = 3
# How big of a standard deviation we have.
std = 2
# The maximum number of indexes in the dataTable we can jump. For example if I am currently have a restaurant
# with and index at 32, and my max_distance is 2, then I will check all restaurants between 30 and 34.
max_distance = 2
# How many rows we have in the database.
sizeOfData = 100

class neighbors:
    def generateData():
        data = []
        # List of restaurant names
        restaurant_names = [f'Restaurant {i}' for i in range(1, 101)]

        # Generate 100 restaurant arrays
        for i in range(sizeOfData):
            array = [restaurant_names[i]]
            array.extend([np.random.randint(1, 101) for _ in range(3)])
            data.append(array)

        sorted_data = sorted(data, key=lambda x: x[1])
        # Print the generated arrays
        for array in sorted_data:
            print(array)

        return sorted_data


    """
    Gets a single neighbor for the current bee.
    @param: The bee object.
    Returns: The index of the neighbor if there is one and None otherwise
    """
    def getNeighbor(bee, data):
        neighbor = None
        distanceCount = 1
        # Get the current restaurant data.
        current = data[bee.position]
        # Loop for number of indexes from the current e.g. plus or minus the max_distance from the current index.
        while distanceCount <= max_distance and neighbor is None:
            # Keeps the indexes in bounds. I assume we don't want this to wrap.
            index1 = data[bee.position + (1 * distanceCount)] if bee.position < sizeOfData else None
            index2 = data[bee.position - (1 * distanceCount)] if bee.position > 0 else None

            # Prevent repeat visits of restaurants.
            if index1 in bee.visitedIndexes: index1 = None
            if index2 in bee.visitedIndexes: index2 = None

            stdCount = 0
            # Loop for the number of standard deviations from the current data.
            while neighbor is None and stdCount is not max_std:
                if index1:
                    if current[bee.primaryFilter] <= index1[bee.primaryFilter] <= current[bee.primaryFilter] + \
                            (std * stdCount) or current[bee.primaryFilterx] >= index1[bee.primaryFilter] >= \
                            current[bee.primaryFilter] - (std * stdCount): neighbor = index1
                if index2:
                    if current[bee.primaryFilter] <= index2[bee.primaryFilter] <= current[bee.primaryFilter] + \
                            (std * stdCount) or current[bee.primaryFilter] >= index2[bee.primaryFilter] >= \
                            current[bee.primaryFilter] - (std * stdCount): neighbor = index2
                stdCount += 1

        return neighbor if neighbor else None


    '''
    Gets a single neighbor for the current bee, gets the best fit neighbor if two are returned.
    @param: index: The index of the current bee.
    @param: dataBaseIndex: The index of the chosen field in the database (e.g. price, location etc..)
    Returns: The neighbor index if there is one and None otherwise
    '''


    def getNeighborFintessBased(bee, data):
        neighbor = None
        distanceCount = 1
        # Get the current restaurant data.
        current_idx = bee.idx
        current = data[current_idx]
        # Loop for number of indexes from the current e.g. plus or minus the max_distance from the current index.
        while distanceCount <= max_distance and neighbor is None:
            # Keeps the indexes in bounds. I assume we don't want this to wrap.
            index1 = current_idx + (1 * distanceCount) if current_idx < sizeOfData - 1 else None
            index2 = current_idx - (1 * distanceCount) if current_idx > 0 else None
            restaurant1 = data[index1]
            restaurant2 = data[index2]

            # Prevent repeat visits of restaurants.
            # TODO: FIX ME, elementwise comparision 
            if index1 in bee.visitedIndexes: index1 = None
            if index2 in bee.visitedIndexes: index2 = None

            stdCount = 1
            # Loop for the number of standard deviations from the current data.
            while neighbor is None and stdCount is not max_std:
                neighborIndex1 = None
                neighborIndex2 = None
                if index1:
                    if current[bee.primaryFilter] <= restaurant1[bee.primaryFilter] <= current[bee.primaryFilter] + \
                            (std * stdCount) or current[bee.primaryFilter] >= restaurant1[bee.primaryFilter] >= \
                            current[bee.primaryFilter] - (std * stdCount): neighborIndex1 = index1
                if index2:
                    if current[bee.primaryFilter] <= restaurant2[bee.primaryFilter] <= current[bee.primaryFilter] + \
                            (std * stdCount) or current[bee.primaryFilter] >= restaurant2[bee.primaryFilter] >= \
                            current[bee.primaryFilter] - (std * stdCount): neighborIndex2 = index2

                if neighborIndex1 and neighborIndex2:
                    # Passes the index to the fitness function
                    fitness1 = f(restaurant1[1:])
                    fitness2 = f(restaurant2[1:])
                    neighbor = neighborIndex1 if fitness1 > fitness2 else neighborIndex2
                elif neighborIndex1:
                    neighbor = neighborIndex1
                elif neighborIndex2:
                    neighbor = neighborIndex2

                stdCount += 1

        return neighbor if neighbor else None
    
    # TEMPORARY NEIGHBOR IMPLEMENTATION
    def getNeighborEuclidean(bee, data):
        data = [np.array(lst[1:], dtype=int) for lst in data]
        target = bee.position
        distances = []
        for lst in data:
            # Skip the target list itself
            if np.array_equal(lst, target):
                continue
            distances.append(np.linalg.norm(lst - target))

        # If all lists are the same as the target, return None
        if not distances:
            return target
        else:
            # Find the index of the list with the minimum distance
            closest_index = np.argmin(distances)

            # Get the closest list
            closest_list = data[closest_index]
            return closest_list

SOLUTION_SPACE = neighbors.generateData() 

# TODO: This needs to be the same size as x
COEFFICIENTS = [0.5, 0.5, 0.5]

# objective function linear combination of all features
def f(x): 
    return np.dot(COEFFICIENTS, x)

"""ABC algorithm"""

def solve(f, num_bees = 5, abandonment_criteria = 0.1):
    # initialize the bees uniformly in the function space
    #TODO change this to match user input
    primaryFilter = 1

    # select a random subset to begin with
    random_subset = random.sample(range(len(SOLUTION_SPACE)), num_bees)
    population = [Bee(np.array(SOLUTION_SPACE[random_subset[idx]][1:]), f, random_subset[idx], primaryFilter, SOLUTION_SPACE[random_subset[idx]][0]) for idx in range(num_bees)]

    # fitness of population at initialization
    for bee in population:
        bee.update_fitness()

    best_idx = np.argmin([bee.fitness for bee in population])
    best_resturant = population[best_idx].name
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

            neighbor_index = neighbors.getNeighborFintessBased(bee, SOLUTION_SPACE)
            if neighbor_index:
                neighbor_restaurant = SOLUTION_SPACE[neighbor_index]
                neighbor_name = neighbor_restaurant[0]
                neighbor_position = neighbor_restaurant[1:]
                new_fitness = f(neighbor_position)

                # compare fitness with parent
                if new_fitness < bee.fitness:
                    bee.position = neighbor_position
                    bee.visitedIndexes.append(neighbor_index)
                    bee.fitness = new_fitness
                    bee.name = neighbor_name

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

                def euclidean_distance(other):
                    other_pos = other.position
                    return np.linalg.norm(other_pos[primaryFilter-1] - bee.position[primaryFilter-1])

                #sorts the population based on the distance from the current bee.position.
                sorted_population = sorted(population, key=euclidean_distance)

                # The first in the population will most likely be the same bee as the current bee so take the second.
                neighbor_bee = sorted_population[1] if bee == sorted_population[0] else sorted_population[0]
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
            best_resturant = population[best_idx].name
            best_soln = population[best_idx].position
            best_fitness = population[best_idx].fitness

    return population[best_idx].name

print(solve(f))
