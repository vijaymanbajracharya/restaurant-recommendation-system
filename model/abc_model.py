import numpy as np
import math
import random
import pandas as pd
import pdb
from bee_class import Bee, OnlookerBee, EmployedBee, ScoutBee

# ARGS
seed = 871623
np.random.seed(seed)

MAX_ITER = 1000

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
max_distance = 10
# How many rows we have in the database.
sizeOfData = 100

# total number of neighbors to look at during onlooker phase (leave as even number)
max_onlooker_distance = 10

class neighbors:
    def generateData(cuisine):
        cuisine = [c.strip().lower() for c in cuisine]
        # preferences = {}
        # preferences['Price'] = float(input("Enter your preference for Price on a scale of 0-5: "))
        # preferences['Distance'] = float(input("Enter your preference for Distance on a scale of 0-5: "))
        # preferences['Cleanliness and Hygiene']=float(input("Enter your preference for Cleanliness and Hygiene on a scale of 0-5: "))
        # preferences['Locality/Neighborhood']=float(input("Enter your preference for Locality/Neighborhood on a scale of 0-5: "))
        # preferences['Portion Sizes']=float(input("Enter your preference for Portion Sizes on a scale of 0-5: "))
        # preferences['Overall Rating']=float(input("Enter your preference for Overall Rating on a scale of 0-5: "))
        file_path = '../data/new_restaurants_data.csv'
        df = pd.read_csv(file_path)
        df['Cuisine'] = df['Cuisine'].str.lower().str.strip()
        filtered_df = df[df['Cuisine'].isin(cuisine)]
        # print(filtered_df)
        filtered_df = filtered_df.drop(columns=['Cuisine'])
        return filtered_df

    '''
    Gets a single neighbor for the current bee, gets the best fit neighbor if two are returned.
    @param: index: The index of the current bee.
    @param: dataBaseIndex: The index of the chosen field in the database (e.g. price, location etc..)
    Returns: The neighbor index if there is one and None otherwise
    '''

    def getNeighborFintessBased(bee, data):
        neighbor = None
        distanceCount = 1
        restaurant1, restaurant2 = None, None
        # Get the current restaurant data.
        current_idx = bee.idx
        current = data[current_idx]
        # Loop for number of indexes from the current e.g. plus or minus the max_distance from the current index.
        while distanceCount <= max_distance and neighbor is None:
            # Keeps the indexes in bounds. I assume we don't want this to wrap.
            index1 = bee.idx + (1 * distanceCount) if (bee.idx + (1 * distanceCount)) < len(data) else None
            index2 = bee.idx - (1 * distanceCount) if (bee.idx - (1 * distanceCount)) > 0 else None
            if index1:
                restaurant1 = data[index1]
            if index2:
                restaurant2 = data[index2]
            # Prevent repeat visits of restaurants.
            if index1 in bee.visitedIndexes: index1 = None
            if index2 in bee.visitedIndexes: index2 = None
            stdCount = 1
            # Loop for the number of standard deviations from the current data.
            while neighbor is None and stdCount is not max_std:
                neighborIndex1 = None
                neighborIndex2 = None
                if index1:
                    # print("restaurant1", restaurant1[bee.primaryFilter])
                    if current[bee.primaryFilter] <= restaurant1[bee.primaryFilter] <= current[bee.primaryFilter] + \
                            (std * stdCount) or current[bee.primaryFilter] >= restaurant1[bee.primaryFilter] >= \
                            current[bee.primaryFilter] - (std * stdCount): neighborIndex1 = index1
                if index2:
                    # print("restaurant2", restaurant2[bee.primaryFilter])
                    if current[bee.primaryFilter] <= restaurant2[bee.primaryFilter] <= current[bee.primaryFilter] + \
                            (std * stdCount) or current[bee.primaryFilter] >= restaurant2[bee.primaryFilter] >= \
                            current[bee.primaryFilter] - (std * stdCount): neighborIndex2 = index2
                # if index1:
                #     for x in range(1,4):
                #         if index1+x<len(data) and ( current[-1]-4 <=data[index1+x][-1]<=current[-1]+4):
                #             neighborIndex1=index1+x
                #             break
                #         elif index1-x>0 and ( current[-1]-4 <=data[index1-x][-1]<=current[-1]+4):
                #             neighborIndex1=index1-x
                #             break
                # if index2:
                #     for x in range(1,4):
                #         if index2+x<len(data) and ( current[-1]-4 <=data[index2+x][-1]<=current[-1]+4):
                #             neighborIndex2=index2+x
                #             break
                #         elif index2+x>0 and ( current[-1]-4 <=data[index2-x][-1]<=current[-1]+4):
                #             neighborIndex2=index2-x
                #             break
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

            distanceCount += 1

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


    """
    Returns the best neighbor based on fitness that is half of max_onlooker_distance up or down from the current index.
    """
    def getOnlookerNeighbor(bee, data):
        neighbors = []
        # Get the current restaurant data.
        current_idx = bee.idx
        distance = max_onlooker_distance / 2
        for i in range(int(distance)):
            # make sure the index is in range
            if current_idx + (i + 1) < len(data):
                indexUp = current_idx + (i + 1)
                #structured as [index, reastaurant data]
                neighbors.append([indexUp, data[indexUp]])
            if current_idx - (i + 1) >= 0:
                indexDown = current_idx - (i + 1)
                # structured as [index, reastaurant data]
                neighbors.append([indexDown, data[indexDown]])

        # Sort the array based on the position of each restauraunt.
        def sort(other):
            return f(other[1][1:])

        sortedNeighbors = sorted(neighbors, key=sort)
        #return the index, and the best restauruant neighbor
        return sortedNeighbors[0][0], sortedNeighbors[0][1]



# TODO: This needs to be the same size as x
COEFFICIENTS = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]


# objective function linear combination of all features
#	Price	Distance	Cleanliness and Hygiene	Locality/Neighborhood	Wait Times	Portion Sizes	Overall Rating

def f(x):
    return (x[0] * 0.2 + (100 - x[1]) * 0.2 + x[2] * 0.2 + x[3] * 0.1 + (100 - x[4]) * 0.1 + x[5] * 0.1 + x[6] * 0.1) / 100


"""ABC algorithm"""


def solve(f, cuisine, num_bees=5, abandonment_limit=10):
    SOLUTION_SPACE = neighbors.generateData(cuisine)
    SOLUTION_SPACE = SOLUTION_SPACE.to_numpy()
    # initialize the bees uniformly in the function space
    # TODO change this to match user input
    primaryFilter = 1
    population = []
    # select a random subset to begin with
    random_subset = random.sample(range(len(SOLUTION_SPACE)), num_bees)
    # print(random_subset)
    for idx in random_subset:
        # Extract the row corresponding to the index, skipping the first element for data
        row = SOLUTION_SPACE[idx]
        name = row[0]
        data = np.array(row[1:])
        primaryFilter = 1
        # Create a new Bee instance
        new_bee = Bee(data, f, idx, primaryFilter, name)
        population.append(new_bee)
    
    # fitness of population at initialization
    for bee in population:
        bee.update_fitness()

    best_idx = np.argmax([bee.fitness for bee in population])
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
                random_candidate_idx = np.random.randint(0, num_bees)\
            
            neighbor_index = neighbors.getNeighborFintessBased(bee, SOLUTION_SPACE)
            if neighbor_index:
                neighbor_restaurant = SOLUTION_SPACE[neighbor_index]
                neighbor_name = neighbor_restaurant[0]
                neighbor_position = neighbor_restaurant[1:]
                new_fitness = f(neighbor_position)

                # compare fitness with parent
                if new_fitness > bee.fitness:
                    bee.position = neighbor_position
                    bee.visitedIndexes.append(neighbor_index)
                    bee.fitness = new_fitness
                    bee.name = neighbor_name
                    bee.idx = neighbor_index
                else:
                    bee.nonImprovementCounter += 1

        # calculate probabilities
        fitness_sum = np.sum([bee.fitness for bee in population])
        prob = [bee.fitness / fitness_sum for bee in population]

        # onlooker bees
        # TODO: Check the logic of this, it should look at the SOLUTION_SPACE not population
        #       If you look at the continuous implementation, during the Onlooker Phase, a
        #       new food source needs to be assigned. This method does not seem to do that
        #       It does not use the neighbor implementation from the bee class so that is not
        #       the issue here. it always returns a food source that already has a bee
        #       on it.
        for i, bee in enumerate(population):
            if np.random.uniform() < prob[i]:
                bee.__class__ = OnlookerBee
                # generate neighborhood source and test its fitness
                neighborhood_source = np.random.randint(0, num_bees)
                while neighborhood_source == i:
                    neighborhood_source = np.random.randint(0, num_bees)

                # TODO: remove after new implementation working
                # def euclidean_distance(other):
                #     other_pos = other.position
                #     return np.linalg.norm(other_pos[primaryFilter - 1] - bee.position[primaryFilter - 1])
                #
                # # sorts the population based on the distance from the current bee.position.
                # sorted_population = sorted(population, key=euclidean_distance)
                #
                # # The first in the population will most likely be the same bee as the current bee so take the second.
                # neighbor_bee = sorted_population[1] if bee == sorted_population[0] else sorted_population[0]

                # generate a new food source similar to employed but slightly different. if the new food source fitness (rather than the neighbor fitness) is
                # higher, then we accept new solution, otherwise we increment nonImprovementCounter.
                potential_index, potential_neighbor = neighbors.getOnlookerNeighbor(bee, SOLUTION_SPACE)

                #checks if the new neighbor is in the population
                def in_population(potential, population):
                    for bee in population:
                        if bee.name == potential[0]:
                            return True
                    return False

                #if it isn't in the population create a new bee.
                if not in_population(potential_neighbor, population):
                    neighbor_bee = Bee(potential_neighbor[1:], f, potential_index, primaryFilter,potential_neighbor[0])
                    new_fitness = neighbor_bee.fitness
                else:
                    neighbor_bee = None
                    new_fitness = 0

                # recruit onlooker bees to richer sources of food               
                if new_fitness > bee.fitness:
                    population[i] = neighbor_bee
                else:
                    bee.nonImprovementCounter += 1

        # scout bees
        for i, bee in enumerate(population):
            if bee.nonImprovementCounter >= abandonment_limit:
                # TODO: maybe dont abandon strictly based on abandonment limit but have some heuristic based on fitness
                mask = ~np.all(SOLUTION_SPACE[:, 1:] == bee.position, axis=1)
                valid_food_sources = SOLUTION_SPACE[mask]
                new_source_idx = np.random.randint(0, valid_food_sources.shape[0])
                population[i] = Bee(np.array(SOLUTION_SPACE[new_source_idx][1:]), f, new_source_idx, primaryFilter, SOLUTION_SPACE[new_source_idx][0])
                population[i].update_fitness()

        # update best solutions
        best_idx = np.argmax([bee.fitness for bee in population])
        if population[best_idx].fitness > best_fitness:
            best_fitness = population[best_idx].fitness

    return population[best_idx].name


# print(solve(f))
def main():
    li = []
    cuisine = input("Enter preferred cuisines separated by comma (e.g., Indian,Chinese): ").split(',')

    for x in range(10):
        li.append(solve(f, cuisine))

    for x in li:
        if x:
            print(x)


main()
