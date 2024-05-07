import numpy as np
import random
import pandas as pd
import sys
from bee_class import Bee, OnlookerBee, EmployedBee, ScoutBee

# ARGS
seed = 871623
np.random.seed(seed)

MAX_ITER = 100

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

USER_INPUT = []


class neighbors:
    def generateData(dataset_path):
        df = pd.read_csv(dataset_path)
        filtered_df = df.drop(columns=['Cuisine'])
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

    # Euclidean Neighbor 
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
                # structured as [index, reastaurant data]
                neighbors.append([indexUp, data[indexUp]])
            if current_idx - (i + 1) >= 0:
                indexDown = current_idx - (i + 1)
                # structured as [index, reastaurant data]
                neighbors.append([indexDown, data[indexDown]])

        # Sort the array based on the position of each restauraunt.
        def sort(other):
            return f(other[1][1:])

        sortedNeighbors = sorted(neighbors, key=sort)
        # return the index, and the best restauruant neighbor
        return sortedNeighbors[0][0], sortedNeighbors[0][1]

# objective function linear combination of all features
#	Price	Distance	Cleanliness and Hygiene	Locality/Neighborhood	Wait Times	Portion Sizes	Overall Rating
#Divide the input by 10 so it goes from 0-1

def f(x):
    return (x[0] * (USER_INPUT[0]/10) + x[1] * (USER_INPUT[1]/10) + x[2] *(USER_INPUT[2]/10) + x[3] * (USER_INPUT[3]/10)
            + x[4] * (USER_INPUT[4]/10) + x[5] * (USER_INPUT[5]/10) + x[6] * (USER_INPUT[6]/10)) / 100


"""ABC algorithm"""

def solve(f, primary_filter, dataset_path, num_bees=2, abandonment_limit=5):
    if num_bees < 2:
        print(f"ERROR: Number of bees has to be atleast 2!")
        return -1
    
    SOLUTION_SPACE = neighbors.generateData(dataset_path)
    SOLUTION_SPACE = SOLUTION_SPACE.to_numpy()
    # initialize the bees uniformly in the function space
    population = []
    primaryFilter = primary_filter
    # select a random subset to begin with
    random_subset = random.sample(range(len(SOLUTION_SPACE)), num_bees)
    # print(random_subset)
    for idx in random_subset:
        # Extract the row corresponding to the index, skipping the first element for data
        row = SOLUTION_SPACE[idx]
        name = row[0]
        data = np.array(row[1:])
        primaryFilter = primary_filter
        # Create a new Bee instance
        new_bee = Bee(data, f, idx, primaryFilter, name)
        population.append(new_bee)

    # fitness of population at initialization
    for bee in population:
        bee.update_fitness()
    
    local_best_idx = np.argmax([bee.fitness for bee in population])
    global_best_idx = population[local_best_idx].idx
    global_best_fitness = population[local_best_idx].fitness

    # optimization
    for i in range(MAX_ITER):
        # employed bees
        for i, bee in enumerate(population):
            # employ the bee population
            bee.__class__ = EmployedBee

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
        for i, bee in enumerate(population):
            if np.random.uniform() < prob[i]:
                bee.__class__ = OnlookerBee
                # generate neighborhood source and test its fitness
                # generate a new food source similar to employed but slightly different. if the new food source fitness (rather than the neighbor fitness) is
                # higher, then we accept new solution, otherwise we increment nonImprovementCounter.
                potential_index, potential_neighbor = neighbors.getOnlookerNeighbor(bee, SOLUTION_SPACE)

                # checks if the new neighbor is in the population
                def in_population(potential, population):
                    for bee in population:
                        if bee.name == potential[0]:
                            return True
                    return False

                # if it isn't in the population create a new bee.
                if not in_population(potential_neighbor, population):
                    neighbor_bee = Bee(potential_neighbor[1:], f, potential_index, primaryFilter, potential_neighbor[0])
                    neighbor_bee.update_fitness()
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
                invalid_food_sources = [SOLUTION_SPACE[idx, :] for idx in bee.visitedIndexes] 
                mask = np.ones(SOLUTION_SPACE.shape[0], dtype=bool)
                for invalid_source in invalid_food_sources:
                    mask &= ~np.all(SOLUTION_SPACE[:, :] == invalid_source, axis=1)

                valid_food_sources = SOLUTION_SPACE[mask]
                new_source_idx = np.random.randint(0, valid_food_sources.shape[0])
                population[i] = Bee(np.array(SOLUTION_SPACE[new_source_idx][1:]), f, new_source_idx, primaryFilter,
                                    SOLUTION_SPACE[new_source_idx][0])
                population[i].update_fitness()

        # update best solutions
        local_best_idx = np.argmax([bee.fitness for bee in population])
        if population[local_best_idx].fitness > global_best_fitness:
            global_best_idx = population[local_best_idx].idx
            global_best_fitness = population[local_best_idx].fitness

    return SOLUTION_SPACE[global_best_idx, :1]

#Finds the largest rating and returns the index
def findPrimaryFilter(valid_preferences):
    if not valid_preferences:
        return None # empty array
    max_index = 0
    for i in range(1, len(valid_preferences)):
        if valid_preferences[i] > valid_preferences[max_index]:
            max_index = i
    return max_index

# print(solve(f))
def recommendRestaurants(dataset_path):
    li = []
    # cuisine = input("Enter preferred cuisines separated by comma (e.g., Indian,Chinese): ").split(',')

    valid_preferences = []
    success = False
    while not success:
        print("Choose your preferences on a scale from 1-10 (inclusive).")
        print(
            "The order of your preferences are: Price	Distance	Cleanliness and Hygiene	Locality/Neighborhood	Wait Times	Portion Sizes	Overall Rating")
        preferences = input("Enter a comma seperated list of 7 values (eg. 1,1,1,1,1,1,1): ").split(',')
        valid = True
        valid_preferences = []
        for value in preferences:
            try:
                if valid:
                    int_value = int(value) #Makes sure the input is an int
                    if 10 >= int_value > 0: #Makes sure value is between 1 and 10 inclusive
                        valid_preferences.append(int_value)
                    else:
                        print(f"Value {int_value} is not between 1 and 10. Try again.")
                        valid = False

            except ValueError:
                print(f"Invalid input: {value}. Try again.")
                valid = False

        if len(valid_preferences) != 7 and valid: #Makes sure the input is the right size
            print(f"Input size of {len(valid_preferences)}. Needs to be of size 7. Try again.")
            valid = False
        if valid:
            success = True

    primaryFilter = findPrimaryFilter(valid_preferences) + 1 #add one because name is the first index
    global USER_INPUT
    USER_INPUT = valid_preferences

    for x in range(10):
        restaurant = solve(f, primaryFilter, dataset_path)
        if restaurant not in li:
            li.append(restaurant)

    print("\nHere are a list of restaurants that fit your preferences:\n")
    for x in li:
        if x:
            print(x)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide the path for the restaurant dataset!")
        sys.exit(1)

    recommendRestaurants(sys.argv[1])
