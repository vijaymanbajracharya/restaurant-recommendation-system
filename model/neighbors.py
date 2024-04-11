import numpy as np
import random
from ABC import f

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

arrays = []


def generateData():
    # List of restaurant names
    restaurant_names = [f'Restaurant {i}' for i in range(1, 101)]

    # Generate 100 restaurant arrays
    for i in range(sizeOfData):
        array = [restaurant_names[i]]
        array.extend([np.random.randint(1, 101) for _ in range(3)])
        arrays.append(array)

    # Print the generated arrays
    for array in arrays:
        print(array)

    return arrays


"""
Gets a single neighbor for the current bee.
@param: The bee object.
Returns: The index of the neighbor if there is one and None otherwise
"""
def getNeighbor(bee):
    neighbor = None
    distanceCount = 1
    # Get the current restaurant data.
    current = arrays[bee.currentIndex]
    # Loop for number of indexes from the current e.g. plus or minus the max_distance from the current index.
    while distanceCount <= max_distance and neighbor is None:
        # Keeps the indexes in bounds. I assume we don't want this to wrap.
        index1 = arrays[bee.currentIndex + (1 * distanceCount)] if bee.currentIndex < sizeOfData else None
        index2 = arrays[bee.currentIndex - (1 * distanceCount)] if bee.currentIndex > 0 else None

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
Returns: The neighbor if there is one and None otherwise
'''


def getNeighborFintessBased(bee):
    neighbor = None
    distanceCount = 1
    # Get the current restaurant data.
    current = arrays[bee.currentIndex]
    # Loop for number of indexes from the current e.g. plus or minus the max_distance from the current index.
    while distanceCount <= max_distance and neighbor is None:
        # Keeps the indexes in bounds. I assume we don't want this to wrap.
        index1 = arrays[bee.currentIndex + (1 * distanceCount)] if bee.currentIndex < sizeOfData else None
        index2 = arrays[bee.currentIndex - (1 * distanceCount)] if bee.currentIndex > 0 else None

        # Prevent repeat visits of restaurants.
        if index1 in bee.visitedIndexes: index1 = None
        if index2 in bee.visitedIndexes: index2 = None

        stdCount = 0
        # Loop for the number of standard deviations from the current data.
        while neighbor is None and stdCount is not max_std:
            neighbor1 = None
            neighbor2 = None
            if index1:
                if current[bee.primaryFilter] <= index1[bee.primaryFilter] <= current[bee.primaryFilter] + \
                        (std * stdCount) or current[bee.primaryFilterx] >= index1[bee.primaryFilter] >= \
                        current[bee.primaryFilter] - (std * stdCount): neighbor1 = index1
            if index2:
                if current[bee.primaryFilter] <= index2[bee.primaryFilter] <= current[bee.primaryFilter] + \
                        (std * stdCount) or current[bee.primaryFilter] >= index2[bee.primaryFilter] >= \
                        current[bee.primaryFilter] - (std * stdCount): neighbor2 = index2

            if neighbor1 and neighbor2:
                # Passes the index to the fitness function
                fitness1 = f(neighbor1)
                fitness2 = f(neighbor2)
                neighbor = neighbor1 if fitness1 > fitness2 else neighbor2

            stdCount += 1

    return neighbor if neighbor else None
