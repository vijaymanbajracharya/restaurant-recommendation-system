import numpy as np
import random

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


# name, price, portion size, overall rating

# This is currently not used but is a very simple bee class.
class Bee:
    currentIndex = -1
    visitedIndexes = []
    currentFitness = 0


bee = Bee()
bee.currentIndex = random.randint(0, sizeOfData)


arrays = []
def generateData():
    # List of restaurant names
    restaurant_names = [f'Restaurant {i}' for i in range(1, 101)]

    # Generate 100 restaurant arrays
    for i in range(sizeOfData):
        array = [restaurant_names[i]]
        array.extend([random.randint(1, 100) for _ in range(3)])
        arrays.append(array)

    # Print the generated arrays
    for array in arrays:
        print(array)

    return arrays

'''
Gets a single neighbor for the current bee.
@param: index: The index of the current bee.
@param: dataBaseIndex: The index of the chosen field in the database (e.g. price, location etc..)
Returns: The neighbor if there is one and None otherwise
'''
def getNeighbor(index, dataBaseIndex):
    neighbor = None
    distanceCount = 1
    # Get the current restaurant data.
    current = arrays[index]
    # Loop for number of indexes from the current e.g. plus or minus the max_distance from the current index.
    while distanceCount <= max_distance and neighbor is None:
        # Keeps the indexes in bounds. I assume we don't want this to wrap.
        temp1 = arrays[index + (1 * distanceCount)] if index < sizeOfData else None
        temp2 = arrays[index - (1 * distanceCount)] if index > 0 else None

        stdCount = 0
        # Loop for the number of standard deviations from the current data.
        while neighbor is None and stdCount is not max_std:
            if temp1:
                if current[dataBaseIndex] <= temp1[dataBaseIndex] <= current[dataBaseIndex] + (std * stdCount) or current[
                    dataBaseIndex] >= temp1[dataBaseIndex] >= current[dataBaseIndex] - (std * stdCount): neighbor = temp1
            if temp2:
                if current[dataBaseIndex] <= temp2[dataBaseIndex] <= current[dataBaseIndex] + (std * stdCount) or current[
                    dataBaseIndex] >= temp2[dataBaseIndex] >= current[dataBaseIndex] - (std * stdCount): neighbor = temp2
            stdCount += 1

    return neighbor if neighbor else None
