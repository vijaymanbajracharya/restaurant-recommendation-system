import numpy as np
from neighbors import sizeOfData
from ABC import f

class Bee:
    def __init__(self, primaryFilter):
        self.currentIndex = np.random.randint(0, sizeOfData)
        self.visitedIndexes = [self.currentIndex]
        self.currentFitness = self.calculateFitness()
        self.primaryFilter = primaryFilter

    def calculateFitness(self):
        """
        Calculate the fitness for this bee.
        """
        return f(self.currentIndex)

    def updateVisited(self):
        """
        Updates the visited index array for this bee.
        """
        self.visitedIndexes.append(self.currentIndex)

class Onlooker(Bee):


class Employed(Bee):


class Scout(Bee):