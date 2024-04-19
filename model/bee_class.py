import numpy as np

# TODO: import these values from data file
lower_bound = -2.0
upper_bound = 2.0

class Bee():
    def __init__(self, position, f, primaryFilter=0) -> None:
        self.position = position
        self.objective_func = f
        self.fitness = 0
        self.visitedIndexes = [self.position]
        self.primaryFilter = primaryFilter
    
    def update_position(self, new_position):
        self.position = new_position

    def update_fitness(self):
        self.fitness = self.objective_func(self.position)

    def display(self):
        print(f"I am a Base Bee")
        return
    
    def _neighbor(self, other_position):
        neighbor = self.position + np.random.uniform(-1, 1, size=len(self.position)) * (self.position - other_position)
        neighbor = np.clip(neighbor, lower_bound, upper_bound)
        neighbor = Bee(neighbor, self.objective_func)
        neighbor.update_fitness()
        return neighbor
        
    
class EmployedBee(Bee):
    def neighbor(self, other_position):
        return self._neighbor(other_position)
    
    def display(self):
        print(f"I am an Employeed Bee")
        return
    
class OnlookerBee(Bee):
    def neighbor(self, other_position):
        return self._neighbor(other_position)
    
    def display(self):
        print(f"I am an Onlooker Bee")

class ScoutBee(Bee):
    def neighbor(self):
        neighbor = Bee(np.random.uniform(lower_bound, upper_bound, size=len(self.position)), self.objective_func)
        neighbor.update_fitness()
        return neighbor

    def display(self):
        print(f"I am a Scout Bee")

    
