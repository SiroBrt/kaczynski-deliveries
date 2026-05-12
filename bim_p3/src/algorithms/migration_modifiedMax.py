from __future__ import annotations

import random
import math
import time

try:
    from ..problem import ProblemInstance
    from ..evaluation import evaluate_solution
except ImportError:
    from problem import ProblemInstance
    from evaluation import evaluate_solution


# ============================================================================
# Vehicle and Individual Classes
# ============================================================================


class Vehicle:
    """Represents a single vehicle's route in the solution."""
    
    __vehicleCapacity: int = math.inf
    __problem: ProblemInstance = None
    
    def __init__(self, path: list[int] = []):
        self.path: list[int] = path
        self.computeVehicleParameters()
    
    @classmethod
    def setMaximumCapacity(cls, maxCapacity: int):
        """Set the maximum capacity for all vehicles."""
        cls.__vehicleCapacity = maxCapacity

    @classmethod
    def setProblemInstance(cls, problem: ProblemInstance):
        """Set the problem instance for all vehicles."""
        cls.__problem = problem

    def computeVehicleParameters(self):
        """Compute distance, time, delay, cost, and remaining capacity for this vehicle's route."""
        current = 0  # Start at depot
        self.distance = 0
        self.time = 0
        self.delay = 0
        self.cost = 0
        self.remainingCapacity = self.__vehicleCapacity
        
        if self.path and self.__problem:
            for cid in self.path:
                customer = self.__problem.customers[cid - 1]
                dist = self.__problem.distance(current, cid)
                arrival = self.time + dist / self.__problem.vehicle_speed
                self.time = max(arrival, customer.ready_time) + customer.service_time
                self.remainingCapacity -= customer.demand
                current = cid


class Individual:
    """Represents a complete solution with multiple vehicles."""
    
    __problem: ProblemInstance = None
    
    def __init__(self):
        self.vehicles: list[Vehicle] = []
        self.score = 0
        self.unvisited = set()
        if self.__problem:
            self.unvisited = set([i for i in range(1, self.__problem.num_customers + 1)])
    
    @classmethod
    def setProblemInstance(cls, problem: ProblemInstance):
        """Set the problem instance for all individuals."""
        cls.__problem = problem
        Vehicle.setProblemInstance(problem)

    def evaluate(self):
        """Evaluate this solution using the evaluation module."""
        self.score = evaluate_solution(self.__problem, self.getAllRoutes()).total_cost

    def __lt__(self, other):
        """Compare individuals by score (for sorting)."""
        return self.score < other.score
    
    def generateVehicles(self):
        """Randomly generate vehicles with random customer assignments."""
        self.vehicles: list[Vehicle] = []
        self.unvisited = set([i for i in range(1, self.__problem.num_customers + 1)])
        num_vehicles = self.__problem.num_vehicles
        
        for _ in range(num_vehicles):
            # Randomly assign customers to this vehicle
            if self.unvisited:
                route = random.sample(
                    sorted(self.unvisited),
                    random.randint(0, min(len(self.unvisited), self.__problem.num_customers // num_vehicles + 1))
                )
                vehicle = Vehicle(route)
                self.vehicles.append(vehicle)
                self.unvisited -= set(route)
            else:
                self.vehicles.append(Vehicle([]))
        
        self.evaluate()

    def getAllRoutes(self) -> list[list[int]]:
        """Get all routes as a list of customer lists."""
        routes = []
        for v in self.vehicles:
            routes.append(v.path)
        return routes

    def mutate(self):
        """Apply mutation operator (swap, move, or insert customer)."""
        num = random.random()
        if num < 1/3:
            # Swap two customers in the same route
            v = random.choice(self.vehicles)
            if len(v.path) >= 2:
                i, j = random.sample(range(len(v.path)), 2)
                v.path[i], v.path[j] = v.path[j], v.path[i]
        elif num < 2/3:
            # Move a customer from one route to another
            if len(self.vehicles) >= 2:
                v1, v2 = random.sample(self.vehicles, 2)
                if v1.path:
                    customer = random.choice(v1.path)
                    v1.path.remove(customer)
                    v2.path.append(customer)
        else:
            # Insert an unvisited customer into a random route
            if self.unvisited:
                customer = random.choice(sorted(self.unvisited))
                v = random.choice(self.vehicles)
                v.path.append(customer)
                self.unvisited.remove(customer)
        
        self.evaluate()

    def updateUnvisited(self):
        """Update the set of unvisited customers."""
        visited = set()
        for v in self.vehicles:
            visited.update(v.path)
        self.unvisited = set(range(1, self.__problem.num_customers + 1)) - visited

    def removeRedundant(self):
        """Remove duplicate customers that appear in multiple routes."""
        unique_customers = set()
        for v in self.vehicles:
            new_path = []
            for cid in v.path:
                if cid not in unique_customers:
                    unique_customers.add(cid)
                    new_path.append(cid)
            v.path = new_path
        self.evaluate()

    def crossover(self, other) -> tuple[Individual, Individual]:
        """Create two children by crossing over routes with another individual."""
        child1 = Individual()
        child2 = Individual()
        
        for v1, v2 in zip(self.vehicles, other.vehicles):
            if random.random() < 0.5:
                child1.vehicles.append(Vehicle(v1.path.copy()))
                child2.vehicles.append(Vehicle(v2.path.copy()))
            else:
                child1.vehicles.append(Vehicle(v2.path.copy()))
                child2.vehicles.append(Vehicle(v1.path.copy()))
        
        child1.updateUnvisited()
        child2.updateUnvisited()
        child1.removeRedundant()
        child2.removeRedundant()
        
        return child1, child2


class Population:
    """Manages a population of individuals and runs the genetic algorithm."""
    
    def __init__(self, problem: ProblemInstance):
        self.individuals: list[Individual] = []
        self.problem = problem
        Individual.setProblemInstance(problem)

    def generatePopulation(self, n: int):
        """Generate n random individuals for the population."""
        self.individuals = [None] * n
        for i in range(n):
            ind = Individual()
            ind.generateVehicles()
            self.individuals[i] = ind
        self.evaluatePopulation()

    def evaluatePopulation(self):
        """Evaluate all individuals in the population."""
        for ind in self.individuals:
            ind.evaluate()

    def sample(self, tournament_size: int = 3) -> Individual:
        """Select an individual using tournament selection."""
        inds = random.sample(self.individuals, min(tournament_size, len(self.individuals)))
        return min(inds)

    @property
    def best(self) -> Individual:
        """Return the best individual in the population."""
        return min(self.individuals)

    def runSelection(
        self,
        generations: int,
        mutation_rate: float = 0.35,
        crossover_rate: float = 0.6,
        elite_size: int = 2,
        tournament_size: int = 3
    ) -> tuple[list[float], Individual]:
        """Run the genetic algorithm for specified generations.
        
        Args:
            generations: Number of generations to run
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of best individuals to preserve
            tournament_size: Size of tournament selection
            
        Returns:
            Tuple of (best_score_history, best_individual)
        """
        self.evaluatePopulation()
        best_score_hist = []
        
        for gen in range(generations):
            new_population = []
            
            # Elitism: preserve the best individuals
            elites = sorted(self.individuals)[:elite_size]
            new_population.extend(elites)
            
            # Generate new individuals through selection, crossover, and mutation
            while len(new_population) < len(self.individuals):
                randNum = random.random()
                
                if randNum < crossover_rate:
                    # Crossover
                    parent1 = self.sample(tournament_size)
                    parent2 = self.sample(tournament_size)
                    child1, child2 = parent1.crossover(parent2)
                    new_population.append(child1)
                    if len(new_population) < len(self.individuals):
                        new_population.append(child2)
                        
                elif randNum < crossover_rate + mutation_rate:
                    # Mutation
                    parent = self.sample(tournament_size)
                    child = Individual()
                    child.vehicles = [Vehicle(v.path.copy()) for v in parent.vehicles]
                    child.updateUnvisited()
                    child.mutate()
                    new_population.append(child)
                    
                else:
                    # Reproduction (copy parent)
                    parent = self.sample(tournament_size)
                    child = Individual()
                    child.vehicles = [Vehicle(v.path.copy()) for v in parent.vehicles]
                    child.updateUnvisited()
                    child.evaluate()
                    new_population.append(child)
            
            self.individuals = new_population
            best_score_hist.append(self.best.score)
        
        return best_score_hist, self.best


# ============================================================================
# Migration Algorithm Function
# ============================================================================

def runMigration(
    problem: ProblemInstance,
    populations_num: int,
    iterations: int,
    population_size: int,
    seed: int
) -> tuple[list[list[int]], list[float]]:
    """
    Solve a Vehicle Routing Problem using a genetic algorithm with migration between multiple populations.
    
    Implements an island model where multiple populations evolve independently, periodically
    exchanging best individuals to prevent local convergence.
    
    Args:
        problem: The VRP problem instance to solve
        populations_num: Number of independent populations (islands)
        iterations: Total number of GA generations across all islands
        population_size: Size of each population
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (best_routes, score_history) where:
            - best_routes: Routes of the best solution found
            - score_history: List of best scores across all generations
    """
    random.seed(seed)
    populations:list[Population] = []
    for i in range(populations_num):
        pop = Population(problem)
        pop.generatePopulation(population_size)
        populations.append(pop)

    best_score_hist = []
    elite_size = max(4, population_size // 10)
    generations_per_epoch = iterations // 10
    
    for epoch in range(10):
        # Run local evolution on each population
        local_best_scores = [math.inf] * generations_per_epoch
        bests_to_migrate = {}

        for i, pop in enumerate(populations):
            score_history, _ = pop.runSelection(
                generations=generations_per_epoch,
                mutation_rate=0.4,
                crossover_rate=0.6,
                elite_size=elite_size,
                tournament_size=5
            )
            bests_to_migrate[i] = sorted(pop.individuals)[:elite_size]
            
            # Track global best across populations and generations
            for j, score in enumerate(score_history):
                local_best_scores[j] = min(local_best_scores[j], score)
        
        best_score_hist.extend(local_best_scores)
        
        # Migration phase: Ring topology (circular migration)
        # Each population sends best individuals to the next population in the ring
        for src in range(populations_num):
            dst = (src + 1) % populations_num  # Ring topology: 0->1->2->...->n-1->0
            populations[dst].individuals[-elite_size:] = bests_to_migrate[src]
    
    # Find and return the best solution across all populations
    bests = [pop.best for pop in populations]
    best = min(bests)
    return best.getAllRoutes(), best_score_hist



def run(
    problem: ProblemInstance,
    iterations: int = 200,
    population_size: int = 200,
    seed: int = 0
) -> tuple[list[list[int]], list[float]]:
    start = time.time()
    results = runMigration(
        problem,
        populations_num=10,
        iterations=iterations,
        population_size=population_size,
        seed=seed
    )
    print(f"Execution time: {time.time() - start}")
    return results