"""
Ant Colony Optimization (ACO) Algorithm

A population-based metaheuristic inspired by the foraging behavior of ants.
Ants deposit pheromones on paths they traverse; frequently used paths become
increasingly attractive to other ants, creating positive feedback that
gradually builds toward good solutions.

Key ideas:
- Pheromone matrix: τ[i][j] ≈ desirability of traveling from i to j
- Heuristic matrix: η[i][j] = 1/distance[i][j] (local information)
- Ant decision: probability of choosing j proportional to τ^α × η^β
- Pheromone update: increase on best paths, evaporate globally each iteration
- Exploration/exploitation trade-off controlled by α, β parameters
"""

from __future__ import annotations

import math
import random

try:
    from ..evaluation import evaluate_solution
    from ..problem import ProblemInstance
except ImportError:
    from evaluation import evaluate_solution
    from problem import ProblemInstance


# ============================================================================
# Route Post-Processing
# ============================================================================


def _split_by_capacity(problem: ProblemInstance, sequence: list[int]) -> list[list[int]]:
    """
    Convert a customer sequence into valid routes respecting vehicle capacity.
    
    Greedily assigns customers to routes, starting a new route whenever
    adding the next customer would exceed vehicle capacity.
    
    Args:
        problem: VRP problem instance
        sequence: List of customer IDs in visitation order
        
    Returns:
        List of routes (each route is a list of customer IDs), limited to
        at most problem.num_vehicles routes.
    """
    routes: list[list[int]] = []
    current_route: list[int] = []
    current_load = 0
    
    for cid in sequence:
        demand = problem.customers[cid - 1].demand
        
        # Start a new route if current would exceed capacity
        if current_route and current_load + demand > problem.vehicle_capacity:
            routes.append(current_route)
            current_route = [cid]
            current_load = demand
        else:
            current_route.append(cid)
            current_load += demand
    
    # Add the last route if non-empty
    if current_route:
        routes.append(current_route)
    
    # Limit to available vehicles
    return routes[: problem.num_vehicles]


# ============================================================================
# ACO Main Algorithm
# ============================================================================


def run_aco(
    problem: ProblemInstance,
    iterations: int = 60,
    ants: int = 20,
    alpha: float = 1.0,
    beta: float = 3.0,
    rho: float = 0.15,
    q: float = 100.0,
    seed: int = 0,
) -> tuple[list[list[int]], list[float]]:
    """
    Run the Ant Colony Optimization algorithm.
    
    Hyperparameters:
        iterations: Number of iterations (each iteration has 'ants' constructions)
        ants: Number of ants per iteration
        alpha: Weight on pheromone τ^α (exploration vs. exploitation)
               α=0: pure greedy, α→∞: purely random
        beta: Weight on heuristic η^β (distance importance)
              β=0: ignore distances, β→∞: pure greedy by distance
        rho: Pheromone evaporation rate ∈ (0, 1]
             rho=0: no evaporation (paths accumulate forever)
             rho=1: complete reset each iteration
        q: Pheromone quantity deposited: Δτ = q / cost
           Higher q → faster convergence (but more prone to early stagnation)
        seed: Random seed for reproducibility
    
    Args:
        problem: VRP problem instance
        iterations: Number of iterations
        ants: Number of ants per iteration
        alpha: Pheromone weight (exploration)
        beta: Heuristic weight (greedy)
        rho: Evaporation rate
        q: Pheromone deposit quantity
        seed: Random seed
        
    Returns:
        Tuple of (best_routes, history) where history is the best cost per iteration.
    """
    rng = random.Random(seed)
    n = problem.num_customers
    
    # Initialize pheromone matrix (all edges equally likely initially)
    pheromone = [[1.0 for _ in range(n + 1)] for _ in range(n + 1)]
    
    # Precompute heuristic matrix: η[i][j] = 1 / distance
    # This biases ants toward short edges
    heuristic = [[0.0 for _ in range(n + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                heuristic[i][j] = 1.0 / (problem.distance(i, j) + 1e-6)

    best_routes: list[list[int]] | None = None
    best_cost = math.inf
    history: list[float] = []

    # Main optimization loop
    for iteration in range(iterations):
        iteration_best_cost = math.inf
        iteration_best_routes = None

        # Ants construct solutions
        for ant_id in range(ants):
            # Build a single solution
            unvisited = set(range(1, n + 1))
            sequence: list[int] = []
            current = 0  # Start at depot
            
            # Greedy random construction based on pheromone + heuristic
            while unvisited:
                candidates = list(unvisited)
                weights = []
                
                # Compute attractiveness of each candidate
                for cid in candidates:
                    # Pheromone component: how "good" this edge has been historically
                    tau = pheromone[current][cid] ** alpha
                    
                    # Heuristic component: distance-based desirability
                    eta = heuristic[current][cid] ** beta
                    
                    # Problem-specific heuristics
                    cust = problem.customers[cid - 1]
                    
                    # Urgency: customers with tight time windows get priority
                    urgency = 1.0 / (1.0 + max(0.0, cust.due_time - cust.ready_time))
                    
                    # Priority boost: high-priority customers are more attractive
                    priority_boost = 1.0 + 0.2 * cust.priority
                    
                    # Combined weight (higher = more likely to choose)
                    weight = tau * eta * (1.0 + urgency) * priority_boost
                    weights.append(weight)

                # Roulette wheel selection: choose next customer probabilistically
                total = sum(weights)
                pick = rng.random() * total
                acc = 0.0
                chosen = candidates[-1]  # Default to last candidate
                for cid, w in zip(candidates, weights):
                    acc += w
                    if acc >= pick:
                        chosen = cid
                        break

                sequence.append(chosen)
                unvisited.remove(chosen)
                current = chosen

            # Convert sequence to valid routes (respecting capacity)
            routes = _split_by_capacity(problem, sequence)
            score = evaluate_solution(problem, routes).total_cost

            # Track best solution in this iteration
            if score < iteration_best_cost:
                iteration_best_cost = score
                iteration_best_routes = routes

            # Track global best
            if score < best_cost:
                best_cost = score
                best_routes = routes

        # Pheromone update phase
        
        # Step 1: Evaporate pheromone (decay)
        # This prevents paths from becoming too good and helps escape local optima
        for i in range(n + 1):
            for j in range(n + 1):
                pheromone[i][j] *= (1.0 - rho)
                # Ensure pheromone doesn't become too small
                pheromone[i][j] = max(0.01, pheromone[i][j])

        # Step 2: Reinforce best path found in this iteration
        # This is "elitist" ACO: only the best solution deposits pheromone
        if iteration_best_routes is not None:
            for route in iteration_best_routes:
                prev = 0  # Start from depot
                for node in route:
                    # Deposit pheromone inversely proportional to solution cost
                    pheromone[prev][node] += q / iteration_best_cost
                    prev = node
                # Close the route back to depot
                pheromone[prev][0] += q / iteration_best_cost

        # Record best cost this iteration
        history.append(best_cost)

    assert best_routes is not None
    return best_routes, history


def run(problem: ProblemInstance, seed: int = 0) -> tuple[list[list[int]], list[float]]:
    """
    Standard interface for the ACO algorithm.
    
    Args:
        problem: VRP problem instance
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (routes, history).
    """
    return run_aco(problem, seed=seed)
