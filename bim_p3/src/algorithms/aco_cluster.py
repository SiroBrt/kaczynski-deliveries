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
from sklearn.cluster import KMeans
import numpy as np

try:
    from ..evaluation import evaluate_solution
    from ..problem import ProblemInstance
    from .aco import run_aco
except ImportError:
    from evaluation import evaluate_solution
    from problem import ProblemInstance
    from .aco import run_aco


# ============================================================================
# Route Post-Processing
# ============================================================================

def refine_routes(routes: list[list[int]], problem: ProblemInstance) -> list[list[int]]:
    """Refine routes by selecting customers based on proximity to the current customer.
    
    Uses a nearest-neighbor approach: starting from the first customer in each route,
    iteratively select the nearest unvisited customer.
    """
    refined_routes = []
    for route in routes:
        if not route:
            refined_routes.append(route)
            continue
        
        # Start with the first customer in the route
        nearest_initial_customer = min(
            set(route),
            key=lambda cid: problem.distance(0, cid)
        )
        ordered_route = [nearest_initial_customer]
        unvisited = set(route) - {nearest_initial_customer}
        
        # Greedily select nearest unvisited customer
        current_customer = nearest_initial_customer
        current_time = problem.customers[current_customer - 1].ready_time + problem.customers[current_customer - 1].service_time
        while unvisited:
            nearest = min(
                unvisited,
                key=lambda cid: problem.distance(current_customer, cid) + 0.1 * (problem.customers[cid - 1].due_time - current_time)
            )
            ordered_route.append(nearest)
            unvisited.remove(nearest)
            current_customer = nearest
            current_time = problem.customers[current_customer - 1].ready_time + problem.customers[current_customer - 1].service_time
        
        refined_routes.append(ordered_route)
    return refined_routes

def cluster(
    problem: ProblemInstance,
    seed: int = 0,
    n_clusters: int = 1,
) -> list[ProblemInstance]:
    """
    Returns a Kmeans model fitted to the customer locations. Used to cluster the customers into groups.
    """

    X = np.array([[c.x, c.y] for c in problem.customers])
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    customer_ids = np.array([c.idx for c in problem.customers])
    customer_ids_by_cluster = [customer_ids[labels == i].tolist() for i in range(problem.num_vehicles)]

    subproblems = [
        ProblemInstance(
            # centroids[i][0], 
            # centroids[i][1], 
            problem.depot_x,
            problem.depot_y,
            [problem.customers[j - 1] for j in cluster],
            problem.vehicle_capacity, 
            1,
            problem.vehicle_speed,
            problem.fixed_vehicle_cost,
            problem.weight_delay,
            problem.weight_unserved,
            problem.weight_capacity,
        ) 
        for i, cluster in enumerate(customer_ids_by_cluster)
    ]

    return subproblems

# ============================================================================
# CLUSTER Main Algorithm
# ============================================================================
    
def run_cluster(
    problem: ProblemInstance,
    iterations: int = 60,
    seed: int = 0,
    tries: int = 10
) -> tuple[list[list[int]], list[float]]:

    subproblems = cluster(problem, seed=seed, n_clusters=problem.num_vehicles)
    subproblem_routes = [
        run_aco(
            subproblem, 
            iterations=int(subproblem.num_customers * 2.5),
            ants=int(subproblem.num_customers * 0.85),
            seed=seed
        )
        for subproblem in subproblems
    ]
    routes = [route[0][0] for route in subproblem_routes]
    for subproblem, route in zip(subproblems, routes):
        for i, customer_idx in enumerate(route):
            route[i] = subproblem.customers[customer_idx - 1].idx

    print([route[0] for route in subproblem_routes])
    # refined_routes = refine_routes(routes, problem)
    return [route[0][0] for route in subproblem_routes], []

def run(problem: ProblemInstance, seed: int = 0) -> tuple[list[list[int]], list[float]]:
    """
    Standard interface for the ACO algorithm.
    
    Args:
        problem: VRP problem instance
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (routes, history).
    """
    return run_cluster(problem, seed=seed)