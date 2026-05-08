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
except ImportError:
    from evaluation import evaluate_solution
    from problem import ProblemInstance


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
        while unvisited:
            nearest = min(
                unvisited,
                key=lambda cid: problem.distance(current_customer, cid)
            )
            ordered_route.append(nearest)
            unvisited.remove(nearest)
            current_customer = nearest
        
        refined_routes.append(ordered_route)
    return refined_routes

def cluster(
    problem: ProblemInstance,
    seed: int = 0,
) -> KMeans:
    """
    Returns a Kmeans model fitted to the customer locations. Used to cluster the customers into groups.
    For now, each cluster will be assigned to a single vehicle, and the customers in that cluster will be served by that vehicle.
    """
    # 1. Extract only 2D positions from customer data for clustering
    X = np.array([[c.x, c.y] for c in problem.customers])

    # 2. Define the model and fit the model to the data
    kmeans = KMeans(n_clusters=problem.num_vehicles, random_state=seed)
    kmeans.fit(X)
    return kmeans

# ============================================================================
# CLUSTER Main Algorithm
# ============================================================================
    
def run_cluster(
    problem: ProblemInstance,
    iterations: int = 60,
    seed: int = 0,
    tries: int = 10
) -> tuple[list[list[int]], list[float]]:

    customer_ids = np.array([c.idx for c in problem.customers])
    kmeans: KMeans = cluster(problem, seed=seed)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # print(f"Cluster Labels: {labels}")
    # print(f"Centroids:\n{centroids}")
    # print(customer_ids[labels == 0])
    
    routes = [customer_ids[labels == i].tolist() for i in range(problem.num_vehicles)]
    # print(routes)
    refined_routes = refine_routes(routes, problem)
    return refined_routes, []

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