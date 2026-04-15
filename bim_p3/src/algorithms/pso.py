"""
Particle Swarm Optimization (PSO) Algorithm

A population-based metaheuristic inspired by the collective behavior of bird flocking
and fish schooling. Particles move through a continuous space, influenced by:
- Their own best position found so far (personal best / pbest)
- The swarm's best position found so far (global best / gbest)
- Random acceleration toward both

Representation:
- Each particle is a vector of continuous values (one per customer)
- Values are interpreted as priorities: higher values = higher routing priority
- The priority vector is decoded into actual routes greedily

Key features:
- Continuous optimization (unlike discrete ACO)
- Simple update equations with few hyperparameters
- Good balance between exploration and exploitation
"""

from __future__ import annotations

import random

try:
    from ..evaluation import evaluate_solution
    from ..problem import ProblemInstance
except ImportError:
    from evaluation import evaluate_solution
    from problem import ProblemInstance


# ============================================================================
# Solution Decoding
# ============================================================================


def _decode(problem: ProblemInstance, priorities: list[float]) -> list[list[int]]:
    """
    Convert a priority vector into actual routes.
    
    Customers are visited in decreasing order of priority. Routes are formed
    by greedily adding customers to the current route until either capacity
    or time window constraints are violated.
    
    Args:
        problem: VRP problem instance
        priorities: List of priority values (one per customer, index i-1 for customer i)
        
    Returns:
        List of routes (each route is a list of customer IDs).
    """
    # Sort customers by priority (highest first)
    order = sorted(
        range(1, problem.num_customers + 1),
        key=lambda i: priorities[i - 1],
        reverse=True
    )
    
    routes: list[list[int]] = []
    current_route: list[int] = []
    current_load = 0
    current_time = 0.0
    current = 0  # Current location (0 = depot)

    # Visit customers in priority order, building routes greedily
    for cid in order:
        cust = problem.customers[cid - 1]
        dist = problem.distance(current, cid)
        arrival = current_time + dist / problem.vehicle_speed
        would_load = current_load + cust.demand
        
        # Check if violating constraints would be "too expensive"
        # (allowing +30 time units of slack for late arrivals)
        too_late = arrival > cust.due_time + 30
        
        # Start a new route if constraints would be violated
        if current_route and (would_load > problem.vehicle_capacity or too_late):
            routes.append(current_route)
            current_route = []
            current_load = 0
            current_time = 0.0
            current = 0

        # Add customer to current route
        current_route.append(cid)
        arrival = current_time + problem.distance(current, cid) / problem.vehicle_speed
        current_time = max(arrival, cust.ready_time) + cust.service_time
        current_load += cust.demand
        current = cid

    # Add the last route if non-empty
    if current_route:
        routes.append(current_route)

    # Limit to available vehicles
    return routes[: problem.num_vehicles]


# ============================================================================
# PSO Main Algorithm
# ============================================================================


def run_pso(
    problem: ProblemInstance,
    particles: int = 25,
    iterations: int = 80,
    inertia: float = 0.72,
    c1: float = 1.49,
    c2: float = 1.49,
    seed: int = 0,
) -> tuple[list[list[int]], list[float]]:
    """
    Run the Particle Swarm Optimization algorithm.
    
    Hyperparameters:
        particles: Number of particles (population size)
                   More particles → more exploration, slower per iteration
        iterations: Number of iterations (update steps)
                    More iterations → better convergence (if not trapped)
        inertia: Weight on velocity (previous motion)
                 w ≈ 0.4-0.9 typical; higher → more exploration, more persistence
        c1: "Cognitive" (personal best) coefficient
            Higher c1 → particles trust their own best more
        c2: "Social" (global best) coefficient
            Higher c2 → particles follow the swarm more closely
            Typical: c1 ≈ c2 ≈ 1.49 (empirically tuned for good convergence)
        seed: Random seed for reproducibility
    
    Velocity update equation:
        v[t+1] = w*v[t] + c1*r1*(pbest - x[t]) + c2*r2*(gbest - x[t])
        where r1, r2 are random ∈ [0, 1]
    
    Position update:
        x[t+1] = x[t] + v[t+1]
    
    Args:
        problem: VRP problem instance
        particles: Number of particles
        iterations: Number of iterations
        inertia: Inertia weight w
        c1: Cognitive coefficient
        c2: Social coefficient
        seed: Random seed
        
    Returns:
        Tuple of (best_routes, history) where history is the best cost per iteration.
    """
    rng = random.Random(seed)
    dim = problem.num_customers
    
    # Initialize particle positions (priorities) randomly
    positions = [
        [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        for _ in range(particles)
    ]
    
    # Initialize particle velocities (small random values)
    velocities = [
        [rng.uniform(-0.1, 0.1) for _ in range(dim)]
        for _ in range(particles)
    ]
    
    # Track personal best for each particle
    pbest = [p[:] for p in positions]
    pbest_scores = []

    # Evaluate initial population
    for p in positions:
        routes = _decode(problem, p)
        pbest_scores.append(evaluate_solution(problem, routes).total_cost)

    # Find global best
    best_idx = min(range(particles), key=lambda i: pbest_scores[i])
    gbest = pbest[best_idx][:]
    gbest_score = pbest_scores[best_idx]
    history = [gbest_score]

    # Main optimization loop
    for iteration in range(iterations):
        # Update each particle
        for i in range(particles):
            # Update each dimension of the particle
            for d in range(dim):
                # Random coefficients for stochasticity
                r1 = rng.random()
                r2 = rng.random()
                
                # Velocity update: combination of inertia, personal best, and social influence
                velocities[i][d] = (
                    inertia * velocities[i][d]                    # Momentum
                    + c1 * r1 * (pbest[i][d] - positions[i][d])  # Cognitive term
                    + c2 * r2 * (gbest[d] - positions[i][d])     # Social term
                )
                
                # Position update
                positions[i][d] += velocities[i][d]

            # Evaluate new position by decoding to routes and scoring
            routes = _decode(problem, positions[i])
            score = evaluate_solution(problem, routes).total_cost
            
            # Update personal best
            if score < pbest_scores[i]:
                pbest[i] = positions[i][:]
                pbest_scores[i] = score
                
                # Update global best if this is the best found so far
                if score < gbest_score:
                    gbest = positions[i][:]
                    gbest_score = score
        
        # Record best cost this iteration
        history.append(gbest_score)

    # Return the best solution found
    return _decode(problem, gbest), history


def run(problem: ProblemInstance, seed: int = 0) -> tuple[list[list[int]], list[float]]:
    """
    Standard interface for the PSO algorithm.
    
    Args:
        problem: VRP problem instance
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (routes, history).
    """
    return run_pso(problem, seed=seed)
