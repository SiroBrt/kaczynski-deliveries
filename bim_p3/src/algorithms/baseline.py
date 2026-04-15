"""
Greedy Baseline Algorithm

A simple greedy heuristic for the VRP that constructs routes sequentially.
For each vehicle, greedily selects the next feasible customer that minimizes
a score combining distance, wait time, lateness, and priority.

This serves as a reference point for comparison with more sophisticated algorithms.
"""

from __future__ import annotations

try:
    from ..problem import ProblemInstance
except ImportError:
    from problem import ProblemInstance


def greedy_baseline(problem: ProblemInstance) -> list[list[int]]:
    """
    Construct routes greedily, one vehicle at a time.
    
    For each vehicle, repeatedly add the feasible customer with the best score.
    A customer is feasible if:
    1. It hasn't been visited yet
    2. Adding it doesn't violate the vehicle capacity
    
    Score formula: distance + 0.5*wait_time + 3.0*lateness - 2.0*priority
    (Lower scores are better; high priority customers get a bonus)
    
    Args:
        problem: The VRP problem instance
        
    Returns:
        List of routes, one per vehicle.
    """
    unvisited = set(range(1, problem.num_customers + 1))
    routes: list[list[int]] = []

    # Build one route per vehicle
    for _ in range(problem.num_vehicles):
        route: list[int] = []
        current = 0                           # Start at depot
        remaining_capacity = problem.vehicle_capacity
        current_time = 0.0

        # Greedily add customers to this route
        while unvisited:
            feasible = []
            
            # Find all feasible customers (capacity constraint + demand check)
            for cid in unvisited:
                cust = problem.customers[cid - 1]
                
                # Skip if adding this customer exceeds capacity
                if cust.demand > remaining_capacity:
                    continue
                
                # Calculate metrics for this customer
                dist = problem.distance(current, cid)
                arrival = current_time + dist / problem.vehicle_speed
                wait = max(0.0, cust.ready_time - arrival)
                lateness = max(0.0, arrival - cust.due_time)
                
                # Score: lower is better
                # Distance is the main criterion
                # Wait time and lateness are penalties
                # High priority customers get a discount
                score = dist + 0.5 * wait + 3.0 * lateness - 2.0 * cust.priority
                feasible.append((score, cid))

            # If no feasible customers, move to next vehicle
            if not feasible:
                break

            # Select the customer with the best score
            feasible.sort()
            _, nxt = feasible[0]
            cust = problem.customers[nxt - 1]
            
            # Update route and vehicle state
            dist = problem.distance(current, nxt)
            arrival = current_time + dist / problem.vehicle_speed
            current_time = max(arrival, cust.ready_time) + cust.service_time
            remaining_capacity -= cust.demand
            route.append(nxt)
            unvisited.remove(nxt)
            current = nxt

        routes.append(route)

    # Handle any remaining unvisited customers (assign to last route)
    if unvisited:
        if routes:
            routes[-1].extend(sorted(unvisited))
        else:
            routes.append(sorted(unvisited))

    return routes


def run(problem: ProblemInstance, seed: int = 0) -> tuple[list[list[int]], list[float]]:
    """
    Standard interface for the baseline algorithm.
    
    Args:
        problem: VRP problem instance
        seed: Ignored (baseline is deterministic)
        
    Returns:
        Tuple of (routes, history) where history is empty for baseline.
    """
    return greedy_baseline(problem), []
