"""
Solution Evaluation

Evaluates the quality and feasibility of proposed routes for a VRP instance.
Computes total cost using a weighted combination of:
  - Distance traveled
  - Time window violations (delay penalties)
  - Number of vehicles used
  - Unserved customers
  - Capacity violations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    from .problem import ProblemInstance
except ImportError:
    from problem import ProblemInstance


# ============================================================================
# Result Data Structure
# ============================================================================


@dataclass
class EvaluationResult:
    """Detailed metrics for a solution.
    
    Attributes:
        total_cost: Final cost value (weighted sum of all penalties)
        total_distance: Total distance traveled by all vehicles
        total_delay: Sum of all time window violations (positive only)
        vehicles_used: Number of vehicles with non-empty routes
        unserved_customers: Number of customers not visited
        capacity_violations: Number of routes exceeding vehicle capacity
        visited_customers: Set of customer IDs that were served
    """
    total_cost: float
    total_distance: float
    total_delay: float
    vehicles_used: int
    unserved_customers: int
    capacity_violations: int
    visited_customers: set[int]


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_solution(
    problem: ProblemInstance,
    routes: list[list[int]],
    weight_delay: float = 10.0,
    weight_unserved: float = 200.0,
    weight_capacity_violation: float = 150.0,
) -> EvaluationResult:
    """
    Evaluate a proposed solution (list of routes) for a VRP instance.
    
    Simulates the execution of all routes, checking feasibility and accumulating
    costs such as distance, time delays, and penalties for violations.
    
    Cost calculation:
        cost = distance * 1.0
              + total_delay * weight_delay
              + vehicles_used * fixed_vehicle_cost
              + unserved_customers * weight_unserved
              + capacity_violations * weight_capacity_violation
    
    Args:
        problem: The VRP problem instance
        routes: List of routes, where each route is a list of customer IDs
        weight_delay: Penalty weight for time window violations (default 10.0)
        weight_unserved: Penalty weight for unserved customers (default 200.0)
        weight_capacity_violation: Penalty weight for capacity violations (default 150.0)
        
    Returns:
        EvaluationResult with detailed metrics about the solution.
    """
    visited: set[int] = set()
    total_distance = 0.0
    total_delay = 0.0
    vehicles_used = 0
    capacity_violations = 0

    # Evaluate each route
    for route in routes:
        if not route:
            # Skip empty routes
            continue
            
        vehicles_used += 1
        current = 0              # Start at depot
        current_time = 0.0       # Time of arrival at current location
        load = 0                 # Current load on vehicle

        # Simulate traversal of each customer in the route
        for customer_id in route:
            cust = problem.customers[customer_id - 1]
            
            # Calculate travel time and arrival
            dist = problem.distance(current, customer_id)
            travel_time = dist / problem.vehicle_speed
            arrival = current_time + travel_time
            
            # Respect ready time window
            service_start = max(arrival, cust.ready_time)
            
            # Calculate delay (any arrival after due_time)
            delay = max(0.0, service_start - cust.due_time)
            
            # Update totals
            total_delay += delay
            total_distance += dist
            current_time = service_start + cust.service_time
            load += cust.demand
            current = customer_id
            visited.add(customer_id)

        # Return to depot
        total_distance += problem.distance(current, 0)
        
        # Check for capacity violations
        if load > problem.vehicle_capacity:
            capacity_violations += 1

    # Count unserved customers
    unserved = problem.num_customers - len(visited)
    
    # Compute total cost as weighted sum of penalties
    # Weights can be tuned to emphasize different aspects
    total_cost = (
        total_distance                                                # Base cost: every km counts
        + weight_delay * total_delay                                  # Time window violations
        + problem.fixed_vehicle_cost * vehicles_used                  # Vehicle deployment cost
        + weight_unserved * unserved                                  # Unserved customers
        + weight_capacity_violation * capacity_violations             # Capacity violations
    )

    return EvaluationResult(
        total_cost=total_cost,
        total_distance=total_distance,
        total_delay=total_delay,
        vehicles_used=vehicles_used,
        unserved_customers=unserved,
        capacity_violations=capacity_violations,
        visited_customers=visited,
    )


# ============================================================================
# Utilities
# ============================================================================


def flatten(routes: Iterable[Iterable[int]]) -> list[int]:
    """
    Flatten a list of routes into a single sequence of customers.
    
    Useful for converting from (route1, route2, ...) to all_customers_visited.
    
    Args:
        routes: Iterable of routes, each route is an Iterable of customer IDs
        
    Returns:
        Flat list of all customer IDs across all routes.
    """
    return [node for route in routes for node in route]
