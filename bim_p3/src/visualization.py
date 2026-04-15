"""
Visualization Utilities

Provides matplotlib-based plotting functions for:
  - Route visualization on a 2D map
  - Algorithm convergence history
"""

from __future__ import annotations

import matplotlib.pyplot as plt

try:
    from .problem import ProblemInstance
except ImportError:
    from problem import ProblemInstance


def plot_routes(problem: ProblemInstance, routes: list[list[int]]):
    """
    Visualize delivery routes on a 2D map.
    
    Displays:
    - Depot location (large square marker)
    - Customer locations (scatter points)
    - Customer labels (ID numbers)
    - Route paths (line segments connecting customers)
    
    Args:
        problem: The VRP problem instance
        routes: List of routes (each route is a list of customer IDs)
        
    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Plot depot
    ax.scatter([problem.depot_x], [problem.depot_y], marker='s', s=120, label='Depot')

    # Plot all customer locations
    xs = [c.x for c in problem.customers]
    ys = [c.y for c in problem.customers]
    ax.scatter(xs, ys, s=40, label='Customers')

    # Add customer ID labels
    for c in problem.customers:
        ax.text(c.x + 1, c.y + 1, str(c.idx), fontsize=8)

    # Draw routes as line segments
    for route in routes:
        if not route:
            continue
        # Each route starts and ends at depot (index 0)
        points = [0] + route + [0]
        px = [problem.coords(i)[0] for i in points]
        py = [problem.coords(i)[1] for i in points]
        ax.plot(px, py, linewidth=1.5)

    # Format plot
    ax.set_title('Delivery routes')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_history(history: list[float], title: str = 'Cost evolution'):
    """
    Plot algorithm convergence history.
    
    Shows the best cost found at each iteration, useful for analyzing
    algorithm performance and convergence behavior.
    
    Args:
        history: List of best cost values (one per iteration)
        title: Title for the plot
        
    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history)
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best cost')
    ax.grid(True, alpha=0.3)
    return fig
