"""
Problem Definition and Instance Generation

Defines the Vehicle Routing Problem with Time Windows (VRPTW) structure
and provides a random instance generator and YAML-based instance loader
for testing and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any
import math
import random

try:
    import yaml
except ImportError:
    yaml = None


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class Customer:
    """A customer location to be served.
    
    Attributes:
        idx: Customer identifier (1-indexed)
        x, y: Coordinates on a 2D map
        demand: Amount of goods to deliver
        ready_time: Earliest time the customer can be served
        due_time: Latest time the customer must be served by (deadline)
        service_time: Time required to serve this customer
        priority: Priority level (1=low, 3=high); high-priority jobs attract more attention
    """
    idx: int
    x: float
    y: float
    demand: int
    ready_time: float
    due_time: float
    service_time: float
    priority: int


@dataclass
class ProblemInstance:
    """A Vehicle Routing Problem instance with time windows.
    
    Attributes:
        depot_x, depot_y: Coordinates of the central depot (start/end point)
        customers: List of Customer objects to be served
        vehicle_capacity: Maximum load each vehicle can carry
        num_vehicles: Number of available vehicles
        vehicle_speed: Travel speed (distance per time unit; default 1.0)
        fixed_vehicle_cost: Operating cost for each vehicle deployed
    """
    depot_x: float
    depot_y: float
    customers: List[Customer]
    vehicle_capacity: int
    num_vehicles: int
    vehicle_speed: float = 1.0
    fixed_vehicle_cost: float = 15.0

    def distance(self, a: int, b: int) -> float:
        """
        Calculate Euclidean distance between two locations.
        
        Args:
            a: Location index (0 = depot, 1..n = customers)
            b: Location index
            
        Returns:
            Euclidean distance between the two locations.
        """
        ax, ay = self.coords(a)
        bx, by = self.coords(b)
        return math.hypot(ax - bx, ay - by)

    def coords(self, idx: int) -> tuple[float, float]:
        """
        Get (x, y) coordinates for a location.
        
        Args:
            idx: Location index (0 = depot, 1..n = customers)
            
        Returns:
            Tuple of (x, y) coordinates.
        """
        if idx == 0:
            return self.depot_x, self.depot_y
        c = self.customers[idx - 1]
        return c.x, c.y

    @property
    def num_customers(self) -> int:
        """Return the number of customers in this problem instance."""
        return len(self.customers)


# ============================================================================
# Instance Generation
# ============================================================================


def generate_instance(
    num_customers: int = 30,
    num_vehicles: int = 4,
    vehicle_capacity: int = 30,
    width: int = 100,
    height: int = 100,
    seed: int = 0,
) -> ProblemInstance:
    """
    Generate a random VRPTW problem instance.
    
    Creates a problem with customers randomly distributed on a square map,
    each with random demand and time windows. The depot is placed at the center.
    
    Args:
        num_customers: Number of customers to generate
        num_vehicles: Number of vehicles to deploy
        vehicle_capacity: Maximum capacity per vehicle
        width: Width of the problem space (x-axis range: [0, width])
        height: Height of the problem space (y-axis range: [0, height])
        seed: Random seed for reproducibility
        
    Returns:
        A ProblemInstance with randomly generated customers and depot.
    """
    rng = random.Random(seed)
    customers: List[Customer] = []
    depot_x, depot_y = width / 2.0, height / 2.0

    # Generate customer data
    for i in range(1, num_customers + 1):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        demand = rng.randint(1, 8)
        
        # Time window: customers have a ready time and a due time
        start = rng.uniform(0, 120)           # Ready time (not before this)
        window = rng.uniform(40, 120)         # Time window width
        due = start + window                  # Due time (not after this)
        
        service = rng.uniform(3, 12)          # Service duration at location
        priority = rng.randint(1, 3)          # Priority: 1 (low) to 3 (high)
        
        customers.append(
            Customer(
                idx=i,
                x=x,
                y=y,
                demand=demand,
                ready_time=start,
                due_time=due,
                service_time=service,
                priority=priority,
            )
        )

    # Create and return the problem instance
    return ProblemInstance(
        depot_x=depot_x,
        depot_y=depot_y,
        customers=customers,
        vehicle_capacity=vehicle_capacity,
        num_vehicles=num_vehicles,
    )


# ============================================================================
# YAML-based Problem Loading
# ============================================================================


def load_problem_from_yaml(filepath: str) -> ProblemInstance:
    """
    Load a VRP problem instance from a YAML configuration file.
    
    The YAML file should contain:
      - metadata: Problem name, description, number of vehicles, capacity, vehicle speed
      - depot: Depot coordinates (x, y)
      - customers: List of customer definitions with coordinates, demand, time windows, etc.
    
    Example structure:
        metadata:
          vehicles: 2
          vehicle_capacity: 50
          vehicle_speed: 60
        depot:
          x: 0.0
          y: 0.0
        customers:
          - id: 1
            x: 10.0
            y: 10.0
            demand: 5
            ready_time: 0.0
            due_time: 100.0
            service_time: 5.0
            priority: 1
    
    Args:
        filepath: Path to the YAML configuration file
        
    Returns:
        A ProblemInstance loaded from the YAML file
        
    Raises:
        ValueError: If YAML module is not available or file format is invalid
    """
    if yaml is None:
        raise ValueError(
            'PyYAML is not installed. Install it with: pip install pyyaml'
        )
    
    # Load YAML file
    with open(filepath, 'r') as f:
        config: dict[str, Any] = yaml.safe_load(f)
    
    if not config:
        raise ValueError('YAML file is empty')
    
    # Extract metadata
    metadata = config.get('metadata', {})
    vehicle_count = metadata.get('vehicles', 1)
    vehicle_capacity = metadata.get('vehicle_capacity', 50)
    vehicle_speed = metadata.get('vehicle_speed', 1.0)
    
    # Extract depot coordinates
    depot_conf = config.get('depot', {})
    depot_x = float(depot_conf.get('x', 0.0))
    depot_y = float(depot_conf.get('y', 0.0))
    
    # Extract customers
    customers_conf = config.get('customers', [])
    customers: List[Customer] = []
    
    for cust_data in customers_conf:
        customer = Customer(
            idx=int(cust_data['id']),
            x=float(cust_data.get('x', 0.0)),
            y=float(cust_data.get('y', 0.0)),
            demand=int(cust_data.get('demand', 1)),
            ready_time=float(cust_data.get('ready_time', 0.0)),
            due_time=float(cust_data.get('due_time', 1000.0)),
            service_time=float(cust_data.get('service_time', 1.0)),
            priority=int(cust_data.get('priority', 1)),
        )
        customers.append(customer)
    
    # Sort customers by ID to ensure consistency
    customers.sort(key=lambda c: c.idx)
    
    # Create and return problem instance
    return ProblemInstance(
        depot_x=depot_x,
        depot_y=depot_y,
        customers=customers,
        vehicle_capacity=vehicle_capacity,
        num_vehicles=vehicle_count,
        vehicle_speed=vehicle_speed,
    )
