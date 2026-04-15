# Bio-inspired Optimisation of Urban Delivery (VRPTW)

## Overview

A delivery company wants to plan deliveries in a simulated city with multiple customers, capacity-limited vehicles, and time constraints. The goal is to design and compare bio-inspired methods to find high-quality solutions in a sufficiently realistic setting.

This assignment tackles a simplified variant of the **Vehicle Routing Problem with Time Windows (VRPTW)** using **ACO**, **PSO**, or a **hybrid combination** of both.

### Learning Objectives

By the end of the assignment, you should be able to:

1. Model a constrained combinatorial optimisation problem
2. Design a suitable representation for PSO and/or ACO
3. Implement and evaluate at least one bio-inspired algorithm
4. Experimentally analyse the effect of hyperparameters and instance size
5. Communicate results with metrics, visualisations, and critical discussion

## Problem Definition

### Scenario

You are given:

- A central depot
- A set of customers distributed on a 2D map
- Each customer has:
  - coordinates `(x, y)`
  - demand (quantity to deliver)
  - time window `[ready_time, due_time]`
  - service time (duration of delivery)
  - priority level
- A fleet of vehicles, each with:
  - maximum capacity
  - fixed speed
  - fixed usage cost
  - optional range limit

The task is to build routes that start and end at the depot, visiting each customer to serve them within their constraints.

### Objective

Minimize a total cost function combining multiple objectives:

$$
\text{cost} = w_1 \cdot \text{total distance} +
              w_2 \cdot \text{accumulated delay} +
              w_3 \cdot \text{vehicles used} +
              w_4 \cdot \text{penalties}
$$

Where penalties include:
- Exceeding vehicle capacity
- Arriving outside a customer's time window
- Leaving customers unserved
- Other domain-specific constraints

### Constraints

Solutions must satisfy the following:

1. Each customer is visited at most once
2. All customers must be served or explicitly penalised
3. Each route starts and ends at the depot
4. The sum of demands on a route must not exceed vehicle capacity
5. Time windows and service times must be respected

## Solution Approach

Choose one or more of the following approaches:

### Option A: Ant Colony Optimization (ACO)

Implement ACO to build delivery routes using pheromone trails.

Design considerations:
- Where to place pheromones (edges, customers, routes)?
- What heuristic information to use (distance, urgency, priority)?
- How to handle pheromone evaporation?
- Whether to implement elitism
- How to repair infeasible solutions

### Option B: Particle Swarm Optimization (PSO)

Implement PSO in one of these ways:
- Optimize a permutation or priority-based encoding
- Partition customers among vehicles
- Hyper-tune ACO parameters
- Locate hubs or refueling stations
- Hybrid PSO + ACO approach

### Option C: Hybrid Strategy

Combine PSO and ACO:
- PSO tunes ACO parameters (α, β, ρ, q)
- PSO finds initial customer partitions, ACO optimizes each sub-route
- ACO builds solutions, PSO refines a continuous representation

## Deliverables

### 1. Source Code

Submit a repository with:
- Modular, well-commented source code
- Clear execution instructions
- Reproducible configuration (seeds, random generator setup)

### 2. Report (~5 pages)

Document:
- Problem modelling and formulation
- Solution representation
- Objective function design
- Algorithm description and design decisions
- Experimental setup and results
- Conclusions

### 3. Experiments

Minimum requirements:
- **Multiple instance sizes**: At least 3 sizes (e.g., 20, 50, 100 customers)
- **Statistical rigor**: At least 5 runs per configuration with different seeds
- **Baseline comparison**: Include comparison with the provided greedy baseline
- **Sensitivity analysis**: Test 2–3 hyperparameters and analyze their impact

---

## Running the code

### What's Included

The package provides a complete skeleton with:

- `src/problem.py` – Problem definition and random instance generator
- `src/evaluation.py` – Solution evaluation and cost calculation
- `src/algorithms/` – Algorithm implementations (baseline, ACO, PSO)
- `src/visualization.py` – Route visualization and convergence plots
- `src/app.py` – WebUI (Streamlit) and CLI interfaces
- `problems/` – Example YAML configuration files

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-Line Usage

Quick examples:

```bash
# Run baseline on random 25-customer problem
python -m src.app --algorithm baseline --customers 25 --vehicles 4 --seed 7

# Run ACO with custom evaluation weights
python -m src.app --algorithm aco --customers 25 --vehicles 4 \
  --weight-delay 8.0 --weight-unserved 180.0

# Load a predefined YAML problem
python -m src.app --algorithm aco --config-file problems/medium.yaml
```

### Run the Web Interface

A web interface is also available:

```bash
streamlit run src/app.py
```

This opens an interactive web UI where you can:
- Select algorithms (dynamically discovered from `src/algorithms/`)
- Configure problem size and vehicle parameters
- Adjust evaluation weights
- Load YAML problem configurations
- Visualize routes and convergence


### Algorithm Overview

- **baseline** – Greedy insertion; use as a reference point
- **aco** – Ant Colony Optimization with pheromone trails
- **pso** – Particle Swarm Optimization with priority encoding

### Extension Ideas

Consider implementing:

- Change the objective function weights or structure
- Add more realistic penalties (time window violations, partial service)
- Tune hyperparameters systematically
- Combine PSO and ACO into a hybrid
- Implement local search operators (2-opt, relocate, 3-opt)
- Dynamic problem scenarios (customers appearing mid-delivery)
- Multiple depots or vehicle types
- Route animation or real-time visualization

---

## Advanced Features

This package includes two powerful features for flexible problem solving:

### 1. Parameterizable Evaluation Weights

Control how different cost factors are weighted in the optimization. The total cost is calculated as:

```
Total Cost = distance × 1.0
           + delay × weight_delay
           + vehicles_used × fixed_vehicle_cost
           + unserved_customers × weight_unserved
           + capacity_violations × weight_capacity_violation
```

Default weights: `weight_delay=10.0`, `weight_unserved=200.0`, `weight_capacity_violation=150.0`

#### Using Custom Weights from CLI

```bash
# Minimize delays
python -m src.app --algorithm aco --customers 25 --vehicles 4 \
  --weight-delay 0.1 --weight-unserved 200.0

# Maximize coverage
python -m src.app --algorithm pso --customers 25 --vehicles 4 \
  --weight-unserved 500.0 --weight-delay 5.0

# Lean operations (cost and coverage focused)
python -m src.app --algorithm baseline --customers 25 --vehicles 4 \
  --weight-delay 20.0 --weight-unserved 300.0
```

#### Using Custom Weights from Streamlit

Open the web interface and use the weight sliders in the sidebar:
- **Delay penalty** (0–100, default: 10.0)
- **Unserved penalty** (0–500, default: 200.0)
- **Capacity violation penalty** (0–500, default: 150.0)

### 2. YAML Configuration Support

Define deterministic problem instances instead of using random generation. This enables reproducibility and real-world scenario modeling.

#### YAML File Format

```yaml
metadata:
  name: Problem Name
  description: Brief description
  vehicles: 2              # Number of vehicles
  vehicle_capacity: 50     # Max capacity per vehicle
  vehicle_speed: 60        # Speed (km/h)

depot:
  x: 0.0                   # Depot X coordinate
  y: 0.0                   # Depot Y coordinate

customers:
  - id: 1
    x: 10.0                # Customer X coordinate
    y: 10.0                # Customer Y coordinate
    demand: 5              # Units demanded
    ready_time: 0.0        # Earliest service time
    due_time: 100.0        # Latest service time
    service_time: 5.0      # Time to serve
    priority: 1            # Priority level (1-3)
  
  - id: 2
    x: 20.0
    y: 25.0
    demand: 8
    ready_time: 5.0
    due_time: 120.0
    service_time: 7.0
    priority: 1
```

#### Example YAML Files

Two example files are provided:

1. **Simple problem** (`problems/simple.yaml`) – 5 customers
   ```bash
   python -m src.app --algorithm baseline --config-file problems/simple.yaml
   ```

2. **Medium problem** (`problems/medium.yaml`) – 10 customers
   ```bash
   python -m src.app --algorithm aco --config-file problems/medium.yaml
   ```

#### Using YAML with CLI

Basic usage:
```bash
python -m src.app --algorithm <name> --config-file <path.yaml>
```

Combined with custom weights:
```bash
python -m src.app --algorithm aco --config-file problems/medium.yaml \
  --weight-delay 12.0 --weight-unserved 220.0 --weight-capacity 160.0
```

Note: When `--config-file` is specified, `--customers`, `--vehicles`, and `--capacity` arguments are ignored.

#### Using YAML with Streamlit

1. Start the interface: `streamlit run src/app.py`
2. In the sidebar, select "Problem source" → "YAML file"
3. Click "Upload YAML configuration file" and select your file
4. Configure algorithm, weights, and seed
5. Click "Run"

#### Creating Custom YAML Files

1. Choose depot coordinates (central location)
2. Define each customer with:
   - Unique ID (starting from 1)
   - Coordinates (x, y) – normalized to similar scale
   - Demand (typical: 1–20)
   - Time window (ready_time, due_time) – in same time units
   - Service time (typical: 1–10 time units)
   - Priority (1–3)
3. Save as `.yaml` or `.yml` file

#### Complete CLI Examples

```bash
# Default settings (random problem)
python -m src.app --algorithm aco --customers 20 --vehicles 3

# Custom weights with random problem
python -m src.app --algorithm pso --customers 25 --vehicles 4 \
  --weight-delay 8.0 --weight-unserved 180.0 --weight-capacity 120.0

# YAML problem with default weights
python -m src.app --algorithm baseline --config-file problems/simple.yaml

# YAML problem with custom weights
python -m src.app --algorithm aco --config-file problems/medium.yaml \
  --weight-delay 12.0 --weight-unserved 220.0 --weight-capacity 160.0

# Reproducible experiment
python -m src.app --algorithm aco --config-file problems/medium.yaml --seed 42 \
  --weight-delay 10.0 --weight-unserved 200.0 --weight-capacity 150.0
```