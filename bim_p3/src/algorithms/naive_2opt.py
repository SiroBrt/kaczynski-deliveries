from sklearn.cluster import KMeans
import numpy as np
import random
import time
import math

try:
    from ..evaluation import evaluate_solution
    from ..problem import ProblemInstance, Customer
    from heavy import auto2opt
except ImportError:
    from evaluation import evaluate_solution
    from problem import ProblemInstance, Customer
    from algorithms.heavy import auto2opt

# def greedy_cluster(problem: ProblemInstance,
#     n_clusters: int = 1,
#     ) -> list[list[int]]:
#     """
#     Copy of greedy_baseline but better
#     """
#     if problem.customers == []:
#         return [[]]
#     unvisited = problem.customers
#     routes: list[list[int]] = []
#     wait_weight = 0.5
#     lateness_weight = 3.0
#
#     # Build one route per vehicle
#     for _ in range(n_clusters):
#         route: list[int] = []
#         depot = Customer(
#                 idx=0,
#                 x=problem.depot_x,
#                 y=problem.depot_y,
#                 demand=0,
#                 ready_time=0.0,
#                 due_time=float("inf"),
#                 service_time=0.0,
#                 priority=0
#                 )
#         current = depot # Start at depot
#         remaining_capacity = problem.vehicle_capacity
#         current_time = 0.0
#
#         # Greedily add customers to this route
#         while unvisited:
#             # Find all feasible customers (capacity constraint + demand check)
#             best = (float("inf"),depot)
#             for cust in unvisited:
#                 # Skip if adding this customer exceeds capacity
#                 if cust.demand > remaining_capacity:
#                     continue
#                 
#                 # Calculate metrics for this customer
#                 dist = math.hypot(current.x - cust.x, current.y - cust.y)
#                 arrival = current_time + dist / problem.vehicle_speed
#                 wait = max(0.0, cust.ready_time - arrival)
#                 lateness = max(0.0, arrival - cust.due_time)
#                 
#                 # Score: lower is better
#                 # Distance is the main criterion
#                 # Wait time and lateness are penalties
#                 score = dist + wait_weight * wait + lateness_weight * lateness
#                 if score < best[0]:
#                     best = (score, cust)
#
#             # If no feasible customers, move to next vehicle
#             if best[0] == float("inf"):
#                 break
#
#             # Select the customer with the best score
#             cust = best[1]
#             
#             # Update route and vehicle state
#             dist = math.hypot(current.x - cust.x, current.y - cust.y)
#             arrival = current_time + dist / problem.vehicle_speed
#             current_time = max(arrival, cust.ready_time) + cust.service_time
#             remaining_capacity -= cust.demand
#             route.append(cust.idx)
#             unvisited.remove(cust)
#             current = cust
#
#         routes.append(route)
#
#     # Handle any remaining unvisited customers (assign to last route)
#     if unvisited:
#         if routes:
#             routes[-1].extend([c.idx for c in unvisited])
#         else:
#             routes.append([c.idx for c in unvisited])
#
#     return routes
#

def cluster(problem: ProblemInstance, seed: int = 0, n_clusters: int = 1):
    """
    Returns a Kmeans model fitted to the customer locations. Used to cluster the customers into a given number of groups.
    """
    X = np.array([[c.x, c.y] for c in problem.customers])
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    kmeans.fit(X)
    labels = kmeans.labels_
    customer_ids = np.array([c.idx for c in problem.customers])
    routes = [customer_ids[labels == i].tolist() for i in range(problem.num_vehicles)]
    return routes

def find_route(
    old_problem: ProblemInstance,
    nodes:[Customer]
    ):
    new_problem = ProblemInstance(
        depot_x=old_problem.depot_x,
        depot_y=old_problem.depot_y,
        customers=nodes,
        vehicle_capacity=old_problem.vehicle_capacity,
        num_vehicles=old_problem.num_vehicles,
        vehicle_speed=old_problem.vehicle_speed,
    )
    
    if len(nodes)<2:
        return[[c.idx for c in nodes]]

    fake_result = cluster(problem=new_problem,n_clusters=min(old_problem.num_vehicles,len(nodes))) 
    result = []
    for route in fake_result:
        result.append(auto2opt(old_problem,route))
    return result

def choose_clients_alternating(problem: ProblemInstance, included_clients):
    """
    included_clients works as an initial guess for the best subset of customers.
    We will then add or remove the most beneficial one at a time until we get to a minimum

    Inspired by variable selection techinques in statistics
    """
    included_eval = evaluate_solution(problem,find_route(problem,included_clients)).total_cost
    current_best = float("inf")
    while True:
        for i in problem.customers:
            if i in included_clients:
                possible_clients = [c for c in included_clients]
                possible_clients.remove(i)
            else:
                possible_clients = [c for c in included_clients] + [i]
            possible_evaluation = evaluate_solution(problem,find_route(problem,possible_clients)).total_cost
            if possible_evaluation <= current_best:
                to_change = i
                current_best = possible_evaluation
        if current_best < included_eval:
            if to_change in included_clients:
                included_clients.remove(to_change)
            else:
                included_clients.append(to_change)
            included_eval = current_best
        else:
            break
    return find_route(problem,included_clients)

    
def run_selection(
    problem: ProblemInstance,
    seed: int = 0,
    tries: int = 3
) -> tuple[list[list[int]], list[float]]:
    random.seed(seed)
    t1 = time.time()
    rutas = []
    evals = []
    rutas.append(choose_clients_alternating(problem,[])) # gives a first solution
    evals.append(evaluate_solution(problem,rutas[0]).total_cost)
    if len(problem.customers) < 30:
        rutas.append(choose_clients_alternating(problem,[c for c in problem.customers])) # improves baseline
        evals.append(evaluate_solution(problem,rutas[1]).total_cost)


    # random starts to get different minima
    for _ in range(tries):
        rutas.append(choose_clients_alternating(problem,random.sample(problem.customers,2*len(problem.customers)//3)))
        evals.append(evaluate_solution(problem,rutas[-1]).total_cost)
    t2 = time.time()
    print(f"naive 2opt: {round(t2-t1,4)} sec, {round(min(evals))}")
    return rutas[evals.index(min(evals))], []



def run(problem: ProblemInstance, seed: int = 0) -> tuple[list[list[int]], list[float]]:
    """
    Use naive approach to select which customers to serve. Then, use greedy_baseline to find route for those clients
    
    Args:
        problem: VRP problem instance
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (routes, history).
    """
    return run_selection(problem, seed=seed)
