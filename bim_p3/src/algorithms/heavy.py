from sklearn.cluster import KMeans
import numpy as np
import random
import math
import time

try:
    from ..evaluation import evaluate_solution
    from ..problem import ProblemInstance, Customer
except ImportError:
    from evaluation import evaluate_solution
    from problem import ProblemInstance, Customer


def evaluate_truck(
    problem: ProblemInstance,
    route: list[int]
    ):
    '''
    Faster evaluate_solution with only one route.
    Ignores vehicle costs, overload and unserved customers
    (should be the same among all routes with the same customers).
    
    Thought for cheap checking of different routes between the same customers
    '''
    total_distance = 0.0
    total_delay = 0.0

    current = 0              # Start at depot
    current_time = 0.0       # Time of arrival at current location

    # Simulate traversal of each customer in the route
    for customer_id in route:
        cust = next((x for x in problem.customers if x.idx == customer_id), None)
        if cust is None:
            continue

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
        current = customer_id

    # Return to depot
    total_distance += problem.distance(current, 0)

    return total_distance + problem.weight_delay * total_delay

def cluster(problem: ProblemInstance, seed: int = 0, cluster_number: int = 1):
    """
    Returns a Kmeans model fitted to the customer locations. Used to cluster the customers into a given number of groups.
    """
    X = np.array([[c.x, c.y] for c in problem.customers])
    kmeans = KMeans(n_clusters=cluster_number, random_state=seed)
    kmeans.fit(X)
    labels = kmeans.labels_
    customer_ids = np.array([c.idx for c in problem.customers])
    routes = [customer_ids[labels == i].tolist() for i in range(problem.num_vehicles)]
    return routes

def auto2opt(
        problem: ProblemInstance,
        lista: list[int]):
    '''
    Untangles routes taking into account delivery times.
    Since it's greedy it may go for local minima
    '''
    route = [c for c in lista]
    current_distance = evaluate_truck(problem, route)
    while True:
        best_route = []
        best_distance = float("inf")
        possible_route = []
        for i in range(len(lista)-1):
            for j in range(1,len(lista)-i):
                possible_route = route[:i]+route[i:i+j+1][::-1]+route[i+j+1:]
                possible_cost = evaluate_truck(problem, possible_route)
                if possible_cost < best_distance:
                    best_route = [c for c in possible_route]
                    best_distance = possible_cost
        if best_distance < current_distance:
            route = best_route
            current_distance = best_distance
        else:
            break
    return route


def find_route(old_problem: ProblemInstance, included:[Customer], clustering, seed: int = 0):
    new_problem = ProblemInstance(
        depot_x=old_problem.depot_x,
        depot_y=old_problem.depot_y,
        customers=included,
        vehicle_capacity=old_problem.vehicle_capacity,
        num_vehicles=old_problem.num_vehicles,
        vehicle_speed=old_problem.vehicle_speed,
    )
    
    if included == []:
        return [[]]
    best_cost = float("inf")
    best_route = [[]]
    upper_limit = min(old_problem.num_vehicles,len(included))

    for cluster_number in range(upper_limit,upper_limit//2,-1):
        # choose clients for each route
        result = clustering(new_problem,seed,cluster_number)

        # untangle
        untangled = []
        for route in result:
            untangled.append(auto2opt(old_problem,route))

        untangled_cost = evaluate_solution(old_problem,untangled).total_cost

        if untangled_cost >= best_cost:
            break
        best_route = [[c for c in r] for r in untangled]
        best_cost = untangled_cost
    return best_route


def choose_clients_alternating(problem: ProblemInstance, included_clients, clustering, seed: int = 0):
    """
    included_clients works as an initial guess for the best subset of customers.
    We will then add or remove the most beneficial one at a time until we get to a minimum

    Inspired by variable selection techinques in statistics
    """
    included_eval = evaluate_solution(problem,find_route(problem,included_clients,clustering,seed)).total_cost
    current_best = float("inf")
    while True:
        for i in problem.customers:
            if i in included_clients:
                possible_clients = [c for c in included_clients]
                possible_clients.remove(i)
            else:
                possible_clients = [c for c in included_clients] + [i]
            possible_evaluation = evaluate_solution(problem,find_route(problem,possible_clients,clustering,seed)).total_cost
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
    return find_route(problem,included_clients,clustering,seed)

def run_heavy(
    problem: ProblemInstance,
    seed: int = 0,
    tries: int = 2
) -> tuple[list[list[int]], list[float]]:
    """
    Finds a good solution by alternating between selecting which customers to serve and finding good routes for those customers.
    """

    random.seed(seed)
    t1 = time.time()
    rutas = [choose_clients_alternating(problem,[c for c in problem.customers], cluster)]
    evals = [evaluate_solution(problem,rutas[0]).total_cost]

    # Random starts to get different minima
    for _ in range(tries):
        rutas.append(choose_clients_alternating(problem,random.sample(problem.customers, len(problem.customers)//2), cluster))
        evals.append(evaluate_solution(problem,rutas[-1]).total_cost)
    t2 = time.time()
    print(f"heavy: {round(t2-t1,4)} sec, {round(min(evals))}")
    routes = rutas[evals.index(min(evals))]
    score = evaluate_solution(problem, routes).total_cost
    return routes, [score]


def run(problem: ProblemInstance, seed: int = 0) -> tuple[list[list[int]], list[float]]:
    """
    Use same approach as "naive.py" to select which customers to serve. Then, use a better way to find route

    time approx:
    - 30 customers: ~5 secs
    - 40 customers: ~30 secs
    - 50 customers: ~5 mins
    - 100 customers: 2-3 hours
    
    Args:
        problem: VRP problem instance
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (routes, history).
    """
    return run_heavy(problem, seed=seed)
