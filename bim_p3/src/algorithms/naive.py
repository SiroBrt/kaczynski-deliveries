import random
try:
    from ..evaluation import evaluate_solution
    from ..problem import ProblemInstance, Customer
    from baseline import greedy_baseline
except ImportError:
    from evaluation import evaluate_solution
    from problem import ProblemInstance, Customer
    from algorithms.baseline import greedy_baseline


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
    fake_result = greedy_baseline(new_problem) 
    # this actually uses de position of the customers in the list instead of the idx attribute, so we have to convert it
    result = []
    new_cust = []
    for route in fake_result:
        result.append([])
        for i in route:
            result[-1].append(nodes[i-1].idx)
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

    
def run_naive(
    problem: ProblemInstance,
    iterations: int = 60,
    seed: int = 0,
    tries: int = 10
) -> tuple[list[list[int]], list[float]]:
    rutas = []
    rutas.append(choose_clients_alternating(problem,[])) # gives a first solution
    rutas.append(choose_clients_alternating(problem,[c for c in problem.customers])) # improves baseline
    evals = []
    evals.append(evaluate_solution(problem,rutas[0]).total_cost)
    evals.append(evaluate_solution(problem,rutas[1]).total_cost)

    # random starts to get different minima
    for i in range(tries):
        rutas.append(choose_clients_alternating(problem,random.sample(problem.customers,len(problem.customers)//2)))
        evals.append(evaluate_solution(problem,rutas[-1]).total_cost)
    return rutas[evals.index(min(evals))], []



def run(problem: ProblemInstance, seed: int = 0) -> tuple[list[list[int]], list[float]]:
    """
    Use naive 
    
    Args:
        problem: VRP problem instance
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (routes, history).
    """
    return run_naive(problem, seed=seed)
