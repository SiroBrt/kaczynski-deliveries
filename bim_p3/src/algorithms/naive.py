try:
    from ..evaluation import evaluate_solution
    from ..problem import ProblemInstance
except ImportError:
    from evaluation import evaluate_solution
    from problem import ProblemInstance


def run_naive(
    problem: ProblemInstance,
    iterations: int = 60,
    seed: int = 0,
) -> tuple[list[list[int]], list[float]]:

    return [[1,2]],[]

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
