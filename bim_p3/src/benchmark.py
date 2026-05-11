"""
Benchmarking and Analysis Framework

Provides tools to:
1. Run algorithms multiple times and collect statistics
2. Compare multiple algorithms on the same problem
3. Benchmark across different problem configurations
4. Generate comparative plots and analysis
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, asdict
from typing import Callable, Any
import statistics

import matplotlib.pyplot as plt
import numpy as np

try:
    from src.problem import ProblemInstance, generate_instance
    from src.algorithms import migration, pso, aco, aco_cluster, aco_pso_cluster, baseline, heavy, naive, naive_2opt, naive_aco_cluster, retarded_aco_cluster
except ImportError:
    from problem import ProblemInstance, generate_instance
    from algorithms import migration, pso, aco, aco_cluster, aco_pso_cluster, baseline, heavy, naive, naive_2opt, naive_aco_cluster, retarded_aco_cluster


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AlgorithmRun:
    """Results from a single algorithm run."""
    algorithm_name: str
    problem_name: str
    seed: int
    best_score: float
    final_score: float
    history: list[float]
    execution_time: float
    num_iterations: int


@dataclass
class AlgorithmStats:
    """Statistics across multiple runs of the same algorithm."""
    algorithm_name: str
    problem_name: str
    num_runs: int
    
    # Best solution across all runs
    best_score_ever: float
    
    # Average and std of final scores
    average_final_score: float
    std_final_score: float
    
    # Average and std of best scores per run
    average_best_score: float
    std_best_score: float
    
    # Average execution time
    average_time: float
    
    # Average history across runs
    average_history: list[float]
    std_history: list[float]
    
    # Individual runs
    runs: list[AlgorithmRun]


@dataclass
class ProblemConfig:
    """Configuration for generating a problem instance.
    
    Note: When running benchmarks, each run will generate a fresh problem instance
    using the run's seed, ensuring reproducibility. The seed parameter here is 
    kept for reference but is not used in the new implementation.
    """
    name: str
    num_customers: int
    num_vehicles: int
    vehicle_capacity: int
    seed: int = 0  # Kept for reference (not used in new implementation)


# ============================================================================
# Single Algorithm Analysis
# ============================================================================

def run_algorithm_multiple_times(
    algorithm_func: Callable,
    algorithm_name: str,
    problem_config: ProblemConfig,
    problem_name: str,
    num_runs: int = 5,
    seeds: list[int] | None = None
) -> AlgorithmStats:
    """
    Run an algorithm multiple times with different seeds. Each seed generates its own problem.
    
    Args:
        algorithm_func: The algorithm's run() function
        algorithm_name: Name of the algorithm for reporting
        problem_config: Problem configuration for generation
        problem_name: Name of the problem for reporting
        num_runs: Number of times to run the algorithm
        seeds: List of seeds (if None, uses range(num_runs))
        
    Returns:
        AlgorithmStats object with aggregated results
    """
    if seeds is None:
        seeds = list(range(num_runs))
    
    runs: list[AlgorithmRun] = []
    
    print(f"\n{'='*70}")
    print(f"Running {algorithm_name} on {problem_name} ({num_runs} times)...")
    print(f"{'='*70}")
    
    for i, algo_seed in enumerate(seeds[:num_runs]):
        print(f"  Run {i+1}/{num_runs} (seed={algo_seed})...", end=" ", flush=True)
        
        # Generate problem with the run seed for reproducibility
        problem = generate_instance(
            num_customers=problem_config.num_customers,
            num_vehicles=problem_config.num_vehicles,
            vehicle_capacity=problem_config.vehicle_capacity,
            seed=algo_seed  # Use algorithm seed for problem generation
        )
        
        start_time = time.time()
        routes, history = algorithm_func(problem, seed=algo_seed)
        execution_time = time.time() - start_time
        
        best_score = min(history) if history else float('inf')
        final_score = history[-1] if history else float('inf')
        
        run = AlgorithmRun(
            algorithm_name=algorithm_name,
            problem_name=problem_name,
            seed=algo_seed,
            best_score=best_score,
            final_score=final_score,
            history=history,
            execution_time=execution_time,
            num_iterations=len(history)
        )
        runs.append(run)
        
        print(f"Best: {best_score:.2f}, Final: {final_score:.2f}, Time: {execution_time:.2f}s")
    
    # Compute statistics
    final_scores = [r.final_score for r in runs]
    best_scores = [r.best_score for r in runs]
    times = [r.execution_time for r in runs]
    
    # Average history (pad to same length, handle empty histories)
    max_len = max((len(r.history) for r in runs if r.history), default=1)
    padded_histories = []
    for r in runs:
        if r.history:
            padded = r.history + [r.history[-1]] * (max_len - len(r.history))
        else:
            # If no history, use infinity values (algorithm produced no results)
            padded = [float('inf')] * max_len
        padded_histories.append(padded)
    
    average_history = [statistics.mean(scores) for scores in zip(*padded_histories)]
    std_history = [statistics.stdev(scores) if len(set(scores)) > 1 else 0 
                   for scores in zip(*padded_histories)]
    
    # Filter out inf values for statistics (algorithms that produced no valid results)
    valid_best_scores = [s for s in best_scores if s != float('inf')]
    valid_final_scores = [s for s in final_scores if s != float('inf')]
    
    stats = AlgorithmStats(
        algorithm_name=algorithm_name,
        problem_name=problem_name,
        num_runs=num_runs,
        best_score_ever=min(valid_best_scores) if valid_best_scores else float('inf'),
        average_final_score=statistics.mean(valid_final_scores) if valid_final_scores else float('inf'),
        std_final_score=statistics.stdev(valid_final_scores) if len(set(valid_final_scores)) > 1 else 0,
        average_best_score=statistics.mean(valid_best_scores) if valid_best_scores else float('inf'),
        std_best_score=statistics.stdev(valid_best_scores) if len(set(valid_best_scores)) > 1 else 0,
        average_time=statistics.mean(times),
        average_history=average_history,
        std_history=std_history,
        runs=runs
    )
    
    return stats


# ============================================================================
# Multi-Algorithm Comparison
# ============================================================================

def benchmark_algorithms_on_problem(
    problem_config: ProblemConfig,
    algorithms: dict[str, Callable] | None = None,
    num_runs: int = 5,
    seeds: list[int] | None = None
) -> dict[str, AlgorithmStats]:
    """
    Compare multiple algorithms on the same problem configuration.
    Each seed generates its own problem instance.
    
    Args:
        problem_config: Problem configuration for generation
        algorithms: Dict of {algorithm_name: algorithm_func}
                   If None, uses all available algorithms
        num_runs: Number of runs per algorithm
        seeds: List of seeds for reproducibility
        
    Returns:
        Dict of {algorithm_name: AlgorithmStats}
    """
    if algorithms is None:
        algorithms = {
            "Migration": migration.run,
            "PSO": pso.run,
            "ACO": aco.run,
            "ACO PSO Cluster": aco_pso_cluster.run,
            "Baseline": baseline.run,
            "Heavy": heavy.run,
            "Naive": naive.run,
            "Naive 2opt": naive_2opt.run,
        }
    
    results = {}
    for algo_name, algo_func in algorithms.items():
        stats = run_algorithm_multiple_times(
            algo_func,
            algo_name,
            problem_config,
            problem_config.name,
            num_runs=num_runs,
            seeds=seeds
        )
        results[algo_name] = stats
    
    return results


# ============================================================================
# Full Benchmark Suite
# ============================================================================

def benchmark_suite(
    problem_configs: list[ProblemConfig],
    algorithms: dict[str, Callable] | None = None,
    num_runs: int = 5,
    seed_offset: int = 0,
    output_dir: str = "benchmark_results"
) -> dict[str, dict[str, AlgorithmStats]]:
    """
    Run full benchmark suite across multiple problem configurations.
    
    For each algorithm run, a unique problem instance is generated using the run's seed.
    This ensures that:
    - Same seed always produces the same problem
    - Different seeds produce different problems
    - Results are fully reproducible
    
    Args:
        problem_configs: List of problem configurations to benchmark
        algorithms: Dict of algorithms (default: all)
        num_runs: Number of runs per configuration per algorithm
        seed_offset: Offset for seeds (useful for multiple runs)
        output_dir: Directory to save results
        
    Returns:
        Nested dict: {problem_name: {algo_name: AlgorithmStats}}
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if algorithms is None:
        algorithms = {
            "Migration": migration.run,
            "PSO": pso.run,
            "ACO": aco.run,
            "ACO PSO Cluster": aco_pso_cluster.run,
            "Baseline": baseline.run,
            "Heavy": heavy.run,
            "Naive": naive.run,
            "Naive 2opt": naive_2opt.run,
        }
    
    all_results = {}
    seeds = list(range(seed_offset, seed_offset + num_runs))
    
    print(f"\n{'#'*70}")
    print(f"BENCHMARK SUITE: {len(problem_configs)} problems × {len(algorithms)} algorithms")
    print(f"{'#'*70}\n")
    
    for config in problem_configs:
        print(f"\n{'*'*70}")
        print(f"Problem: {config.name}")
        print(f"  Customers: {config.num_customers}, Vehicles: {config.num_vehicles}")
        print(f"  Capacity: {config.vehicle_capacity}")
        print(f"{'*'*70}")
        
        # Benchmark all algorithms on this problem configuration
        # (each run will generate its own problem instance with the run seed)
        results = benchmark_algorithms_on_problem(
            config,
            algorithms=algorithms,
            num_runs=num_runs,
            seeds=seeds
        )
        
        all_results[config.name] = results
        
        # Save intermediate results
        _save_results(results, f"{output_dir}/{config.name}_results.json")
    
    # Generate comparison plots
    _plot_benchmark_results(all_results, output_dir)
    
    return all_results


# ============================================================================
# Results Visualization
# ============================================================================

def _plot_benchmark_results(
    results: dict[str, dict[str, AlgorithmStats]],
    output_dir: str
) -> None:
    """Generate comparative plots for benchmark results."""
    
    problems = list(results.keys())
    
    # 1. Best scores comparison across problems
    fig, ax = plt.subplots(figsize=(12, 6))
    for algo_name in results[problems[0]].keys():
        best_scores = [results[p][algo_name].best_score_ever for p in problems]
        ax.plot(problems, best_scores, marker='o', label=algo_name, linewidth=2)
    
    ax.set_xlabel("Problem Configuration", fontsize=12)
    ax.set_ylabel("Best Score Found", fontsize=12)
    ax.set_title("Algorithm Comparison: Best Scores", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_best_scores.png", dpi=150)
    plt.close()
    
    # 2. Average final score comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    for algo_name in results[problems[0]].keys():
        avg_scores = [results[p][algo_name].average_final_score for p in problems]
        ax.plot(problems, avg_scores, marker='s', label=algo_name, linewidth=2)
    
    ax.set_xlabel("Problem Configuration", fontsize=12)
    ax.set_ylabel("Average Final Score", fontsize=12)
    ax.set_title("Algorithm Comparison: Average Final Scores", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_average_scores.png", dpi=150)
    plt.close()
    
    # 3. Execution time comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    for algo_name in results[problems[0]].keys():
        times = [results[p][algo_name].average_time for p in problems]
        ax.plot(problems, times, marker='^', label=algo_name, linewidth=2)
    
    ax.set_xlabel("Problem Configuration", fontsize=12)
    ax.set_ylabel("Average Execution Time (seconds)", fontsize=12)
    ax.set_title("Algorithm Comparison: Execution Times", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_execution_times.png", dpi=150)
    plt.close()
    
    # 4. Convergence curves for each problem
    for problem_name, problem_results in results.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for algo_name, stats in problem_results.items():
            history = stats.average_history
            ax.plot(history, label=algo_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Best Score", fontsize=12)
        ax.set_title(f"Convergence Curves: {problem_name}", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/convergence_{problem_name}.png", dpi=150)
        plt.close()
    
    print(f"\n✅ Plots saved to {output_dir}/")


def _save_results(results: dict[str, AlgorithmStats], filepath: str) -> None:
    """Save benchmark results to JSON file."""
    data = {}
    for algo_name, stats in results.items():
        data[algo_name] = {
            "best_score_ever": stats.best_score_ever,
            "average_final_score": stats.average_final_score,
            "std_final_score": stats.std_final_score,
            "average_best_score": stats.average_best_score,
            "std_best_score": stats.std_best_score,
            "average_time": stats.average_time,
            "num_runs": stats.num_runs,
        }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {filepath}")


# ============================================================================
# Summary and Reporting
# ============================================================================

def print_benchmark_summary(results: dict[str, dict[str, AlgorithmStats]]) -> None:
    """Print human-readable summary of benchmark results."""
    
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}\n")
    
    for problem_name, problem_results in results.items():
        print(f"\n📊 {problem_name}")
        print(f"{'-'*80}")
        
        # Create comparison table
        print(f"{'Algorithm':<15} {'Best':>12} {'Avg Final':>12} {'Std':>10} {'Time':>10}")
        print(f"{'-'*80}")
        
        for algo_name, stats in sorted(
            problem_results.items(),
            key=lambda x: x[1].best_score_ever
        ):
            print(f"{algo_name:<15} {stats.best_score_ever:>12.2f} "
                  f"{stats.average_final_score:>12.2f} {stats.std_final_score:>10.2f} "
                  f"{stats.average_time:>10.2f}s")
        
        # Find winner
        best_algo = min(problem_results.items(), key=lambda x: x[1].best_score_ever)
        print(f"\n🏆 Winner: {best_algo[0]} (score: {best_algo[1].best_score_ever:.2f})")


# ============================================================================
# Helper Functions
# ============================================================================

def create_default_benchmark_suite() -> list[ProblemConfig]:
    """Create a standard benchmark suite with varying problem sizes."""
    return [
        ProblemConfig("small_few_vehicles", num_customers=20, num_vehicles=3, vehicle_capacity=40, seed=0),
        ProblemConfig("small_many_vehicles", num_customers=20, num_vehicles=5, vehicle_capacity=20, seed=1),
        ProblemConfig("medium_few_vehicles", num_customers=35, num_vehicles=4, vehicle_capacity=50, seed=2),
        ProblemConfig("medium_many_vehicles", num_customers=35, num_vehicles=7, vehicle_capacity=30, seed=3),
        ProblemConfig("large_few_vehicles", num_customers=50, num_vehicles=5, vehicle_capacity=60, seed=4),
        ProblemConfig("large_many_vehicles", num_customers=50, num_vehicles=10, vehicle_capacity=30, seed=5),
    ]


if __name__ == "__main__":
    # Example usage - creates and runs benchmark suite
    # Each run generates its own problem with a unique seed, ensuring reproducibility
    configs = create_default_benchmark_suite()
    
    results = benchmark_suite(
        problem_configs=configs,
        num_runs=3,  # 3 runs with seeds [0, 1, 2]
        output_dir="benchmark_results"
    )
    
    print_benchmark_summary(results)
