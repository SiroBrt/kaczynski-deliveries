"""
Bio-inspired Routing Lab: Unified CLI and Streamlit Application

This module provides both command-line and web-based interfaces for solving
the Vehicle Routing Problem with Time Windows (VRPTW) using various bio-inspired
algorithms. Algorithms are discovered dynamically from the algorithms/ folder.

Usage:
  - CLI:       python -m src.app --algorithm <name> --customers 30 --seed 0
  - Streamlit: streamlit run src/app.py
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
import types

try:
    from .evaluation import evaluate_solution
    from .problem import generate_instance, load_problem_from_yaml
    from .visualization import plot_history, plot_routes
except ImportError:
    from evaluation import evaluate_solution
    from problem import generate_instance, load_problem_from_yaml
    from visualization import plot_history, plot_routes


# ============================================================================
# Algorithm Discovery
# ============================================================================


def _discover_algorithms() -> dict[str, types.ModuleType]:
    """
    Discover available algorithms by scanning the algorithms/ folder.
    
    Each algorithm module must expose a run(problem, seed) function that returns
    (routes: list[list[int]], history: list[float]).
    
    Returns:
        Dictionary mapping algorithm names to loaded modules.
    """
    algorithms_dir = pathlib.Path(__file__).parent / 'algorithms'
    result: dict[str, types.ModuleType] = {}
    
    # Iterate through all Python files in the algorithms folder
    for path in sorted(algorithms_dir.glob('*.py')):
        # Skip private/internal modules (starting with underscore)
        if path.stem.startswith('_'):
            continue
            
        mod: types.ModuleType | None = None
        
        # Try both package and direct imports to support different execution contexts
        for pkg_prefix in ('src.algorithms', 'algorithms'):
            try:
                mod = importlib.import_module(f'{pkg_prefix}.{path.stem}')
                break
            except (ImportError, ModuleNotFoundError):
                continue
        
        # Only register modules that have a run() function
        if mod is not None and callable(getattr(mod, 'run', None)):
            result[path.stem] = mod
    
    return result


def _running_under_streamlit() -> bool:
    """Check if Streamlit has been imported (i.e., we're running via streamlit run)."""
    import sys
    return 'streamlit' in sys.modules


# ============================================================================
# Streamlit Interface
# ============================================================================


def _run_streamlit() -> None:
    """
    Launch the interactive Streamlit web interface.
    
    Features:
    - Sidebar controls for algorithm selection and problem parameters
    - Support for loading problems from YAML files or generating random instances
    - Real-time metrics display (cost, distance, delay, unserved customers)
    - Route visualization on a 2D map
    - Algorithm convergence history plot
    - Configurable evaluation weights
    """
    import streamlit as st
    import tempfile
    import os

    def _render_cost_formula_details(
        total_distance: float,
        total_delay: float,
        vehicles_used: int,
        unserved_customers: int,
        capacity_violations: int,
        fixed_vehicle_cost: float,
        weight_delay: float,
        weight_unserved: float,
        weight_capacity: float,
        total_cost: float,
    ) -> None:
        """Show formula and explain each term that contributes to the total cost."""
        st.markdown('### Cost formula')
        st.latex(r'C = D + w_d \cdot L + c_v \cdot V + w_u \cdot U + w_c \cdot K')
        st.markdown(
            """
- $D$: total distance traveled
- $w_d$: delay penalty weight
- $L$: total delay (time window violations)
- $c_v$: fixed vehicle cost
- $V$: vehicles used
- $w_u$: unserved customer penalty weight
- $U$: number of unserved customers
- $w_c$: capacity violation penalty weight
- $K$: number of capacity violations
"""
        )

        st.markdown('### Current run breakdown')
        delay_term = weight_delay * total_delay
        vehicle_term = fixed_vehicle_cost * vehicles_used
        unserved_term = weight_unserved * unserved_customers
        capacity_term = weight_capacity * capacity_violations
        st.write(f'Distance term: {total_distance:.2f}')
        st.write(f'Delay term: {weight_delay:.2f} x {total_delay:.2f} = {delay_term:.2f}')
        st.write(f'Vehicle term: {fixed_vehicle_cost:.2f} x {vehicles_used} = {vehicle_term:.2f}')
        st.write(f'Unserved term: {weight_unserved:.2f} x {unserved_customers} = {unserved_term:.2f}')
        st.write(f'Capacity term: {weight_capacity:.2f} x {capacity_violations} = {capacity_term:.2f}')
        st.write(f'Total cost: {total_cost:.2f}')

    if hasattr(st, 'dialog'):
        @st.dialog('Cost Computation Details')
        def _show_cost_formula_modal(**kwargs) -> None:
            _render_cost_formula_details(**kwargs)
    else:
        def _show_cost_formula_modal(**kwargs) -> None:
            st.warning('This Streamlit version does not support modal dialogs. Showing details inline.')
            _render_cost_formula_details(**kwargs)

    algos = _discover_algorithms()

    # Page configuration
    st.set_page_config(page_title='Bio-inspired Routing Lab', layout='wide')
    st.title('Bio-inspired Routing Lab')
    st.write('Simple environment to generate instances, run an algorithm, and visualize results.')

    # Sidebar controls for problem configuration
    with st.sidebar:
        st.header('Configuration')
        
        # Choose between YAML file or random generation
        problem_source = st.radio('Problem source', ['Random generation', 'YAML file'])
        
        if problem_source == 'YAML file':
            uploaded_file = st.file_uploader('Upload YAML configuration file', type=['yaml', 'yml'])
            config_path = None
            if uploaded_file is not None:
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml') as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    config_path = tmp.name
        else:
            # Random generation parameters
            customers = st.slider('Customers', 10, 120, 30, 5)
            vehicles = st.slider('Vehicles', 1, 10, 4, 1)
            capacity = st.slider('Capacity', 10, 60, 30, 5)
            config_path = None
        
        algorithm = st.selectbox('Algorithm', list(algos.keys()))
        seed = st.number_input('Seed', min_value=0, value=0, step=1)
        
        st.divider()
        st.subheader('Evaluation Weights')
        weight_delay = st.slider('Delay penalty', min_value=0.0, max_value=100.0, value=10.0, step=0.5)
        weight_unserved = st.slider('Unserved penalty', min_value=0.0, max_value=500.0, value=200.0, step=5.0)
        weight_capacity = st.slider('Capacity violation penalty', min_value=0.0, max_value=500.0, value=150.0, step=5.0)
        
        run = st.button('Run')

    if run:
        # Load or generate problem instance
        try:
            if problem_source == 'YAML file':
                if config_path is None:
                    st.error('Please upload a YAML configuration file.')
                else:
                    problem = load_problem_from_yaml(config_path)
                    st.success(f'Loaded problem with {len(problem.customers)} customers')
            else:
                # Generate random instance
                problem = generate_instance(
                    num_customers=customers,
                    num_vehicles=vehicles,
                    vehicle_capacity=capacity,
                    seed=int(seed),
                )
        except Exception as e:
            st.error(f'Error loading problem: {str(e)}')
            return

        # Run the selected algorithm
        routes, history = algos[algorithm].run(problem, seed=int(seed))
        result = evaluate_solution(
            problem, 
            routes,
            weight_delay=weight_delay,
            weight_unserved=weight_unserved,
            weight_capacity_violation=weight_capacity,
        )

        # Persist last successful run so UI actions (like opening info modal)
        # survive Streamlit reruns triggered by button clicks.
        st.session_state['last_run'] = {
            'problem': problem,
            'routes': routes,
            'history': history,
            'result': result,
            'algorithm': algorithm,
            'weight_delay': weight_delay,
            'weight_unserved': weight_unserved,
            'weight_capacity': weight_capacity,
        }

        # Clean up temporary file if it was created
        if config_path and os.path.exists(config_path):
            try:
                os.unlink(config_path)
            except:
                pass

    if 'last_run' in st.session_state:
        last_run = st.session_state['last_run']
        problem = last_run['problem']
        routes = last_run['routes']
        history = last_run['history']
        result = last_run['result']
        algorithm = last_run['algorithm']
        weight_delay = last_run['weight_delay']
        weight_unserved = last_run['weight_unserved']
        weight_capacity = last_run['weight_capacity']

        # Display key metrics in a 4-column layout
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            info_col, cost_col = st.columns([1, 5], gap='small')
            show_cost_info = info_col.button('i', key='cost_info_button', help='How cost is computed')
            cost_col.metric('Cost', f'{result.total_cost:.2f}')
        c2.metric('Distance', f'{result.total_distance:.2f}')
        c3.metric('Delay', f'{result.total_delay:.2f}')
        c4.metric('Unserved', result.unserved_customers)

        if show_cost_info:
            _show_cost_formula_modal(
                total_distance=result.total_distance,
                total_delay=result.total_delay,
                vehicles_used=result.vehicles_used,
                unserved_customers=result.unserved_customers,
                capacity_violations=result.capacity_violations,
                fixed_vehicle_cost=problem.fixed_vehicle_cost,
                weight_delay=weight_delay,
                weight_unserved=weight_unserved,
                weight_capacity=weight_capacity,
                total_cost=result.total_cost,
            )

        # Visualize routes on a 2D map
        st.subheader('Routes')
        st.pyplot(plot_routes(problem, routes))

        # Display convergence history if available
        if history:
            st.subheader('Algorithm evolution')
            st.pyplot(plot_history(history, title=f'Cost evolution ({algorithm.upper()})'))

        # Show routes in list format
        with st.expander('Routes as list'):
            st.write(routes)
    else:
        st.info('Adjust parameters and press Run.')


# ============================================================================
# Command-Line Interface
# ============================================================================


def _run_cli() -> None:
    """
    Launch command-line interface for non-interactive algorithm evaluation.
    
    Parses arguments from the command line and outputs results to stdout.
    """
    algos = _discover_algorithms()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Bio-inspired Routing Lab CLI')
    parser.add_argument(
        '--algorithm',
        choices=list(algos.keys()),
        default=next(iter(algos)),
        help='Algorithm to use (discovered from algorithms/ folder)'
    )
    parser.add_argument('--config-file', type=str, default=None, help='YAML file with problem configuration (if provided, ignores --customers, --vehicles, --capacity)')
    parser.add_argument('--customers', type=int, default=30, help='Number of customers (ignored if --config-file is provided)')
    parser.add_argument('--vehicles', type=int, default=4, help='Number of vehicles (ignored if --config-file is provided)')
    parser.add_argument('--capacity', type=int, default=30, help='Vehicle capacity (ignored if --config-file is provided)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--weight-delay', type=float, default=10.0, help='Penalty weight for time window violations (default: 10.0)')
    parser.add_argument('--weight-unserved', type=float, default=200.0, help='Penalty weight for unserved customers (default: 200.0)')
    parser.add_argument('--weight-capacity', type=float, default=150.0, help='Penalty weight for capacity violations (default: 150.0)')
    args = parser.parse_args()

    # Validate arguments
    if args.customers < 1:
        raise ValueError('Number of customers must be at least 1')
    if args.vehicles < 1:
        raise ValueError('Number of vehicles must be at least 1')
    if args.capacity < 1:
        raise ValueError('Vehicle capacity must be at least 1')

    # Generate or load problem instance
    if args.config_file:
        # Load from YAML file
        try:
            problem = load_problem_from_yaml(args.config_file)
            print(f'\nLoaded problem configuration from: {args.config_file}')
            print(f'Problem size: {len(problem.customers)} customers, {problem.num_vehicles} vehicles, '
                  f'capacity {problem.vehicle_capacity}')
        except FileNotFoundError:
            raise ValueError(f'Configuration file not found: {args.config_file}')
        except ValueError as e:
            raise ValueError(f'Error loading configuration file: {e}')
    else:
        # Generate random problem instance
        problem = generate_instance(
            num_customers=args.customers,
            num_vehicles=args.vehicles,
            vehicle_capacity=args.capacity,
            seed=args.seed,
        )
        print(f'\nGenerated random problem instance')
        print(f'Problem size: {args.customers} customers, {args.vehicles} vehicles, '
              f'capacity {args.capacity}')

    # Run the selected algorithm
    routes, history = algos[args.algorithm].run(problem, seed=args.seed)
    result = evaluate_solution(
        problem, 
        routes,
        weight_delay=args.weight_delay,
        weight_unserved=args.weight_unserved,
        weight_capacity_violation=args.weight_capacity,
    )

    # Output results to stdout
    print(f'\n{"="*60}')
    print(f'Algorithm: {args.algorithm.upper()}')
    print(f'{"="*60}')
    print(f'Random seed: {args.seed}')
    print(f'\n{"─"*60}')
    print('Results:')
    print(f'{"─"*60}')
    print(f'Total cost:        {result.total_cost:.2f}')
    print(f'Total distance:    {result.total_distance:.2f}')
    print(f'Total delay:       {result.total_delay:.2f}')
    print(f'Vehicles used:     {result.vehicles_used}')
    print(f'Unserved customers: {result.unserved_customers}')
    if result.capacity_violations > 0:
        print(f'Capacity violations: {result.capacity_violations}')
        print(f'Final best cost:   {history[-1]:.2f}')
    print(f'\nRoutes: {routes}')
    print(f'{"="*60}\n')


# ============================================================================
# Entry Point
# ============================================================================


if _running_under_streamlit():
    _run_streamlit()
elif __name__ == '__main__':
    _run_cli()
