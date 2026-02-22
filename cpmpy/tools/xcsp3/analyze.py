"""
Collection of visualisation tools for processing the result of a `benchmark.py` run.

Best used though its CLI, a command-line tool to visualize and analyze solver performance 
based on CSV output files.

E.g. to compare the results of multiple solvers:

.. code-block:: console

    python analyze.py <results_dir>

Positional Arguments
--------------------
files : str
    One or more CSV files (or a single directory) containing performance data to analyze.

Optional Arguments
------------------
--time_limit : float, optional
    Maximum time limit (in seconds) to display on the x-axis of the plot.

--plot, -o : str, optional
    Path to save the generated plot image (e.g., "plot.png").

--show : bool, optional
    Display plots interactively using matplotlib's interactive backend.

--small : int, optional
    Threshold for filtering instances by largest table size (default: 25). Instances with max rows <= this value are excluded.

--sort-legend : str, optional
    Sort order for legend/lines. Options: 'alpha' (lexicographic, default), 'performance' (by instances solved).

--compare : str [str], optional
    Compare solvers. Takes 1 or 2 arguments:
    - --compare A: Compare all solvers against A (baseline). Creates correlation plots and scatter plots for all solvers vs A.
    - --compare A B: Compare A vs B. Creates scatter plot and correlation analysis with A as baseline.
    Each string must match exactly one solver alias (substring match). Raises exception if multiple matches found.

--tex : path, optional
    Path to save LaTeX tables generated from the analysis results.

--paper : bool, optional
    Use larger font sizes (2x) in plots suitable for papers/publications.
"""

import builtins
import subprocess
import argparse
import ast
import json
import pathlib
import statistics
import re
import pandas as pd
import numpy as np

# Set interactive backend before importing pyplot
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        pass  # Use default backend

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.markers import MarkerStyle

# Different marker shapes for visual distinction (use all filled markers from matplotlib)
marker_shapes = list(MarkerStyle.filled_markers)

FIELDNAMES = [
    "year",
    "track",
    "instance",
    "alias",
    "solver",
    "solver_kwargs",
    "time_total",
    "time_parse",
    "time_model",
    "time_post",
    "time_solve",
    "status",
    "objective_value",
    "solution",
    "exception",
    "traceback",
    "intermediate",
    "checker_result",
    "n_cuts",
    "n_cuts_explained",
    "n_cuts_unexplained",
    "time_cb",
    "constraints",
]

METADATA_COLS = ["method", "area", "rows", "min", "max", "count", "mean", "median", "stdev"]



def _extract_cost(solution_str):
    """
    Extract numeric cost from solution string like '<instantiation ... cost="69">'
    """
    if isinstance(solution_str, str):
        match = re.search(r'cost="(\d+)"', solution_str)
        if match:
            return float(match.group(1))
    return np.nan


OPT = 'OPTIMUM FOUND'
UNS = 'UNSATISFIABLE'
SAT = 'SATISFIABLE'
MEM = 'MEMORY'
ERR = 'ERROR'
UNK = 'UNKNOWN'

def xcsp3_plot(df, time_limit=None, metric="time_solve", filter_by="solved", solved_only=False, sort_legend='alpha'):
    # Get unique solvers
    solvers = df['alias'].unique()

    # # Determine the status to plot (Opt if at least one opt, otherwise sad)
    # if filter_by == "solved":
    #     status_filter = solved
    # elif filter_by == "feasible":
    #     status_filter = (OPT, UNS, SAT)
    # else:
    #     raise Exception

    # Count total unique instances before filtering
    n_instances = len(df.groupby(['problem', 'instance']))

    df = df[df["solved"]]  # only those that reached the desired status
    # print(df[["solver", "instance", "status", metric]])

    # Count how many instances each solver solved (with correct status)
    solver_counts = df['alias'].value_counts()

    # Sort solvers based on sort_legend parameter
    if sort_legend == 'performance':
        # Sort by number of instances solved (descending)
        solvers_sorted = solver_counts.sort_values(ascending=False).index.tolist()
    else:  # 'alpha' or default
        # Sort lexicographically
        solvers_sorted = sorted(solvers)

    # Create figure
    fig = plt.figure(figsize=(10, 6))

    for solver in solvers_sorted:
        # Get data for this solver
        solver_data = df[df['alias'] == solver]
        
        # Sort by time_total
        solver_data = solver_data.sort_values(metric)
        
        # If time_limit is set, truncate data
        if time_limit is not None:
            solver_data = solver_data[solver_data[metric] <= time_limit]
        
        # Build x and y values
        x = [0.0] + solver_data[metric].tolist()
        y = [0] + list(range(1, len(solver_data) + 1))
        
        # Plot the performance curve
        plt.plot(x, y, label=f"{solver} ({len(solver_data)})", linewidth=2.5)
    
    # Set plot properties
    plt.xlabel('Time (seconds)')
    # plt.ylabel(f'Number of {filter_by} instances (status in [{','.join(s[:3] for s in status_filter)}])')
    plt.ylabel(f'Number of solved instances')
    # Get unique year-track combinations
    year_track_pairs = df[['year', 'track']].drop_duplicates()
    datasets = ', '.join([f'{row.year}:{row.track}' for _, row in year_track_pairs.iterrows()])
    plt.title(f"Performance Plot ({datasets}, {n_instances} instances){' (solved only)' if solved_only else ''}")
    plt.grid(True)
    plt.legend()
    
    # Set x-axis limit if specified
    if time_limit is not None:
        plt.xlim(0, time_limit)

    return fig

def xcsp3_scatter_plot(df, solver1=None, solver2=None, metric="time_solve", inst_metric="median", time_limit=None, track=None, inst_metric_range=None, paper=False):
    """
    Create a scatter plot comparing the performance of two solvers.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing solver results with columns: alias, instance, and metric
    solver1 : str, optional
        Name of the first solver (x-axis). If None, uses the first solver in the dataframe.
    solver2 : str, optional
        Name of the second solver (y-axis). If None, uses the second solver in the dataframe.
    metric : str, default="time_solve"
        The metric to compare (e.g., "time_solve", "time_total")
    time_limit : float, optional
        Maximum time limit for the axes. If specified, unsolved instances are plotted at time_limit.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated scatter plot figure
    """
    # Get unique solvers
    solvers = df['alias'].unique()

    # Determine which solvers to compare
    if solver1 is None:
        solver1 = solvers[0]
    if solver2 is None:
        solver2 = solvers[1] if len(solvers) > 1 else solvers[0]

    if solver1 not in solvers or solver2 not in solvers:
        raise ValueError(f"Solvers {solver1} and/or {solver2} not found in dataframe. Available: {solvers}")

    # Select columns for merge - include metadata for tooltips
    base_cols = ['problem', 'instance', metric, 'solved', inst_metric, 'small']
    metadata_cols = ['area', 'rows', 'median', 'stdev', 'min', 'max', 'time_post']

    # Combine and deduplicate columns (inst_metric might already be in metadata_cols)
    cols = list(dict.fromkeys(base_cols + metadata_cols))

    df1 = df[df['alias'] == solver1][cols].copy()
    df2 = df[df['alias'] == solver2][cols].copy()

    # # Remove duplicates (keep first occurrence) to avoid duplicate rows in merge
    # df1 = df1.drop_duplicates(subset=['problem', 'instance'], keep='first')
    # df2 = df2.drop_duplicates(subset=['problem', 'instance'], keep='first')

    # Merge on problem and instance to get paired data
    merged = df1.merge(df2, on=['problem', 'instance'], suffixes=('_1', '_2'))

    PAR = 2 * time_limit
    # Handle unsolved instances
    if time_limit is not None:
        merged[f'{metric}_1'] = merged[f'{metric}_1'].fillna(PAR)
        merged[f'{metric}_2'] = merged[f'{metric}_2'].fillna(PAR)
    else:
        # Drop instances where either solver didn't solve it
        merged = merged.dropna(subset=[f'{metric}_1', f'{metric}_2'])

    # Remove instances where neither solver solved it
    merged = merged[merged['solved_1'] | merged['solved_2']]

    # # Reset index to ensure clean array extraction
    merged = merged.reset_index(drop=True)

    # Extract x and y coordinates
    x = merged[f'{metric}_1'].values
    y = merged[f'{metric}_2'].values

    inst_metrics = merged[f'{inst_metric}_1'].values
    problems = merged['problem'].values
    instances = merged['instance'].values

    # Extract additional metadata for tooltips
    area = merged['area_1'].values
    rows = merged['rows_1'].values
    medians = merged['median_1'].values
    stdevs = merged['stdev_1'].values
    mins = merged['min_1'].values
    maxs = merged['max_1'].values
    time_posts_1 = merged['time_post_1'].values
    time_posts_2 = merged['time_post_2'].values

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create colormap and set colors for special cases
    cmap = plt.get_cmap('viridis').copy()

    # Create alpha array - lower alpha for red dots (below small threshold)
    alpha_values = np.where(merged["small_1"], 0.2, 0.8)

    # Get unique problem classes
    unique_problems = sorted(problems.unique())

    problem_markers = dict(zip(unique_problems,
                             [marker_shapes[i % len(marker_shapes)]
                              for i in range(len(unique_problems))]))

    # Plot scatter points for each problem class with different markers
    scatter_plots = []
    for problem in unique_problems:
        mask = problems == problem
        n_instances = mask.sum()
        scatter = ax.scatter(
            x[mask],
            y[mask],
            c=inst_metrics[mask],
            alpha=alpha_values[mask],
            s=50,
            cmap=cmap,
            edgecolors='black',
            linewidth=0.5,
            marker=problem_markers[problem],
            label=f"{problem} ({n_instances})",
            norm=LogNorm(
                vmin=inst_metric_range[0] if inst_metric_range else inst_metrics.min(),
                vmax=inst_metric_range[1] if inst_metric_range else inst_metrics.max(),
            ),
        )
        scatter_plots.append(scatter)

    # Add colorbar using the first scatter plot
    cbar = plt.colorbar(scatter_plots[0], ax=ax)
    cbar.set_label(f'table {inst_metric}', rotation=270, labelpad=20)

    min_max = [0.1, PAR]
    padding = 1.5
    # Extend diagonal lines to the padded plot limits
    diagonal_range = [min_max[0], min_max[1] * padding]
    ax.plot(diagonal_range, diagonal_range, 'k--', linewidth=1.5, label='Equal performance', zorder=0)

    # Add 10% improvement lines (parallel to diagonal)
    improvement_factor = 2
    ax.plot(diagonal_range, [m * improvement_factor for m in diagonal_range], 'k:', linewidth=1, alpha=0.5, zorder=0)
    ax.plot(diagonal_range, [m / improvement_factor for m in diagonal_range], 'k:', linewidth=1, alpha=0.5, zorder=0)

    # Add time limit borders and grey out areas outside
    if time_limit is not None:
        # Set full background to grey
        ax.set_facecolor('lightgrey')

        # Fill the valid area (within time limit) with default background color
        ax.fill_between([min_max[0], time_limit], min_max[0], time_limit, color='white', zorder=-1)

    # Set plot properties
    ax.set_xlabel(f'{solver1} - {metric} (seconds)')
    ax.set_ylabel(f'{solver2} - {metric} (seconds)')

    # Get unique year-track combinations
    year_track_pairs = df[['year', 'track']].drop_duplicates()
    datasets = ', '.join([f'{row.year}:{row.track}' for _, row in year_track_pairs.iterrows()])

    # Count wins
    solver1_wins = sum(1 for i in range(len(x)) if x[i] < y[i])
    solver2_wins = sum(1 for i in range(len(x)) if x[i] > y[i])
    ties = sum(1 for i in range(len(x)) if np.isclose(x[i], y[i], rtol=1))

    title = f"{solver1} vs {solver2}"
    if track:
        title += f" ({track})"
    # title += f"{solver1} faster: {solver1_wins}, {solver2} faster: {solver2_wins}, Ties: {ties}"
    # title += f"{solver1} faster: {solver1_wins}, {solver2} faster: {solver2_wins}, Ties: {ties}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Use log scale if there's a wide range
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set axis limits with padding to prevent dots from being cut off
    if time_limit is not None:
        ax.set_xlim(min_max[0], min_max[1] * padding)
        ax.set_ylim(min_max[0], min_max[1] * padding)

    # Place legend inside plot area at top-left with transparency (hide for paper)
    if not paper:
        ax.legend(loc='upper left', framealpha=0.5)
    plt.tight_layout()

    # Add hover labels for instance names
    try:
        import mplcursors
        cursor = mplcursors.cursor(scatter_plots, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            idx = sel.index
            text = (f"{problems[idx]}-{instances[idx]}\n"
                   f"area: {area[idx]:.0f},"
                   f"rows: {rows[idx]:.0f}, median: {medians[idx]:.1f}, "
                   f"stdev: {stdevs[idx]:.1f if not np.isnan(stdevs[idx]) else 'N/A'}\n"
                   f"min: {mins[idx]:.0f}, max: {maxs[idx]:.0f}\n"
                   f"time_post: {time_posts_1[idx]:.3f}s / {time_posts_2[idx]:.3f}s")
            sel.annotation.set_text(text)
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
            sel.annotation.set_zorder(1000)  # Draw on top of colorbar
    except ImportError:
        # Fallback to manual hover implementation
        annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                           bbox=dict(boxstyle="round", fc="white", alpha=0.9),
                           arrowprops=dict(arrowstyle="->"),
                           zorder=1000)  # Draw on top of colorbar
        annot.set_visible(False)

        def hover(event):
            if event.inaxes == ax:
                # Check all scatter plots
                found = False
                for scatter in scatter_plots:
                    cont, ind = scatter.contains(event)
                    if cont:
                        idx = ind["ind"][0]
                        annot.xy = (x[idx], y[idx])
                        text = (f"{problems[idx]}-{instances[idx]}\n"
                               f"rows: {rows[idx]:.0f}, median: {medians[idx]:.1f}, "
                               f"stdev: {stdevs[idx]:.1f}\n"
                               f"min: {mins[idx]:.0f}, max: {maxs[idx]:.0f}\n"
                               f"time_post: {time_posts_1[idx]:.3f}s / {time_posts_2[idx]:.3f}s")
                        annot.set_text(text)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        found = True
                        break
                if not found:
                    if annot.get_visible():
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

    return fig

def get_cost(row):
    """
    Get the achieved objective value from the provided row.
    If intermediate solutions are available, get the best found (not neccesarily proven optimal).
    """
    intermediate = row['intermediate']

    # Try to parse string representations safely
    if isinstance(intermediate, str):
        try:
            intermediate = ast.literal_eval(intermediate)
        except (ValueError, SyntaxError):
            intermediate = None

    # If it's a valid list of tuples, return the last objective
    if isinstance(intermediate, list) and len(intermediate) > 0:
        try:
            return intermediate[-1][1]
        except (IndexError, TypeError):
            pass

    # Fallback to extracting from solution
    return _extract_cost(row['solution'])

def xcsp3_objective_performance_profile(df):
    # Parse cost from the solution string
    df = df.copy()
    # print(df["intermediate"])
    df['cost'] = df.apply(get_cost, axis=1)
    # df['cost'] = df['solution'].apply(extract_cost)

    # Pivot to get costs per instance per solver
    pivot = df.pivot_table(index='instance', columns='solver', values='cost')

    # Drop instances not solved by all solvers (for fair comparison)
    pivot = pivot.dropna(how='all')

    # Compute the best (minimum) cost per instance
    best_costs = pivot.min(axis=1)

    # Compute performance ratios: solver_cost / best_cost
    perf_ratios = pivot.divide(best_costs, axis=0)

    # Replace inf or NaN with a large number for safe plotting
    perf_ratios = perf_ratios.replace([np.inf, np.nan], np.max(perf_ratios.values) * 10)

    # Compute a score for sorting: fraction of instances with ratio ≤ 1.1 (or similar)
    score_threshold = 1.1
    solver_scores = (perf_ratios <= score_threshold).mean().sort_values(ascending=False)
    sorted_solvers = solver_scores.index.tolist()

    # τ range for plotting
    tau_vals = np.linspace(1, perf_ratios.max().max(), 500)

    # Plotting
    fig = plt.figure(figsize=(10, 6))

    for solver in sorted_solvers:
        y_vals = [(perf_ratios[solver] <= tau).mean() for tau in tau_vals]
        plt.plot(tau_vals, y_vals, label=solver, linewidth=2.5)

    plt.xlabel(r'Objective ratio $\tau$')
    plt.ylabel('Fraction of instances')
    year_track_pairs = df[['year', 'track']].drop_duplicates()
    datasets = ', '.join([f'{row.year}:{row.track}' for _, row in year_track_pairs.iterrows()])
    plt.title(f'Objective Performance Profile ({datasets})')
    plt.grid(True)
    plt.legend()
    plt.xlim(left=1)

    return fig

def check_inconsistent_instances(df):
    """
    Check for instances that have both SAT and UNS results across different solvers/runs,
    or where a solver claims OPT but has a sub-optimal objective (worse than the best known).
    This indicates an inconsistency that should be investigated.
    Sets status to ERROR for the inconsistent rows.
    """
    inconsistent = []
    suboptimal = []

    for (track, problem, instance), group in df.groupby(['track', 'problem', 'instance']):
        statuses = set(group['status'].unique())

        # Check if both SAT and UNS appear
        if SAT in statuses and UNS in statuses:
            # Get solvers that returned UNS and SAT
            uns_results = group[group['status'] == UNS][['alias', 'time_total']]
            uns_solvers = [(row['alias'], row['time_total']) for _, row in uns_results.iterrows()]
            sat_results = group[group['status'] == SAT][['alias', 'time_total']]
            sat_solvers = [(row['alias'], row['time_total']) for _, row in sat_results.iterrows()]

            inconsistent.append({
                'track': track,
                'problem': problem,
                'instance': instance,
                'statuses': statuses,
                'uns_solvers': uns_solvers,
                'sat_solvers': sat_solvers
            })

            # Set status to ERROR only for the UNS rows of this instance
            mask = (df['track'] == track) & (df['problem'] == problem) & (df['instance'] == instance) & (df['status'] == UNS)
            df.loc[mask, 'status'] = ERR
            sat_info = ', '.join([f"{solver} ({time:.2f}s)" for solver, time in sat_solvers])
            err_msg = f"Inconsistent: UNS but SAT from [{sat_info}]"
            df.loc[mask, 'traceback'] = df.loc[mask, 'traceback'].fillna('') + '\n' + err_msg

        # Check for sub-optimal OPT claims
        # If a solver claims OPT but another solver found a better objective, the OPT claim is wrong
        opt_results = group[group['status'] == OPT]
        verified_results = group[group['status'].isin([SAT, OPT])]
        all_objectives = verified_results['obj'].dropna()
        if len(opt_results) > 0 and len(all_objectives) > 0:
            method = group['method'].iloc[0]
            is_minimize = method == 'minimize'

            # Get the best known objective across ALL results
            best_obj = all_objectives.min() if is_minimize else all_objectives.max()

            # Check if any OPT result has a worse objective than the best known
            for idx, row in opt_results.iterrows():
                if pd.isna(row['obj']):
                    continue
                is_suboptimal = (row['obj'] > best_obj) if is_minimize else (row['obj'] < best_obj)
                if is_suboptimal:
                    # Find which solvers found the better objective
                    better_results = group[group['obj'] == best_obj]
                    better_solvers = better_results['alias'].tolist()

                    suboptimal.append({
                        'track': track,
                        'problem': problem,
                        'instance': instance,
                        'method': method,
                        'solver': row['alias'],
                        'solver_obj': row['obj'],
                        'solver_status': row['status'],
                        'best_obj': best_obj,
                        'better_solvers': better_solvers
                    })

                    # Set status to ERROR for the OPT result with suboptimal objective
                    df.loc[idx, 'status'] = ERR
                    better_solvers_str = ', '.join(better_solvers)
                    cmp = '>' if is_minimize else '<'
                    method_str = 'MIN' if is_minimize else 'MAX'
                    err_msg = f"Sub-optimal OPT ({method_str}): claimed optimal={row['obj']} {cmp} best known={best_obj} from [{better_solvers_str}]"
                    existing = df.loc[idx, 'traceback']
                    df.loc[idx, 'traceback'] = ('' if pd.isna(existing) else existing + '\n') + err_msg

    return inconsistent, suboptimal

def reorder_cols(df, cols):
    return df[cols + [col for col in df.columns if col not in cols]]


def match_solver(search_str, aliases, strict=False):
    """
    Find a solver alias matching the given search string (substring match).

    Parameters
    ----------
    search_str : str
        The substring to search for in solver aliases
    aliases : list or array-like
        Available solver aliases to search through
    strict : bool, default=False
        If True, raise ValueError on no match or multiple matches.
        If False, return None on no match or multiple matches.

    Returns
    -------
    str or None
        The matching alias, or None if no unique match found (when strict=False)

    Raises
    ------
    ValueError
        If strict=True and no match or multiple matches found
    """
    matches = [alias for alias in aliases if search_str in alias]
    if len(matches) == 0:
        if strict:
            raise ValueError(f"No solver alias found containing '{search_str}'. Available aliases: {list(aliases)}")
        return None
    if len(matches) > 1:
        if strict:
            raise ValueError(f"Multiple solver aliases match '{search_str}': {matches}. Please use a more specific string.")
        return None
    return matches[0]


def save_plot(fig, path, name):
    """
    Save a matplotlib figure to both PNG and SVG formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    path : pathlib.Path
        The base path for saving (without extension)
    name : str
        The plot name/suffix to append to the path
    """
    plot_path = path / name
    fig.savefig(f"{plot_path}.png", bbox_inches='tight', dpi=150)
    fig.savefig(f"{plot_path}.svg", bbox_inches='tight')
    print(f"Plot saved to {plot_path}.{{png,svg}}")


def load_and_process_csvs(files, time_limit=None, no_errors=False, intermediate=False, small=None, glob_alias=None, glob_instance=None):
    """
    Load and process CSV files containing benchmark results.

    Parameters
    ----------
    files : list
        List of CSV file paths or directories
    time_limit : float, optional
        Time limit for PAR-2 calculations
    no_errors : bool
        Filter out instances with errors
    intermediate : bool
        Only keep instances present for all solvers
    small : int, optional
        Filter out instances with max table size <= this value
    glob_alias : list, optional
        Filter by solver alias patterns
    glob_instance : list, optional
        Filter by instance patterns

    Returns
    -------
    pd.DataFrame
        Processed dataframe with all solver results
    """
    # Gather all CSV files
    csv_files = []
    for (i, path_str) in enumerate(reversed(files)):
        path = pathlib.Path(path_str)
        assert path.exists(), path
        if path.is_file() and path.suffix == '.csv':
            csv_files.append((i, pathlib.Path(path)))
        elif path.is_dir():
            csv_files.extend((i, pathlib.Path(p)) for p in path.rglob('*.csv'))
        else:
            print(f"Warning: {path} is not a valid CSV file or directory")

    if not csv_files:
        print("No CSV files found.")
        return None

    # Read and merge all CSV files
    dfs = []
    for i, file in csv_files:
        print(f"Reading {file}")
        df = pd.read_csv(file, names=FIELDNAMES, skiprows=1, index_col=False, dtype={'traceback': str, 'exception': str})
        df["run"] = chr(65 + i) if True else str(file.parent)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Find problem names
    df['problem'] = df['instance'].map(lambda x: x.split("-")[0])
    df['instance'] = df['instance'].map(lambda x: "-".join(x.split("-")[1:]).split(".")[0])

    # Rename
    df = df.rename(columns={"objective_value": "obj"})

    # Create metadata dataframe for unique instances only
    df["file_name"] = df["year"].map(str) + "/" + df["track"] + "/" + df["problem"] + "-" + df["instance"] + ".json"

    def get_metadata(x):
        with open(x) as f:
            metadata = json.load(f)
        rowss = [t["rows"] for t in metadata["tables"]]

        def mean(a):
            return statistics.mean(a) if a else None

        def median(a):
            return statistics.median(a) if a else None

        def stdev(a):
            return statistics.stdev(a) if a else None

        def min_(a):
            return min(a) if a else None

        def max_(a):
            return max(a) if a else None

        return [metadata.get("method", None), sum(t["area"] for t in metadata["tables"]), sum(rowss), min_(rowss), max_(rowss), len(rowss), mean(rowss), median(rowss), stdev(rowss) if len(rowss) > 1 else None]

    # Get unique instances to avoid reading the same file multiple times
    unique_instances = df[["year", "track", "problem", "instance", "file_name"]].drop_duplicates()

    # Read metadata only for unique instances
    metadata_values = unique_instances["file_name"].map(get_metadata).to_list()
    unique_instances[METADATA_COLS] = pd.DataFrame(
        metadata_values, index=unique_instances.index
    )

    # Merge metadata back into main dataframe
    df = df.drop(columns=["file_name"]).merge(
        unique_instances.drop(columns=["file_name"]),
        on=["year", "track", "problem", "instance"],
        how="left"
    )

    # Filter out instances with max table size <= small threshold
    df["small"] = (df["max"] <= small) if small is not None else False

    # Let solve include post time
    df["time_solve"] = df["time_solve"] + df["time_post"].fillna(0)

    # Change status to MEM for out of memory errors
    gurobi_oom_mask = df['traceback'].notna() & df['traceback'].astype(str).str.contains("Out of memory|MemoryError", na=False)
    df.loc[gurobi_oom_mask, 'status'] = MEM

    if (df.groupby(by=['problem','instance','alias']).size() > 1).any():
        df['alias'] = df['alias'] + "-" + df['run']

    # Filter by alias
    if glob_alias:
        df = df[df['alias'].map(lambda g: any(g_ in g for g_ in glob_alias))].copy()
    # df = df[~df['alias'].str.contains("mdd", na=False)]

    # Filter by instance
    if glob_instance:
        track_match = df['track'].map(lambda g: any(g_ in g for g_ in glob_instance))
        problem_match = df['problem'].map(lambda g: any(g_ in g for g_ in glob_instance))
        instance_match = df['instance'].map(lambda g: any(g_ in g for g_ in glob_instance))
        df = df[track_match | problem_match | instance_match].copy()

    # # Rename tracks for cleaner display
    # df['track'] = df['track'].replace({
    #     'COP22to25': 'COP',
    #     'CSP22to25': 'CSP'
    # })

    # Set solved status based on track type
    df["unknown"] = df["status"] == UNK
    df["error"] = df["status"] == ERR
    df["memory"] = df["status"] == MEM
    df["feasible"] = df["status"].isin((OPT, SAT, UNS))

    # For COP tracks: solved if status is OPT or UNS
    # For other tracks: solved if status is SAT or UNS
    is_cop = df["track"].str.contains("COP", na=False)
    df["solved"] = ((is_cop & df["status"].isin([OPT, UNS])) |
                    (~is_cop & df["status"].isin([SAT, UNS])))

    # Replace time_solve to NaN if not solved
    df["time_solve"] = df["time_solve"].mask(~df["solved"])

    if intermediate:
        # Filter to only keep instances that occur for all solvers
        total_solvers = df['alias'].nunique()
        instance_solver_counts = df.groupby(['problem', 'instance'])['alias'].nunique()
        valid_instances = instance_solver_counts[instance_solver_counts == total_solvers].index
        df = df.set_index(['problem', 'instance']).loc[valid_instances].reset_index()

    if no_errors:
        # Filter out instances where any solver got an error status
        error_instances = df[df['error']].groupby(['problem', 'instance']).size().index
        df = df.set_index(['problem', 'instance'])
        df = df.drop(error_instances, errors='ignore')
        df = df.reset_index()

    assert not df.empty

    # Add PAR-2 time columns if time_limit is provided
    if time_limit is not None:
        TIMES = ("post", "solve", "total")
        for t in TIMES:
            df[f"time_{t}_p2"] = df[f"time_{t}"].fillna(value=time_limit * 2)

    return df



def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze XCSP3 solver performance data')
    parser.add_argument('files', nargs='+', help='List of CSV files or directories to analyze')
    parser.add_argument('--time-limit', type=float, default=None, help='Maximum time limit in seconds to show on x-axis')
    parser.add_argument('--plot', '-p', type=pathlib.Path, default=None, help='Path to save the plot image (e.g., plot.png)')
    parser.add_argument('--show', nargs='*', choices=['cactus', 'scatter'], default=None,
                        help='Display plots interactively. Specify which plots: cactus, scatter, or both. Use --show without args to show all.')
    parser.add_argument('--sync', type=pathlib.Path, default=None, help='Location to sync files from')
    parser.add_argument('--save', type=pathlib.Path, default=None, help='Location to save post-processed full csv to')
    parser.add_argument('--tex', type=pathlib.Path, default=None, help='Path to save LaTeX tables generated from the analysis')
    parser.add_argument('--no-errors', action='store_true', help='Omit instances which have an error for any solver')
    parser.add_argument('-i', '--intermediate', action='store_true', help='Only show instances which occur for all solvers (intermediate mode)')
    parser.add_argument('--small', type=int, help='Threshold for filtering by largest table size (default: 25)')
    parser.add_argument('-g', '--glob-alias', type=str, nargs="*", default=None, help='Glob alias')
    parser.add_argument('--glob-instance', type=str, nargs="*", default=None, help='Glob instance')
    parser.add_argument('--sort-legend', type=str, choices=['alpha', 'performance'], default='alpha',
                        help='Sort order for legend/lines: alpha (lexicographic, default) or performance (by instances solved)')
    parser.add_argument('--compare', type=str, nargs='+', metavar='SOLVER', default=None,
                        help='Compare solvers: --compare A (compare all vs A), --compare A B (compare A vs B). Creates scatter and correlation plots with A as baseline.')
    parser.add_argument('--metric', type=str, default='t_solv_p2',
                        help='Metric to use for scatter/correlation analysis (default: t_solv_p2). Options: t_solv_p2 (solve time), t_post_p2 (posting time), t_totl_p2 (total time). Requires --time-limit.')
    parser.add_argument('--paper', action='store_true', default=False,
                        help='Use larger font sizes (2x) in plots suitable for papers/publications')
    args = parser.parse_args()
    analyze(**vars(args))

def analyze(files=[], time_limit=None, plot=None, show=None, sync=None, no_errors=False, save=False, intermediate=False, small=None, glob_alias=None, glob_instance=None, tex=None, sort_legend='alpha', compare=None, metric='t_solv_p2', paper=False):

    # Set font sizes for publication-ready plots
    if paper:
        plt.rcParams.update({
            'font.size': 20,
            'axes.titlesize': 24,
            'axes.labelsize': 20,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 18,
        })

    if sync:
        cmd = ["rsync", "-r", sync / files[0], "results"]
        print("CMD", " ".join(str(c) for c in cmd))
        subprocess.run(cmd)

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.expand_frame_repr", False)

    # Use shared function to load and process CSVs
    df = load_and_process_csvs(
        files=files,
        time_limit=time_limit,
        no_errors=no_errors,
        intermediate=intermediate,
        small=small,
        glob_alias=glob_alias,
        glob_instance=glob_instance
    )

    if df is None:
        return

    # Print some stats

# def xcsp3_stats(df, time_limit=None, save=None, solved=[OPT, UNS], tex=None):

    # Start stats
    if False:  # TODO FutureWarning: The behavior of Series.idxmax with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError
        for phase in ['parse', 'model', 'post']:
            slowest_idx = df[f"time_{phase}"].idxmax()
            if slowest_idx is not None and not pd.isna(slowest_idx):
                print(f"Slowest {phase}: {df.loc[slowest_idx, f'time_{phase}']}s ({df.loc[slowest_idx, 'instance']}, {df.loc[slowest_idx, 'solver']})")

    # Show each unique solver+solver_kwargs row
    print("\n== Solvers ==")
    solver_configs = df[['alias', 'solver', 'solver_kwargs']].drop_duplicates().sort_values('alias')
    for _, row in solver_configs.iterrows():
        print(f"  {row['alias']}: {row['solver']} {row['solver_kwargs']}")

    problems = df['problem'].unique()
    print("\nProblems", df['problem'].unique())
    # df = df[df["problem"].isin(problems[:2])]

    pd.set_option('display.float_format', '{:0.1f}'.format)

    df["post"] = ~df["time_post"].isna()
    df["cuts"] = df["n_cuts"] + df["n_cuts_explained"]
    df["lp_cuts"] = df["n_cuts_explained"]
    df["no_cuts"] = df["n_cuts_unexplained"]
    df["cb_rel"] = 100 * (df["time_cb"] / df["time_total"])
    df["time_pc"] = (df["time_cb"] / df["cuts"]) * 1000
    df = df.sort_values(by=["problem", "instance", "alias"])


    if False:
        print("RESULTS")
        df_ =df[[
              # .where(df["status"] == OPT)
            "problem",
            "instance",
            # "small",
            "rows",
            # "mean",
            "median",
            "stdev",
            # "alias",
            "status",
            "method",
            "obj",
            "time_total",
            "time_parse",
            "time_post",
            "time_solve",
            # "time_cb",
            "cb_rel",
            # "time_pc",
            "cuts",
            # "lp_cuts",
            # "no_cuts",
            "constraints",
            # "exception",
        ]]

        if len(df_) == 1:
            print(df_.loc[0])
        else:
            print(df_.sort_values(by=
                       [
                           # "rows",
                           "problem",
                           "instance",
                           "alias"
                           ]
                       # ["time_total"]
                       ))

    # print(df[["instance", "status", "is_err"]].sort_values(by=["instance"]))

    TIMES = ("post", "solve", "total")

    if time_limit is not None:
        for t in TIMES:
            df[f"time_{t}_p2"] = df[f"time_{t}"].fillna(value=time_limit * 2)

    # Check for inconsistent instances
    check_inconsistent_instances(df)


    for grouping_type, grouping in (
            # ("per_alias", ['alias', 'problem']),
            ("per_instance", ['track', 'problem', 'instance', 'alias']),
            ("per_problem", ['track', 'problem', 'alias']),
            ("per_track", ['track', 'alias']),
        ):

        groups = df.assign(
                time_solve=df["time_solve"].where(df["solved"], 2 * time_limit)
            ).groupby(grouping).agg(
                # track = ("track", "first"),
                # alias = ("alias", "first"),
                insts = ("problem", 'count'),
                min_ = ("min", 'min'),
                max_ = ("max", 'max'),
                median = ('median', 'mean'),
                rows = ('rows', 'mean'),
                # stdev = ('stdev', 'first'),
                # t_totl_hr = ('time_total', 'sum'),
                t_totl_p2 = ('time_total_p2', 'mean'),
                t_post_p2 = ('time_post_p2', 'mean'),
                t_solv_p2 = ('time_solve_p2', 'mean'),
                # insts = ('status', 'count'),
                status = ('status', 'first'),
                err = ('error', 'sum'),
                unk = ('unknown', 'sum'),
                mem = ('memory', 'sum'),
                post = ('post', 'sum'),
                feas = ('feasible', 'sum'),
                solv = ('solved', 'sum'),
                cuts = ('cuts', 'mean'),
                lp_cuts = ('lp_cuts', 'mean'),
                no_cuts = ('no_cuts', 'mean'),
                constraints = ('constraints', 'mean'),
                cb_rel = ('cb_rel', 'mean'),
                time_pc = ('time_pc', 'mean'),
                method = ('method', 'first'),
                obj = ('obj', 'first'),
                )

        # Filter out instances which have not been solved by at least one solver
        if grouping_type == "per_instance":
            # Get the instance-level keys (excluding 'alias')
            instance_keys = [k for k in grouping if k != 'alias']
            # For each instance, check if any solver solved it
            solved_mask = groups.groupby(instance_keys)['solv'].transform('max') > 0
            groups = groups[solved_mask]

        track = groups.index.get_level_values('track')[0]
        is_cop = "COP" in track

        # TODO sort by given key from CLI arg
        # groups_ = groups_.reset_index().sort_values(by=["min_", *grouping]).set_index(grouping)

        groups_ = groups[[
            *([
                "insts",
                "min_",
                "max_",
                "err",
                "unk",
                "mem",
            ]),
            *(["feas"] if is_cop else []),
            *([
                # "feas",
                "post",
                "solv",
            ] if not grouping_type == "per_instance" else ["median", "status"]),
            # *(["insts"] if PER_PROBLEM else ["insts"]),
            *([
                "t_post_p2",
                "t_solv_p2",
            ]),
            *(["method", "obj"] if grouping_type == "per_instance" else []),
            *([
                "cuts",
                "lp_cuts",
                "no_cuts",
                "constraints",
                "cb_rel",
                # "time_pc",
            ]),
        ]]


        if "t_totl_hr" in groups_:
            groups_["t_totl_hr"] = groups_["t_totl_hr"].map(lambda x: x / 3600)
        if "rows" in groups_:
            groups_["rows"] = groups_["rows"].map(lambda x: f"{x:.1e}")

        print(f"\n== {grouping_type} ==")
        print(groups_)


        if compare:
            # Show diff for all consecutive rows
            diff_cols = [c for c in groups.columns if c in ["t_post_p2", "t_solv_p2", "unk", "mem", "post", "solv"]]
            # Use first compare argument as baseline
            aliases = groups.index.get_level_values('alias').unique()
            baseline = match_solver(compare[0], aliases)
            if baseline is None:
                continue
            print(f"Using baseline: {baseline}")

            # Get baseline data
            baseline_data = groups.xs(baseline, level='alias')[diff_cols]

            # # Filter out instances that are unsolved by either baseline or any compared solver
            # # Keep only instances where both baseline AND at least one other solver solved it
            # baseline_solved_mask = baseline_data['solv'] > 0
            # baseline_solved_instances = baseline_data[baseline_solved_mask].index
            # # For each instance, check if baseline AND the current solver both solved it
            # groups_index_without_alias = groups.index.droplevel('alias')
            # baseline_solved_filter = groups_index_without_alias.isin(baseline_solved_instances)
            # solver_solved_filter = groups['solv'] > 0
            # # Keep instances where baseline OR solver solved it (at least one)
            # groups = groups[baseline_solved_filter | solver_solved_filter]

            # Update baseline_data after filtering
            baseline_data = groups.xs(baseline, level='alias')[diff_cols]
            print(baseline_data)

            # Compute diff against baseline for each solver
            diff_ = groups.copy()
            for col in diff_cols:
                # Subtract baseline values from all solvers
                # For MultiIndex, we need to align by the non-alias levels
                diff_[col] = groups[col] - baseline_data[col]

            # Drop the baseline itself from the diff
            diff_ = diff_.drop(index=baseline, level="alias", errors='ignore')

            print(f"DIFF (relative to baseline {baseline})")
            print(diff_)

            # Compute correlations for each solver separately
            if grouping_type == "per_instance":
                # Get baseline data
                baseline_data = groups.xs(baseline, level='alias')[diff_cols]

                # Time metrics - use the specified metric
                if metric not in diff_.columns:
                    raise ValueError(f"Metric '{metric}' not found in data. Available metrics: {diff_.columns.tolist()}")
                time_cols = [metric]

                # Select available columns
                available_metadata = [col for col in METADATA_COLS if col in diff_.columns and col != 'method']
                # available_metadata = ["rows"]
                available_time = [col for col in time_cols if col in diff_.columns]
                corr_cols = available_time + available_metadata

                if corr_cols:
                    # Iterate over tracks if present in the grouping
                    if 'track' in diff_.index.names:
                        tracks = diff_.index.get_level_values('track').unique()
                    else:
                        tracks = [None]

                    for track in tracks:
                        # Filter by track if applicable
                        if track is not None:
                            track_diff = diff_.xs(track, level='track')
                            track_baseline = baseline_data.xs(track, level='track')
                        else:
                            track_diff = diff_
                            track_baseline = baseline_data

                        # Get unique solvers (aliases)
                        all_solvers = track_diff.index.get_level_values('alias').unique()

                        # Determine which solvers to create correlation plots for
                        if len(compare) == 1:
                            # Compare all other solvers against baseline
                            solvers_to_plot = [s for s in all_solvers if s != baseline]
                        elif len(compare) == 2:
                            # Only plot for the specific solver being compared
                            solver_str2 = compare[1]
                            matches2 = [alias for alias in all_solvers if solver_str2 in alias]
                            if len(matches2) == 1:
                                solvers_to_plot = matches2
                            else:
                                solvers_to_plot = []  # Skip if no unique match
                        else:
                            solvers_to_plot = []

                        for solver in sorted(solvers_to_plot):
                            # Get data for this solver
                            solver_data = track_diff.xs(solver, level='alias')

                            # Compute correlation for this solver
                            correlation = solver_data[corr_cols].corr()

                            # Mask upper triangle (including diagonal) to avoid showing duplicate info
                            mask = np.triu(np.ones_like(correlation, dtype=bool))
                            correlation_masked = correlation.mask(mask)

                            print(f"\n== Correlation Matrix for {solver} (diff vs baseline) ==")
                            with pd.option_context('display.float_format', '{:.6f}'.format):
                                print(correlation_masked)

                            # Create scatter plots with fitted lines for t_solv_p2 correlations
                            for time_col in time_cols:
                                for metadata_col in available_metadata:
                                    if metadata_col in solver_data.columns:
                                        # Remove NaN values for plotting
                                        plot_data = solver_data[[metadata_col, time_col]].dropna()

                                        if len(plot_data) > 0:
                                            fig, ax = plt.subplots(figsize=(12, 8))

                                            problems = plot_data.index.get_level_values('problem')
                                            unique_problems = sorted(problems.unique())

                                            # Create color and marker maps
                                            # colors = plt.cm.tab20(np.linspace(0, 1, len(unique_problems)))
                                            # problem_colors = dict(zip(unique_problems, colors))

                                            # Different marker shapes for visual distinction
                                            problem_markers = dict(zip(unique_problems,
                                                                     [marker_shapes[i % len(marker_shapes)]
                                                                      for i in range(len(unique_problems))]))

                                            ALPHA_EASY = None

                                            if ALPHA_EASY:
                                                # Determine instances solved in under 10 seconds by all solvers
                                                # Use the original (non-diff) time data from groups
                                                if track is not None:
                                                    groups_track = groups.xs(track, level='track')
                                                else:
                                                    groups_track = groups
                                                # Get max time across all solvers for each instance
                                                max_time_per_instance = groups_track.groupby(['problem', 'instance'])[time_col].max()
                                                fast_instances = set(max_time_per_instance[max_time_per_instance < ALPHA_EASY].index)
                                            else:
                                                fast_instances = None


                                            # Scatter plot colored and shaped by problem class
                                            for problem in unique_problems:
                                                mask = problems == problem
                                                problem_data = plot_data[mask]
                                                n_instances = len(problem_data)
                                                # Determine alpha for each point: 0.5 if solved under 10s by all solvers
                                                alpha_values = np.array([
                                                    0.8 if fast_instances is None or (problem, inst) not in fast_instances else 0.2
                                                    for inst in problem_data.index.get_level_values('instance')
                                                ])
                                                ax.scatter(
                                                        problem_data[metadata_col],
                                                        problem_data[time_col],
                                                         alpha=alpha_values,
                                                         s=50,
                                                         label=f"{problem} ({n_instances})",
                                                         # color=problem_colors[problem],
                                                         marker=problem_markers[problem]
                                                         )
                                        else:
                                            # Fallback: single color if no problem info
                                            ax.scatter(plot_data[metadata_col], plot_data[time_col],
                                                     alpha=0.6, s=50, label='Data points')

                                        # Fit line
                                        if len(plot_data) > 1:
                                            z = np.polyfit(plot_data[metadata_col], plot_data[time_col], 1)
                                            p = np.poly1d(z)
                                            x_line = np.linspace(plot_data[metadata_col].min(),
                                                               plot_data[metadata_col].max(), 100)
                                            ax.plot(x_line, p(x_line), 'r-', linewidth=2,
                                                  label=f'Fitted line (r={correlation.loc[metadata_col, time_col]:.3f})')

                                        # Set axis scales and limits
                                        ax.set_xscale('log')

                                        # Set y-axis limits to -PAR..PAR with padding
                                        PAR = 2 * time_limit
                                        padding = 0.1  # 10% padding
                                        ax.set_ylim(-PAR * (1 + padding), PAR * (1 + padding))

                                        ax.set_xlabel(metadata_col)
                                        ax.set_ylabel(f'{time_col} differential')
                                        ax.set_title(f'{baseline} - {solver}: {metadata_col} vs {time_col} differential ({track})')

                                        # Place legend outside plot area to avoid covering data (hide for paper)
                                        if not paper:
                                            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                                                    borderaxespad=0.)
                                        ax.grid(True, alpha=0.3)

                                        plt.tight_layout()

                                        # Save plot
                                        if plot:
                                            save_plot(fig, plot, f"correlation-{track}-{baseline}-{solver}-{metadata_col}-{time_col}")

                                        plt.close(fig)


    # Collect all scatter plots for show logic
    fig_scatters = []

    for track, groups in df.groupby(by="track"):
        is_cop = "COP" in track
        if tex is not None:
            tex.mkdir(exist_ok=True, parents=True)
            n_instances = len((groups["problem"] + "-" + groups["instance"]).unique())

            track = track[:3]

            def rename_idx(x):
                if "base" in x:
                    return "\\baseline"
                elif "cutoff" in x:
                    return "\\cutoff"
                elif "none" in x:
                    return "\\explain"
                else:
                    return f"\\{x.split('-')[1:][0]}"

            # Aggregate per alias for tex output
            tex_df = groups.groupby('alias').agg(
                unk=('unknown', 'sum'),
                mem=('memory', 'sum'),
                post=('post', 'sum'),
                feas=('feasible', 'sum'),
                solv=('solved', 'sum'),
                t_solv_p2=('time_solve_p2', 'mean'),
                cuts=('cuts', 'mean'),
                cb_rel=('cb_rel', 'mean'),
            )
            # tex_df = tex_df.rename(index=rename_idx)
            print(tex_df)
            latex_output = tex_df[
                [
                    *(["unk", "mem", "post"]),
                    *(["feas"] if is_cop else []),
                    *(["solv", "t_solv_p2", "cuts", "cb_rel"])
                ]
            ].to_latex(
                na_rep="",
                header=[
                    *(["\\unk", "\\mem", "\\pst"]),
                    *(["\\sat"] if is_cop else []),
                    *(["\\sol", "time [s]", "cuts", "CB [\\%]"]),
                ],
                float_format="%.1f",
                caption=f"{n_instances} {track} instances",
                label=f"tbl:res:{track.lower()}",
                escape=True,
            )

            # Move caption to bottom of table
            lines = latex_output.split('\n')
            caption_lines = []
            other_lines = []

            # Separate caption/label lines from other lines
            for line in lines:
                if line.strip().startswith('\\caption') or line.strip().startswith('\\label'):
                    caption_lines.append(line)
                else:
                    other_lines.append(line)

            # Reconstruct with caption at the bottom
            if caption_lines:
                # Find the \end{table} line
                for i, line in enumerate(other_lines):
                    if '\\end{table}' in line:
                        # Insert caption lines before \end{table}
                        other_lines = other_lines[:i] + caption_lines + other_lines[i:]
                        break
                latex_output = '\n'.join(other_lines)
            else:
                latex_output = '\n'.join(other_lines)

            # Save to file
            with open(tex / f"table-{track}.tex", 'w') as f:
                f.write(latex_output)
            print(f"LaTeX table saved to {tex}")

        # Determine which plots to generate and show
        show_cactus = False
        show_scatter = False
        if show is not None:
            # If show is an empty list, show all plots
            if len(show) == 0:
                show_cactus = True
                show_scatter = True
            else:
                show_cactus = 'cactus' in show
                show_scatter = 'scatter' in show

        # Generate plots based on plot argument or show argument
        generate_cactus = plot is not None or show_cactus

        fig_cactus = None

        if generate_cactus:
            # Generate performance/cactus plot
            fig_cactus = xcsp3_plot(
                groups.reset_index(),
                time_limit,
                filter_by="feasible",
                sort_legend=sort_legend,
            )

            if plot:
                save_plot(fig_cactus, plot, f"cactus-{track}")

        # Generate scatter plot(s) if --compare option is provided
        fig_scatters = []
        aliases = sorted(groups["alias"].unique())
        # Compute global inst_metric range for consistent colorbar across plots
        inst_metric_col = "rows"
        inst_metric_range = (groups[inst_metric_col].min(), groups[inst_metric_col].max())
        if compare is not None:
            # Find baseline solver
            baseline_solver = match_solver(compare[0], aliases)
            if baseline_solver is None:
                continue

            # Determine which solvers to compare against baseline
            if len(compare) == 1:
                # Compare all other solvers against baseline
                compare_solvers = [alias for alias in aliases if alias != baseline_solver]
            elif len(compare) == 2:
                # Compare specific solver against baseline
                solver2 = match_solver(compare[1], aliases)
                if solver2 is None:
                    continue
                compare_solvers = [solver2]
            else:
                raise ValueError(f"--compare takes 1 or 2 arguments, got {len(compare)}")

            # Create scatter plot for each comparison
            for solver2 in compare_solvers:
                print(f"Creating scatter plot: {baseline_solver} vs {solver2}")

                # Map aggregated metric names to raw column names for scatter plot
                metric_map = {
                    't_solv_p2': 'time_solve_p2',
                    't_post_p2': 'time_post_p2',
                    't_totl_p2': 'time_total_p2',
                }
                scatter_metric = metric_map.get(metric, metric)

                fig_scatter = xcsp3_scatter_plot(
                    groups.reset_index(),
                    solver1=baseline_solver,
                    solver2=solver2,
                    metric=scatter_metric,
                    time_limit=time_limit,
                    inst_metric="rows",
                    track=track,
                    inst_metric_range=inst_metric_range,
                    paper=paper,
                )
                fig_scatters.append(fig_scatter)

                if plot:
                    # Include both solver names in filename for clarity
                    solver1_short = baseline_solver.replace('gurobi-', '').replace('-', '')
                    solver2_short = solver2.replace('gurobi-', '').replace('-', '')
                    save_plot(fig_scatter, plot, f"scatter-{track}-{solver1_short}--{solver2_short}")

    # Set status to ERR for rows that don't pass the checker
    def checker_failed(checker_result):
        if pd.isna(checker_result):
            return False
        lines = checker_result.split("\n")
        assert len(lines) >= 2, f"Unexpected checker_result format: {checker_result}"
        return not lines[-2].startswith("OK")

    checker_fail_mask = df['checker_result'].apply(checker_failed)
    df.loc[checker_fail_mask, 'status'] = ERR

    # Check if obj matches checker_result objective
    for idx, row in df.iterrows():
        try:
            if pd.isna(row['obj']) or pd.isna(row['checker_result']):
                continue
            row['checker_result']
            lines = row['checker_result'].split("\n")
            if len(lines) >= 2:
                checker_obj = float(lines[-2].split("\t")[-1])
                if row['obj'] != checker_obj:
                    df.loc[idx, 'status'] = ERR
                    err_msg = f"Objective mismatch: solver reported {row['obj']} but checker found {checker_obj}"
                    existing = df.loc[idx, 'traceback']
                    df.loc[idx, 'traceback'] = ('' if pd.isna(existing) else existing + '\n') + err_msg
        except Exception as e:
            # raise Exception(f"Exception in checker {row['checker_result']}") from e
            df.loc[idx, 'status'] = ERR
            err_msg = f"Exception in checker {row['checker_result']}"


    print("== ERRORS ==")
    for idx, row in df.iterrows():
        if row["status"] == ERR:
            # Skip errors containing the range object error
            if pd.notna(row['exception']) and "object has no attribute" in str(row['exception']):
                continue

            print(f"\n[2025/{row['track']}/{row['problem']}-{row['instance']}.xml - {row['alias']}]")
            print(f"Status: {row['status']} | Time: {row['time_total']:.2f}s")
            if pd.notna(row['exception']):
                print(f"Exception: {row['exception']}")
            if pd.notna(row['traceback']):
                print(f"Traceback:\n{row['traceback']}")

            print(f"Solution: {row['solution']}")
            print(f"Exception: {row['checker_result']}")
        elif row['status'] == MEM:
            assert pd.isna(row["exception"]) or "Out of memory" in row['exception'] or "MemoryError" in row['exception'] or "Unable to allocate" in row['exception'] or "Invalid argument to Model.addLConstr", row
            # assert pd.isna(row["traceback"]) or "Out of memory" in row['traceback'] or "MemoryError" in row['traceback'] or "Unable to allocate" in row['traceback']
        else:
            # assert pd.isna(row['exception']), row
            # assert pd.isna(row['traceback']), row
            assert pd.isna(row['checker_result']) or row['checker_result'].split("\n")[-2].startswith("OK"), row

    # for idx, row in df.iterrows():
    #     if pd.notna(row['checker_result']) and not row['checker_result'].split("\n")[-2].startswith("OK"):
    #         print(f"Exception: {row['checker_result']}")

    # Close figures we don't want to show before calling plt.show()
    if show is not None:
        if fig_cactus is not None and not show_cactus:
            plt.close(fig_cactus)
        # Close scatter plots if not showing them
        if not show_scatter:
            for fig in fig_scatters:
                plt.close(fig)
        plt.show()





    # if not errors.empty:
        # exc = df[df["status"] == ERR]
        # exc["file"] = exc["problem"] + "-" + exc["instance"]
        # print(", ".join(str(x)[:100] for x in df["exception"].unique()))
        # print(", ".join(str(x)[:100] for x in df["exception"].unique()))


    # Save convenience
    if save:
        df[
                [
                    'problem',
                    'instance',
                    # 'run',
                    'alias',
                    'status',
                    'obj',
                    ] +
                [f'time_{t}' for t in (
                    "total",
                    "parse",
                    "model",
                    "post",
                    "solve",
                    "cb",
                )] + [
                    'cb_rel',
                    'n_cuts',
                    'n_cuts_explained',
                    'n_cuts_unexplained',
                    'exception',
                ]].to_csv(save, float_format='%.1f')

if __name__ == '__main__':
    # Create custom print function that writes to both console and file
    # Open output file for logging
    output_file = open('./analysis.txt', 'w')

    original_print = builtins.print
    def print_to_both(*args, **kwargs):
        original_print(*args, **kwargs)
        kwargs_file = kwargs.copy()
        kwargs_file['file'] = output_file
        kwargs_file['flush'] = True
        original_print(*args, **kwargs_file)

    # Replace built-in print
    builtins.print = print_to_both

    try:
        main()
    finally:
        # Restore original print and close file
        builtins.print = original_print
        output_file.close()
        print(f"Analysis output saved to ./analysis.txt")
