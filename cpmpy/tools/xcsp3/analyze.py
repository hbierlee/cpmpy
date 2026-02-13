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

--scatter : str str, optional
    Create a scatter plot for two specific solvers. Takes two strings A and B that match solver aliases.
    Each string must match exactly one solver alias (substring match). Raises exception if multiple matches found.

--tex : path, optional
    Path to save LaTeX tables generated from the analysis results.

--paper : bool, optional
    Use larger font sizes (2x) in plots suitable for papers/publications.
"""

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

def xcsp3_scatter_plot(df, solver1=None, solver2=None, metric="time_solve", inst_metric="median", time_limit=None):
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
    # cmap.set_under('red')

    # cmap.set_bad('blue')  # Color for masked values (small instances)

    # Create alpha array - lower alpha for red dots (below small threshold)
    alpha_values = np.where(merged["small_1"], 0.2, 0.8)

    # Plot scatter points colored by median
    scatter = ax.scatter(
        x,
        y,
        c=inst_metrics,
        alpha=alpha_values,
        s=50,
        cmap=cmap,
        edgecolors='black',
        linewidth=0.5,
        norm=LogNorm(
            vmin=inst_metrics.min(),
            vmax=inst_metrics.max(),
        ),
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'table {inst_metric}', rotation=270, labelpad=20)

    # # Add min and max as ticks
    # effective_vmin = max(vmin, 1e-10) if SET_MIN is None else SET_MIN
    # existing_ticks = cbar.get_ticks()
    # new_ticks = sorted(set([effective_vmin, vmax] + list(existing_ticks)))
    # cbar.set_ticks(new_ticks)

    # Plot diagonal line (y=x)
    max_val = max(x.max(), y.max())
    min_val = min(x.min(), y.min())

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

    ax.legend()
    plt.tight_layout()

    # Add hover labels for instance names
    try:
        import mplcursors
        cursor = mplcursors.cursor(scatter, hover=True)
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
                else:
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
    Check for instances that have both SAT and UNS results across different solvers/runs.
    This indicates an inconsistency that should be investigated.
    """
    inconsistent = []

    for (problem, instance), group in df.groupby(['problem', 'instance']):
        statuses = set(group['status'].unique())

        # Check if both SAT and UNS appear
        if SAT in statuses and UNS in statuses:
            # Get only the solvers that returned UNS
            uns_results = group[group['status'] == UNS][['alias', 'time_total']]
            uns_solvers = [(row['alias'], row['time_total']) for _, row in uns_results.iterrows()]

            inconsistent.append({
                'problem': problem,
                'instance': instance,
                'statuses': statuses,
                'uns_solvers': uns_solvers
            })

    if inconsistent:
        print("\n== INCONSISTENT INSTANCES (both SAT and UNS) ==")
        for item in inconsistent:
            uns_info = ', '.join([f"{solver} ({time:.2f}s)" for solver, time in item['uns_solvers']])
            print(f"{item['problem']}-{item['instance']}: UNS from [{uns_info}]")
        print(f"\nTotal inconsistent instances: {len(inconsistent)}")

    return inconsistent

def reorder_cols(df, cols):
    return df[cols + [col for col in df.columns if col not in cols]]

    
    
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
    parser.add_argument('--solved-only', action='store_true', help='Only show instances which have been solved by all solvers')
    parser.add_argument('-i', '--intermediate', action='store_true', help='Only show instances which occur for all solvers (intermediate mode)')
    parser.add_argument('--small', type=int, help='Threshold for filtering by largest table size (default: 25)')
    parser.add_argument('-g', '--glob-alias', type=str, nargs="*", default=None, help='Glob alias')
    parser.add_argument('--glob-instance', type=str, nargs="*", default=None, help='Glob instance')
    parser.add_argument('--sort-legend', type=str, choices=['alpha', 'performance'], default='alpha',
                        help='Sort order for legend/lines: alpha (lexicographic, default) or performance (by instances solved)')
    parser.add_argument('--scatter', type=str, nargs=2, metavar=('A', 'B'), default=None,
                        help='Create scatter plot for two specific solvers whose aliases contain strings A and B')
    parser.add_argument('--paper', action='store_true', default=False,
                        help='Use larger font sizes (2x) in plots suitable for papers/publications')
    args = parser.parse_args()
    analyze(**vars(args))

def analyze(files=[], time_limit=None, plot=None, show=None, sync=None, no_errors=False, save=False, solved_only=False, intermediate=False, small=None, glob_alias=None, glob_instance=None, tex=None, sort_legend='alpha', scatter=None, paper=False):

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

    import subprocess
    if sync:
        cmd = ["rsync", "-r", sync / files[0], "results"]
        print("CMD", " ".join(str(c) for c in cmd))
        subprocess.run(cmd)
    


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
        return

    # Read and merge all CSV files
    dfs = []
    for i, file in csv_files:
        print("Reading", file)
        df = pd.read_csv(file, names=FIELDNAMES, skiprows=1, index_col=False)
        df["run"] = chr(65 + i) if True else str(file.parent)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.expand_frame_repr", False)


    # find problem names
    df['problem'] = df['instance'].map(lambda x: x.split("-")[0])
    df['instance'] = df['instance'].map(lambda x: "-".join(x.split("-")[1:]).split(".")[0])

    # rename
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
    unique_instances[["method", "area", "rows", "min", "max", "count", "mean", "median", "stdev"]] = pd.DataFrame(
        metadata_values, index=unique_instances.index
    )

    # Merge metadata back into main dataframe
    df = df.drop(columns=["file_name"]).merge(
        unique_instances.drop(columns=["file_name"]),
        on=["year", "track", "problem", "instance"],
        how="left"
    )


    # df[df["method"] == "minimize"]["obj"] *= -1  # higher is better

    # df["obj"] = df.where(df["method"] == "minimize", -df["obj"], df["obj"])
    # df["obj"] = df.map(lambda x: -x["obj"] if x["method"] == "minimize" else x["obj"])
    # df = df.drop(df[df["max"] <= 100].index)

    # Filter out instances with max table size <= small threshold
    
    df["small"] = (df["max"] <= small) if small is not None else False
    # df = df.drop(df[df["small"]].index)

    # let solve include post time?
    df["time_solve"] = df["time_solve"] + df["time_post"].fillna(0)

    # Change status to MEM for Gurobi out of memory errors
    gurobi_oom_mask = df['traceback'].notna() & df['traceback'].str.contains("gurobipy._exception.GurobiError: Out of memory", na=False)
    df.loc[gurobi_oom_mask, 'status'] = MEM

    if (df.groupby(by=['problem','instance','alias']).size() > 1).any():
        df['alias'] = df['alias'] + "-" + df['run']

    for col, glob in [("alias", glob_alias), ("problem", glob_instance), ("track", glob_instance)]:
        if glob:
            df = df.drop(df[~df[col].map(lambda g: any(g_ in g for g_ in glob))].index)


    # print(df.where(df["status"] == OPT).groupby(by=['problem', 'instance'])['obj'].nunique())
    # print(df.mask(df["status"] == OPT).groupby(by=['problem', 'instance']).agg(lambda x: ','.join(str(x_) for x_ in x.unique()))['obj'])

    df['alias'] += '.'

    # is_cop = "COP" in track

    # solved = [OPT, UNS] if is_cop else [SAT, UNS]
    solved = [OPT, UNS]

    df["unknown"] = df["status"] == UNK
    df["error"] = df["status"] == ERR
    df["memory"] = df["status"] == MEM
    df["feasible"] = df["status"].isin((OPT, SAT, UNS))
    df["solved"] = df["status"].isin(solved)


    # replace time_solve to NaN if not solved
    df["time_solve"] = df["time_solve"].mask(~df["status"].isin(solved))

    if solved_only:
        df = df[df[['problem', 'instance']].apply(lambda x: set(df[(df['problem'] == x['problem']) & (df['instance'] == x['instance'])]["status"].unique()).issubset(solved), axis=1)]

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

    # Print some stats

# def xcsp3_stats(df, time_limit=None, save=None, solved=[OPT, UNS], tex=None):

    # Start stats
    if False:  # TODO FutureWarning: The behavior of Series.idxmax with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError
        for phase in ['parse', 'model', 'post']:
            slowest_idx = df[f"time_{phase}"].idxmax()
            if slowest_idx is not None and not pd.isna(slowest_idx):
                print(f"Slowest {phase}: {df.loc[slowest_idx, f'time_{phase}']}s ({df.loc[slowest_idx, 'instance']}, {df.loc[slowest_idx, 'solver']})")

    # TODO show each unique solver+solver_kwargs row
    # print("Solvers", df[['alias', 'solver_kwargs']].nunique())

    problems = df['problem'].unique()
    print("Problems", df['problem'].unique())
    # df = df[df["problem"].isin(problems[:2])]

    # Check for inconsistent instances
    check_inconsistent_instances(df)

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

    for grouping_type, grouping in (
            # ("per_alias", ['alias', 'problem']),
            ("per_instance", ['track', 'problem', 'instance', 'alias']),
            ("per_problem", ['track', 'problem', 'alias']),
            ("per_track", ['track', 'alias']),
        ):

        groups = df.assign(
                time_solve=df["time_solve"].where(df["status"].isin(solved), 2 * time_limit)
            ).groupby(grouping).agg(
                # track = ("track", "first"),
                # alias = ("alias", "first"),
                insts = ("problem", 'count'),
                min_ = ("min", 'min'),
                max_ = ("max", 'max'),
                median = ('median', 'first'),
                rows = ('rows', 'first'),
                # stdev = ('stdev', 'first'),
                # t_totl_hr = ('time_total', 'sum'),
                t_totl_p2 = ('time_total_p2', 'mean'),
                t_post_p2 = ('time_post_p2', 'mean'),
                t_solv_p2 = ('time_solve_p2', 'mean'),
                # diff = ("diff", 'mean'),
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

        # is_cop = "COP" in track

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
            # *(["feas"] if is_cop else []),
            *([
                "feas",
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

        # Show diff for all consecutive rows
        diff_cols = [c for c in groups.columns if c in ["t_post_p2", "t_solv_p2", "unk", "mem", "post", "solv"]]

        # TODO add cli arg to determine baseline alias
        baseline = "base_gurobi-gleb."
        baseline = None
        if baseline:
            diff_ = groups
            # TODO show diff for each solver to the baseline (rather than to each other by comparing to the previous row)
            diff_[diff_cols] = groups[diff_cols].diff()
            diff_ = diff_.drop(index=baseline, level=grouping[-1] if diff_.index.nlevels > 1 else None)

            print("DIFF")
            print(diff_)

            # Compute correlations
            if grouping_type == "per_instance":
                correlation = diff_[['t_solv_p2', 'median']].corr()
                print("\n== Correlation between t_solv_p2 and median ==")
                with pd.option_context('display.float_format', '{:.6f}'.format):
                    print(correlation)

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

            tex_df = groups
            # tex_df = groups.rename(index=rename_idx)
            latex_output = tex_df[
                [
                    *(["unk", "mem", "post"]),
                    *(["feas"] if is_cop else [] ),
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
            with open(tex.with_name(tex.name + "_table.tex"), 'w') as f:
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
        fig_scatter = None

        if generate_cactus:
            # Generate performance/cactus plot
            fig_cactus = xcsp3_plot(
                groups.reset_index(),
                time_limit,
                filter_by="feasible",
                solved_only=solved_only,
                sort_legend=sort_legend,
            )

            plot_ = plot.with_name(plot.name + "-" + track)
            if plot_:
                # plot.mkdir(exist_ok=True, parents=True)
                cactus = plot_.with_name(plot_.name + "-cactus")
                fig_cactus.savefig(cactus.with_suffix(".png"), bbox_inches='tight')
                fig_cactus.savefig(cactus.with_suffix(".svg"), bbox_inches='tight')
                print(f"Plot saved to {cactus}.{{png,svg}}")

        # Generate custom scatter plot if --scatter option is provided
        fig_scatter_custom = None
        aliases = sorted(groups["alias"].unique())
        if scatter is not None:
            solver_str1, solver_str2 = scatter

            # Find all aliases containing the provided strings
            matches1 = [alias for alias in aliases if solver_str1 in alias]
            matches2 = [alias for alias in aliases if solver_str2 in alias]

            # Validate that each string matches exactly one solver
            if len(matches1) == 0:
                raise ValueError(f"No solver alias found containing '{solver_str1}'. Available aliases: {aliases}")
            if len(matches1) > 1:
                raise ValueError(f"Multiple solver aliases match '{solver_str1}': {matches1}. Please use a more specific string.")
            if len(matches2) == 0:
                raise ValueError(f"No solver alias found containing '{solver_str2}'. Available aliases: {aliases}")
            if len(matches2) > 1:
                raise ValueError(f"Multiple solver aliases match '{solver_str2}': {matches2}. Please use a more specific string.")

            solver1 = matches1[0]
            solver2 = matches2[0]

            print(f"Creating custom scatter plot: {solver1} vs {solver2}")

            fig_scatter_custom = xcsp3_scatter_plot(
                groups.reset_index(),
                solver1=solver1,
                solver2=solver2,
                time_limit=time_limit,
                # inst_metric="area",
                # inst_metric="max",
                inst_metric="rows",
            )

            if plot_:
                scatter = plot_.with_name(plot_.name + "_scatter")
                fig_scatter_custom.savefig(scatter.with_suffix(".png"), bbox_inches='tight')
                fig_scatter_custom.savefig(scatter.with_suffix(".svg"), bbox_inches='tight')
                print(f"Custom scatter plot saved to {scatter}.{{png,svg}}")

    errors = df[df["status"] == ERR][["problem","instance","alias","status","time_total", "exception", "traceback"]]
    if not errors.empty:
        print("== ERRORS ==")
        for idx, error in errors.iterrows():
            # Skip errors containing the range object error
            if pd.notna(error['exception']) and "object has no attribute" in str(error['exception']):
                continue

            print(f"\n[{error['problem']}-{error['instance']} - {error['alias']}]")
            print(f"Status: {error['status']} | Time: {error['time_total']:.2f}s")
            if pd.notna(error['exception']):
                print(f"Exception: {error['exception']}")
            if pd.notna(error['traceback']):
                print(f"Traceback:\n{error['traceback']}")

    # Close figures we don't want to show before calling plt.show()
    if show is not None:
        if fig_cactus is not None and not show_cactus:
            plt.close(fig_cactus)
        if fig_scatter is not None and not show_scatter:
            plt.close(fig_scatter)
        # Custom scatter plot is shown if scatter option is enabled or show all
        if fig_scatter_custom is not None and not show_scatter:
            plt.close(fig_scatter_custom)
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
    main()
