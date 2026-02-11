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

def xcsp3_plot(df, time_limit=None, metric="time_solve", filter_by="solved", solved_only=False):
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

    # Sort solvers descending by number of instances solved
    solvers_sorted = solver_counts.sort_values(ascending=False).index.tolist()
    
    # Create figure
    fig = plt.figure(figsize=(10, 6))
    
    for solver in sorted(solvers): # Sort solver names for consistent ordering
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

def xcsp3_scatter_plot(df, solver1=None, solver2=None, metric="time_solve", inst_metric="median", time_limit=None, small=25):
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
    metadata_cols = ['rows', 'median', 'stdev', 'min', 'max', 'time_post']

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
    cbar.set_label(f'{inst_metric} table size', rotation=270, labelpad=20)

    # # Add min and max as ticks
    # effective_vmin = max(vmin, 1e-10) if SET_MIN is None else SET_MIN
    # existing_ticks = cbar.get_ticks()
    # new_ticks = sorted(set([effective_vmin, vmax] + list(existing_ticks)))
    # cbar.set_ticks(new_ticks)

    # Plot diagonal line (y=x)
    max_val = max(x.max(), y.max())
    min_val = min(x.min(), y.min())

    min_max = [0.1, PAR]
    ax.plot(min_max, min_max, 'k--', linewidth=1.5, label='Equal performance', zorder=0)

    # Add 10% improvement lines (parallel to diagonal)
    improvement_factor = 2
    ax.plot(min_max, [m * improvement_factor for m in min_max], 'k:', linewidth=1, alpha=0.5, zorder=0)
    ax.plot(min_max, [m / improvement_factor for m in min_max], 'k:', linewidth=1, alpha=0.5, zorder=0)

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

    title = f"Scatter Plot: {solver1} vs {solver2} ({datasets})\n"
    title += f"{solver1} faster: {solver1_wins}, {solver2} faster: {solver2_wins}, Ties: {ties}"
    title += f" | Cutoff: {small}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Use log scale if there's a wide range
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set axis limits
    if time_limit is not None:
        ax.set_xlim(*min_max)
        ax.set_ylim(*min_max)

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

def xcsp3_stats(df, time_limit=None, save=None, solved=[OPT, UNS], tex=False):

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
            ("per_inst", ['problem','instance','alias']),
            ("per_problem", ['problem', 'alias']),
            ("per_alias", ['alias'])
            ):

        groups = df.assign(
                time_solve=df["time_solve"].where(df["status"].isin(solved), 2 * time_limit)
            ).groupby(grouping).agg(
                alias = ("alias", "first"),
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

        track, = df["track"].unique()
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
                "post",
                "solv",
            ] if not grouping_type == "per_inst" else ["median", "status"]),
            # *(["insts"] if PER_PROBLEM else ["insts"]),
            *([
                "t_post_p2",
                "t_solv_p2",
            ]),
            *(["method","obj"] if grouping_type == "per_inst" else []),
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
            if grouping_type == "per_inst":
                correlation = diff_[['t_solv_p2', 'median']].corr()
                print("\n== Correlation between t_solv_p2 and median ==")
                with pd.option_context('display.float_format', '{:.6f}'.format):
                    print(correlation)

        if tex and grouping_type == "per_alias":
            n_instances = len((df["problem"] + "-" + df["instance"]).unique())

            track, = df["track"].unique()
            track = track[:3]
            print(groups.index)
            def rename_idx(x):
                if "base" in x:
                    return "\\baseline"
                elif "cutoff" in x:
                    return "\\cutoff"
                elif "none" in x:
                    return "\\explain"
                else:
                    print(x)
                    return f"\\{x.split('-')[1:][0]}"

            tex = groups
            # tex = groups.rename(index=rename_idx)
            print(tex[
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
            ))


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
    parser.add_argument('--tex', action='store_true', default=None, help='Output tables in tex')
    parser.add_argument('--no-errors', action='store_true', help='Omit instances which have an error for any solver')
    parser.add_argument('--solved-only', action='store_true', help='Only show instances which have been solved by all solvers')
    parser.add_argument('-i', '--intermediate', action='store_true', help='Only show instances which occur for all solvers (intermediate mode)')
    parser.add_argument('--small', type=int, default=25, help='Threshold for filtering by largest table size (default: 25)')
    parser.add_argument('-g', '--glob-alias', type=str, nargs="*", default=None, help='Glob alias')
    parser.add_argument('--glob-instance', type=str, nargs="*", default=None, help='Glob instance')
    args = parser.parse_args()
    analyze(**vars(args))

def analyze(files=[], time_limit=None, plot=None, show=None, sync=None, no_errors=False, save=False, solved_only=False, intermediate=False, small=25, glob_alias=None, glob_instance=None, tex=False):

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


        return [metadata.get("method", None), sum(rowss), min_(rowss), max_(rowss), len(rowss), mean(rowss), median(rowss), stdev(rowss) if len(rowss) > 1 else None]

    df[["method", "rows", "min", "max", "count", "mean", "median", "stdev"]] = pd.DataFrame(df["file_name"].map(get_metadata).to_list(),index=df.index )


    # df[df["method"] == "minimize"]["obj"] *= -1  # higher is better

    # df["obj"] = df.where(df["method"] == "minimize", -df["obj"], df["obj"])
    # df["obj"] = df.map(lambda x: -x["obj"] if x["method"] == "minimize" else x["obj"])
    # df = df.drop(df[df["max"] <= 100].index)

    # Filter out instances with max table size <= small threshold
    df["small"] = df["max"] <= small
    # df = df.drop(df[df["small"]].index)

    # let solve include post time?
    df["time_solve"] = df["time_solve"] + df["time_post"].fillna(0)

    # Change status to MEM for Gurobi out of memory errors
    gurobi_oom_mask = df['traceback'].notna() & df['traceback'].str.contains("gurobipy._exception.GurobiError: Out of memory", na=False)
    df.loc[gurobi_oom_mask, 'status'] = MEM

    if (df.groupby(by=['problem','instance','alias']).size() > 1).any():
        df['alias'] = df['alias'] + "-" + df['run']

    for col, glob in [("alias", glob_alias), ("problem", glob_instance), ("track", "COP")]:
        if glob:
            df = df.drop(df[~df[col].map(lambda g: any(g_ in g for g_ in glob))].index)


    # print(df.where(df["status"] == OPT).groupby(by=['problem', 'instance'])['obj'].nunique())
    # print(df.mask(df["status"] == OPT).groupby(by=['problem', 'instance']).agg(lambda x: ','.join(str(x_) for x_ in x.unique()))['obj'])

    df['alias'] += '.'

    track, = df["track"].unique()
    is_cop = "COP" in track

    solved = [OPT, UNS] if is_cop else [SAT, UNS]

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
    xcsp3_stats(df, time_limit=time_limit, save=save, tex=tex)

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
    generate_scatter = plot is not None or show_scatter

    fig_cactus = None
    fig_scatter = None

    if generate_cactus:
        # Generate performance/cactus plot
        fig_cactus = xcsp3_plot(df, time_limit, filter_by="feasible", solved_only=solved_only)
        if plot:
            cactus = plot.with_name(f"{plot}_cactus")
            fig_cactus.savefig(cactus.with_suffix(".png"), bbox_inches='tight')
            fig_cactus.savefig(cactus.with_suffix(".svg"), bbox_inches='tight')
            print(f"Plot saved to {cactus}.{{png,svg}}")

    # Generate scatter plot if exactly 2 solvers
    aliases = sorted(df["alias"].unique())
    if generate_scatter and len(aliases) == 2:
        fig_scatter = xcsp3_scatter_plot(
                df,
                solver1=aliases[0],
                solver2=aliases[1],
                time_limit=time_limit,
                inst_metric="rows",
                small=small,
                # metric="time_post"
                )
        if plot:
            scatter = plot.with_name(f"{plot}_scatter")
            fig_scatter.savefig(scatter.with_suffix(".png"), bbox_inches='tight')
            fig_scatter.savefig(scatter.with_suffix(".svg"), bbox_inches='tight')
            print(f"Plot saved to {scatter}.{{png,svg}}")

    # Close figures we don't want to show before calling plt.show()
    if show is not None:
        if fig_cactus is not None and not show_cactus:
            plt.close(fig_cactus)
        if fig_scatter is not None and not show_scatter:
            plt.close(fig_scatter)
        plt.show()


if __name__ == '__main__':
    main()
