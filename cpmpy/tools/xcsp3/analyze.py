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
    Path to save the generated plot image (e.g., "plot.png"). If not provided, the plot will be displayed interactively.
"""

import argparse
import ast
import json
import pathlib
import statistics
import re
import matplotlib
import pandas as pd
import numpy as np
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

    # # Determine the status to plot (Opt if at least one opt, otherwise sat)
    # if filter_by == "solved":
    #     status_filter = solved
    # elif filter_by == "feasible":
    #     status_filter = (OPT, UNS, SAT)
    # else:
    #     raise Exception

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
    plt.title(f"Performance Plot ({datasets}) {' (solved only)' if solved_only else ''}")
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

    df1 = df[df['alias'] == solver1][['instance', metric, 'solved', inst_metric]].copy()
    df2 = df[df['alias'] == solver2][['instance', metric, 'solved', inst_metric]].copy()

    # Merge on instance to get paired data
    merged = df1.merge(df2, on='instance', suffixes=('_1', '_2'))

    # Handle unsolved instances
    if time_limit is not None:
        merged[f'{metric}_1'] = merged[f'{metric}_1'].fillna(time_limit)
        merged[f'{metric}_2'] = merged[f'{metric}_2'].fillna(time_limit)
    else:
        # Drop instances where either solver didn't solve it
        merged = merged.dropna(subset=[f'{metric}_1', f'{metric}_2'])

    # Extract x and y coordinates
    x = merged[f'{metric}_1'].values
    y = merged[f'{metric}_2'].values

    inst_metrics = merged[f'{inst_metric}_1'].values
    print(f"Median values - min: {inst_metrics.min()}, max: {inst_metrics.max()}, unique: {len(np.unique(inst_metrics))}")
    print(f"Sample median values: {inst_metrics[:10] if len(inst_metrics) >= 10 else inst_metrics}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Use log normalization if the range is large
    vmin, vmax = inst_metrics.min(), inst_metrics.max()
    # inst_metrics = inst_metrics.where(inst_metrics <= 25, 0, inst_metrics)

    # Plot scatter points colored by median
    scatter = ax.scatter(
            x,
            y,
            c=inst_metrics,
            alpha=0.6,
            s=50,
            cmap='viridis',
            edgecolors='black',
            linewidth=0.5,
            norm=LogNorm(
                vmin=max(vmin, 1e-10),
                # vmin=100,
                vmax=vmax
                ),
            )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'{inst_metric} table size', rotation=270, labelpad=20)

    # Plot diagonal line (y=x)
    max_val = max(x.max(), y.max())
    min_val = min(x.min(), y.min())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Equal performance', zorder=0)

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

    ax.set_title(f"Scatter Plot: {solver1} vs {solver2} ({datasets})\n"
                 f"{solver1} faster: {solver1_wins}, {solver2} faster: {solver2_wins}, Ties: {ties}")
    ax.grid(True, alpha=0.3)

    # Use log scale if there's a wide range
    if max_val / min_val > 100:
        ax.set_xscale('log')
        ax.set_yscale('log')

    # Set axis limits
    if time_limit is not None:
        ax.set_xlim(0, time_limit * 1.1)
        ax.set_ylim(0, time_limit * 1.1)

    ax.legend()
    plt.tight_layout()

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

def xcsp3_stats(df, time_limit=None, save=None, solved=[OPT, UNS], tex=False):

    if False:  # TODO FutureWarning: The behavior of Series.idxmax with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError
        for phase in ['parse', 'model', 'post']:
            slowest_idx = df[f"time_{phase}"].idxmax()
            if slowest_idx is not None and not pd.isna(slowest_idx):
                print(f"Slowest {phase}: {df.loc[slowest_idx, f'time_{phase}']}s ({df.loc[slowest_idx, 'instance']}, {df.loc[slowest_idx, 'solver']})")

    print("Solvers", df[['alias', 'solver_kwargs']])

    print("Problems", df['problem'].unique())


    diff = "Δ"
    df.insert(df.columns.get_loc("time_solve"), diff, df.groupby(['instance'])["time_solve"].diff())
    # df["rows"], df["count"] = df["file_name"].map(get_metadata)
    pd.set_option('display.float_format', '{:0.1f}'.format)

    df["post"] = ~df["time_post"].isna()
    df["cuts"] = df["n_cuts"] + df["n_cuts_explained"]
    df["cb_rel"] = 100 * (df["time_cb"] / df["time_total"])
    df["time_pc"] = (df["time_cb"] / df["cuts"]) * 1000
    df = df.sort_values(by=["problem", "instance", "alias"])


    SHOW_DIFF = False
    if SHOW_DIFF:
        df = df.drop(df[df["alias"] == "base_gurobi"].index)

    if True:
        print("RESULTS")
        print(df
              # .where(df["status"] == OPT)
              [[
            "problem",
            "instance",
            "rows",
            # "mean",
            "median",
            "stdev",
            ] + ([] if SHOW_DIFF else ["alias"])
             + [
            diff,
            "status",
            "method",
            "obj",
            "time_total",
            "time_parse",
            "time_post",
            "time_solve",
            "time_cb",
            "cb_rel",
            "time_pc",
            "cuts",
            "constraints",
            # "exception",
        ]].sort_values(by=
                       [
                           "rows",
                           "problem",
                           "instance",
                           "alias"
                           ]
                       # ["time_total"]
                       + ([] if SHOW_DIFF else ["alias"])
                       ))
    if SHOW_DIFF:
        exit(0)

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


        groups = df.groupby(grouping).agg(
                alias = ("alias", "first"),
                insts = ("problem", 'count'),
                min_ = ("min", 'min'),
                max_ = ("max", 'max'),
                # median = ('median', 'first'),
                # stdev = ('stdev', 'first'),
                # t_totl_hr = ('time_total', 'sum'),
                t_totl_p2 = ('time_total_p2', 'mean'),
                t_post_p2 = ('time_post_p2', 'mean'),
                t_solv_p2 = ('time_solve_p2', 'mean'),
                # insts = ('status', 'count'),
                err = ('error', 'sum'),
                unk = ('unknown', 'sum'),
                mem = ('memory', 'sum'),
                post = ('post', 'sum'),
                feas = ('feasible', 'sum'),
                solv = ('solved', 'sum'),
                cuts = ('cuts', 'mean'),
                constraints = ('constraints', 'mean'),
                cb_rel = ('cb_rel', 'mean'),
                time_pc = ('time_pc', 'mean'),
                method = ('method', 'first'),
                obj = ('obj', 'first'),
                )

        track, = df["track"].unique()
        is_cop = "COP" in track


        groups = groups[[
            # *(["insts"] if PER_PROBLEM else ["insts"]),
            *([] if grouping_type == "per_alias" else [
                "min_",
                "max_",
                # "median",
                # "stdev",
                ]),
            *([
                "t_post_p2",
                "t_solv_p2",
                ]),
            *(["method","obj"] if grouping_type == "per_inst" else []),
            *([
                "insts",
                "err",
                "unk",
                "mem",
                "post",
                ]),
            *([
                "feas",
            ] if is_cop else []),
            *([
                "solv",
                "cuts",
                "constraints",
                "cb_rel",
                "time_pc",
            ]),
        ]]


        # groups = groups.sort_index(level=["problem"], by="rows")
        # groups = groups.sort_values(by="rows", ascending=False)
        if "t_totl_hr" in groups:
            groups["t_totl_hr"] = groups["t_totl_hr"].map(lambda x: x / 3600)
        if "rows" in groups:
            groups["rows"] = groups["rows"].map(lambda x: f"{x:.1e}")
        # groups.loc[('Total')] = groups.sum(numeric_only=True)

        print(f"\n== {grouping_type} ==")
        print(groups)

        if tex and grouping_type == "per_alias":
            n_instances = len((df["problem"] + "-" + df["instance"]).unique())

            # n_instances = df.groupby(['problem','instance']).nunique()

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
                    *(["\sat"] if is_cop else []),
                    *(["\sol", "time [s]", "cuts", "CB [\%]"]),
                ],
                float_format="%.1f",
                caption=f"{n_instances} {track} instances",
                label=f"tbl:res:{track.lower()}",
                escape=True,
            ))


    errors = df[df["status"] == ERR][["problem","instance","alias","status","time_total", "exception", "traceback"]]
    if not errors.empty:
        print("== ERRORS ==")
        # exc = df[df["status"] == ERR]
        # exc["file"] = exc["problem"] + "-" + exc["instance"]
        print(errors)
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
    parser.add_argument('--sync', type=pathlib.Path, default=None, help='Location to sync files from')
    parser.add_argument('--save', type=pathlib.Path, default=None, help='Location to save post-processed full csv to')
    parser.add_argument('--tex', action='store_true', default=None, help='Output tables in tex')
    parser.add_argument('--no-errors', action='store_true', help='Omit instances which have an error for any solver')
    parser.add_argument('--solved-only', action='store_true', help='Only show instances which have been solved by all solvers')
    parser.add_argument('--glob-alias', type=str, nargs="*", default=None, help='Glob alias')
    parser.add_argument('--glob-instance', type=str, nargs="*", default=None, help='Glob instance')
    args = parser.parse_args()
    analyze(**vars(args))

def analyze(files=[], time_limit=None, plot=None, sync=None, no_errors=False, save=False, solved_only=False, glob_alias=None, glob_instance=None, tex=False):

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
        df["run"] = chr(65 + i)
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
    # df = df.drop(df[df["max"] <= 25].index)
    print(df[["instance", "median", "stdev", "min", "max"]])
    # assert False






    # let solve include post time?
    df["time_solve"] = df["time_solve"] + df["time_post"].fillna(0)

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



    # # temporarily drop all instances where there are any errors
    # print(df)
    # print(df.columns)
    # # statuses = df.apply(lambda x: df[(df['problem'] == x['problem']) & (df['instance'] == x['instance'])]["status"].unique())
    # statuses = df.group(by=["problem", "instance"]).unique()
    # print(statuses)
    # assert False

    # if no_errors:
    #     df = df.drop(df[df['instance'].map(lambda x: ERR in df[df["instance"] == x & df["instance"] == x]["status"].unique())].index)

    if solved_only:
        df = df[df[['problem', 'instance']].apply(lambda x: set(df[(df['problem'] == x['problem']) & (df['instance'] == x['instance'])]["status"].unique()).issubset(solved), axis=1)]

    assert not df.empty

    # Print some stats
    xcsp3_stats(df, time_limit=time_limit, save=save, tex=tex)

    if plot:
        # Save or show plot
        fig = xcsp3_plot(df, time_limit, filter_by="feasible", solved_only=solved_only)
        fig.savefig(plot.with_suffix(".png"), bbox_inches='tight')
        fig.savefig(plot.with_suffix(".svg"), bbox_inches='tight')
        print(f"Plot saved to {plot}.{{png,svg}}")

        aliases = sorted(df["alias"].unique())
        if len(aliases) == 2 and plot:
            fig = xcsp3_scatter_plot(
                    df,
                    solver1=aliases[0],
                    solver2=aliases[1],
                    time_limit=time_limit,
                    inst_metric="min",
                    # metric="time_post"
                    )
            scatter = plot.with_name("scatter")
            fig.savefig(scatter.with_suffix(".png"), bbox_inches='tight')
            fig.savefig(scatter.with_suffix(".svg"), bbox_inches='tight')
            print(f"Plot saved to {scatter}.{{png,svg}}")

    # else:
    #     plt.show()

    if len(df) == 1:
        print(df.loc[0])



if __name__ == '__main__':
    main()
