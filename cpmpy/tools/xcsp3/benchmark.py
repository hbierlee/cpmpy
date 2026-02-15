"""
Benchmark solvers on XCSP3 Instances by replicating the XCSP3 competition.

A command-line tool for benchmarking constraint solvers on XCSP3 competition instances.
Supports parallel execution, time/memory limits, and solver configuration.

Required Arguments
------------------
--year : int
    The competition year (e.g., 2023).

--track : str
    The competition track (e.g., "CSP", "COP", "MiniCOP").

--solver : str
    The name of the solver to benchmark (e.g., "ortools", "exact", "choco").

Optional Arguments
------------------
--workers : int, default=4
    Number of parallel workers to use.

--time-limit : int, default=300
    Time limit in seconds per instance.

--mem-limit : int, default=8192
    Memory limit in megabytes per instance.

--output-dir : str, default='results'
    Directory where result CSV files will be saved.

--verbose
    If set, display full xcsp3 output during execution.

--intermediate
    If set, report intermediate solutions (if supported by the solver).
"""

import cProfile
import csv
import os
import signal
import subprocess
import time
import pathlib
import pprint
import lzma
import sys
import argparse
import warnings
import traceback
import multiprocessing
import itertools
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple
from io import StringIO
from datetime import datetime
from filelock import FileLock
from concurrent.futures import ThreadPoolExecutor
import gurobipy

import cpmpy as cp
from cpmpy.tools.xcsp3.experiments import get_experiments
from cpmpy.tools.xcsp3.dataset import XCSP3Dataset
from cpmpy.tools.xcsp3 import read_xcsp3
from cpmpy.tools.xcsp3.xcsp3_cpmpy import xcsp3_cpmpy, init_signal_handlers, ExitStatus, TIME_BUFFER, CheckError
import analyze


class Tee:
    """
    A stream-like object that duplicates writes to multiple underlying streams.
    """
    def __init__(self, *streams):
        """
        Arguments:
            *streams: Any number of file-like objects that implement a write() method,
                      such as sys.stdout, sys.stderr, or StringIO.
        """
        self.streams = streams

    def write(self, data):
        """
        Write data to all underlying streams.

        Args:
            data (str): The string to write.
        """
        for s in self.streams:
            s.write(data)

    def flush(self):
        """
        Flush all underlying streams to ensure all data is written out.
        """
        for s in self.streams:
            s.flush()

class PipeWriter:
    """
    Stdout wrapper for a multiprocessing pipe.
    """
    def __init__(self, conn):
        self.conn = conn
    def write(self, data):
        if data:  # avoid empty writes
            try:
                self.conn.send(data)
            except:
                pass
    def flush(self):
        pass  # no buffering


def xcsp3_wrapper(conn, kwargs, verbose, profile):
    """
    Wraps a call to xcsp3_cpmpy as to correctly 
    forward stdout to the multiprocessing pipe (conn).
    Also sends a last status report though the pipe.

    Status report can be missing when process has been terminated by a SIGTERM.
    """
    
    original_stdout = sys.stdout

    pipe_writer = PipeWriter(conn)

    if not verbose:
        warnings.filterwarnings("ignore")
        sys.stdout = pipe_writer # only forward to pipe
    else:
        sys.stdout = Tee(original_stdout, pipe_writer) # forward to pipe and console

    try:
        init_signal_handlers() # configure OS signal handlers

        if profile:
            cProfile.runctx('xcsp3_cpmpy(**kwargs, verbose=verbose)', globals(), locals(), profile)
        else:
            xcsp3_cpmpy(**kwargs, verbose=verbose)
        conn.send({"status": "ok"})
    except TimeoutError as e: # capture exceptions and report in state
        tb_str = traceback.format_exc()
        conn.send({"status": ExitStatus.unknown.value, "exception": e, "traceback": tb_str})
    except MemoryError as e: # capture exceptions and report in state
        tb_str = traceback.format_exc()
        conn.send({"status": ExitStatus.memory.value, "exception": e, "traceback": tb_str})
    except gurobipy._exception.GurobiError as e:
        tb_str = traceback.format_exc()
        status = ExitStatus.memory if e.errno == 10001 else ExitStatus.error
        conn.send({"status": status.value, "exception": e, "traceback": tb_str})
    except Exception as e: # capture exceptions and report in state
        tb_str = traceback.format_exc()
        conn.send({"status": ExitStatus.error.value, "exception": e, "traceback": tb_str})
    finally:
        sys.stdout = original_stdout
        conn.close()

# exec_args = (filename, metadata, solver, time_limit, mem_limit, check_time_limit, output_file, verbose) 
def execute_instance(args: Tuple[str, dict, str, str, dict, dict, int, int, int, int, pathlib.Path, bool, bool, str, pathlib.Path]) -> None:
    """
    Solve a single XCSP3 instance and write results to file immediately.
    
    Args is a list of:
        filename: Path to the XCSP3 instance file
        metadata: Dictionary containing instance metadata (year, track, name)
        alias: Alias for the solver configto use
        solver: Name of the solver to use
        solver_kwargs: Solver init kwargs
        solve_kwargs: Solver solve kwargs
        time_limit: Time limit in seconds
        mem_limit: Memory limit in MB
        check_time_limit: Check time limit in seconds
        output_file: Path to the output CSV file
        verbose: Whether to show solver output
        profile: profile
    """
    
    filename, metadata, alias, solver, solver_kwargs, solve_kwargs, time_limit, mem_limit, check_time_limit, cores, output_file, verbose, intermediate, checker_path, profile = args
    output_file = pathlib.Path(output_file)

    # Fieldnames for the CSV file
    result = dict.fromkeys(analyze.FIELDNAMES)  # init all fields to None
    result['year'] = metadata['year']
    result['track'] = metadata['track']
    result['instance'] = metadata['name']
    result['solver'] = solver if isinstance(solver, str) else solver().name
    result['alias'] = alias
    result['solver_kwargs'] = str(solver_kwargs) + "-" + str(solve_kwargs)
    # result['solve_kwargs'] = str(solve_kwargs)
    # TODO result['options'] = str(solver_kwargs)

    # Decompress before timers start
    file_path = filename
    if str(filename).endswith(".lzma"):
        # Decompress the XZ file
        with lzma.open(filename, 'rt', encoding='utf-8') as f:
            xml_file = StringIO(f.read()) # read to memory-mapped file
            filename = xml_file
            
    # Start total timing
    total_start = time.time()
    
    # Call xcsp3 in separate process
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = multiprocessing.Pipe() # communication pipe between processes
    process = ctx.Process(target=xcsp3_wrapper, args=(
                                                    child_conn, 
                                                      {
                                                          "benchname": filename, 
                                                          "metadata": metadata,
                                                          "alias": alias, 
                                                          "solver": solver, 
                                                          "time_limit": time_limit, 
                                                          "check_time_limit": check_time_limit, 
                                                          "mem_limit": mem_limit, 
                                                          "intermediate": intermediate, 
                                                          "time_buffer": TIME_BUFFER,
                                                          "cores": cores,
                                                          "solve_kwargs": solve_kwargs,
                                                          "solver_kwargs": solver_kwargs,
                                                        }, 
                                                    verbose,
                                                    profile)
                          )
    process.start()
       
    sol_time = None # For annotation intermediate solutions (when they were received)
    
    # Default status if nothing returned by subprocess
    # -> process exited prematurely due to sigterm
    status = {"status": ExitStatus.unknown.value}

    # Parse the output to get status, solution and timings
    complete_solution = None
    timeout = time_limit + check_time_limit
    while True:
        line = parent_conn.recv() if parent_conn.poll(timeout=0.1) else None
        dt = None if timeout is None else time.time() - total_start

        if not process.is_alive():
            break
        elif line is None:
            pass
        elif isinstance(line, str):
            if line.startswith('s '):
                result['status'] = line[2:].strip()
            elif line.startswith('v ') and result['solution'] is None:
                # only record first line, contains 'type' and 'cost'
                solution = line.split("\n")[0][2:].strip()
                result['solution'] = str(solution)
                complete_solution = line
                if "cost" in solution:
                    result['objective_value'] = solution.split('cost="')[-1][:-2]
            elif line.startswith('o '):
                obj = int(line[2:].strip())
                if result['intermediate'] is None:
                    result['intermediate'] = []
                result['intermediate'] += [(sol_time, obj)]
                result['objective_value'] = obj
                obj = None
            elif line.startswith('c Solution'):
                parts = line.split(', time = ')
                # Get solution time from comment for intermediate solution -> used for annotating 'o ...' lines
                sol_time = float(parts[-1].replace('s', '').rstrip())
            elif line.startswith('c took '):
                # Parse timing information
                parts = line.split(' seconds to ')
                if len(parts) == 2:
                    time_val = float(parts[0].replace('c took ', ''))
                    action = parts[1].strip()
                    if action.startswith('parse'):
                        result['time_parse'] = time_val
                    elif action.startswith('convert'):
                        result['time_model'] = time_val
                    elif action.startswith('post'):
                        result['time_post'] = time_val
                    elif action.startswith('solve'):
                        result['time_solve'] = time_val
            elif line.startswith('c Stat'): # c Stat=x=y
                parts = line.split('=')
                field = parts[1]
                result[field] = parts[2]
                if field not in analyze.FIELDNAMES:
                    analyze.FIELDNAMES.append(field)

        # Received a new status from the subprocess
        elif isinstance(line, dict):
            status = line
            break

        else:
            raise()

        if dt > timeout:  # a buffer time is already given to the solver
            break

    process.join(timeout=1)

    # Replicate competition convention on how jobs get terminated
    if process.is_alive():
        # Send sigterm to let process know it reached its time limit
        os.kill(process.pid, signal.SIGTERM)
        # 1 second grace period
        process.join(timeout=1)
        # Kill if still alive
        if process.is_alive():
            os.kill(process.pid, signal.SIGKILL)
            process.join()

    result['time_total'] = time.time() - total_start

    # Parse the exit status
    if status["status"] != "ok":
        # Ignore timeouts
        # if "TimeoutError" in repr(status["exception"]):
        #     pass

        # All other exceptions, put in solution field
        # result["exception"] = status['exception']
        result |= status

    if checker_path is not None and complete_solution is not None:
        checker_output, checker_time = run_solution_checker(
            JAR=checker_path,
            instance_location=file_path,
            out_file="'" + complete_solution.replace("\n\r", " ").replace("\n", " ").replace("v   ", "").replace("v ", "")+ "'",
            verbose=verbose,
            cpm_time=result.get('time_solve', 0)  # or total solve time you have
        )

        if checker_output is not None:
            result['checker_result'] = checker_output
        else:
            result['checker_result'] = None

    # Use a lock file to prevent concurrent writes
    lock_file = output_file.with_suffix(".lock")
    lock = FileLock(lock_file)
    try:
        with lock:
            # Pre-check if file exists to determine if we need to write header
            write_header = not os.path.exists(output_file)
            # # TODO fix dynamic fieldnames
            # if output_file.exists():
            #     import pandas as pd
            #     df = pd.read_csv(output_file)
            #     if not set(fieldnames).subset(set(df.columns)):
            #         df.append(result)

            with open(output_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=analyze.FIELDNAMES)
                if write_header:
                    writer.writeheader()
                writer.writerow(result)
    finally:
        # Optional: cleanup if the lock file somehow persists
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except Exception:
                pass  # avoid crashing on cleanup


def run_solution_checker(JAR, instance_location, out_file, verbose, cpm_time):

    start = time.time()
    command = ["java", "-jar", JAR, "'" + str(instance_location) + "'" + " " + str(out_file)]
    command = " ".join(command)
    test_res_str = subprocess.run(command, capture_output=True, text=True, shell=True)
    checker_time = time.time() - start

    if verbose:
        for line in test_res_str.stdout.split("\n"):
            print("c " + line)
        print(f"c cpmpy time: {cpm_time}")
        print(f"c validation time: {checker_time}")
        print(f"c elapsed time: {cpm_time + checker_time}")
    
    return test_res_str.stdout.split("\n")[-2], checker_time


def get_table_metadata_(table):
    X, tab = table.args
    cols = [cp.expressions.utils.dom_size(x) for x in X]
    rows = len(tab)
    return {"cols": cols, "rows": rows, "area": rows * sum(cols)}

def get_table_metadata(model):
    tables = []
    for c in model.constraints:
        if isinstance(c, cp.expressions.core.Expression) and c.name == "table":
            tables.append(get_table_metadata_(c))
    match model.objective_is_min:
        case True:
            method = "minimize"
        case False:
            method = "maximize"
        case None:
            method = "satisfy"
        case _:
            raise ValueError(model.objective_is_min)

    return { "method": method, "area": sum(t["area"] for t in tables), "tables": tables}



def xcsp3_benchmark(
    year: int,
    track: str,
    solver: str,
    alias: Optional[str] = None,
    solver_kwargs: Optional[dict] = {},
    solve_kwargs: Optional[dict] = {},
    workers: int = 1,
    time_limit: int = 300,
    mem_limit: Optional[int] = 4096, cores: int=1,
    check_time_limit: int = 10,
    output_dir: str = 'results',
    no_timestamp: bool = False,
    verbose: bool = False, intermediate: bool = False,
    checker_path: Optional[str] = None,
    glob_instance: Optional[str] = None,
    first: Optional[bool] = False,
    profile: Optional[pathlib.Path] = False,
    results: Optional[list] = None,
    filter_feasible: bool = False,
    filter_easy: Optional[float] = None,
) -> str:
    """
    Benchmark a solver on XCSP3 instances.
    
    Args:
        year (int): Competition year (e.g., 2023)
        track (str): Track type (e.g., COP, CSP, MiniCOP)
        solver (str): Solver name (e.g., ortools, exact, choco, ...)
        alias (str): Solver alias (e.g., ortools-parallel, ...)
        workers (int): Number of parallel workers
        time_limit (int): Time limit in seconds per instance
        check_time_limit (int): Check time limit in seconds per instance
        glob_instance (int): Filter instances
        mem_limit (int): Memory limit in MB per instance
        output_dir (str): Output directory for CSV files
        verbose (bool): Whether to show solver output
        
    Returns:
        str: Path to the output CSV file
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    alias = (solver if isinstance(solver, str) else solver().name) if alias is None else alias
    output_file = pathlib.Path(f"xcsp3_{year}_{track}_{alias}")
    if no_timestamp is False:
        # Get current timestamp in a filename-safe format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_file.with_name(f"{output_file.name}_{timestamp}")
    
    # Define output file path with timestamp
    output_file = output_dir / output_file.with_suffix(".csv")
    output_file.unlink(missing_ok=True)
    
    # Initialize dataset
    problems = set()

    areas = dict()
    def update_metadata_table(metadata):
        metadata["problem"] = metadata["name"].split("-")[0]
        print(f"Updating table metadata for {metadata['name']}", end=None)
        if 'tables' not in metadata and (metadata["problem"] not in areas or areas[metadata["problem"]]):
            try:
                model = read_xcsp3(metadata['path'])
                print("read")
                metadata = { **metadata, **get_table_metadata(model) }
                print("area = ", metadata["area"])
                areas[metadata["problem"]] = metadata["area"]
            except Exception as e:
                metadata["error"] = str(e)
                metadata["area"] = 0
                metadata["tables"] = []
        else:
            metadata["area"] = 0
            metadata["tables"] = []
            print("cached as 0")

        return metadata

    dataset = XCSP3Dataset(
            year=year,
            track=track,
            download=True,
            target_transform=update_metadata_table
            )
    if glob_instance is not None:
        dataset = ((filename, metadata) for filename, metadata in dataset if glob_instance in filename)
    if first:
        dataset_ = []
        for k, g in itertools.groupby(dataset, key=lambda f_m: f_m[0].split("-")[0]):
            dataset_.append(min(g, key=lambda g_:g_[0]))
        dataset = dataset_
    # dataset = [(filename, metadata) for filename, metadata in dataset if min((t["rows"] for t in metadata["tables"]), default=0) >= 25]
    dataset = [(filename, metadata) for filename, metadata in dataset if metadata["area"]]

    # Filter by results from CSV files if requested
    if results is not None:
        print(f"Loading results from: {results}")
        df_results = analyze.load_and_process_csvs(files=results, time_limit=time_limit)

        # Start with all instances in results
        filtered_set = set((prob + "-" + inst) for prob, inst in df_results.groupby(['problem', 'instance']).size().index)
        print(f"Found {len(filtered_set)} instances in results")

        # Filter by feasibility if requested
        if filter_feasible:
            print("Filtering to feasible instances only")
            feasible_instances = df_results[df_results["feasible"]].groupby(['problem', 'instance']).size()
            feasible_set = set((prob + "-" + inst) for prob, inst in feasible_instances.index)
            print(f"Found {len(feasible_set)} feasible instances")
            filtered_set = filtered_set & feasible_set

        # Filter by solve time if requested
        if filter_easy is not None:
            print(f"Filtering to instances solved in at most {filter_easy} seconds")
            easy_instances = df_results[
                df_results["solved"] & (df_results["time_solve"] <= filter_easy)
            ].groupby(['problem', 'instance']).size()
            easy_set = set((prob + "-" + inst) for prob, inst in easy_instances.index)
            print(f"Found {len(easy_set)} easy instances (solved <= {filter_easy}s)")
            filtered_set = filtered_set & easy_set

        # Apply filtering to dataset
        if filter_feasible or filter_easy is not None:
            original_size = len(dataset)
            dataset = [
                (filename, metadata)
                for filename, metadata in dataset
                if metadata["name"].split(".")[0] in filtered_set
            ]
            filters = []
            if filter_feasible:
                filters.append("feasibility")
            if filter_easy is not None:
                filters.append("time")
            print(f"Filtered dataset from {original_size} to {len(dataset)} instances based on {' and '.join(filters)}")
    exit(0)

    assert dataset

    # Process instances in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks and track their futures
        futures = [executor.submit(execute_instance,  # below: args
                                   (filename, metadata, alias, solver, solver_kwargs, solve_kwargs, time_limit, mem_limit, check_time_limit, cores, output_file, verbose, intermediate, checker_path, profile))
                   for filename, metadata in dataset]
        # Process results as they complete
        for i,future in enumerate(tqdm(futures, total=len(futures), desc=f"Running {alias}")):
            try:
                _ = future.result(timeout=time_limit+60)  # for cleanliness sake, result is empty
            except TimeoutError:
                pass
            except Exception as e:
                print(f"Job {i}: {dataset[i][1]['name']}, ProcessPoolExecutor caught: {e}")

        # raise()
        # TODO [thomas] ?
        # raise Exception()
    
    return output_file

def main(args):
    args_ = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k not in ("analyze", "glob_alias", "dry", "reverse_experiments", "debug")
    }

    if not args_["verbose"]:
        warnings.filterwarnings("ignore")
    
    experiments = get_experiments(
        overrides=args_,
        filters=[("alias", args.glob_alias)] if args.glob_alias else []
    )

    for e in experiments:
        if type(e["solver"]) != str and e["solver"]().name == "lazy_gurobi":
            if args.debug:
                e["solver_kwargs"]["env"]["debug"] = True
            e["solver_kwargs"]["env"]["verbosity"] = 0

    if args.reverse_experiments:
        experiments.reverse()

    print("Solvers")
    # Print only unique alias/solver_kwargs combinations
    seen_configs = set()
    for e in experiments:
        config_key = (e["alias"], str(sorted(e.get("solver_kwargs", {}).items())))
        if config_key not in seen_configs:
            seen_configs.add(config_key)
            print(f"  {e['alias']}")
            pprint.pprint(e["solver_kwargs"], indent=4)

    assert experiments

    if args.dry:
        exit(0)


    for experiment in experiments:
        print("Run", experiment)
        output_file = xcsp3_benchmark(**experiment)
        print(f"Results added to {output_file}")


    output_dir = pathlib.Path(next(e["output_dir"] for e in experiments))
    csvs = list(output_dir.glob("*.csv"))
    assert csvs
    import pandas as pd
    dfs = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    dfs.to_csv(pathlib.Path(output_dir.name).with_suffix(".csv"))

    if args.analyze:
        analyze.analyze([output_dir], time_limit=args.time_limit)
    return dfs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Benchmark solvers on XCSP3 instances')
    parser.add_argument('--year', type=int, help='Competition year (e.g., 2023)')
    parser.add_argument('--track', type=str, help='Track type (e.g., COP, CSP, MiniCOP)')
    parser.add_argument('--solver', type=str, help='Solver name (e.g., ortools, exact, choco, ...)')
    parser.add_argument('--alias', type=str, help='Solver config alias (e.g., ortools-par, ...)')
    parser.add_argument('--workers', type=int, help='Number of parallel workers')
    parser.add_argument('--time-limit', type=int, help='Time limit in seconds per instance')
    parser.add_argument('--check-time-limit', type=int, help='Check time limit in seconds per instance')
    parser.add_argument('--glob-alias', type=str, nargs="+", default=None, help='Solver config alias (e.g., ortools-par, ...)')
    parser.add_argument('--glob-instance', type=str, default=None, help='Filter instances according to glob expression')
    parser.add_argument('--first', action='store_true', help='Run only first instance of each problem')
    parser.add_argument('--mem-limit', type=int, help='Memory limit in MB per instance')
    parser.add_argument('--cores', type=int, help='Number of cores to assign to a single instance')
    parser.add_argument('--output-dir', type=str, help='Output directory for CSV files')
    parser.add_argument('--no-timestamp', action='store_true', help='Add timestamp to file names')
    parser.add_argument('--verbose', action='store_true', help='Show solver output')
    parser.add_argument('--intermediate', action='store_true', help='Report on intermediate solutions')
    parser.add_argument('--checker-path', type=str, help='Path to the XCSP3 solution checker JAR file')
    parser.add_argument('--profile', type=pathlib.Path, help='Profile')
    parser.add_argument('--analyze', action='store_true', help='Analyze results')
    parser.add_argument('--dry', action='store_true', help='Dry run')
    parser.add_argument('--debug', action='store_true', help='Debug run')
    parser.add_argument('--reverse-experiments', action='store_true', help='Reverse experiment order')
    parser.add_argument('--results', type=str, nargs='+', help='CSV files/directories with previous benchmark results for filtering')
    parser.add_argument('--filter-feasible', action='store_true', help='When used with --results, filter to only run instances found feasible')
    parser.add_argument('--filter-easy', type=float, help='When used with --results, only keep instances solved in at most this many seconds by any solver')

    main(parser.parse_args())
