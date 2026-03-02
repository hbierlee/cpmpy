import math
import time
import pprint
import itertools
import pathlib
import pickle
import random
import traceback
import tracemalloc

import numpy as np
import pandas as pd
import pytest
import sys

import cpmpy as cp
from cpmpy.expressions.utils import argvals
from cpmpy.transformations.get_variables import get_variables, get_variables_model
from cpmpy.tools.xcsp3 import read_xcsp3
from cpmpy.expressions.utils import show_assignment, dom_size
from cpmpy.solvers.lazy_gurobi import CPM_lazy_gurobi, normalize_table, Heuristic, Coverlift, TableData
from cpmpy.solvers.ortools import CPM_ortools
from cpmpy.tools.xcsp3.experiments import get_solvers

INT_VAR_NAMES = list("xyzuvwprq")
SEED = 42
random.seed(SEED)


def generate_table_from_example():
    x = cp.intvar(1, 4, name="x")
    y = cp.intvar(1, 3, name="y")
    z = cp.intvar(1, 3, name="z")
    X = (x, y, z)
    T = [[2, 1, 1], [3, 2, 2], [4, 3, 3], [1, 2, 3], [2, 1, 2]]

    return cp.Model(cp.Table(X, T))


def generate_two_tables():
    x = cp.intvar(1, 4, name="x")
    y = cp.intvar(1, 3, name="y")
    z = cp.intvar(2, 4, name="z")
    w = cp.intvar(1, 2, name="w")
    return cp.Model(
        cp.Table((x, y, z), [[2, 1, 2], [3, 2, 2], [4, 3, 3], [1, 2, 3], [2, 1, 4]]),
        cp.Table((z, y, w), [[2, 1, 1], [3, 2, 1], [2, 3, 1], [2, 3, 1], [4, 1, 2]]),
    )


def _load_xcsp3(path):
    """Load an XCSP3 model from a file path."""
    sys.argv = ["-nocompile"]  # Stop pyxcsp3 from complaining on exit
    return read_xcsp3(pathlib.Path(path))


def generate_models_w_tables(hardness=(0, 2), glob=None):
    """Generator yielding (name, model) tuples for various test cases

    Args:
        hardness: Tuple (min, max) for test case hardness levels
        glob: Filter test cases by name substring (or tuple of substrings, all must match)
    """
    glob_filters = (glob,) if isinstance(glob, str) else (glob or ())

    def matches(name):
        return all(g in name for g in glob_filters)

    def _generate():
        a, b = hardness

        # yield ("stillife_bug", _load_xcsp3("2025/COP22to25/StillLife-05-05_c24.xml"))
        # # yield ("bug", _load_xcsp3("2025/COP22to25/DC-rijndael-xor5-d1-t0-r04-keysize7-plainsize4_c22.xml"))
        # yield ("bug", _load_xcsp3("2025/COP22to25/AircraftLanding-table-airland08_c22.xml"))
        # return

        # for f in pathlib.Path("test_cases").glob("*.pkl"):
        #     with open(f, "rb") as fp:
        #         case = pickle.load(fp)
        #     print(case)
        #     yield case
        # return

        if a <= 0 <= b:
            # Basic test cases
            # yield ("alldiff", cp.Model(cp.AllDifferent(cp.intvar(1, 3, shape=3, name="x"))))
            yield ("single_row", generate_table_from_data([[1, 1]]))
            yield ("two_rows", generate_table_from_data([[1, 1], [2, 2]]))
            yield ("singleton_dom", generate_table_from_data([[1, 1]], ub=1))
            yield ("singleton_dom_infeasible", generate_table_from_data([[1, 2]], ub=1))
            yield ("feasible_diagonal", generate_table_from_data([[1, 1], [2, 2]], ub=3))
            yield ("feasible_swap", generate_table_from_data([[1, 2], [2, 1]], ub=3))
            yield ("infeasible_alldiff", with_constraints(generate_table_from_data([[1, 1], [2, 2]], ub=3), with_alldiff=True))
            yield (
                "alldiff_min",
                with_constraints(generate_table_from_data([[1, 2], [2, 1]], ub=3), with_alldiff=True, with_min=True),
            )
            yield (
                "issue_9_mix_bool_and_intvar.",
                cp.Model(
                    cp.Table(
                        [cp.intvar(0, 8, name="x"), cp.boolvar(name="p")],
                        [[0, 0], [1, 0], [2, 0], [2, 1], [3, 1], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0]],
                    )
                ),
            )
            yield ("example", generate_table_from_example())
            yield ("example_min", with_constraints(generate_table_from_example(), with_min=True))
            yield ("example_alldiff", with_constraints(generate_table_from_example(), with_alldiff=True))
            yield ("COM", generate_table_from_data([[2, 2], [2, 4], [4, 4], [4, 6]], ub=6))
            # Edge cases
            yield ("single_column", cp.Model(cp.Table([cp.intvar(1, 5, name="x")], [[2], [4]])))
            yield (
                "complete_table",
                cp.Model(
                    cp.Table(
                        [cp.intvar(1, 2, name="x"), cp.intvar(1, 2, name="y")],
                        [[i, j] for i in range(1, 3) for j in range(1, 3)],
                    )
                ),
            )
            yield (
                "duplicate_rows",
                cp.Model(
                    cp.Table([cp.intvar(1, 3, name="x"), cp.intvar(1, 3, name="y")], [[1, 2], [2, 1], [1, 2], [2, 1], [1, 2]])
                ),
            )

            yield (
                "constant_column_b",
                cp.Model(
                    cp.Table(
                        [cp.intvar(1, 3, name="x"), cp.intvar(1, 3, name="y")],
                        [[1, 1], [1, 2], [1, 3]],
                    )
                ),
            )

            yield (
                "constant_column_c",
                cp.Model(
                    cp.Table(
                        [cp.intvar(1, 4, name="x"), cp.intvar(1, 4, name="y")],
                        [[1, 1], [2, 2], [3, 3], [3, 4]],
                    )
                ),
            )

            yield (
                "constant_column",
                cp.Model(
                    cp.Table(
                        [cp.intvar(1, 5, name="x"), cp.intvar(1, 3, name="y"), cp.intvar(1, 4, name="z")],
                        [[1, 2, 3], [1, 1, 2], [1, 3, 4], [1, 2, 1]],
                    )
                ),
            )

        if a <= 1 <= b:
            yield (
                "no_valid_tuples",
                cp.Model(cp.Table([cp.intvar(1, 3, name="x"), cp.intvar(1, 3, name="y")], [[4, 5], [5, 6], [6, 7]])),
            )
            yield ("fixed_feasible", cp.Model(cp.Table([cp.intvar(3, 3, name="a"), cp.intvar(1, 3, name="y")], [[3, 2]])))
            yield ("fixed_infeasible", cp.Model(cp.Table([cp.intvar(3, 3, name="a"), cp.intvar(1, 3, name="y")], [[4, 2]])))
            yield ("bool_vars", cp.Model(cp.Table([cp.boolvar(name="p"), cp.intvar(1, 3, name="y")], [[0, 2], [1, 3]])))
            x = cp.intvar(0, 3, name="x", shape=2)

            #   0 0 0 1 1 1
            # [[1 0 0 0 0 1]
            #  [0 1 0 0 1 0]
            #  [0 0 1 1 0 0]]
            yield ("soccer_table", cp.Model(*[cp.InDomain(x_i, [0, 1, 3]) for x_i in x], cp.Table(x, [[0, 3], [1, 1], [3, 0]])))
            # p <-> q
            yield ("hcpizza_table", cp.Model(cp.Table([cp.boolvar(name="p"), cp.intvar(1, 6)], [[0, 1], [1, 6]])))

            airland = (3, 5)
            # T_enc =
            # [[1 0 0 0 1 0 0 0 1 0 0 0]
            #  [0 1 0 0 0 1 0 0 0 1 0 0]
            #  [0 0 1 0 0 0 1 0 0 0 1 0]
            #  [0 0 0 1 0 0 0 1 0 0 0 1]]
            yield (
                "airland_table",
                cp.Model(
                    cp.Table(
                        cp.intvar(1, airland[1], shape=airland[0], name="x"), [[i] * airland[0] for i in range(1, airland[1])]
                    )
                ),
            )

            # yield ("random_gaps", generate_table(5, 10, 10, k=1, gaps=0.5))
            yield ("random_gaps", generate_table(5, 5, 3, k=1, gaps=0.5))
            yield ("negated_bool", cp.Model(cp.Table([~cp.boolvar(name="p"), cp.intvar(1, 3, name="y")], [[0, 2], [1, 3]])))
            yield ("single_tuple", cp.Model(cp.Table([cp.intvar(1, 5, name="x"), cp.intvar(1, 5, name="y")], [[1, 1]])))
            yield (
                "diagonal",
                cp.Model(cp.Table([cp.intvar(1, 3, name="x"), cp.intvar(1, 3, name="y")], [[1, 1], [2, 2], [3, 3]])),
            )
            yield (
                "anti_diagonal",
                cp.Model(cp.Table([cp.intvar(1, 3, name="x"), cp.intvar(1, 3, name="y")], [[1, 3], [2, 2], [3, 1]])),
            )
            yield ("t2x2x2_bug", with_constraints(generate_table(2, 2, 2)))
            yield (
                "t2x2x3_alldiff_min",
                with_constraints(
                    generate_table(2, 2, 3),
                    with_alldiff=True,
                    with_min=True,
                ),
            )

            # yield (f"tmp_t5x100x10", with_constraints(generate_table(5, 100, 10)))
            yield ("two_tables", generate_two_tables())

            yield ("many_rows_1", with_constraints(generate_table(2, 20, 5)))
            # yield ("many_rows_2", with_constraints(generate_table(2, 95, 10)))
            # yield ("many_rows_3", with_constraints(generate_table(3, 100, 5))) # -4.5%

            # yield (
            #     "sparse_table",
            #     cp.Model(cp.Table([cp.intvar(1, 100, name="x"), cp.intvar(1, 100, name="y")], [[1, 2], [50, 75], [99, 100]])),
            # )

        if a <= 2 <= b:
            yield (
                "Spot-bug",
                generate_table_from_data(
                    [
                        [0, 0],
                        [0, 1],
                        [0, 2],
                        [0, 3],
                        [1, 0],
                        [1, 1],
                        [1, 2],
                        [2, 0],
                        [2, 1],
                        [2, 2],
                        [2, 3],
                        [3, 0],
                        [3, 1],
                        [3, 2],
                        [3, 3],
                    ],
                    lb=0,
                    ub=3,
                ),
            )

        # Generated table tests
        # for gaps in (None, 0.5):
        for k in (
            1,
            5,
            10,
            # 25,
        ):
            # suffix = "_gaps" if gaps else "_nogaps"
            suffix = f"_k{k}"

            def generate_table_(*args, **kwargs):
                return generate_table(*args, **kwargs, k=k)

            yield (
                f"t4x4x4{suffix}",
                with_constraints(
                    generate_table_(4, 4, 4),
                    with_alldiff=False,
                    with_min=False,
                ),
            )

            yield (f"t2x2x3_k2{suffix}", with_constraints(generate_table_(2, 2, 3)))

            yield (
                f"t3x3x4{suffix}",
                with_constraints(
                    generate_table_(3, 3, 4),
                ),
            )

            yield (
                f"t4x3x4_k2{suffix}",
                with_constraints(generate_table_(4, 3, 4)),
            )

            if a <= 3 <= b:
                yield (f"t5x100x10{suffix}", with_constraints(generate_table_(5, 100, 10)))
                yield (f"t10x200x15{suffix}", with_constraints(generate_table_(10, 200, 15)))
                yield (f"t10x500x10{suffix}", with_constraints(generate_table_(10, 500, 10)))
                yield (f"t15x300x20{suffix}", with_constraints(generate_table_(15, 300, 20)))

            if a <= 4 <= b:
                yield (f"t20x500x25{suffix}", with_constraints(generate_table_(20, 500, 25)))
                yield (f"t25x1000x30{suffix}", with_constraints(generate_table_(25, 1000, 30)))

                # yield (
                #     f"big{suffix}",
                #     with_constraints(generate_table_(50, 5000, 10000)),
                # )

        if a <= 3 <= b:
            yield (
                "wide_table",
                cp.Model(
                    cp.Table(
                        cp.intvar(1, 3, shape=10, name="x"),
                        [[1, 2, 3, 1, 2, 3, 1, 2, 3, 1], [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]],
                    )
                ),
            )

        if a <= 5 <= b:
            # yield ("hcpizza", _load_xcsp3("2025/COP22to25/HCPizza-20-20-2-8-01_c23.xml"))
            # yield ("fillomino", _load_xcsp3("2025/CSP22to25/Fillomino-5-0_c24.xml"))
            yield ("soccer", _load_xcsp3("2025/CSP22to25/Soccer-20-12-20-1_c24.xml"))
            yield ("airland", _load_xcsp3("2025/COP22to25/AircraftLanding-table-airland01_c22.xml"))
            yield ("crossword", _load_xcsp3("2025/CSP22to25/Crossword-m18-ogd2008-vg-04-05_c22.xml"))  # good result but high CB

            airland = _load_xcsp3("2025/COP22to25/AircraftLanding-table-airland01_c22.xml")
            airland.constraints = [next(c for c in airland.constraints if c.name == "table")]
            airland.constraints += [c for c in airland.constraints if c.name != "table"]
            yield ("airland_first", airland)

            # yield ("opt_bug", "2025/COP22to25/Fortress1-05_c25.xml")
            # yield ("opt_bug", "2025/COP22to25/Fortress1-05_c25.xml")

            # TODO fix
            # [2025/COP22to25/TankAllocation2-0000_c25.xml - lazy_gurobi-none]
            # Status: ERROR | Time: 3.22s
            # Exception: BV[x[13][0] == -1]
            # Traceback:
            # Traceback (most recent call last):
            #   File "/cw/dtailocal/henk/Projects/cpmpy/cpmpy/tools/xcsp3/benchmark.py", line 141, in xcsp3_wrapper
            #     xcsp3_cpmpy(**kwargs, verbose=verbose)
            #   File "/cw/dtailocal/henk/Projects/cpmpy/cpmpy/tools/xcsp3/xcsp3_cpmpy.py", line 815, in xcsp3_cpmpy
            #     raise e
            #   File "/cw/dtailocal/henk/Projects/cpmpy/cpmpy/tools/xcsp3/xcsp3_cpmpy.py", line 745, in xcsp3_cpmpy
            #     s.solve(**solver_args, time_limit=time_limit)
            #   File "/cw/dtailocal/henk/Projects/cpmpy/cpmpy/solvers/lazy_gurobi.py", line 1139, in solve
            #     raise e
            #   File "/cw/dtailocal/henk/Projects/cpmpy/cpmpy/solvers/lazy_gurobi.py", line 1108, in solve
            #     hassol = super().solve(
            #              ^^^^^^^^^^^^^^
            #   File "/cw/dtailocal/henk/Projects/cpmpy/cpmpy/solvers/gurobi.py", line 814, in solve
            #     raise getattr(self.native_model, "_callback_exception", None) or Exception("Gurobi was interrupted (perhaps the solution callback called model.terminate())")
            #   File "/cw/dtailocal/henk/Projects/cpmpy/cpmpy/solvers/lazy_gurobi.py", line 794, in solution_callback
            #     x_enc_a = {x_enc_i: cbGetVal(x_enc_i, what.cbGetSolution) for x_enc_i in all_xs}
            #                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            #   File "/cw/dtailocal/henk/Projects/cpmpy/cpmpy/solvers/lazy_gurobi.py", line 780, in cbGetVal
            #     else cbGet(self._varmap[cpm_var])
            #                ~~~~~~~~~~~~^^^^^^^^^
            # KeyError: BV[x[13][0] == -1]
            # yield ("varmap_bug", _load_xcsp3("2025/COP22to25/TankAllocation2-0000_c25.xml"))

    for name, model in _generate():
        name = f"{name}."
        if matches(name):
            yield (name, model)


def generate_table_from_data(T, lb=1, ub=2):
    """Generate a table constraint with the given `rows` and with var domains of size `d`"""

    X = cp.intvar(lb, ub, shape=len(T[0]), name="x")
    return cp.Model(cp.Table(X, T))


def generate_table(n, m, d, k=1, gaps=None, allow_duplicate_vars=True, ensure_feasible=True):
    """Generate `k` table constraints with `n` variables with domains of size `d`, and with `m` rows

    Args:
        n: Number of variables
        m: Number of rows in each table
        d: Domain size (1 to d)
        k: Number of table constraints
        gaps: If not None, randomly remove values from domains (fraction to keep)
        allow_duplicate_vars: Allow the same variable to appear multiple times in a table
        ensure_feasible: If True, guarantee feasibility by including a random solution in each table
    """
    model = cp.Model()
    X = cp.intvar(1, d, shape=(n,), name=INT_VAR_NAMES[:n] if n < len(INT_VAR_NAMES) else "x")

    # Track actual domains for each variable
    domains = {x: list(range(1, d + 1)) for x in X}

    if gaps:
        for x in X:
            dom = sorted(random.sample(range(1, d + 1), max(2, round(d * gaps))))
            assert len(set(dom)) > 1, dom
            model += cp.InDomain(x, dom)
            domains[x] = dom

    # Generate a random feasible assignment if needed
    feasible_assignment = None
    if ensure_feasible:
        feasible_assignment = {x: random.choice(domains[x]) for x in X}

    for _ in range(k):
        if k > 1:
            k_ = n // 2
            X = list(X)
            Y = random.choices(X, k=k_) if allow_duplicate_vars else random.sample(X, k=k_)
        else:
            Y = X
        if len(Y):
            # Generate m-1 random rows
            rows_to_generate = m - 1 if ensure_feasible else m
            T = [[random.randint(1, d) for _ in enumerate(Y)] for _ in range(rows_to_generate)]

            # Add the feasible assignment at a random index
            if ensure_feasible:
                feasible_row = [feasible_assignment[y] for y in Y]
                random_idx = random.randint(0, len(T))
                T.insert(random_idx, feasible_row)

            model += cp.Table(Y, T)
    return model


def allsols(slv, model, solution_limit=100, max_search=None, time_limit=None):

    search = math.prod(dom_size(x) for x in get_variables_model(model))
    if max_search is not None and search > max_search:
        print("skip", search, max_search)
        return None

    sols = []
    xs = tuple(sorted(slv.user_vars, key=lambda x: x.name))

    def display():
        sol = tuple(argvals(xs))
        sols.append(sol)
        # print(f"checking ({len(sols)} <= {search})", sol)
        if time_limit is not None and time.time() - dt > time_limit:
            raise TimeoutError

        for c in model.constraints:
            assert c.value(), f"Constraint {c} failed for assignment\n\n{show_assignment(get_variables(c))}"

    dt = time.time()

    slv.solveAll(display=display, solution_limit=solution_limit)
    actual_sat = bool(sols)
    assert len(sols) < solution_limit, f"increase sol limit, found {sols}"
    return sols if len(sols) < solution_limit else None


def check_model(model, exp=None, checked=True, expected_sat=None, expected_obj=None, expected_sols=None):
    if True:
        print("== Model ==")
        print(model)
        print(", ".join(f"{x} in {x.lb}..{x.ub}" for x in sorted(get_variables_model(model), key=lambda x: x.name)))

    print("== ENV == ")
    pprint.pprint(exp)

    try:
        if True:
            slv_ = CPM_ortools(cpm_model=model) if exp["solver"] == "ortools" else exp["solver"](**exp["solver_kwargs"])
            print("ENCODING")
            for i, c in enumerate(model.constraints, start=1):
                print(f"C{i}", repr(c)[:100])
                print("  ", ", ".join(f"{x} in {x.lb}..{x.ub}" for x in get_variables(c)))
                for ci in slv_.transform([c]):
                    print("  ", ci)
                    # print("  ", slv._csemap)
            print("ENCODED")

        print("expected feasible = ", expected_sat, expected_obj, len(expected_sols) if expected_sols is not None else None)

        slv = (
            CPM_ortools(cpm_model=model)
            if exp["solver"] == "ortools"
            else exp["solver"](cpm_model=model, **exp["solver_kwargs"], time_limit=TIME_LIMIT)
        )

        print("solving for actual feasibility", slv)
        if expected_sols is not None:
            print("allsols")
            actual_sols = allsols(slv, model, time_limit=TIME_LIMIT, solution_limit=SOL_LIMIT, max_search=MAX_SEARCH)
            assert frozenset(actual_sols) == frozenset(expected_sols)
            actual_sat = len(actual_sols) > 0
        else:
            print("onesol")
            actual_sat = slv.solve(time_limit=TIME_LIMIT)
            print("STATUS", slv.status())
            print("OBJ", slv.objective_value())
            if actual_sat is None:
                raise TimeoutError
        print("actual feasible", actual_sat)

        if hasattr(slv, "stats"):
            print("stats = ", slv.stats())

        # if hasattr(slv, "stats"):
        #     slv.stats()

        if actual_sat:
            X = cp.transformations.get_variables.get_variables_model(model)
            # print("assignment", show_assignment(X))

            assert all(x.value() is not None for x in X), (
                f"Expected all variables to be assigned, but found: {show_assignment(X)}"
            )

            violations = [c for c in model.constraints if c.value() is False]
            assert not violations, (
                f"For assignment:\n\n{show_assignment(X)}\n\nThe following constraints fail:\n\n'"
                + "\n\n".join(str(v) for v in violations)
            )

        assert expected_obj is None or expected_obj == slv.objective_value()
        assert expected_sat is None or expected_sat == actual_sat, f"Expected equisat, but {expected_sat=} and {actual_sat=}"

        print("PASS.")
    except AssertionError as e:
        with open("/tmp/failed_model.pkl", "wb") as f:
            pickle.dump(model, f)

        raise e
        if exp["debug"]:
            raise e
        else:
            print("try debug", e)

            check_model(model, exp={**exp, "debug": True})


def show_sols(sols, T):
    return ", ".join(f"*{sol}" if list(sol) in T.tolist() else f"{sol}" for sol in sorted(sols))


def with_constraints(model, with_alldiff=False, with_min=True):
    X = cp.transformations.get_variables.get_variables_model(model)
    if with_alldiff:
        model += cp.AllDifferent(X)
    if with_min:
        model.minimize(sum(X) + 42)
    return model


def get_envs():

    debug_env = {
        "verbosity": VERBOSITY,
        "debug": 1,
        "max_iterations": 5000,
        "seed": 42,
        "checked": CHECKED

    }

    if False:
        yield {
            "alias": "dev",
            "solver": CPM_lazy_gurobi,
            "solver_kwargs": {
                "env": {
                    **debug_env,
                    "fractional": False,
                    "coverlift": Coverlift.INPUT,
                    "shrink": False,
                    "negatives": 0,
                    "checked": False,
                    # "heuristic": Heuristic.INPUT,
                    "heuristic": Heuristic.GREEDY,
                    # "heuristic": Heuristic.REDUCE,
                    "cutoff": 0,
                    "verbosity": 3,
                    "variant": 1,
                },
            },
            "solve_kwargs": {"Seed": 42},
        }

    for e in get_solvers():
        if "ort" in e["alias"]:
            continue
        elif "lazy" in e["alias"]:
            e["solver_kwargs"]["env"] |= debug_env
        elif "base" in e["alias"]:
            e["solver_kwargs"] |= {"named": True, "verbose": True}
            cp.transformations.int2bool.IntVarEnc.NAMED = True
        yield e


@pytest.fixture
def env():
    return next(get_envs())


def load_model(path):
    if pathlib.Path(path).exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        return cp.Model()


@pytest.mark.timeout(60)
class TestTables:
    def test_repro_explain(self, env):
        path = pathlib.Path("fail.pkl")
        if path.exists():
            with open(path, "rb") as f:
                cut = pickle.load(f)
                X_enc, A_enc, T_enc, parts, frm, env = cut
            slv = CPM_lazy_gurobi(
                env=env | {"verbosity": 3, "debug": True, "checked": False},
                # env={
                #     **env,
                #     **{"verbosity": 3, "debug": True, "checker": False, "coverlift": False, "shrink": False, "fractional": True},
                # },
            )

            COLS = None

            # print("parts", len(np.unique(parts)))
            # COLS = np.isin(parts, range(83,85))
            if COLS is not None:
                A_enc = A_enc[COLS]
                T_enc = T_enc[:, COLS]
                parts = parts[COLS]
                X_enc = X_enc[COLS]

            print(" ", np.array(X_enc))
            
            if env["negatives"]:
                n = len(A_enc) // 2
                tbl = TableData(X_enc[:n], T_enc[:,:n], parts[:n], None, slv)
            else:
                tbl = TableData(X_enc, T_enc, parts, None, slv)
            explanation = tbl.explain(A_enc, frm=frm)
            if explanation is None:
                print("Infeasible")
            else:
                expr = slv.explanation_to_expr(explanation, A_enc, X_enc, T_enc, frm)
                slv.check_explanation(expr, X_enc, A_enc, T_enc, parts, frm)

    @pytest.mark.skip()
    def test_coverlift(self, env):
        slv = CPM_lazy_gurobi(
            cpm_model=cp.Model(generate_table_from_example().constraints),
            env={**env, **{"shrink": False, "debug": True, "example2": True}},
        )
        tbl = slv.tables[0]
        X_enc, T_enc, parts = tbl.X_enc, tbl.T_enc, tbl.parts

        # example 2 / 11
        S = {1, 5, 8}

        C_enc = {s: 1 for s in S}
        k = len(S) - 1

        c = slv.gencoverlift(S, C_enc, k, T_enc, A_enc)

        print("c", c)
        # TODO assert

    def test_normalize_table(self, env):
        x = cp.intvar(1, 4, name="x")
        # y = cp.intvar(1, 4, name="y")
        z = cp.intvar(2, 4, name="z")
        c = cp.Table((x, x, z), [[2, 1, 2], [1, 2, 2], [2, 2, 3], [3, 3, 3]])
        assert len(set(c.args[0])) < len(c.args[0])
        c = normalize_table(c)
        print(c)
        assert len(set(c.args[0])) == len(c.args[0])
        assert c.args[1] == [[2, 3], [3, 3]]

    #
    # 0100 100 100
    # 0010 010 010
    # 0001 001 001
    # 1000 010 001
    # 1111 222 333
    #

    def test_playground(self, env):
        # x = np.arange(8).reshape(2, 4)
        T = np.array([[0, 1, 0, 1], [0, 1, 1, 0]], dtype=bool)
        parts = np.array(
            [
                0,
                0,
                1,
                1,
            ]
        )
        T = T & [1, 0, 1, 1]

        T = np.array(
            [
                [
                    *[0, 1, 0, 0],
                    *[1, 0, 0],
                    *[1, 0, 0],
                ],
                [
                    *[0, 0, 1, 0],
                    *[0, 1, 0],
                    *[0, 1, 0],
                ],
                [
                    *[0, 0, 0, 1],
                    *[0, 0, 1],
                    *[0, 0, 1],
                ],
                [
                    *[1, 0, 0, 0],
                    *[0, 1, 0],
                    *[0, 0, 1],
                ],
                [
                    *[0, 1, 0, 0],
                    *[1, 0, 0],
                    *[0, 1, 0],
                ],
            ]
        )
        parts = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        a = np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 0])

        parts_ = np.add.accumulate(np.unique_counts(parts).counts)
        parts_ -= parts_[0]

        print("", a, "a")
        choices = a & [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        # print("", choices, "choices")
        # print(T)
        T = T[[0, 4], :][:, choices]
        # print(T.astype(int))
        # print("", parts, "p")
        y = np.bitwise_or.reduceat(
            T,
            parts_,
            axis=1,
        ).sum(0)
        print("Remaining R", y)

    @pytest.mark.skip()
    def test_shrink(self, env):
        env = {"verbosity": 4}
        X, k = CPM_lazy_gurobi(env=env).shrink(
            np.array([True, True, True]),
            np.array(
                [
                    [True, True, False],
                    [False, False, True],
                    [True, True, False],
                    [False, False, False],
                    [True, True, False],
                ]
            ),
        )
        assert k == 1
        assert (X == np.array([False, True, True])).all()

        return
        # TODO X = floats
        X, k = CPM_lazy_gurobi(env=env).shrink(
            np.array([0.5, 0.5, 0.0]),
            np.array(
                [
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                ]
            ).astype(bool),
        )
        assert k == 1
        assert (X == np.array([False, True, True])).all()

    @pytest.mark.skip()
    def test_explain(self, env):
        random.seed(SEED)
        for e, A_enc, env_ in [
            # (
            #     generate_table_from_example(),
            #     np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 0]),
            #     {"coverlift": True, "example2": True},
            # ),
            (
                generate_table_from_example(),
                np.array([0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0]),
                {"fractional": True, "example_frac": True},
            ),
            # (
            #     generate_table_from_example(),
            #     np.array([0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0]),
            #     {"fractional": True, "shrink": True, "heuristic": Heuristic.GREEDY},
            # ), # TODO fractional counter example where shrink does not work for frac sol
            # (generate_table_from_data([(2, 1), (2, 1), (2, 1), (1, 2)], 2), np.array([0, 1, 0, 1])),
            # (generate_table(3, 3, 3), None),
            # (generate_table(10, 2000, 10), None),
            # (generate_table(100, 500, 25), None),
        ]:
            # found feasible
            # shrink=False, heuristic=input, fractional=False, coverlift=True
            # time_cb = 28.54794931411743
            # cuts (MIPSOL) = 790
            # cuts (MIPNODE, explained) = 0
            # cuts (MIPNODE, unexplainable) = 0
            # .

            if A_enc is None:
                A_enc = []
                for x in e.constraints[0].args[0]:
                    A = dom_size(x) * [False]
                    A[random.randint(x.lb, x.ub) - 1] = True
                    A_enc += A
            A_enc = np.array(A_enc)

            slv = CPM_lazy_gurobi(
                # cpm_model=cp.Model(generate_table_from_example().constraints),
                # cpm_model=cp.Model(cp.Table((cp.intvar(1, 1), cp.intvar(1, 2)), [(1, 2)])),
                cpm_model=e,
                env={
                    **env,
                    **{
                        # "fractional": True,
                        "coverlift": False,
                        "max_iterations": None,
                        "heuristic": Heuristic.GREEDY,
                        # "heuristic": Heuristic.GREEDY,
                        "shrink": True,
                        "debug": True,
                        "verbosity": 3,
                        "negatives": 0,
                        # "checker": True,
                    },
                    **env_,
                },
            )
            if False:
                slv.solve()
                slv.stats()
                print(slv.env["cuts"])
                slv.print_cuts()
                tbl = slv.tables[0]
                X_enc, T_enc, parts = tbl.X_enc, tbl.T_enc, tbl.parts

                def list_to_A_enc(A_enc):
                    return dict(zip(X_enc, A_enc))

                # A_enc = list_to_A_enc(A_enc)
            else:
                tbl = slv.tables[0]
                X_enc, T_enc, parts = tbl.X_enc, tbl.T_enc, tbl.parts
                # explanations = list(slv._explain_assignment(A_enc, frm="MIPSOL"))
                frm = "MIPSOL"
                frm = "MIPNODE-OPT"

                if False:
                    # sum([1, -1] * [⟦x == 2⟧, ⟦y == 1⟧]) <= 0
                    slv.env["cuts"].append({"from": "MIPSOL"})
                    C_enc = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
                    k = 1
                    C_enc = np.array([0, 1, 0, 0, -1, 0, 0, 0, 0, 0])
                    k = 0
                    explanation = (C_enc != 0, C_enc, k)
                else:
                    explanation = tbl.explain(A_enc, frm=frm)

                e = slv.explanation_to_expr(explanation, A_enc, X_enc, T_enc, frm)
                print(e)

        # print("ERR", e.value)

        # slv.check_explanation(explanation, X_enc, A_enc, T_enc)
        # assert (  # Example 7; no longer in use since explain_frac2
        #     slv.explain([0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], T_enc) == {1, 5, 8}
        # )

        # assert (  # Example 9
        #     slv.explain([0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0], T_enc, parts) == {1, 5}
        # )

    @pytest.mark.skip()
    def test_repro_model(self, env):
        m = load_model("/tmp/failed_model.pkl")
        print("Repro model:", m)
        check_model(m, exp=env)

    def test_enc(self, env):
        x = cp.intvar(1, 5, name="x")
        y = cp.intvar(1, 5, name="y")
        m = cp.Model(cp.Table([x, y], [[1, 2]]), cp.InDomain(x, [2, 4]))
        slv = CPM_lazy_gurobi(cpm_model=m)
        print("Repro model:", slv.transform(m.constraints))

    def test_sols(self, env):
        from cpmpy.solvers.utils import solutions

        x = cp.boolvar(3, name="x")
        print(x)
        m = cp.Model(cp.sum([2, 3, 5] * x) <= 6)
        print(m)
        X, sols = solutions(m, verbosity=2)

        # m = generate_table_from_example()
        # model = cp.Model(cp.Table(X, T), cp.AllDifferent(X))
        # check_model(model, env=env)


def idfn(a):
    if isinstance(a, dict):
        return a["alias"]
    else:
        name, repeat = a[0], a[1]
        return f"{name}_r{repeat}" if repeat > 1 else name


ALLSOLS = True
SOLVE_EXPECTED = True  # Set to False to skip solving for expected values
TIME_LIMIT = 30
REPEAT = 3
SOL_LIMIT = 10e5
CHECKED = False
MAX_SEARCH=10e4
VERBOSITY=1
HARDNESS=2


def _generate_cases_with_expected():
    """Generate test cases and precompute expected feasibility/objective once per model."""
    for name, model in generate_models_w_tables(hardness=(0, HARDNESS)):
        # print("Get expected", name)
        # Solve once to get expected values (if enabled)
        if SOLVE_EXPECTED:
            model_ = model.deepcopy()
            expected_sat = model_.solve()
            expected_obj = model_.objective_value() if model.has_objective() else None
            expected_sols = (
                allsols(CPM_ortools(cpm_model=model), model, max_search=MAX_SEARCH, solution_limit=SOL_LIMIT)
                if ALLSOLS and not model.has_objective()
                else None
            )
        else:
            expected_sat = None
            expected_obj = None
            expected_sols = None

        # Yield repeated cases with precomputed expected values
        for j in range(1, REPEAT + 1):
            yield (name, j, model, expected_sat, expected_obj, expected_sols)


@pytest.mark.timeout(60)
class TestModels:
    _cached_cases = None

    @classmethod
    def setup_class(cls):
        """Cache test cases with expected values to avoid re-solving for each env."""
        if cls._cached_cases is None:
            # print("Generating expected vals")
            cls._cached_cases = list(_generate_cases_with_expected())

    @classmethod
    def _get_cached_cases(cls):
        if cls._cached_cases is None:
            cls.setup_class()
        return cls._cached_cases

    @pytest.mark.parametrize(
        ("env", "case"),
        itertools.product(
            get_envs(),
            list(_generate_cases_with_expected()),
        ),
        ids=idfn,
    )
    def test_models(self, case, env):
        name, rep, model, expected_sat, expected_obj, expected_sols = case
        env["solve_kwargs"]["Seed"] = rep
        print(f"== INSTANCE {name} ==")
        check_model(
            model,
            exp=env,
            checked=SOLVE_EXPECTED,
            expected_sat=expected_sat,
            expected_obj=expected_obj,
            expected_sols=expected_sols,
        )


FILTER_PRESETS = {
    "dev": [
        (
            "alias",
            ("negatives",),
        )
    ],
    "all": [],
    "none": [("alias", ("none",))],
    "bool": [
        (
            "alias",
            (
                "bool",
                "none",
                "negatives",
            ),
        )
    ],
    "mdd-reduce": [
        (
            "alias",
            (
                "mdd-reduce-input",
                "mdd-noreduce-input",
            ),
        )
    ],
    "mdd": [
        (
            "alias",
            (
                "bool",
                "mdd-reduce",
            ),
        )
    ],
    "neg": [
        (
            "alias",
            (
                "none",
                "neg",
            ),
        )
    ],
    "shrink": [
        (
            "alias",
            (
                "none",
                "shrink",
            ),
        )
    ],
    "vary": [
        (
            "alias",
            (
                "none",
                "variant",
            ),
        )
    ],
    "coverlift": [
        (
            "alias",
            (
                "none",
                "coverlift_input",
            ),
        )
    ],
    "com": [
        (
            "alias",
            (
                "none",
                "coverlift",
            ),
        )
    ],
}


def benchmark_table_constraints(
    envs=None,
    glob=None,
    hardness=None,
    filter_preset=None,
    verbosity=None,
    max_iterations=None,
    track_memory=False,
    checked=None,
    time_limit=None,
    choices=None,
    xcsp3_path=None,
):
    """Benchmark all table constraints from generate_edge_case_tables() and print stats dataframe

    Args:
        envs: List of environment configurations to test (uses default if None)
        glob: Filter test cases by name substring
        hardness: Tuple (min, max) for test case hardness levels (default: (0, 3))
        filter_preset: Name of filter preset from FILTER_PRESETS (default: "dev")
        verbosity: Verbosity level for solver (default: 2)
    """
    if hardness is None:
        hardness = (0, 3)
    if filter_preset is None:
        filter_preset = "dev"

    filters = FILTER_PRESETS.get(filter_preset, FILTER_PRESETS["dev"])

    envs = get_solvers(
        features=cp.tools.xcsp3.experiments.FEATURES + [("variant", (0, 1))],
        # features=[
        #     ("variant", (0, 1)),
        #     (
        #         "fractional",
        #         (
        #             False,
        #             # True,
        #         ),
        #     ),
        #     ("coverlift", (Coverlift.INPUT, Coverlift.COM_MAX)),
        #     # ("coverlift", (Coverlift.INPUT,)),
        #     ("shrink", (False, True)),
        #     ("negatives", (0, 2)),
        #     # ("negatives", (0,)),
        # ],
        # add_all=True,
        add_none=True,
        filters=filters,
    )
    print(envs)

    assert envs, f"No environments matched filter preset '{filter_preset}'"

    results = []

    if verbosity is None:
        verbosity = 2

    if hardness[1] <= 1:
        if checked is None:
            checked = True
        if time_limit is None:
            time_limit = 10
        debug = True
        track_memory = True
    else:
        if checked is None:
            checked = False
        debug = False
        # max_iterations stays as provided (or None)
        if time_limit is None:
            time_limit = 60 if hardness[1] >= 3 else 10

    # Generate all test cases once (to ensure same cases for all envs)
    if xcsp3_path:
        # Use only the specified XCSP3 instance
        name = pathlib.Path(xcsp3_path).stem
        test_cases = [(name, _load_xcsp3(xcsp3_path))]
    else:
        test_cases = list(generate_models_w_tables(hardness=hardness, glob=glob))

    # Save each test case as a pickle file
    test_cases_dir = pathlib.Path("test_cases")
    test_cases_dir.mkdir(exist_ok=True)
    for name, model in test_cases:
        pickle_path = test_cases_dir / f"{name}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump((name, model), f)
        print(f"Saved test case: {pickle_path}")

    for env in envs:
        env_alias = env.get("alias", "unknown")
        print(f"\n{'=' * 80}")
        print(f"Testing with environment: {env_alias} (TO = {time_limit})")
        print(f"{'=' * 80}\n")
        # print("ENV", env)
        if "lazy" in env_alias:
            env["solver_kwargs"]["env"]["verbosity"] = verbosity
            env["solver_kwargs"]["env"]["checked"] = checked
            env["solver_kwargs"]["env"]["max_iterations"] = max_iterations
            env["solver_kwargs"]["env"]["debug"] = debug
            env["solver_kwargs"]["env"]["choices"] = [int(i) for i in choices] if choices is not None else None

        for name, model in test_cases:
            # TODO ?
            for x in get_variables_model(model):
                x._occurs = False

            print(f"Running {name} with {env_alias}...")
            print(env)
            if verbosity >= 2:
                print("== Model == ")
                print(model)

            try:
                # Create solver with the environment
                if track_memory:
                    tracemalloc.start()
                dt = time.time()
                slv = env["solver"](cpm_model=model, **env["solver_kwargs"], time_limit=time_limit)
                dt = time.time() - dt
                print("POSTED IN", dt)

                # Solve the model
                solve_start = time.time()
                if time_limit - dt < 0:
                    raise TimeoutError
                has_sol = slv.solve(time_limit=time_limit - dt)
                sdt = time.time() - solve_start

                # Verify solution satisfies all constraints
                if has_sol:
                    for con in model.constraints:
                        assert con.value(), f"Constraint not satisfied: {con} by {show_assignment(get_variables(con))}"

                # Get peak memory usage
                peak_mem_mb = None
                grb_mem_mb = None
                if track_memory:
                    _, peak_mem = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peak_mem_mb = peak_mem / (1024 * 1024)

                    # Get Gurobi's peak memory if available
                    if hasattr(slv, "grb_model"):
                        grb_mem_mb = slv.grb_model.getAttr("MaxMemUsed")

                obj = slv.objective_value() if model.has_objective() else None
                if has_sol is None:
                    print(f"  TIMEOUT after {dt:.2f}s")
                    raise TimeoutError
                else:
                    obj_str = f" | obj={obj}" if obj is not None else ""
                    if track_memory:
                        print(
                            f"SOLVED IN {sdt:.2f}s{obj_str} | Mem: {peak_mem_mb:.1f}MB (Python) {grb_mem_mb:.1f}MB (Gurobi)"
                            if grb_mem_mb
                            else f"SOLVED IN {sdt:.2f}s{obj_str} | Mem: {peak_mem_mb:.1f}MB"
                        )
                    else:
                        print(f"SOLVED IN {sdt:.2f}s{obj_str}")

                # Get stats
                stats = slv.stats()
                if track_memory:
                    stats["mem_python_mb"] = peak_mem_mb
                    if grb_mem_mb is not None:
                        stats["mem_gurobi_mb"] = grb_mem_mb

                # Add test case info and environment to stats
                timeout = has_sol is None
                result = {
                    "env": env_alias,
                    "name": name,
                    "status": slv.status().exitstatus,
                    "obj": obj,
                    "time_solve": sdt,
                    **stats,
                }
                results.append(result)

            except TimeoutError:
                print("timeout")
                pass
            except Exception as e:
                raise e
                print(f"  ERROR: {e}")
                traceback.print_exc()
                results.append({"env": env_alias, "name": name, "satisfiable": None, "error": str(e)})
            finally:
                if tracemalloc.is_tracing():
                    tracemalloc.stop()

    # Create and print dataframe
    pd.set_option("display.float_format", "{:0.2f}".format)
    df = pd.DataFrame(results)

    cols = ["constraints", "n_cuts"]

    if verbosity == 0:
        cols += ["time_solve"]

    for col in cols:
        if col not in df.columns:
            df[col] = 0
    df["constraints"] = df["constraints"].fillna(0).astype(int)
    df["n_cuts"] = df["n_cuts"].fillna(0).astype(int)

    df["cons+cuts_num"] = df["constraints"] + df["n_cuts"]
    df["cons+cuts"] = df["constraints"].astype(str) + " + " + df["n_cuts"].astype(str) + " = " + df["cons+cuts_num"].astype(str)

    # df["cb_rel"] = df["time_cb"].fillna(0.) / df["time_solve"]
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_string())
    print("\n")

    assert not df.empty, "No benchmark results collected - check if test cases ran successfully"

    # Print comparison summary if multiple environments
    if len(envs) > 1:
        print("=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        # Pivot table to compare key metrics across environments
        # Preserve original order of test cases and environments
        original_order = [name for name, _ in test_cases]
        df["name"] = pd.Categorical(df["name"], categories=original_order, ordered=True)
        env_order = [e.get("alias", "unknown") for e in envs]
        print(env_order)
        df["env"] = pd.Categorical(df["env"], categories=env_order, ordered=True)

        values = [
            # "constraints",
            # "n_cuts",
            "cons+cuts",
        ]

        if checked:
            values += ["avg_strength"]
            values += ["avg_power"]

        if not checked and verbosity == 0:
            if hardness[0] >= 5:
                values += ["time_solve"]
            values += ["mem_python_mb"]
            # values += ["mem_gurobi_mb"]

        comparison = df.pivot_table(
            index="name",
            columns="env",
            values=[v for v in values if v in df.columns],
            aggfunc="first",
            sort=False,  # Don't sort, use categorical order
        )

        # Add relative difference columns for each metric
        env_names = [e.get("alias", "unknown") for e in envs]
        base_env = env_names[0]

        # Create a separate pivot for numeric diff calculation (cons+cuts uses cons+cuts_num)
        diff_values = [("cons+cuts_num" if v == "cons+cuts" else v) for v in values if v in df.columns or v == "cons+cuts"]
        diff_comparison = df.pivot_table(
            index="name",
            columns="env",
            values=[v for v in diff_values if v in df.columns],
            aggfunc="first",
            sort=False,
        )

        for metric in [v for v in values if v in df.columns]:
            diff_metric = "cons+cuts_num" if metric == "cons+cuts" else metric
            if (diff_metric, base_env) in diff_comparison.columns:
                base_col = diff_comparison[(diff_metric, base_env)]
                # Skip diff calculation for non-numeric columns
                if not pd.api.types.is_numeric_dtype(base_col):
                    continue
                for other_env in env_names[1:]:
                    if (diff_metric, other_env) in diff_comparison.columns:
                        other_col = diff_comparison[(diff_metric, other_env)]
                        # Calculate relative difference: (other - base) / base * 100
                        rel_diff = ((other_col - base_col) / base_col * 100).round(1)
                        comparison[(metric, f"Δ%({other_env})")] = rel_diff

        # Sort columns to group metric, envs, and diffs together
        comparison = comparison.sort_index(axis=1, level=0)

        print("\nTime (time_cb) and Cuts by Environment:")
        print(comparison.to_string())
        print("\n")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark table constraints")
    parser.add_argument(
        "-x",
        "--hardness",
        type=int,
        nargs=2,
        default=[0, 5],
        metavar=("MIN", "MAX"),
        help="Hardness range for test cases (default: 0 3)",
    )
    parser.add_argument(
        "--filter",
        "-f",
        dest="filter_preset",
        choices=list(FILTER_PRESETS.keys()),
        default="dev",
        help=f"Filter preset (default: dev). Available: {', '.join(FILTER_PRESETS.keys())}",
    )
    parser.add_argument(
        "--glob",
        "-g",
        type=str,
        default=None,
        help="Filter test cases by name substring",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="table_benchmark_results.csv",
        help="Output CSV file (default: table_benchmark_results.csv)",
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=int,
        default=0,
        help="Verbosity level for solver (default: 2)",
    )
    parser.add_argument(
        "--max-iterations",
        "-m",
        type=int,
        default=None,
        help="Maximum iterations for lazy solver (default: auto based on hardness)",
    )
    parser.add_argument(
        "--track-memory",
        action="store_true",
        default=False,
        help="Track memory usage during solve (adds overhead)",
    )
    parser.add_argument(
        "--checked",
        "-c",
        action="store_true",
        default=False,
        help="Enable solution checking/verification",
    )
    parser.add_argument(
        "--choices",
        nargs="*",
        default=None,
        help="Fix choices",
    )
    parser.add_argument(
        "--time-limit",
        "-t",
        type=int,
        default=None,
        help="Time limit in seconds for each solve (default: auto based on hardness)",
    )
    parser.add_argument(
        "--xcsp3",
        type=str,
        default=None,
        help="Find and run XCSP3 instance matching this pattern (searches in 2025/ directory)",
    )

    args = parser.parse_args()

    # Handle --xcsp3 pattern matching
    xcsp3_glob = None
    if args.xcsp3:
        import glob as glob_module
        pattern = f"2025/**/*{args.xcsp3}*.xml*"
        matches = sorted(glob_module.glob(pattern, recursive=True))
        if not matches:
            print(f"No XCSP3 instances found matching pattern: {pattern}")
            sys.exit(1)
        elif len(matches) == 1:
            xcsp3_glob = matches[0]
            print(f"Found XCSP3 instance: {xcsp3_glob}")
        else:
            print(f"Multiple XCSP3 instances found matching '{args.xcsp3}':")
            for i, m in enumerate(matches[:20]):
                print(f"  {i}: {m}")
            if len(matches) > 20:
                print(f"  ... and {len(matches) - 20} more")
            sys.exit(1)

    print(f"Running table constraints benchmark...")
    print(f"  Hardness: {tuple(args.hardness)}")
    print(f"  Filter preset: {args.filter_preset}")
    print(f"  Verbosity: {args.verbosity}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Track memory: {args.track_memory}")
    print(f"  Checked: {args.checked}")
    print(f"  Time limit: {args.time_limit}")
    if args.glob:
        print(f"  Glob: {args.glob}")
    if xcsp3_glob:
        print(f"  XCSP3: {xcsp3_glob}")

    df = benchmark_table_constraints(
        hardness=tuple(args.hardness),
        filter_preset=args.filter_preset,
        glob=args.glob,
        verbosity=args.verbosity,
        max_iterations=args.max_iterations,
        track_memory=args.track_memory,
        checked=args.checked if args.checked else None,
        time_limit=args.time_limit,
        choices=args.choices,
        xcsp3_path=xcsp3_glob,
    )

    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
