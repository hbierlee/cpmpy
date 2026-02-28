import cpmpy as cp
import numpy as np
import pandas as pd
import pathlib
import sys
import time

from cpmpy.tools.xcsp3 import read_xcsp3
from cpmpy.transformations.get_variables import get_variables
from cpmpy.tools.xcsp3.benchmark import xcsp3_benchmark, get_table_metadata_
from cpmpy.tools.xcsp3.experiments import experiment, ablate
from cpmpy.tools.xcsp3.xcsp3_cpmpy import ExitStatus, TIME_BUFFER
from cpmpy.solvers.gurobi import CPM_gurobi, Encoding
from cpmpy.expressions.variables import _BoolVarImpl, _IntVarImpl
from cpmpy.solvers.lazy_gurobi import CPM_lazy_gurobi

TIMEOUT = 5


def raise_error():
    raise Exception("Raised error")


def timeout(time_limit):
    time.sleep(5 * time_limit)


CPM_base_solver = CPM_lazy_gurobi if False else CPM_gurobi


class CPM_gurobi_transform_error(CPM_base_solver):
    def transform(self, *args, **kwargs):
        raise_error()


class CPM_gurobi_transform_timeout(CPM_base_solver):
    def transform(self, _):
        timeout(self.time_limit)
        return []


class CPM_gurobi_solve_error(CPM_base_solver):
    def transform(self, *args, **kwargs):
        return []

    def solve(self, *args, **kwargs):
        raise_error()


class CPM_gurobi_solve_timeout(CPM_base_solver):
    def transform(self, *args, **kwargs):
        return []

    def solve(self, *args, **kwargs):
        timeout(self.time_limit)


# TODO make work for lazy_gurobi as base
class CPM_gurobi_callback_timeout(CPM_gurobi):
    def transform(self, *args, **kwargs):
        return []

    def solve(self, *args, **kwargs):
        super().solve(*args, **kwargs, solution_callback=lambda *args: timeout(self.time_limit))


class CPM_gurobi_callback_error(CPM_gurobi):
    def transform(self, *args, **kwargs):
        return []

    def solve(self, *args, **kwargs):
        def solution_callback(*args):
            raise_error()

        super().solve(*args, **kwargs, solution_callback=lambda *args: raise_error())


class CPM_gurobi_solve_memoryout(CPM_base_solver):
    def transform(self, *args, **kwargs):
        return []

    def solve(self, *args, **kwargs):
        time.sleep(TIMEOUT / 2)
        raise MemoryError


class CPM_gurobi_solve_grb_memoryout(CPM_base_solver):
    def transform(self, *args, **kwargs):
        return []

    def solve(self, *args, **kwargs):
        time.sleep(TIMEOUT / 2)
        import gurobipy

        raise gurobipy._exception.GurobiError(10001, "Out of memory")


class CPM_gurobi_solve_incorrect(CPM_base_solver):
    def transform(self, cpm_expr):
        self.constraints = cpm_expr
        self.user_vars |= set(get_variables(self.constraints))
        return []

    def solve(self, *args, **kwargs):
        for x in self.user_vars:
            x._value = x.lb
        self.cpm_status.exitstatus = cp.solvers.solver_interface.ExitStatus.OPTIMAL
        self.objective_value_ = 0
        return True


import pytest


def idfn(a):
    if isinstance(a, dict):
        return a["solver"]
    elif isinstance(a, tuple):
        return ",".join(f"{a.value}" for a in a)


@pytest.mark.order("last")
class TestBenchmark:
    @pytest.mark.parametrize(
        "idx, experiment, expected_status",
        [
            (
                idx,
                {
                    **experiment(
                        [
                            {
                                # "alias": CPM_base_solver.__bases__,
                                # "glob_instance": "AlteredStates-02_c25.xml",
                                "glob_instance": "Fortress1-08_c25.xml",
                                "verbose": True,
                                "track": "COP25",
                                "time_limit": TIMEOUT,
                                "check_time_limit": 3,
                                "checker_path": "jbang  --main org.xcsp.parser.callbacks.SolutionChecker org.xcsp:xcsp3-tools:2.5",
                                "solver_kwargs": {"encoding": Encoding.GLEB},
                            }
                        ]
                    )[0],
                    **exp,
                },
                expected_status,
            )
            for idx, (exp, expected_status) in enumerate(
                [
                    ({"solver": CPM_base_solver, "time_limit": 10}, (ExitStatus.sat, ExitStatus.optimal)),
                    ({"solver": CPM_gurobi_transform_error}, ExitStatus.error),
                    ({"solver": CPM_gurobi_transform_timeout}, ExitStatus.unknown),
                    ({"solver": CPM_gurobi_solve_timeout}, ExitStatus.unknown),
                    ({"solver": CPM_gurobi_callback_timeout}, ExitStatus.unknown),
                    ({"solver": CPM_gurobi_callback_error}, ExitStatus.error),
                    ({"solver": CPM_gurobi_solve_error}, ExitStatus.error),
                    ({"solver": CPM_gurobi_solve_memoryout}, ExitStatus.memory),
                    ({"solver": CPM_gurobi_solve_grb_memoryout}, ExitStatus.memory),
                    (
                        {
                            "solver": CPM_base_solver,
                            "glob_instance": "SchedulingOS-gp-05-05_c25.xml",
                            "time_limit": 10,
                        },
                        (ExitStatus.unknown, ExitStatus.sat),
                    ),
                    # TODO test the right error is raised
                    ({"solver": CPM_gurobi_solve_incorrect, "time_limit": 10}, ExitStatus.error),
                    (  # small dup. table
                        {
                            "solver": CPM_lazy_gurobi,
                            "time_limit": 20,
                            "glob_instance": "Fortress1-03_c25.xml",
                        },
                        (ExitStatus.sat, ExitStatus.optimal),
                    ),
                ],
                start=1,
            )
        ],
        ids=idfn,
    )
    def test_benchmark(self, idx, experiment, expected_status, tmp_path):
        experiment["output_dir"] = "test_benchmark" / tmp_path
        print("e", experiment["output_dir"])

        if isinstance(expected_status, ExitStatus):
            expected_status = (expected_status,)

        # x = cp.boolvar(shape=3)
        # experiment["solver"](cp.Model(2*x + 3*x + 5*x <= 6))

        dt = time.time()
        print("E", experiment)
        out = xcsp3_benchmark(**experiment)
        dt = time.time() - dt

        df = pd.read_csv(out).loc[0]
        print(df)
        status = ExitStatus(df["status"])
        assert status in expected_status, f"Unexpected status for {experiment['solver']}\n\n{df['exception']}"
        assert (
            dt
            < experiment["time_limit"] + experiment["check_time_limit"] + (2 if experiment["checker_path"] else 0) + TIME_BUFFER
        )

        feasible = status in (ExitStatus.unsat, ExitStatus.optimal, ExitStatus.sat)
        print(df)

        assert not np.isnan(df["time_total"])
        if feasible:
            assert not np.isnan(df["time_solve"])
            assert not np.isnan(df["time_post"])

        assert status is not ExitStatus.error or df["exception"]

    def test_ablate(self):
        assert ablate([("a", (False, True)), ("b", (0, 5, 2))], add_none=True, add_all=True) == [
            ("none", {"a": False, "b": 0}),
            ("a", {"a": True, "b": 0}),
            ("b_5", {"a": False, "b": 5}),
            ("b_2", {"a": False, "b": 2}),
            ("all", {"a": True, "b": 2}),
        ]

    def test_dev(self):
        # Parse and create CPMpy model
        x, y, z = cp.intvar(0, 3, shape=3, name=tuple("xyz"))
        m = cp.Model([x + y <= 3, (x == 2) | (y == 2), cp.all([x == 2])])

        sys.argv = ["-nocompile"]  # Stop pyxcsp3 from complaining on exit
        for m in (
            # "2025/COP25-dev/dev-1.xml",
            # "2025/CSP25/Accordion-11-01_c25.xml.lzma",
            # "2025/COP25/Fortress1-03_c25.xml.lzma",
            # test_lazy_gurobi.with_constraints(
            #     test_lazy_gurobi.generate_table_from_example(), with_alldiff=True
            # ),
            # "2025/COP25/IHTC-i01_c25.xml",
            # "2025/COP25/RoadefPlaning2-2021-04_c25.xml",
            # "2025/COP25/FAPP-aux-ex1_c25.xml",
            # "2025/COP25/RoadefPlaning2-2024-11_c25.xml",
            # "2025/CSP22to25/Soccer-20-12-20-1_c24.xml.lzma",
            # "2025/CSP22to25/CoveringArray-3-05-2-10_c23.xml.lzma",
            # m,
            "2025/COP22to25/AircraftLanding-table-airland01_c22.xml",
        ):
            if isinstance(m, str):
                print("MM", m)
                m = read_xcsp3(pathlib.Path(m))
            # slv = CPM_lazy_gurobi()
            # import scalene

            import psutil

            import humanfriendly

            process = psutil.Process()
            import difflib

            # scalene.scalene_profiler.start()
            PRINT = True
            slvs = (
                CPM_gurobi(),
                CPM_lazy_gurobi(env={"cutoff": 0}),
            )
            reps = {}
            # counters = (_BoolVarImpl.counter, _IntVarImpl.counter)
            for slv in slvs:
                with open(f"/tmp/{slv.name}.txt", "w") as f:
                    _BoolVarImpl.counter, _IntVarImpl.counter = (0, 0)

                    reps[slv.name] = ""

                    t = time.time()
                    print(slv.name, file=f)
                    rows = []
                    for i, c in enumerate(m.constraints[:] if True else (m.constraints,)):
                        # TODO weirdly the BV counters are not properly reset if all constraints are transformed at once
                        row = {}
                        mem = process.memory_info().rss
                        rep = repr(c)
                        print(f"C{i} ({c.__class__.__name__})", rep[:100], file=f)
                        # print("  ", slv._csemap, file=f)
                        reps[slv.name] += rep
                        row["c"] = rep
                        if getattr(c, "name", None) == "table":
                            md = get_table_metadata_(c)
                            print(md, file=f)
                            del md["cols"]
                            row |= md
                        for c_ in slv.transform([c] if not isinstance(c, list) else c):
                            # row["t"] = repr(c)

                            rep = repr(c_)
                            reps[slv.name] += rep
                            row["t"] = rep
                            if PRINT:
                                print("  ", c_, file=f)
                        if time.time() - t > 60:
                            return
                        diff = process.memory_info().rss - mem
                        row["mem"] = diff
                        rows.append(row)
                        print(f"  M = {humanfriendly.format_size(diff)}", f)

                # pd.set_option("display.max_colwidth", None)
                # pd.set_option("display.max_columns", None)
                # pd.set_option("display.max_rows", None)
                # pd.set_option("display.expand_frame_repr", False)

                df = pd.DataFrame(data=rows)
                df.insert(
                    df.columns.get_loc("mem") + 1,
                    value=df["mem"].map(humanfriendly.format_size),
                    column="mem_hf",
                )
                df = df.sort_values(by="mem")
                # df["mem"] = humanfriendly.format_size(df["mem"])
                print(df)
                print(humanfriendly.format_size(df["mem"].sum()))

                # np.savetxt(f"/tmp/{slv.name}.txt", df.values)
            print("done")

            # diff = difflib.ndiff(reps["gurobi"], reps["lazy_gurobi"])
            # import itertools
            # print(''.join(itertools.islice(diff, 0, 10)), end="")

        # m.solve(solver="gurobi")
        # parser = _parse_xcsp3("../2025/COP25-dev/dev-1.xml")
        # model = _load_xcsp3(parser)
