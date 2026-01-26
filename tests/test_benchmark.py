import pandas as pd
import numpy as np
import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables
from cpmpy.tools.xcsp3.benchmark import xcsp3_benchmark
from cpmpy.tools.xcsp3.experiments import experiment, ablate
from cpmpy.tools.xcsp3.xcsp3_cpmpy import ExitStatus, TIME_BUFFER
from cpmpy.solvers.gurobi import CPM_gurobi
from cpmpy.solvers.lazy_gurobi import CPM_lazy_gurobi
import time

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


class TestBenchmark:
    @pytest.mark.parametrize(
        "idx, experiment, expected_status",
        [
            (
                idx,
                {
                    **experiment(
                        [
                            [
                                {
                                    "alias": "base_gurobi",
                                    "glob_instance": "Fortress1-08_c25.xml",
                                    "verbose": True,
                                    "time_limit": TIMEOUT,
                                    "check_time_limit": 3,
                                    "output_dir": "/tmp/test_benchmark_results",
                                }
                            ]
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
    def test_benchmark(self, idx, experiment, expected_status):
        if isinstance(expected_status, ExitStatus):
            expected_status = (expected_status,)

        # x = cp.boolvar(shape=3)
        # experiment["solver"](cp.Model(2*x + 3*x + 5*x <= 6))

        dt = time.time()
        out = xcsp3_benchmark(**experiment)
        dt = time.time() - dt

        df = pd.read_csv(out).loc[0]
        print(df)
        status = ExitStatus(df["status"])
        assert status in expected_status, f"Unexpected status for {experiment['solver']}\n\n{df['exception']}"
        assert dt < experiment["time_limit"] + experiment["check_time_limit"] + TIME_BUFFER

        feasible = status in (ExitStatus.unsat, ExitStatus.optimal, ExitStatus.sat)
        print(df)

        assert not np.isnan(df["time_total"])
        if feasible:
            assert not np.isnan(df["time_solve"])
            assert not np.isnan(df["time_post"])

    def test_ablate(self):
        assert ablate([("a", (False, True)), ("b", (0, 2, 5))], add_none=True, add_all=True) == [
            ("none", {"a": False, "b": 0}),
            ("a", {"a": True, "b": 0}),
            ("b-2", {"a": False, "b": 2}),
            ("b-5", {"a": False, "b": 5}),
            ("all", {"a": True, "b": 2}),
        ]
