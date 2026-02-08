import math
import itertools
import pathlib
import pickle
import random

import numpy as np
import pytest

import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables_model
from cpmpy.expressions.utils import show_assignment, dom_size
from cpmpy.solvers.lazy_gurobi import CPM_lazy_gurobi, normalize_table, Heuristic
from cpmpy.solvers.gurobi import CPM_gurobi, Encoding
from cpmpy.tools.xcsp3.experiments import get_experiments


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


def generate_edge_case_tables():
    """Generator yielding various edge case table constraints"""

    # Single column table (1 variable)
    yield cp.Model(cp.Table([cp.intvar(1, 5, name="x")], [[2], [4]]))

    # Complete table (all possible tuples)
    yield cp.Model(
        cp.Table(
            [cp.intvar(1, 2, name="x"), cp.intvar(1, 2, name="y")],
            [[i, j] for i in range(1, 3) for j in range(1, 3)],
        )
    )

    # Duplicate rows in table
    yield cp.Model(
        cp.Table(
            [cp.intvar(1, 3, name="x"), cp.intvar(1, 3, name="y")], [[1, 2], [2, 1], [1, 2], [2, 1], [1, 2]]
        )
    )

    # Constant column (one variable always same value)
    yield cp.Model(
        cp.Table(
            [cp.intvar(1, 5, name="x"), cp.intvar(1, 3, name="y"), cp.intvar(1, 4, name="z")],
            [[1, 2, 3], [1, 1, 2], [1, 3, 4], [1, 2, 1]],
        )
    )

    # Sparse table (large domain, few tuples)
    yield cp.Model(
        cp.Table([cp.intvar(1, 100, name="x"), cp.intvar(1, 100, name="y")], [[1, 2], [50, 75], [99, 100]])
    )

    # No valid tuples (unsatisfiable - tuples outside domains)
    yield cp.Model(cp.Table([cp.intvar(1, 3, name="x"), cp.intvar(1, 3, name="y")], [[4, 5], [5, 6], [6, 7]]))

    # table with fixed vals, feas/infeasible
    yield cp.Model(cp.Table([cp.intvar(3, 3, name="a"), cp.intvar(1, 3, name="y")], [[3, 2]]))
    yield cp.Model(cp.Table([cp.intvar(3, 3, name="a"), cp.intvar(1, 3, name="y")], [[4, 2]]))

    # table with bool vars
    yield cp.Model(cp.Table([cp.boolvar(name="p"), cp.intvar(1, 3, name="y")], [[0, 2], [1, 3]]))

    # table with neg bool vars
    yield cp.Model(cp.Table([~cp.boolvar(name="p"), cp.intvar(1, 3, name="y")], [[0, 2], [1, 3]]))


    # Single tuple only
    yield cp.Model(cp.Table([cp.intvar(1, 5, name="x"), cp.intvar(1, 5, name="y")], [[3, 3]]))

    # Very wide table (many columns)
    yield cp.Model(
        cp.Table(
            cp.intvar(1, 3, shape=10, name="x"),
            [[1, 2, 3, 1, 2, 3, 1, 2, 3, 1], [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]],
        )
    )

    # Diagonal pattern
    yield cp.Model(cp.Table([cp.intvar(1, 3, name="x"), cp.intvar(1, 3, name="y")], [[1, 1], [2, 2], [3, 3]]))

    # Anti-diagonal pattern
    yield cp.Model(cp.Table([cp.intvar(1, 3, name="x"), cp.intvar(1, 3, name="y")], [[1, 3], [2, 2], [3, 1]]))


def generate_table_from_data(T, d):
    """Generate a table constraint with the given `rows` and with var domains of size `d`"""

    X = cp.intvar(1, d, shape=len(T[0]), name="x")
    return cp.Model(cp.Table(X, T))


def generate_table(n, m, d, k=1, allow_duplicate_vars=False):
    """Generate `k` table constraints with `n` variables with domains of size `d`, and with `m` rows"""
    X = cp.intvar(1, d, shape=(n,), name="x")
    # model = cp.Model(x == x for x in X)
    model = cp.Model()
    random.seed(SEED)
    for _ in range(k):
        if k > 1:
            k_ = n // 2
            X = list(X)
            Y = random.choices(X, k=k_) if allow_duplicate_vars else random.sample(X, k=k_)
        else:
            Y = X
        if len(Y):
            T = [[random.randint(1, d) for _ in enumerate(Y)] for _ in range(m)]
            model += cp.Table(Y, T)
    return model


def assert_integer_solution(A_enc):
    for a_enc_i in A_enc:
        assert math.isclose(a_enc_i, round(a_enc_i), abs_tol=1e-5), (
            f"Expected integer solution for MIP, but got {a_enc_i} in {A_enc}"
        )


def check_model(model, env=None):
    print("== Model ==")
    print(model)


    print(", ".join(f"{x} in {x.lb}..{x.ub}" for x in get_variables_model(model)))
    expected_sat = model.deepcopy().solve()
    print("expected feasible = ", expected_sat)
    try:
        slv = env["solver"](cpm_model=model, **env["solver_kwargs"])

        print("solver", slv)
        actual_sat = slv.solve()
        print("actual feasible", actual_sat)
        print("stats = ", slv.stats())

        if True:
            slv_ = env["solver"](**env["solver_kwargs"])
            print("ENCODING")
            for i, c in enumerate(model.constraints, start=1):
                print(f"C{i}", repr(c)[:100])
                for ci in slv_.transform([c]):
                    print("  ", ci)
                    # print("  ", slv._csemap)

        # if hasattr(slv, "stats"):
        #     slv.stats()

        if expected_sat and actual_sat:
            X = cp.transformations.get_variables.get_variables_model(model)
            print("assignment", show_assignment(X))

            assert all(x.value() is not None for x in X), (
                f"Expected all variables to be assigned, but found: {show_assignment(X)}"
            )

            violations = [c for c in model.constraints if c.value() is False]
            assert not violations, (
                f"For assignment:\n\n{show_assignment(X)}\n\nThe following constraints fail:\n\n'"
                + "\n\n".join(str(v) for v in violations)
            )

        assert expected_sat == actual_sat, f"Expected equisat, but {expected_sat=} and {actual_sat=}"

        print("PASS.")
    except AssertionError as e:
        with open("/tmp/failed_model.pkl", "wb") as f:
            pickle.dump(model, f)

        raise e
        if env["debug"]:
            raise e
        else:
            print("try debug", e)

            check_model(model, env={**env, "debug": True})


def show_sols(sols, T):
    return ", ".join(f"*{sol}" if list(sol) in T.tolist() else f"{sol}" for sol in sorted(sols))


def with_constraints(model, with_alldiff=False, with_min=False):
    X = cp.transformations.get_variables.get_variables_model(model)
    if with_alldiff:
        model += cp.AllDifferent(X)
    if with_min:
        model.minimize(sum(X))
    return model


SEED = 42
SEED = None


def get_envs():

    debug_env = {
        "verbosity": 2,
        "debug": 1,
        "max_iterations": 500,
        "seed": 42,
    }

    yield {
        "alias": "dev",
        "solver": CPM_lazy_gurobi,
        "solver_kwargs": {
            "env": {
                **debug_env,
                "fractional": False,
                "coverlift": True,
                "negatives": 0,
                # "checker": cp.Model(),
                "heuristic": Heuristic.GREEDY,
                # "heuristic": Heuristic.REDUCE,
                "cutoff": 0,
            }
        },
    }

    for e in get_experiments():
        if "base" in e["alias"]:
            continue

        yield e | {"solver_kwargs": {"env": debug_env}}

    for encoding in [Encoding.DEFAULT, Encoding.GLEB]:
        yield {
            "alias": f"base_gurobi-{encoding}",
            "solver": CPM_gurobi,
            "solver_kwargs": {
                "verbose": 0,
                "encoding": encoding,
            },
        }


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
    @pytest.mark.skip()
    def test_repro_explain(self, env):
        path = pathlib.Path("/tmp/failed_cut.pkl")
        if path.exists():
            with open(path, "rb") as f:
                X_enc, A_enc, T_enc, parts, frm, A_enc_ = pickle.load(f)
            slv = CPM_lazy_gurobi(
                env={**env, **{"verbosity": 3, "debug": True, "checker": None}},
            )

            COLS = None

            print("parts", len(np.unique(parts)))
            # COLS = np.isin(parts, range(83,85))
            if COLS is not None:
                A_enc = A_enc[COLS]
                T_enc = T_enc[:, COLS]
                parts = parts[COLS]
                A_enc_ = A_enc_[COLS]
                X_enc = X_enc[COLS]

            print(" ", np.array(X_enc))
            explanation = slv.explain(A_enc, T_enc, parts, frm=frm)
            if explanation is None:
                print("Infeasible")
            else:
                slv.explanation_to_expr(explanation, A_enc, X_enc, T_enc, frm)

    @pytest.mark.skip()
    def test_coverlift(self, env):
        slv = CPM_lazy_gurobi(
            cpm_model=cp.Model(generate_table_from_example().constraints),
            env={**env, **{"shrink": False, "debug": True, "example2": True}},
        )
        X_enc, T_enc, parts, table = slv.tables[0]

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
        print("", choices, "choices")
        print(T)
        T = T[[0, 4], :][:, choices]
        print(T.astype(int))
        print("", parts, "p")
        y = np.bitwise_or.reduceat(
            T,
            parts_,
            axis=1,
        ).sum(0)
        print("Remaining R", y)

    def test_shrink(self, env):
        X, k = CPM_lazy_gurobi(env={"verbosity": 4}).shrink(
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

    def test_explain(self, env):
        random.seed(SEED)
        for e, A_enc in [
            (generate_table_from_example(), np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 0])),
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
            # print("A_enc", A_enc)

            slv = CPM_lazy_gurobi(
                # cpm_model=cp.Model(generate_table_from_example().constraints),
                # cpm_model=cp.Model(cp.Table((cp.intvar(1, 1), cp.intvar(1, 2)), [(1, 2)])),
                cpm_model=e,
                env={
                    **env,
                    **{
                        "fractional": False,
                        "coverlift": False,
                        "max_iterations": None,
                        "heuristic": Heuristic.GREEDY,
                        # "heuristic": Heuristic.GREEDY,
                        "shrink": True,
                        "debug": True,
                        "verbosity": 2,
                        "negatives": 0,
                        "checker": cp.Model(),
                    },
                },
            )
            if False:
                slv.solve()
                slv.stats()
                print(slv.env["cuts"])
                slv.print_cuts()
                X_enc, T_enc, parts, table = slv.tables[0]

                def list_to_A_enc(A_enc):
                    return dict(zip(X_enc, A_enc))

                # A_enc = list_to_A_enc(A_enc)
            else:
                X_enc, T_enc, parts, table = slv.tables[0]
                # explanations = list(slv._explain_assignment(A_enc, frm="MIPSOL"))
                frm = "MIPSOL"

                if False:
                    # sum([1, -1] * [⟦x == 2⟧, ⟦y == 1⟧]) <= 0
                    slv.env["cuts"].append({"from": "MIPSOL"})
                    C_enc = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
                    k = 1
                    C_enc = np.array([0, 1, 0, 0, -1, 0, 0, 0, 0, 0])
                    k = 0
                    explanation = (C_enc != 0, C_enc, k)
                else:
                    explanation = slv.explain(A_enc, T_enc, parts, frm=frm)

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
        check_model(m, env=env)

    def test_enc(self, env):
        x = cp.intvar(1, 5, name="x")
        y = cp.intvar(1, 5, name="y")
        m = cp.Model(cp.Table([x, y], [[1, 2]]), cp.InDomain(x, [2, 4]))
        slv = CPM_lazy_gurobi(cpm_model=m)
        print("Repro model:", slv.transform(m.constraints))

    def test_table_enc(self, env):
        x = cp.intvar(1, 4, name="x")
        y = cp.intvar(1, 3, name="y")
        z = cp.intvar(1, 3, name="z")
        X = (x, y, z)
        T = [[2, 1, 1], [3, 2, 2], [4, 3, 3], [1, 2, 3], [2, 1, 2]]
        model = cp.Model(cp.Table(X, T), cp.AllDifferent(X))
        check_model(model, env=env)

    def test_sols(self, env):
        from cpmpy.solvers.utils import solutions

        x = cp.boolvar(3, name="x")
        print(x)
        m = cp.Model(cp.sum([2, 3, 5] * x) <= 6)
        print(m)
        X, sols = solutions(m, verbosity=2)
        print("xx", sols)

        # m = generate_table_from_example()
        # model = cp.Model(cp.Table(X, T), cp.AllDifferent(X))
        # check_model(model, env=env)


def idfn(a):
    if isinstance(a, dict):
        return a["alias"]
    else:
        return f"{a[0]}-{a[1]}"


REPEAT = 3


@pytest.mark.timeout(60)
class TestModels:
    @pytest.mark.parametrize(
        ("env", "case"),
        itertools.product(
            get_envs(),
            (
                (i, j, t)
                for j, _ in enumerate(range(REPEAT), start=1)  # to repeat the test
                for i, t in enumerate(
                    [
                        *[
                            cp.Model(cp.AllDifferent(cp.intvar(1, 3, shape=3))),
                            cp.Model(cp.Table([cp.intvar(0, 5)], [])),
                            cp.Model(~cp.Table([cp.intvar(0, 5)], [])),
                            # generate_table_from_data([tuple()], 5),
                            generate_table_from_data([[1, 1]], 3),  # single row (actually exists in xcsp3)
                            generate_table_from_data([[1, 1], [2, 2]], 3),  # Feasible (often 0 explanations)
                            generate_table_from_data([[1, 2], [2, 1]], 3),  # Feasible
                            with_constraints(
                                generate_table_from_data([[1, 1], [2, 2]], 3), with_alldiff=True
                            ),  # Infeasible
                            with_constraints(
                                generate_table_from_data([[1, 2], [2, 1]], 3),
                                with_alldiff=True,
                                with_min=True,
                            ),
                            generate_table_from_example(),
                            with_constraints(
                                generate_table_from_example(),
                                with_alldiff=True,
                            ),
                            generate_two_tables(),
                        ],
                        *list(generate_edge_case_tables()),  # Edge case tables
                        *[
                            table
                            for allow_duplicate_vars in (False, True)
                            for table in [
                                with_constraints(
                                    generate_table(2, 2, 3, allow_duplicate_vars=allow_duplicate_vars),
                                    with_alldiff=True,
                                    with_min=True,
                                ),
                                with_constraints(
                                    generate_table(4, 4, 4, allow_duplicate_vars=allow_duplicate_vars),
                                    with_alldiff=False,
                                    with_min=False,
                                ),
                                with_constraints(
                                    generate_table(2, 2, 3, k=2, allow_duplicate_vars=allow_duplicate_vars)
                                ),
                                with_constraints(
                                    generate_table(3, 3, 4, allow_duplicate_vars=allow_duplicate_vars),
                                    # with_alldiff=False,
                                    # with_min=True,
                                ),
                                with_constraints(
                                    generate_table(2, 2, 2, allow_duplicate_vars=allow_duplicate_vars)
                                ),  # minimized 1/1000 bug
                                with_constraints(
                                    generate_table(5, 100, 10, allow_duplicate_vars=allow_duplicate_vars)
                                ),
                                with_constraints(
                                    generate_table(4, 3, 4, k=2, allow_duplicate_vars=allow_duplicate_vars)
                                ),  # TRICKY BUG FINDER NO CHIOCE
                            ]
                        ],
                    ],
                    start=1,
                )
            ),
        ),
        ids=idfn,
    )
    def test_models(self, case, env):
        import pprint

        pprint.pprint(env)
        print("env", env)
        _, _, model = case
        check_model(model, env=env)
