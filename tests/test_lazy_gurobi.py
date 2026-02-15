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
from cpmpy.solvers.ortools import CPM_ortools
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
    """Generator yielding (name, model) tuples for various test cases"""

    # Basic test cases
    yield ("alldiff", cp.Model(cp.AllDifferent(cp.intvar(1, 3, shape=3))))
    # yield ("singleton_dom", generate_table_from_data([[1, 1]], 1))
    # yield ("singleton_dom_inf", generate_table_from_data([[1, 2]], 1))
    yield ("single_row", generate_table_from_data([[1, 1]], 3))
    yield ("feasible_diagonal", generate_table_from_data([[1, 1], [2, 2]], 3))
    yield ("feasible_swap", generate_table_from_data([[1, 2], [2, 1]], 3))
    yield ("infeasible_alldiff", with_constraints(generate_table_from_data([[1, 1], [2, 2]], 3), with_alldiff=True))
    yield ("alldiff_min", with_constraints(generate_table_from_data([[1, 2], [2, 1]], 3), with_alldiff=True, with_min=True))
    yield ("from_example", generate_table_from_example())
    yield ("from_example_alldiff", with_constraints(generate_table_from_example(), with_alldiff=True))
    yield ("two_tables", generate_two_tables())

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
        cp.Model(cp.Table([cp.intvar(1, 3, name="x"), cp.intvar(1, 3, name="y")], [[1, 2], [2, 1], [1, 2], [2, 1], [1, 2]])),
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
    yield (
        "sparse_table",
        cp.Model(cp.Table([cp.intvar(1, 100, name="x"), cp.intvar(1, 100, name="y")], [[1, 2], [50, 75], [99, 100]])),
    )
    yield (
        "no_valid_tuples",
        cp.Model(cp.Table([cp.intvar(1, 3, name="x"), cp.intvar(1, 3, name="y")], [[4, 5], [5, 6], [6, 7]])),
    )
    yield ("fixed_feasible", cp.Model(cp.Table([cp.intvar(3, 3, name="a"), cp.intvar(1, 3, name="y")], [[3, 2]])))
    yield ("fixed_infeasible", cp.Model(cp.Table([cp.intvar(3, 3, name="a"), cp.intvar(1, 3, name="y")], [[4, 2]])))
    yield ("bool_vars", cp.Model(cp.Table([cp.boolvar(name="p"), cp.intvar(1, 3, name="y")], [[0, 2], [1, 3]])))
    x = cp.intvar(0, 3, name="x", shape=2)
    yield ("soccer_problem", cp.Model(*[cp.InDomain(x_i, [0, 1, 3]) for x_i in x], cp.Table(x, [[0, 3], [1, 1], [3, 0]])))

    # yield ("random_gaps", generate_table(5, 10, 10, k=1, gaps=0.5))
    yield ("random_gaps", generate_table(5, 5, 3, k=1, gaps=0.5))
    yield ("negated_bool", cp.Model(cp.Table([~cp.boolvar(name="p"), cp.intvar(1, 3, name="y")], [[0, 2], [1, 3]])))
    yield ("single_tuple", cp.Model(cp.Table([cp.intvar(1, 5, name="x"), cp.intvar(1, 5, name="y")], [[3, 3]])))
    yield (
        "wide_table",
        cp.Model(
            cp.Table(
                cp.intvar(1, 3, shape=10, name="x"),
                [[1, 2, 3, 1, 2, 3, 1, 2, 3, 1], [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]],
            )
        ),
    )
    yield ("diagonal", cp.Model(cp.Table([cp.intvar(1, 3, name="x"), cp.intvar(1, 3, name="y")], [[1, 1], [2, 2], [3, 3]])))
    yield ("anti_diagonal", cp.Model(cp.Table([cp.intvar(1, 3, name="x"), cp.intvar(1, 3, name="y")], [[1, 3], [2, 2], [3, 1]])))

    # Generated table tests
    for gaps in (None, 0.5):
        suffix = "_gaps" if gaps else "_nogaps"

        def generate_table_(*args, **kwargs):
            return generate_table(*args, **kwargs)

        yield (
            f"t2x2x3_alldiff_min{suffix}",
            with_constraints(
                generate_table_(2, 2, 3),
                with_alldiff=True,
                with_min=True,
            ),
        )
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
        yield (f"t2x2x2_bug{suffix}", with_constraints(generate_table_(2, 2, 2)))
        yield (
            f"t4x3x4_k2{suffix}",
            with_constraints(generate_table_(4, 3, 4)),
        )
        yield (
            f"sml{suffix}",
            with_constraints(generate_table_(2, 5, 3)),
        )
        yield (
            f"mid{suffix}",
            with_constraints(generate_table_(4, 5, 4, k=3)),
        )
        # yield (
        #     f"big{suffix}",
        #     with_constraints(generate_table_(50, 5000, 10000)),
        # )
        yield (f"t5x100x10{suffix}", with_constraints(generate_table_(5, 100, 10)))


def generate_table_from_data(T, d):
    """Generate a table constraint with the given `rows` and with var domains of size `d`"""

    X = cp.intvar(1, d, shape=len(T[0]), name="x")
    return cp.Model(cp.Table(X, T))


def generate_table(n, m, d, k=1, gaps=None, allow_duplicate_vars=True):
    """Generate `k` table constraints with `n` variables with domains of size `d`, and with `m` rows"""
    model = cp.Model()
    X = cp.intvar(1, d, shape=(n,), name="x")
    if gaps:
        for x in X:
            dom = sorted(random.sample(range(1, d + 1), max(2, round(d * gaps))))
            assert len(set(dom))>1, dom
            model += cp.InDomain(x, dom)
            print('x', x, dom)
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

    print(", ".join(f"{x} in {list(x.dom())}" for x in get_variables_model(model)))
    expected_sat = model.deepcopy().solve()
    print("expected feasible = ", expected_sat)
    try:
        slv = (
            CPM_ortools(cpm_model=model) if env["solver"] == "ortools" else env["solver"](cpm_model=model, **env["solver_kwargs"])
        )

        print("solver", slv)
        actual_sat = slv.solve()
        print("actual feasible", actual_sat)

        if hasattr(slv, "stats"):
            print("stats = ", slv.stats())

        # if True:
        #     slv_ = env["solver"](**env["solver_kwargs"])
        #     print("ENCODING")
        #     for i, c in enumerate(model.constraints, start=1):
        #         print(f"C{i}", repr(c)[:100])
        #         for ci in slv_.transform([c]):
        #             print("  ", ci)
        #             # print("  ", slv._csemap)

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
                "shrink": False,
                "negatives": 0,
                # "checker": cp.Model(),
                "heuristic": Heuristic.GREEDY,
                # "heuristic": Heuristic.REDUCE,
                "cutoff": 0,
                # "verbosity": 2,
            }
        },
    }

    for e in get_experiments():
        if "base" in e["alias"]:
            continue

        yield e | {"solver_kwargs": {"env": debug_env}}

    for encoding in Encoding:
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
    def test_repro_explain(self, env):
        path = pathlib.Path("/tmp/failed_cut.pkl")
        if path.exists():
            with open(path, "rb") as f:
                X_enc, A_enc, T_enc, parts, frm = pickle.load(f)
            slv = CPM_lazy_gurobi(
                env={
                    **env,
                    **{"verbosity": 3, "debug": True, "checker": None, "coverlift": False, "shrink": True},
                },
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
            # print("A_enc", A_enc)

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
                        # "checker": cp.Model(),
                    },
                    **env_,
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
        name, repeat = a[0], a[1]
        return f"{name}_r{repeat}" if repeat > 1 else name


REPEAT = 1


@pytest.mark.timeout(60)
class TestModels:
    @pytest.mark.parametrize(
        ("env", "case"),
        itertools.product(
            get_envs(),
            (
                (name, j, model)
                for j in range(1, REPEAT + 1)  # to repeat the test
                for name, model in generate_edge_case_tables()
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
