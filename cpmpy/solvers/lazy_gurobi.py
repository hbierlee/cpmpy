import itertools
import json
import logging
import math
import os
import pathlib
import pickle
import pprint
import sys
import time
from enum import Enum

import numpy as np
import pandas as pd

import cpmpy as cp
from cpmpy.expressions.core import Comparison, Operator
from cpmpy.expressions.utils import is_false_cst, show_assignment
from cpmpy.expressions.variables import NegBoolView, _BoolVarImpl
from cpmpy.solvers.gurobi import CPM_gurobi

# https://github.com/ed-lam/cpaior2025-master-class/blob/5c727db2a103ded7971bb89693fe5bb69d509c76/common.py#L9
# Functions for approximate comparison of floating point numbers
EPS = 1e-6


def is_eq(x, y):
    return abs(x - y) <= EPS


def is_lt(x, y):
    return x - y < -EPS


def is_le(x, y):
    return x - y <= EPS


def is_gt(x, y):
    return x - y > EPS


def is_ge(x, y):
    return x - y >= -EPS


def eps_floor(x):
    return math.floor(x + EPS)


def eps_ceil(x):
    return math.ceil(x - EPS)


def eps_round(x):
    return math.ceil(x - 0.5 + EPS)


def eps_frac(x):
    return x - eps_floor(x)


def is_integral(x):
    return eps_frac(x) <= EPS


def assign_mipsol(A_enc):
    return [1 if a > 0.5 else 0 for a in A_enc]


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(sorted(obj))
        elif isinstance(obj, Comparison):
            return str(obj)
        elif isinstance(obj, _BoolVarImpl):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


INDEX = 1

DEBUG_NP_PRINTOPTIONS = {
    "threshold": sys.maxsize,
    "linewidth": np.inf,
    "formatter": {"float_kind": "{:.2f}".format},
}


def show_ind(a, index=INDEX):
    return a + index


def show_set(S, index=INDEX):
    return f"{{{', '.join(str(show_ind(s, index=index)) for s in sorted(S))}}}"


class Infeasible(Exception):
    pass


def rows(T, i, j=1):
    """Row indices where T[i]==j"""
    return set(int(i) for i in np.where(T[:, i] == j)[0].flatten())


def cols(T, i, j=1):
    """Col indices where T[i]==j"""
    # TODO replace for rows(T.T, ..)?
    return set(int(i) for i in np.where(T[i, :] == j)[0].flatten())


def is_integer(v):
    return math.isclose(v, v > 0.5, abs_tol=1e-5)


def is_integer_solution(A_enc):
    return all(is_integer(a) for a in A_enc)


def encode_x(a, lb, ub):
    X = (ub - lb + 1) * [0]
    try:
        X[a - lb] = 1
    except IndexError:
        # Fortress1-03_c25 case
        # ['0..0', '0..1']
        # [[0, 1], [1, 0]]
        pass
    return X


def union(sets):
    sets = tuple(sets)
    return set.union(*sets) if sets else set()


def encode(X, T):
    enc = []
    for t in T:
        row_i = []
        for x, row in zip(X, t):
            row_i += encode_x(row, x.lb, x.ub)
        enc += [row_i]
    return np.array(enc)
    return np.array([xij for x, t in zip(X, T) for row in t for xij in encode_x(row, x.lb, x.ub)])


class Heuristic(Enum):
    INPUT = 1
    GREEDY = 2
    REDUCE = 3


def normalize_table(table):
    """Merge columns with duplicate variables (removing rows where values are different)"""
    X, T = table.args

    to_delete = set()
    for i, x in enumerate(X):
        for j, y in enumerate(X[i + 1 :], start=i + 1):
            if x.name == y.name:
                T = [r for r in T if r[i] == r[j]]
                to_delete.add(j)
    for d in to_delete:
        table.args[0] = [x for i, x in enumerate(X) if i not in to_delete]
        table.args[1] = [[ri for i, ri in enumerate(r) if i not in to_delete] for r in T]

    return table


class CPM_lazy_gurobi(CPM_gurobi):
    def __init__(self, env=None, cpm_model=None, **kwargs):
        self.env = {
            "debug": False,
            "verbosity": 0,
            "log": None,
            "heuristic": Heuristic.GREEDY,
            "shrink": True,
            "fractional": True,
            "coverlift": False,
            "cuts": [],
            "max_iterations": None,
            "seed": 42,
            "checker": cp.Model(),
            "tables": [],
            "found_feasible": False,
            "example2": False,
            **({} if env is None else env),
        }
        self.indent = 0

        if self.env["verbosity"] >= 4:
            np.set_printoptions(**DEBUG_NP_PRINTOPTIONS)
            pprint.pprint(env)

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename=self.env["log"], level=logging.DEBUG, filemode="w", force=True, format="%(message)s"
        )

        self.ivarmap = {}

        self.tables = []

        if self.env["debug"]:
            self.env["solutions"] = frozenset(
                cp.solvers.utils.solutions(cpm_model, projected_solution_limit=None)
            )

        super().__init__(lazy=True, cpm_model=cpm_model, **kwargs)
        self.native_model.Params.LazyConstraints = 1
        # self.native_model.Params.Threads = 1
        # self.native_model.Params.PreCrush = 1
        if self.env["seed"] is not None:
            self.native_model.Params.Seed = self.env["seed"]
        if self.env["verbosity"] >= 4:
            self.native_model.Params.LogFile = "/tmp/gurobi.log"
            self.native_model.Params.OutputFlag = 1
            self.native_model.write("/tmp/gurobi.lp")

        if self.env["debug"] and cpm_model is not None:
            self.env["feasible"] = cpm_model.solve()

    def log(self, *mess, verbosity=1, end="\n", indent=None):
        if verbosity <= self.env["verbosity"]:
            indent = self.indent if indent is None else indent
            mess = " " * indent * 2 + " ".join(str(m) for m in mess) + end
            self.logger.debug(mess)
            print(mess, end="")
            # print(mess, end="", flush=self.env["debug"])

    def print_cuts(self):
        cuts_df = pd.DataFrame.from_dict(self.env["cuts"])
        if self.env["verbosity"] >= 4:
            pd.set_option("display.max_rows", None)
            pd.set_option("display.max_colwidth", None)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)
        drop = ["failure", "cut", "size"]
        show_df = cuts_df.drop(drop, axis=1, errors="ignore").to_string()
        if self.env["debug"]:
            self.log(show_df, verbosity=2)

        with open("cuts.json", mode="w") as f:
            json.dump(self.env["cuts"], f, cls=SetEncoder, ensure_ascii=False, indent=2)

    def stats(self):
        if self.env["debug"]:
            self.log(
                ", ".join(f"{k}={self.env[k]}" for k in ["shrink", "heuristic", "fractional", "coverlift"]),
                verbosity=0,
            )
        self.print_cuts()

        cuts = [c for c in self.env["cuts"] if "size" in c]
        cuts_mipsol = [c for c in cuts if c["from"] == "MIPSOL"]
        assert all(c["size"] > 0 for c in cuts_mipsol)
        cuts_mipnode = [c for c in cuts if c["from"] == "MIPNODE-OPT"]
        cuts_mipnode_exp = [c for c in cuts_mipnode if c["size"] > 0]
        cuts_mipnode_unexp = [c for c in cuts_mipnode if c["size"] == 0]
        if self.env["debug"]:
            self.log(f"time_cb = {self.env['time_cb']}")
            self.log(f"cuts (MIPSOL) = {len(cuts_mipsol)}")
            self.log(f"cuts (MIPNODE, explained) = {len(cuts_mipnode_exp)}")
            self.log(f"cuts (MIPNODE, unexplainable) = {len(cuts_mipnode_unexp)}")
        return {
            "time_cb": self.env["time_cb"],
            "n_cuts": len(cuts_mipsol),
            "n_cuts_explained": len(cuts_mipnode_exp),
            "n_cuts_unexplained": len(cuts_mipnode_unexp),
        }

    def choose(self, A, T_enc, R, heuristic=Heuristic.GREEDY):
        if self.env["debug"]:
            self.log(f"Choose from {show_set(A)} from remaining choices {show_set(R)}", verbosity=2)
        if len(A) == 0:
            return None
        match heuristic:
            case Heuristic.INPUT:
                return min(i for i in A)
            case Heuristic.GREEDY:
                return min(A, key=lambda i: len(R.intersection(rows(T_enc, i))))
            case Heuristic.REDUCE:
                choice = min(
                    (i for i in A if R.intersection(rows(T_enc, i)) != R),
                    key=lambda i: len(rows(T_enc, i)),
                    default=None,  # TODO [?] check this edge-case
                )
                return choice if choice is not None else self.choose(A, T_enc, R, heuristic=Heuristic.GREEDY)

    def shrink(self, C, T):
        for i in C:
            if self.env["debug"]:
                self.log(f"shrinking {i + INDEX} in {show_set(C)}", verbosity=3)

            if len(C) <= 1:  # slightly different from
                return C

            S = set.intersection(*[rows(T, l) for l in (C - {i})])
            if len(S) == 0:
                C = C - {i}
        return C

    def gencoverlift(self, S, C_enc, k, T_enc):
        if self.env["debug"]:
            self.log("gencoverlift", verbosity=3)
        R = set(range(len(T_enc)))  # rows
        C = set(range(len(T_enc.T)))  # cols

        def tight(R, RS):
            # TODO [peter] incorrect def in alg?
            # return {r for r in R if sum(T_enc[r, i] for i in S) == k}
            return {r for r in R if is_eq(RS[r], k)}

        RS = [sum(C_enc[i] * T_enc_r[i] for i in S) for T_enc_r in T_enc]
        R_tight = tight(R, RS)
        X = union(cols(T_enc, r) for r in R_tight)
        if self.env["debug"]:
            self.log("S", show_set(S), verbosity=3)
            self.log("C_enc, k", C_enc, k, verbosity=3)
            self.log("RS", RS, verbosity=3)
            self.log("R", show_set(R_tight), verbosity=3)
            self.log("X", show_set(X), verbosity=3)
            self.log("C", show_set(C), verbosity=3)
        # TODO [peter] C missing from alg

        def choose(S):
            self.log("choose from", show_set(S), verbosity=3)
            return min(S)

        i = 0
        while C - X:
            j = [3, 7, 0][i] if self.env["example2"] else choose(C - X)

            a_j = min((k - RS[r] for r in R - R_tight if T_enc[r, j] == 1), default=None)
            if a_j is None:
                break

            S.add(j)  # S = S + {j}
            if self.env["debug"]:
                self.log("j", show_ind(j), verbosity=3)
                self.log("S", show_set(S), verbosity=3)

            # assert j not in C_enc # TODO [peter] can happen?
            C_enc[j] = a_j
            RS = RS + a_j * T_enc.T[j]
            N_tight = tight(R - R_tight, RS)
            R_tight = R_tight.union(N_tight)

            X = X.union(union(cols(T_enc, r) for r in N_tight))

            if self.env["debug"]:
                self.log("a_j", a_j, verbosity=3)
                self.log("A", a_j * T_enc.T[j], verbosity=3)
                self.log("RS", RS, verbosity=3)
                self.log("N_tight", show_set(N_tight), verbosity=3)
                self.log("R_tight", show_set(R_tight), verbosity=3)
                self.log("X", show_set(X), verbosity=3)
                i += 1
                self.check_max_iterations(i)

        if self.env["debug"]:
            self.log("S", show_set(S), verbosity=3)
            self.log("terms", C_enc, verbosity=3)
        return S, C_enc, k

    def explain(self, A_enc, T_enc, parts, frm=None):
        """The `explain_frac2` alg."""

        # TODO convert T_enc to set of tuples?

        if self.env["debug"]:
            self.log("Explain", end="\n")
            self.log(" ", np.array(A_enc), verbosity=2, indent=0)
            if frm == "MIPSOL":
                self.log("", np.array(assign_mipsol(A_enc)), verbosity=2, indent=0)
            self.log(np.array(T_enc), verbosity=2, indent=0)
            # self.log(
            #     "",
            #     np.array(
            #         [
            #             i + 1
            #             for i, p in enumerate([0] + parts_)
            #             for _ in range(p, parts_[i] if i < len(parts_) else p)
            #         ]
            #     ),
            #     verbosity=2,
            #     indent=0,
            # )
            # self.log(np.array(parts_), verbosity=2, indent=0)
            self.log("", indent=0)

        assert len(A_enc) == len(T_enc.T)

        self.env["cuts"].append({"from": frm})

        m = len(T_enc)  # number of cols
        W = set(i for i, a in enumerate(A_enc) if is_eq(a, 1.0))  #
        # F = set(i for i, a in enumerate(A_enc) if is_gt(a, 0.0) and is_lt(a, 1.0))
        F = set(i for i, a in enumerate(A_enc) if not is_integral(a))
        # assert not is_integer_solution(A_enc[i] for i in F), f"F should hold only fractional, but was: {F}"
        X = set()  # columns added to cut
        R = set(range(len(T_enc)))  # remaining columns
        D = set(r for r in range(m) if cols(T_enc, r) <= W.union(F))  # difficult rows; either frac/whole
        U = set(i for i in F if all(T_enc[r, i] == 0 for r in D))  # frac except difficult

        if self.env["debug"]:
            self.log(f"W = {show_set(W)}", verbosity=3)
            self.log(f"F = {show_set(F)}", verbosity=3)
            self.log(f"D = {show_set(D)}", verbosity=3)
            self.log(f"U = {show_set(U)}", verbosity=3)

        if F:
            if not U:
                if self.env["debug"]:
                    self.log("unexplainable", indent=2)
                    self.log("because U is empty", verbosity=3, indent=4)
                return
            else:
                i = self.choose(U, T_enc, R, heuristic=self.env["heuristic"])
                if self.env["debug"]:
                    self.log(f"chosen {i + INDEX}", verbosity=3, indent=self.indent + 2)

                # TODO [peter] should be T_hat[i]?
                R = rows(T_enc, i)
                X = {i}
                V = {parts[i]}
                s = 0
        else:
            R = set(range(m))
            X = set()
            V = set()
            s = -1

        for iteration in itertools.count(start=1):
            self.check_max_iterations(iteration)
            self.indent = 1
            if self.env["debug"]:
                self.log(f"X = {show_set(X)}", verbosity=3)
                self.log(f"V = {show_set(V)}", verbosity=3)
                self.log(f"R = {show_set(R)}", verbosity=3)
                self.log(f"s = {s}", verbosity=3)

            if not R:
                break
            choices = set(range(len(set(parts)))) - V
            if not choices:
                # with open("/tmp/failed_cut_nc.pkl", "wb") as f:
                #     print("store", (A_enc, T_enc, parts, frm))
                #     pickle.dump((A_enc, T_enc, parts, frm), f)
                return  # TODO [peter]

            l = next(l for l in choices)
            if self.env["debug"]:
                self.log(f"choose integer l = {INDEX + l} in {show_set(choices)}", verbosity=3)

            V.add(l)
            s += 1

            def C(v, l):
                # TODO can be further improve by iterating over the relevant part 
                c = set(i for i in range(len(v)) if is_gt(v[i], 0.0) and parts[i] == l)
                # c = set(i for i in range(len(v)) if is_gt(v[i], 0.0) and parts[i] == l)
                if self.env["debug"]:
                    self.log(f"C({v}, {INDEX + l}) = {show_set(c)}", verbosity=3)
                return c

            C_ = C(A_enc, l)
            sets = union(rows(T_enc, i) for i in C_)
            if self.env["debug"]:
                self.log(f"Union = {show_set(sets)}", verbosity=3)
            R = R.intersection(sets)
            X = X.union(C_)

        if self.env["debug"]:
            self.log(f"by explanation of size ({len(X)}): {show_set(X)}")

        if self.env["debug"]:
            self.env["cuts"][-1]["cut"] = X.copy()

        if self.env["shrink"]:
            X_shrunk = self.shrink(X, T_enc)
            shrunk = len(X) - len(X_shrunk)
            if shrunk:
                if self.env["debug"]:
                    self.log(f"shrunk by {shrunk}: {show_set(X)} --> {show_set(X_shrunk)}", indent=2)
                    self.env["cuts"][-1] = {
                        **self.env["cuts"][-1],
                        "pre_shrunk": self.env["cuts"][-1]["cut"],
                        "cut": X_shrunk,
                    }
            self.env["cuts"][-1]["shrunk"] = shrunk

        C_enc = {s: 1 for s in X}
        k = len(X) - 1

        if self.env["debug"]:
            self.log(f"cut == {show_set(X)}, {C_enc}, {k}", indent=2)
        if self.env["coverlift"]:
            X, C_enc, k = self.gencoverlift(X, C_enc, k, T_enc)

        self.env["cuts"][-1]["size"] = len(X)

        if self.env["debug"]:
            self.log(f"gcut== {show_set(X)}, {C_enc}, {k}", indent=2)
        return X, C_enc, k

    def check_max_iterations(self, i):
        # Loop termination for debug purposes
        if self.env["max_iterations"] is not None:
            assert i <= self.env["max_iterations"]

    def get_solution_callback(self):
        from gurobipy import GRB

        all_xs = {x_enc_i for x_enc, _, _, _ in self.tables for x_enc_i in x_enc}

        def solution_callback(what, where):
            time_cb = time.time()

            try:
                x_enc_a = None
                frm = None

                def cbGetVal(cpm_var, cbGet):
                    if isinstance(cpm_var, NegBoolView):
                        return 1.0 - cbGet(self.solver_var(~cpm_var))
                    return cbGet(self.solver_var(cpm_var))

                match where:
                    case GRB.Callback.MIPNODE:
                        if not self.env["fractional"]:
                            return
                        # Optimal solution to LP relaxation
                        if what.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                            x_enc_a = {x_enc_i: cbGetVal(x_enc_i, what.cbGetNodeRel) for x_enc_i in all_xs}
                            frm = "MIPNODE-OPT"
                        else:
                            return
                    case GRB.Callback.MIPSOL:  # Integer solution
                        x_enc_a = {x_enc_i: cbGetVal(x_enc_i, what.cbGetSolution) for x_enc_i in all_xs}
                        frm = "MIPSOL"
                        if self.env["debug"]:
                            assert is_integer_solution(x_enc_a.values()), (
                                f"Expected integer solution for MIP, but got {x_enc_a}"
                            )
                    case _:
                        return

                if self.env["debug"]:
                    self.log(frm, x_enc_a, verbosity=2)
                feasible = True  # assume feasible
                for explanation in self._explain_assignment(x_enc_a, frm=frm):
                    feasible = False  # any explanation means not feasible
                    expr = self.transform(explanation)
                    assert len(expr) == 1
                    expr = expr[0]

                    if isinstance(expr, Comparison) and expr.name == "<=":
                        assert isinstance(expr.args[0], Operator)
                        expr, k = explanation.args
                        cut = (
                            self._make_numexpr(expr) <= k
                            # gp.quicksum([self.solver_var(x) for x in expr.args[0].args])
                            # <= explanation.args[1]
                        )
                        what.cbLazy(cut)
                    elif is_false_cst(expr):
                        raise Infeasible
                    else:
                        assert False, f"Unsupported expl: {expr}"

                    # if explanation:
                    #     # grbs = [self.solver_var(X_enc[c]) for c in explanation]
                    #     # cut = gp.quicksum(grbs) <= len(grbs) - 1.0
                    #     # what.cbLazy(cut)
                    # elif explanation is False:
                    #     raise Infeasible

                self.check_max_iterations(len(self.env["cuts"]))
                if feasible and frm == "MIPSOL":
                    if self.env["debug"]:
                        self.log("found feasible", verbosity=2)
                    self.env["found_feasible"] = feasible
            except Exception as e:
                what._callback_exception = e
                what.terminate()
            finally:
                time_cb = time.time() - time_cb
                if self.env["debug"]:
                    self.log(f"end callback, dt = {time_cb}", verbosity=3)
                self.env["time_cb"] += time_cb
                # assert time_cb < 1.0 or self.env["debug"]

        return solution_callback

    def _explain_assignment(self, x_enc_a, frm=None):
        # If fully integer, we can check if the tables are feasible yet
        for i, (X_enc, T_enc, parts, table) in enumerate(self.tables, start=INDEX):
            A_enc = [x_enc_a[x_enc_i] for x_enc_i in X_enc]
            A_enc_ = assign_mipsol(A_enc)

            # TODO figure out when can be skipped
            # if frm == "MIPNODE-OPT" and is_integer_solution(A_enc) and
            if frm == "MIPSOL":
                if A_enc_ in T_enc.tolist():
                    if self.env["debug"]:
                        self.log(f"table {i}/{len(self.tables)} feasible by {A_enc_}\n\n{T_enc}")
                    # assert False
                    # assert table.value() # TODO after assigning _value
                    continue
                elif self.env["debug"]:
                    self.log(f"table {i}/{len(self.tables)} INfeasible by {A_enc_}\n\n{T_enc}")

            try:
                # encode assignment
                if self.env["debug"]:
                    self.log(" ", np.array(X_enc), verbosity=2, indent=0)
                explanation = self.explain(A_enc, T_enc, parts, frm=frm)
                if self.env["debug"]:
                    # self.check_explanation(explanation, X_enc, A_enc, T_enc)
                    self.env["cuts"][-1]["x"] = str(X_enc)
                    self.env["cuts"][-1]["table"] = str(T_enc)
                    # self.env["cuts"][-1]["explanation"] = explanation_expr
                    self.env["cuts"][-1]["failure"] = list(zip(X_enc, A_enc))
                    self.env["cuts"][-1]["failure_"] = [
                        f"{x}={a}"
                        for x, a in [
                            (f"{self.names[x.name]}" if hasattr(self, "names") else f"{x}", f"{a:.2f}")
                            for x, a in zip(X_enc, A_enc)
                            if a > 0.0
                        ]
                    ]
                if explanation:
                    X, C_enc, k = explanation
                    expr = cp.sum(C_enc[i] * X_enc[i] for i in X) <= k
                    if self.env["debug"]:
                        self.log(
                            f"cons == {expr}",
                            indent=2,
                        )
                    self.env["cuts"][-1]["expr"] = expr

                    # if self.env["debug"]:
                    #     self.check_explanation(expr, X_enc, A_enc, T_enc, table)

                    if frm == "MIPSOL" and self.env["debug"]:
                        for x, a in zip(X_enc, A_enc_):
                            x._value = a
                        assert expr.value() is False, (
                            f"Did not cut off assignment:\n\n{show_assignment(X_enc)}\n\nwith exp {expr} for table:\n\n {np.array(A_enc_)}\n{T_enc}"
                        )
                        for T_enc_i in T_enc:
                            if True:
                                for x_i, a_i_j in zip(X_enc, T_enc_i):
                                    x_i._value = bool(a_i_j)
                                assert expr.value() is True, (
                                    f"Explanation:\n\n{expr}\n\ncut off row\n\n{T_enc_i}\n({show_assignment(X_enc)})\n\nfor failure {A_enc_}"
                                )

                    yield expr
                elif frm == "MIPSOL":  # unsat
                    raise Infeasible
            except Infeasible:
                raise Infeasible
            except Exception as e:
                with open("/tmp/failed_cut.pkl", "wb") as f:
                    pickle.dump((A_enc, T_enc, parts, frm), f)
                raise e

    def add(self, cons):
        if self.env["debug"]:
            self.env["checker"] += [con for con in cons if con.name != "table"]
        return super().add(cons)

    __add__ = add  # avoid redirect in superclass

    def check_explanation(self, explanation, X_enc, A_enc, T_enc, table):
        self.env["checker"] += explanation

        actual_solutions = frozenset(
            cp.solvers.utils.solutions(self.env["checker"], X=self.user_vars, projected_solution_limit=None)
        )
        expected_solutions = self.env["solutions"]
        self.env["cuts"][-1]["n_sols"] = len(actual_solutions)
        assert expected_solutions.issubset(actual_solutions), f"{expected_solutions} </= {actual_solutions}"

        # m = cp.Model(explanation, [x == a for x, a in zip(X_enc, A_enc)])
        # assert not m.solve(), f"Explanation {explanation} did not remove failure {A_enc}"

        # sols = []
        # for sol in solutions(self.user_vars, self.env["checker"], verbosity=1, projected_solution_limit=None):
        #     for x, a in sol.items():
        #         x._value = a
        #     # for c in self.env["tables"]:
        #     #     print("C", c)
        #     #     # for c in self.env["checker"].constraints:
        #     #     assert c.value(), f"Fail on {sol}: {c}"
        #     sols.append(sol)
        # print("SOLS", len(sols), sols)

        # check that each row in the table is still allowed

        # for row in table.args[1]:
        # P = self.env["checker"].copy()
        # P += cp.all(x == v for x, v in zip(table.args[0], row))
        # assert P.solve() or not self.env["feasible"], (
        #     f"Explanation {explanation} removed row, but feasible={self.env['feasible']}: {row}\n\n{P}"
        # )

        # for table in self.env["tables"]:
        #     assert frozenset(tuple(sol[k] for k in table.args[0])).issubset(table.args[0])

        # assert len(sols) >= len(T_enc), f"{sols} != {len(T_enc)} for checker {self.env['checker']}"

    def solve(self, time_limit=None, solution_callback=None, env=None, **kwargs):
        """
        Call the gurobi solver with cut generation
        """

        if env is not None:
            self.env = {**self.env, **env}

        self.env["cuts"] = []
        self.env["time_cb"] = 0.0

        assert solution_callback is None, "For now, no solution_callback in `CPM_lazy_gurobi`"

        self.log("Solving.. ")

        try:
            solution_callback = self.get_solution_callback()
            hassol = super().solve(solution_callback=solution_callback, time_limit=time_limit, **kwargs)

            if hassol:
                if self.env["debug"]:
                    assert self.env["feasible"]
                if not self.env["found_feasible"]:
                    print("WARN: not found feas")
                # assert self.env["found_feasible"]

            if getattr(self.native_model, "_callback_exception", None):
                raise self.grb_model._callback_exception or Exception(
                    "Gurobi was interrupted (perhaps the solution callback called model.terminate())"
                )

        except Infeasible:
            hassol = False
        except Exception as e:
            self.log("Exception", e)
            self.env["verbosity"] = 4
            self.stats()
            raise e

        if "PYTEST_CURRENT_TEST" not in os.environ:
            for field, stat in self.stats().items():
                print(f"c Stat={field}={stat}")

        # TODO recheck https://or.stackexchange.com/questions/12591/ensure-gurobi-uses-callback-on-all-feasible-solutions It looks like if the solution at the end of the root node is integer, gurobi doesn't pass through callbacks for fractional solutions.

        return hassol

    def get_x_encs(self, X):
        return [x_enc_i for x in X for x_enc_i in self.ivarmap[x]._xs]

    def transform(self, cpm_expressions):
        cpm_cons = []  # all but tables
        cpm_expressions = super().transform(cpm_expressions)
        for cpm_expr in cpm_expressions:
            if cpm_expr.name == "table":
                if len(set(cpm_expr.args[0])) < len(cpm_expr.args[0]):
                    cpm_expr = normalize_table(cpm_expr)
                X, T = cpm_expr.args
                if len(T) == 0:
                    return [cp.BoolVal(False)]
                assert len(set(X)) == len(X), f"Dup. int vars in table for {cpm_expr}"

                T_enc = encode(X, T)

                if self.env["debug"]:
                    self.log("X =", ", ".join(f"{x} in {x.lb}..{x.ub}" for x in X), verbosity=3)
                    self.log("T =", verbosity=3)
                    self.log(T, verbosity=3)
                    self.log("T_enc =", verbosity=3)
                    self.log(T_enc, verbosity=3)

                for x in X:
                    x_enc, exactly_one_con = cp.transformations.int2bool._encode_int_var(
                        self.ivarmap, x, "direct", csemap=self._csemap
                    )
                    expr, k = x_enc.encode_term()
                    # TODO if only BV, then need to assign (but no need to assign if decoding constraint present)
                    # Note: do not use self += [..] to avoid poluting user_vars
                    cons = self.transform([*exactly_one_con, cp.sum(c * b for c, b in expr) + k == x])
                    cpm_cons += cons

                    if self.env["debug"]:
                        for c in cons:
                            self.env["checker"] += c

                x_encs = [self.ivarmap[x.name]._xs for x in X]
                parts = [i for i, x_enc in enumerate(x_encs) for _ in range(len(x_enc))]
                X_enc = [x_enc_i for x_enc in x_encs for x_enc_i in x_enc]
                assert len(set(X_enc)) == len(X_enc), f"Dup. bool vars in table for {cpm_expr}"
                self.tables.append((X_enc, T_enc, parts, cpm_expr))
            else:
                cpm_cons.append(cpm_expr)

        # self.names = {
        #     x.name: f"x{i}"
        #     for i, x in enumerate(cp.transformations.get_variables.get_variables(cpm_cons), start=1)
        # }

        return cpm_cons

    def _check_repeat_failure(self):
        if self.env["debug"]:
            prev = next(
                (c for c in self.env["cuts"][:-1] if self.env["cuts"][-1]["failure"] == c["failure"]),
                None,
            )

            assert prev is None, f"""Encountered same failure twice:

        {pprint.pformat(self.env["cuts"][-1])}

        should have been prevented by earlier cut:

        {pprint.pformat(prev)}
        """
