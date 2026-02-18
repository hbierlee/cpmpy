import itertools
import json
import collections
import pickle
import pprint
import sys
import time

import numpy as np
import pandas as pd

import cpmpy as cp
from cpmpy.expressions.core import Comparison, Operator
from cpmpy.expressions.utils import is_true_cst, is_false_cst, show_assignment, dom_size
from cpmpy.solvers.gurobi import CPM_gurobi, Feature
from cpmpy.expressions.variables import NegBoolView, _BoolVarImpl
from cpmpy.transformations.linearize import only_positive_bv

CHECKER_TIME_LIMIT = None

# Using Gurobi's default tolerance values:
# https://www.gurobi.com/documentation/current/refman/parameters.html#sec:Parameters
INT_FEAS_TOL = 1e-5  # Gurobi's IntFeasTol: for checking integrality
# FEAS_TOL = 1e-6  # Gurobi's FeasibilityTol: for checking constraint satisfaction
FEAS_TOL = 1e-5  # relaxed tolerance; slightly slower but easier to work with


def none(A):
    return not A.any()


# Based on https://github.com/ed-lam/cpaior2025-master-class/blob/5c727db2a103ded7971bb89693fe5bb69d509c76/common.py#L9
# Functions for approximate comparison of floating point numbers
def is_eq(x, y):
    return abs(x - y) <= FEAS_TOL


def is_lt(x, y):
    return x - y < -FEAS_TOL


def is_le(x, y):
    return x - y <= FEAS_TOL


def is_gt(x, y):
    return x - y > FEAS_TOL


def is_ge(x, y):
    return x - y >= -FEAS_TOL


def eps_floor(x):
    return np.floor(x + INT_FEAS_TOL)


def eps_ceil(x):
    return np.ceil(x - INT_FEAS_TOL)


def eps_round(x):
    return np.ceil(x - 0.5 + INT_FEAS_TOL)


def is_integral(x):
    return np.abs(x - np.round(x)) <= INT_FEAS_TOL


def assign_mipsol(A_enc):
    return [1 if a > 0.5 else 0 for a in A_enc]


def get_table_area(c):
    return len(c.args[1]) * sum(cp.expressions.utils.dom_size(x) for x in c.args[0])


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(sorted(obj))
        elif isinstance(obj, Comparison):
            return str(obj)
        elif isinstance(obj, _BoolVarImpl):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def terms(cpm_expr):
    if isinstance(cpm_expr, Comparison):
        assert cpm_expr.name == "<="
        lin_exp, k = cpm_expr.args
        match lin_exp.name:
            case "wsum":
                ws, xs = lin_exp.args
            case "sum":
                xs = lin_exp.args
                ws = [1] * len(xs)
            case _:
                raise TypeError
        return ws, xs, k
    else:
        raise TypeError


def without(A, B):
    if len(A) == 0 or len(B) == 0:
        return A
    mask = ~np.any(np.all(A[:, None] == B, axis=2), axis=1)
    return A[mask]


INDEX = 1

DEBUG_NP_PRINTOPTIONS = {
    "threshold": sys.maxsize,
    "linewidth": np.inf,
    "formatter": {"float_kind": "{:.2f}".format},
}


def show_table(T_enc, index=INDEX):
    return np.astype(T_enc, int) if T_enc.dtype == bool else T_enc


def show_nz(A, index=INDEX):
    return A.nonzero()[0] + 1


def show_ind(a, index=INDEX):
    return a + index


def show_set(S, index=INDEX):
    return f"{{{', '.join(str(show(s, index=index)) for s in sorted(S))}}}"


def show(S, index=INDEX):
    if isinstance(S, collections.abc.Iterable):
        return show_set(S, index=index)
    elif isinstance(S, (int, np.integer)):
        return show_ind(S, index=index)
    elif isinstance(S, (bool, np.bool_)):
        return "T" if S else "F"
    else:
        raise TypeError(f"{S}, {type(S)}")


class Infeasible(Exception):
    pass


def rows(T, i, j=1):
    """Row indices where T[i]==j"""
    return cols(T.T, i, j=j)
    # return set(int(i) for i in np.where(T[:, i] == j)[0].flatten())


def cols(T, i, j=1):
    """Col indices where T[i]==j"""
    # TODO replace for rows(T.T, ..)?
    return set(i for i in np.where(T[i, :] == j)[0].flatten())


def union(sets):
    sets = tuple(sets)
    return set.union(*sets) if sets else set()


class Heuristic(Feature):
    INPUT = "input"
    GREEDY = "greedy"
    REDUCE = "reduce"


class Coverlift(Feature):
    No = "no"
    INPUT = "input"
    COM_MIN = "com_min"
    COM_MAX = "com_max"

    def __bool__(self):
        return self is not Coverlift.No


def normalize_table(table):
    """Merge columns with duplicate variables (removing rows where values are different)"""
    X, T = table.args

    to_delete = set()
    for i, x in enumerate(X):
        for j, y in enumerate(X[i + 1 :], start=i + 1):
            if x.name == y.name:
                T = [r for r in T if r[i] == r[j]]
                to_delete.add(j)
    if to_delete:
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
            "cutoff": 0,
            "shrink": False,
            "fractional": False,
            "coverlift": Coverlift.No,
            "negatives": 0,
            "cuts": [],
            "max_iterations": None,
            "seed": 42,
            "checked": False,
            "checker": None,
            "tables": [],
            "found_feasible": False,
            "model": None,
            "example2": False,
            "example_frac": False,
            "feasible": None,
            **({} if env is None else env),
        }
        self.indent = 0

        if self.env["verbosity"] >= 4:
            np.set_printoptions(**DEBUG_NP_PRINTOPTIONS)
            pprint.pprint(env)

        # self.logger = logging.getLogger(__name__)
        # logging.basicConfig(
        #     filename=self.env["log"], level=logging.DEBUG, filemode="w", force=True, format="%(message)s"
        # )

        self.ivarmap = {}

        self.tables = []

        if self.env["checked"]:
            self.env["checker"] = cp.Model()

        super().__init__(
            name="lazy_gurobi",
            cpm_model=cpm_model,
            verbose=self.env["verbosity"] >= 4,
            **kwargs,
        )

        if self.tables:
            self.native_model.Params.LazyConstraints = 1
        # self.native_model.Params.Threads = 1
        # self.native_model.Params.PreCrush = 1
        if self.env["seed"] is not None:
            self.native_model.Params.Seed = self.env["seed"]
        if self.env["verbosity"] >= 4:
            self.native_model.Params.LogFile = "/tmp/gurobi.log"
            self.native_model.Params.OutputFlag = 1
            self.native_model.write("/tmp/gurobi.lp")

        if self.env["checked"] and cpm_model is not None:
            self.log("CHECKED: solve model to determine expected feasibility")
            self.env["feasible"] = cpm_model.solve()
            self.env["model"] = cpm_model

        if self.env["checked"]:
            _, self.env["expected_solutions"] = cp.solvers.utils.solutions(
                cpm_model,
                X=sorted(self.user_vars, key=lambda x: x.name),
                projected_solution_limit=None,
                time_limit=CHECKER_TIME_LIMIT,
            )
            self.env["remain"] = without(self.solutions_checker(), self.env["expected_solutions"])
            if self.env["verbosity"]:
                self.log("SOLS", len(self.env["expected_solutions"]))
                self.log("TO REMOVE", self.env["remain"])

    def log(self, *mess, verbosity=1, end="\n", indent=None):
        assert self.env["verbosity"]
        if verbosity <= self.env["verbosity"]:
            indent = self.indent if indent is None else indent
            mess = " " * indent * 2 + " ".join(str(m) for m in mess) + end
            # assert verbosity >= 3 or len(mess) < 100, f"Long message {mess}"
            # self.logger.debug(mess)
            print(mess, end="")
            # print(mess, end="", flush=self.env["debug"])

    def print_cuts(self):
        cuts_df = pd.DataFrame.from_dict(self.env["cuts"])
        if self.env["verbosity"] >= 0:
            pd.set_option("display.max_rows", None)
            pd.set_option("display.max_colwidth", None)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)
        drop = ["failure", "cut", "size"]
        show_df = cuts_df.drop(drop, axis=1, errors="ignore").to_string()
        if self.env["log"]:
            self.log(show_df, verbosity=2)

        # with open("cuts.json", mode="w") as f:
        #     json.dump(self.env["cuts"], f, cls=SetEncoder, ensure_ascii=False, indent=2)

    def stats(self):
        if self.env["log"]:
            self.log(
                ", ".join(f"{k}={self.env[k]}" for k in ["shrink", "heuristic", "fractional", "coverlift"]),
                verbosity=0,
            )

        if self.env["verbosity"] >= 4:
            self.print_cuts()

        cuts = [c for c in self.env["cuts"] if "size" in c]
        cuts_mipsol = [c for c in cuts if c["from"] == "MIPSOL"]
        assert all(c["size"] > 0 for c in cuts_mipsol)
        cuts_mipnode = [c for c in cuts if c["from"] == "MIPNODE-OPT"]
        cuts_mipnode_exp = [c for c in cuts_mipnode if c["size"] > 0]
        cuts_mipnode_unexp = [c for c in cuts_mipnode if c["size"] == 0]
        if self.env["verbosity"]:
            self.log(f"time_cb = {self.env['time_cb']}")
            self.log(f"cuts (MIPSOL) = {len(cuts_mipsol)}")
            self.log(f"cuts (MIPNODE, explained) = {len(cuts_mipnode_exp)}")
            self.log(f"cuts (MIPNODE, unexplainable) = {len(cuts_mipnode_unexp)}")
        return super().stats() | {
            "time_cb": self.env["time_cb"],
            "n_cuts": len(cuts_mipsol),
            "n_cuts_explained": len(cuts_mipnode_exp),
            "n_cuts_unexplained": len(cuts_mipnode_unexp),
            "avg_strength": (sum(s["strength"] for s in cuts) / len(cuts)) if cuts and self.env["checked"] else None,
        }

    def choose(self, choices, T_enc, R, parts, A_enc, heuristic=Heuristic.GREEDY, make_pos_choice=True):
        if self.env["verbosity"]:
            self.log(f"Choose from {show_nz(choices)} to allow remaining rows R={show_nz(R)}", verbosity=3)
            self.log(show_table(T_enc[R, :]), verbosity=3)
            self.log("", parts, "parts", verbosity=3)
            self.log("", show_table(A_enc), "A_enc", verbosity=3)
            self.log("", choices.astype(int), "choices", verbosity=3)

        if none(choices):
            return None
        match heuristic:
            case Heuristic.INPUT:
                return np.argmax(choices)
            case Heuristic.GREEDY:
                T_enc = T_enc if make_pos_choice else ~T_enc
                parts_ = np.add.accumulate(np.unique_counts(parts[choices]).counts)
                parts_ -= parts_[0]
                # map back to the right part index
                return choices.nonzero()[0][
                    np.bitwise_or.reduceat(
                        # get only the relevant rows and columns
                        T_enc[np.ix_(R, choices)],
                        # for the columns of each part
                        parts_,
                        # see if there is any 1 in the row
                        axis=1,
                    )
                    # sum the number of 1s for each part
                    .sum(0)
                    # find the sm
                    .argmin()
                ]

            case Heuristic.REDUCE:
                assert False
                # choice = min(
                #     (i for i in A if R.intersection(rows(T_enc, i)) != R),
                #     key=lambda i: len(rows(T_enc, i)),
                #     default=None,  # TODO [?] check this edge-case
                # )
                # return choice if choice is not None else self.choose(A, T_enc, R, heuristic=Heuristic.GREEDY)

    def shrink(self, X, T_enc):
        k = 0
        if self.env["verbosity"]:
            self.log(f"start shrink for X={show_nz(X)}", verbosity=2)
            self.log(show_table(T_enc), verbosity=3)
        for i in X.nonzero()[0]:
            X[i] = False

            if self.env["verbosity"]:
                self.log(f"shrinking {i + INDEX} in {show_nz(X)}", verbosity=3)
                self.log(show_table(T_enc[:, X]), verbosity=3)
                self.log(show_table(T_enc[:, X].all(1, keepdims=True)), verbosity=3)

            # 1 0 | 0 all
            # 0 1 | 0
            # 1 0 | 0
            # 0 0 | 0
            # 1 1 | 1
            # 1,3,5 and 2,5 = 5

            # start shrink [1 2 3]
            # [[1 1 0]
            #  [0 0 1]
            #  [1 1 0]
            #  [0 0 0]
            #  [1 1 0]]
            # shrinking 1 in [1 2 3]
            # [[1 0]
            #  [0 1]
            #  [1 0]
            #  [0 0]
            #  [1 0]] # 'and' over all rows
            # [[0]
            #  [0]
            #  [0]
            #  [0]
            #  [0]]  # we find no row where b_2 and b_3 are both true, thus at most 1 is added to LHS
            # shrunk 1
            # shrinking 2 in [2 3]

            if T_enc[:, X].all(axis=1).any():
                if self.env["verbosity"]:
                    self.log(f"keep", i, f"because {show_nz(T_enc[:, X])} {T_enc[:, X]}", verbosity=3)
                X[i] = True  # keep i
            else:
                k += 1
                if self.env["verbosity"]:
                    self.log(f"shrunk", show(i), verbosity=3)
        return X, k

    def gencoverlift(self, S, C_enc, k, T_enc, A_enc, heuristic=Coverlift.INPUT):
        if self.env["verbosity"]:
            self.log("gencoverlift", verbosity=2)
            self.log(show_table(T_enc), verbosity=3)

        R = np.ones(len(T_enc), dtype=bool)

        # TODO add to alg: RS>=k, and X |= S

        def tight(R, RS):
            # TODO [peter] incorrect def in alg?
            return R & (RS == k)

        # RS = np.fromiter((sum(C_enc[i] * T_enc_r[i] for i in S) for T_enc_r in T_enc), dtype=float)
        # X = union(cols(T_enc, r) for r in R_tight.nonzero()[0])

        # TODO perf do we need RS for every row?
        # RS = (C_enc[S] * T_enc[:, S]).sum(axis=1)
        RS = (C_enc * T_enc).sum(axis=1)
        R_tight = tight(R, RS)
        X = (T_enc.T & R_tight).any(1)
        X = X | S

        if self.env["verbosity"]:
            self.log(f"S = {show_nz(S)}", verbosity=3)
            self.log("C_enc, k", C_enc, k, verbosity=3)
            self.log("RS (row slack?)", RS, verbosity=3)
            self.log(f"R_tight = {show_nz(R_tight)}", verbosity=3)
            self.log(f"X = {show_nz(X)}", verbosity=3)
            self.log(f"choices = {show_nz(~X)}", verbosity=3)
        # TODO [peter] C missing from alg

        i = 0

        # centre of mass heuristic
        if heuristic in (Coverlift.COM_MIN, Coverlift.COM_MAX):
            com = T_enc.sum(axis=0) / len(T_enc)

        while not X.all():
            assert (
                not self.env["example2"]
                or (
                    R_tight
                    == [
                        [False, True, False, False, True],
                        [False, True, True, False, True],
                        [True, True, True, False, True],
                    ][i]
                ).all()
            ), f"{i}; {R_tight}"

            assert (
                not self.env["example2"]
                or (
                    X
                    == [
                        [False, True, True, False, True, True, False, False, True, False],
                        [False, True, True, True, True, True, True, False, True, True],
                        [False, True, True, True, True, True, True, True, True, True],
                        # [i in s for i in range(len(T_enc.T))]
                        # for s in [{2, 3, 5, 6, 9}, {2, 3, 4, 5, 6, 7, 9, 10}, {2, 3, 4, 5, 6, 7, 8, 9, 10}]
                    ][i]
                ).all()
            ), f"{i}; {[i + 1 for i in X.nonzero()[0]]}"

            match heuristic:
                case _ if self.env["example2"]:
                    j = [3, 7, 0][i]
                case Coverlift.INPUT:
                    # j = np.nanargmax(np.where(~X, A_enc, np.nan))
                    j = np.argmax(~X)
                case Coverlift.COM_MIN:
                    j = np.nanargmin(com - (np.where(~X, A_enc, np.nan)))
                case Coverlift.COM_MAX:
                    j = np.nanargmax(com - (np.where(~X, A_enc, np.nan)))
            assert not X[j]

            if self.env["verbosity"]:
                self.log(f"Lift j = {show(j)}", verbosity=2)

            non_tight = (~R_tight) & T_enc[:, j]
            # print(f'NON TIGHT = {show_nz(non_tight)}', )

            if non_tight.any():
                a_j = np.min(k - RS[non_tight])
            else:
                a_j = k + 1  # infinite

            RS = RS + a_j * T_enc.T[j]
            N_tight = tight(~R_tight, RS)
            R_tight |= N_tight
            # 11 -> 0
            # 10 -> 1
            # 01 -> 0
            # 00 -> 0
            if self.env["verbosity"]:
                self.log("a_j", a_j, verbosity=3)
                self.log("RS (row slack?)", RS, verbosity=3)
                self.log(f"N_tight = {show_nz(N_tight)}", verbosity=3)
                self.log(f"R_tight = {show_nz(R_tight)}", verbosity=3)
                self.log(f"X = {show_nz(X)}", verbosity=3)
                # self.log("A", a_j * T_enc.T[j], verbosity=3)

            # assert not S[j] and C_enc[j] == 0, f"Already chosen {show(j)} in {show_nz(S)}"

            S[j] = True
            C_enc[j] = a_j

            X |= T_enc[N_tight, :].any(0)
            X[j] = True
            # X = T_enc[R_tight, :].any(0)  # slightly slower

            assert not self.env["example2"] or a_j == [2, 1, 1][i]

            assert not self.env["example2"] or (RS == [[1.0, 2.0, 0.0, 1.0, 2.0], [1.0, 2.0, 2.0, 1.0, 2.0], RS][i]).all(), (
                f"{i}; {RS}"
            )

            if self.env["verbosity"]:
                self.log(f"S = {show_nz(S)}", verbosity=3)
                self.log("C_enc, k", C_enc, k, verbosity=3)
                i += 1

            if self.env["debug"]:
                self.check_max_iterations(i)

            if R_tight.all():
                break

        return S, C_enc, k

    def show_cut(self, X, C_enc, k):
        if self.env["verbosity"]:
            self.log(
                f"X == {X}",
                verbosity=2,
            )
            self.log(
                f"cut == {' + '.join(f'{c} * b_{show(i)}' for i, c in enumerate(C_enc) if c)} <= {k}",
                indent=2,
                verbosity=2,
            )

    def explain(self, A_enc, T_enc, parts, frm=None):
        """The `explain_frac2` alg."""

        def assert_example(A, B):
            assert (A.nonzero()[0] == [i - 1 for i in B]).all(), A.nonzero()[0]

        # TODO convert T_enc to set of tuples?

        if self.env["verbosity"]:
            self.log(f"Explain frm={frm}", end="\n", verbosity=2)
            self.log("", np.astype(A_enc > 0.5, int) if frm == "MIPSOL" else A_enc, verbosity=3, indent=0)
            # self.log("", A_enc, verbosity=2, indent=0)
            # if frm == "MIPSOL":
            #     self.log("", np.array(assign_mipsol(A_enc)), verbosity=2, indent=0)
            self.log(np.astype(T_enc, int), verbosity=3, indent=0)
            self.log(f"p{np.astype(parts, int)}", verbosity=3, indent=0)
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

        assert len(A_enc) == len(T_enc.T)

        C_enc = np.zeros(len(A_enc), dtype=int)
        A_enc_pos = is_gt(A_enc, 0.0)

        self.env["cuts"].append({"from": frm})

        m = len(T_enc)  # number of cols
        # W = np.argwhere(is_ge(A_enc, 1.0))  # find a == 1.0
        W = is_ge(A_enc, 1.0)

        # W = set(i for i, a in enumerate(A_enc) if is_ge(a, 1.0))  # find a == 1.0
        # W = set(i for i, a in enumerate(A_enc) if is_eq(a, 1.0))  # find a == 1.0

        # F = set(i for i, a in enumerate(A_enc) if not is_integral(a))
        # W = set(W.flatten())
        # F = set(i for i in set(range(len(A_enc))) - W if is_gt(A_enc[i], 0.0))  # find 0 < a < 1
        # TODO check only if MIPNODE-OPT
        F = (~W) & A_enc_pos

        # F = set(np.argwhere(is_gt(np.delete(A_enc, W), 0.0)).flatten())  # much slower
        # F = set(np.argwhere(is_gt(A_enc, 0.0) & is_lt(A_enc, 1.0)).flatten())  # slightly slower

        # # assert not is_integer_solution(A_enc[i] for i in F), f"F should hold only fractional, but was: {F}"
        # X = set()  # columns added to cut
        # R = set(range(m))  # remaining columns

        # D = set(r for r in range(m) if cols(T_enc, r) <= W.union(F))  # difficult rows; either frac/whole
        # [   2 3   4     ]
        # [ 0 1 1 0 1 0 0 ]  Y
        # [ 0 0 1 0 1 0 0 ]  Y
        # [ 0 0 1 0 1 1 0 ]  N
        # WF = np.zeros(len(T_enc.T), dtype=bool)
        if self.env["verbosity"]:
            self.log(f"W = {show_nz(W)}", verbosity=3)
            self.log(f"F = {show_nz(F)}", verbosity=3)

        if F.any():
            # D are the difficult rows which only contains 1s for each W/F columns
            if self.env["verbosity"]:
                self.log(show_table((W | F) >= T_enc))
            D = ((W | F) >= T_enc).all(1)

            if self.env["verbosity"]:
                self.log(f"D = {show_nz(D)}", verbosity=3)
            # then U are rows fractional columns which are not in D
            U = F > ~((~T_enc[D, :]).all(0))  # tricky

            if self.env["verbosity"]:
                self.log(f"U = {show_nz(U)}", verbosity=3)

            if none(U):
                if self.env["verbosity"]:
                    self.log("unexplainable, because U is empty", indent=2, verbosity=3)
                return True
            else:
                R = np.ones(m, dtype=bool)
                choice = self.choose(U, T_enc, R, parts, A_enc, heuristic=self.env["heuristic"])

                if self.env["example_frac"]:
                    assert_example(W, [6])
                    assert_example(F, [2, 3, 8, 9])
                    assert_example(D, [2])
                    assert_example(U, [2, 8])
                    choice = 2 - 1

                # TODO [peter] should be T_hat[choice]?
                choices = np.ones(len(T_enc.T), dtype=bool)
                choices[parts[choice] == parts] = False
                X = np.zeros(len(T_enc.T), dtype=bool)
                X[choice] = True
                C_enc[choice] = 1
                R = T_enc[:, choice]
                k = 0

                if self.env["example_frac"]:
                    assert_example(X, [2])
                    assert k == 0
                    assert_example(R, [1, 5])
                    assert parts[choice] == 1 - 1

                if self.env["verbosity"]:
                    self.log(f"intially chosen from U; {show(choice)}", verbosity=3, indent=self.indent + 2)
                    self.log(f"choices {show_nz(choices)}", verbosity=3, indent=self.indent + 2)
                    self.log(f"R ({R.sum()}) = {show_nz(R)}", verbosity=3, indent=self.indent + 2)
                    self.log(f"X {show_nz(X)}", verbosity=3, indent=self.indent + 2)
                    self.log(f"C_enc {C_enc}", verbosity=3, indent=self.indent + 2)

        else:
            R = np.ones(m, dtype=bool)
            X = np.zeros(len(T_enc.T), dtype=bool)
            choices = np.ones(len(T_enc.T), dtype=bool)
            k = -1

        if self.env["verbosity"]:
            self.log(f"R ({R.sum()})", verbosity=2, indent=self.indent + 2)

        for iteration in itertools.count():
            if none(R):
                break

            neg_choices = choices & ~A_enc_pos
            # make_pos_choice = frm == "MIPNODE-OPT" or R.sum() >= self.env["negatives"] or not neg_choices.any()
            make_pos_choice = R.sum() >= self.env["negatives"] or not neg_choices.any()
            choice = self.choose(
                # neg_choices if neg_choices.any() else choices,
                (choices & A_enc_pos) if make_pos_choice else neg_choices,
                T_enc,
                R,
                parts if make_pos_choice else np.arange(len(T_enc.T)),
                A_enc,
                # heuristic=self.env["heuristic"] if make_pos_choice else Heuristic.INPUT,
                heuristic=self.env["heuristic"],
                make_pos_choice=make_pos_choice,
            )
            # choice, is_pos = [(1, True), (4, False)][iteration]
            is_pos = A_enc_pos[choice]

            if choice is None:
                # this can happen only if assignment is feasible
                raise Exception("No choices left for ", frm)

            l_parts = parts[choice] == parts
            assert not C_enc[choice], f"chosen {choice}"
            if is_pos:
                C_ = l_parts & A_enc_pos
                choices[l_parts] = False
                R = R & T_enc[:, C_].any(1)
                X |= C_
                C_enc[X] = 1
                k += 1
            else:
                assert False
                C_ = choice
                choices[choice] = False
                R = R & (~T_enc[:, choice])
                X[choice] = True
                C_enc[choice] = -1

            self.check_max_iterations(iteration)

            if self.env["verbosity"]:
                self.log(f"Ak = {A_enc[X].sum()} < {k}", verbosity=3)
                self.log(
                    f"chosen {'pos' if is_pos else 'neg'} col. {show_ind(choice)} of part {parts[choice]}",
                    verbosity=3,
                    indent=self.indent + 2,
                )
                self.log(f"choices {show_nz(choices)}", verbosity=3, indent=self.indent + 2)
                self.log(f"C_ {show_nz(C_)}", verbosity=3, indent=self.indent + 2)
                self.log(f"is_pos {is_pos} {choice}", verbosity=3, indent=self.indent + 2)
                if F.any():
                    self.log("FRAC", frm, verbosity=2, indent=self.indent + 2)
                self.log(f"R choice={choice} -> ({R.sum()})", verbosity=2, indent=self.indent + 2)
                self.log(f"= {show_nz(R)}", verbosity=3, indent=self.indent + 3)
                self.log(f"X {show_nz(X)}", verbosity=3, indent=self.indent + 2)
                self.log(f"C_enc {C_enc}", verbosity=3, indent=self.indent + 2)
            # [     1   1     ]
            # [ 0 1 1 0 0 0 0 ]  Y
            # [ 0 0 0 0 1 0 0 ]  Y
            # [ 0 0 0 0 1 1 0 ]  Y
            # [ 0 0 0 1 0 1 0 ]  N
            # [ 0 0 0 0 0 1 0 ]  N
            #       -   -
            # [     1   0     ]  Y
            # [     0   1     ]  Y
            # [     0   1     ]  Y
            # [     0   0     ]  N
            # [     0   0     ]  N
            # keep only rows which are in R and which have an 1 where A_enc has a 1

        if self.env["verbosity"]:
            self.log(f"by explanation of size ({sum(X)})", verbosity=2)
            self.log(show_nz(X), verbosity=3)
            self.log("C_enc", C_enc, verbosity=3)
            self.env["cuts"][-1]["cut"] = X.copy()
            assert C_enc[X].all()

        self.show_cut(X, C_enc, k)

        if self.env["shrink"]:
            X_shrunk, shrunk = self.shrink(X, T_enc)
            C_enc[~X_shrunk] = 0
            k -= shrunk
            if self.env["verbosity"] and shrunk:
                self.log(f"shrunk by {shrunk}", indent=2, verbosity=2)
                self.env["cuts"][-1] = {
                    **self.env["cuts"][-1],
                    "shrunk": shrunk,
                    "cut": X_shrunk,
                }
            self.env["cuts"][-1]["shrunk"] = shrunk
            self.show_cut(X, C_enc, k)

        if self.env["coverlift"]:
            if self.env["verbosity"]:
                Xl = X.sum()
            X, C_enc, k = self.gencoverlift(X, C_enc, k, T_enc, A_enc, heuristic=self.env["coverlift"])
            if self.env["verbosity"]:
                self.log("coverlift added ", X.sum() - Xl, verbosity=2)

            self.show_cut(X, C_enc, k)

        self.env["cuts"][-1]["size"] = len(X)

        return X, C_enc, k

    def check_max_iterations(self, i):
        # Loop termination for debug purposes
        if self.env["max_iterations"] is not None:
            assert i <= self.env["max_iterations"], "Out of iterations"

    def solution_callback_inner(self, x_enc_a, frm):
        feasible = True  # assume feasible
        for expr in self._explain_assignment(x_enc_a, frm=frm):
            if isinstance(expr, Comparison) and expr.name == "<=":
                feasible = False  # any expr means not feasible
                assert isinstance(expr.args[0], Operator)
                expr, k = expr.args
                yield expr, k
            elif is_true_cst(expr):
                continue
            elif is_false_cst(expr):
                if self.env["verbosity"]:
                    self.log("explanation raised infeasible")
                raise Infeasible
            else:
                assert False, f"Unsupported expl: {expr}"

        self.check_max_iterations(len(self.env["cuts"]))
        if feasible and frm == "MIPSOL":
            if self.env["verbosity"]:
                self.log("found feasible", verbosity=2)
            self.env["found_feasible"] = feasible

    def get_solution_callback(self):
        from gurobipy import GRB

        all_xs = {x_enc_i for x_enc, _, _, _ in self.tables for x_enc_i in x_enc}

        def solution_callback(what, where):
            time_cb = time.time()

            try:
                x_enc_a = None
                frm = None

                def cbGetVal(cpm_var, cbGet):
                    # if isinstance(cpm_var, NegBoolView):
                    #     return 1.0 - cbGet(self.solver_var(~cpm_var))
                    # return cbGet(self.solver_var(cpm_var))
                    # shortcut some stuff from solver_var ; we know the var exists
                    return (
                        (1.0 - cbGet(self._varmap[cpm_var._bv]))
                        if isinstance(cpm_var, NegBoolView)
                        else cbGet(self._varmap[cpm_var])
                    )

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
                    case _:
                        return

                for expr, k in self.solution_callback_inner(x_enc_a, frm):
                    cut = self._make_numexpr(expr) <= k
                    what.cbLazy(cut)
            except Exception as e:
                self.native_model._callback_exception = e
                what.terminate()
            finally:
                time_cb = time.time() - time_cb
                if self.env["verbosity"]:
                    self.log(f"end callback, dt = {time_cb}", verbosity=4)
                self.env["time_cb"] += time_cb

        return solution_callback

    def explanation_to_expr(self, explanation, A_enc, X_enc, T_enc, frm):

        if is_true_cst(explanation):
            expr = explanation
        else:
            (X, C_enc, k) = explanation
            expr = cp.sum(C_enc[X] * X_enc[X]) <= k
            if self.env["verbosity"]:
                self.show_cut(X, C_enc, k)
                self.log(f"  == {expr}", indent=2, verbosity=2)

        if isinstance(expr, (bool, np.bool_)):
            expr = cp.BoolVal(expr)

        if self.env["debug"]:
            self.env["cuts"][-1]["expr"] = expr

        # if self.env["debug"]:
        #     self.check_explanation(expr, X_enc, A_enc, T_enc, table)

        return expr

    def _explain_assignment(self, x_enc_a, frm=None):
        # If fully integer, we can check if the tables are feasible yet
        if self.env["verbosity"]:
            self.log("EXPLAIN", frm, x_enc_a, verbosity=2)

        for i, (X_enc, T_enc, parts, table) in enumerate(self.tables, start=INDEX):
            # A_enc = np.array([x_enc_a[x_enc_i] for x_enc_i in X_enc])
            if frm == "MIPSOL":
                A_enc = np.fromiter((x_enc_a[x_enc_i] > 0.5 for x_enc_i in X_enc), dtype=bool)
                is_integer = True
            else:
                A_enc = np.fromiter((x_enc_a[x_enc_i] for x_enc_i in X_enc), dtype=float)
                is_integer = False
                if is_integral(A_enc).all():
                    A_enc = A_enc > 0.5
                    assert A_enc.dtype == bool
                    is_integer = True

            # TODO figure out when can be skipped
            if is_integer and (T_enc == A_enc).all(1).any():
                if self.env["verbosity"]:
                    self.log(
                        f"table {i}/{len(self.tables)} feasible: ({show_nz((T_enc == A_enc).all(1))})",
                        verbosity=3,
                    )
                    self.log(
                        f"by {A_enc}\n\n{np.astype(T_enc, int)}",
                        verbosity=3,
                        indent=2,
                    )
                continue
            elif self.env["verbosity"]:
                self.log(
                    f"table {i}/{len(self.tables)} INfeasible",
                    verbosity=2,
                )
                self.log(
                    f"by {show_table(A_enc)}\n\n{np.astype(A_enc, int)}",
                    verbosity=2,
                    indent=2,
                )

            try:
                # encode assignment
                if self.env["verbosity"]:
                    self.log(" ", np.array(X_enc), verbosity=3, indent=0)
                explanation = self.explain(A_enc, T_enc, parts, frm=frm)

                if self.env["verbosity"]:
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
                if explanation is True:
                    assert frm == "MIPNODE-OPT"
                    yield True
                elif explanation:
                    expr = self.explanation_to_expr(explanation, A_enc, X_enc, T_enc, frm)
                    yield expr
                    if self.env["debug"] or self.env["checked"]:
                        self.check_explanation(expr, X_enc, A_enc, T_enc, frm)
                elif frm == "MIPSOL":  # unsat
                    raise Infeasible
            except Infeasible:
                raise Infeasible
            except Exception as e:
                failure = (X_enc, A_enc, T_enc, parts, frm, self.env)
                path = f"/tmp/failed_cut.pkl"
                with open(path, "wb") as f:
                    print("Saved failed cut/env under ", path)
                    pickle.dump(failure, f)
                raise e

    def add(self, cons):
        if not isinstance(cons, list):
            cons = [cons]
        return super().add(cons)

    __add__ = add  # avoid redirect in superclass

    def solutions_checker(self, time_limit=None):
        time_limit = CHECKER_TIME_LIMIT if time_limit is None else time_limit
        return cp.solvers.utils.solutions(
            self.env["checker"],
            X=sorted(self.user_vars, key=lambda x: x.name),
            projected_solution_limit=None,
            time_limit=time_limit,
        )[1]

    def check_explanation(self, expr, X_enc, A_enc, T_enc, frm):
        # A_enc = A_enc > 0.5
        for x, a in zip(X_enc, A_enc):
            x._value = a

        def value(expr):
            if is_true_cst(expr):
                return True
            (expr,) = only_positive_bv([expr])
            ws, xs, k = terms(expr)  # sum(ws*xs) <= k
            lhs = sum(w * x.value() for w, x in zip(ws, xs))
            return bool(is_le(lhs, k))  # np -> python bool

        case = f"The explanation\n\n{expr}\n== {value(expr)}\n\n from assignment {frm}\n\n{show_assignment(X_enc)}\n\nfor A_enc:\n\n{show_table(A_enc)}\n\n for tables:\n\n{show_table(T_enc)}\n\n  "

        if not is_true_cst(expr):
            assert value(expr) is False, f"Did not cut off assignment:\n\n{case}"

        for i, T_enc_i in enumerate(T_enc):
            for x_i, a_i_j in zip(X_enc, T_enc_i):
                x_i._value = a_i_j
            assert value(expr) is True, f"Cut off row {show(i)} for case:\n\n{case}\n\n{show_table(T_enc_i)}"

        if self.env["checked"]:
            # if self.env["checker"] and self.env["feasible"]:
            repeated = expr in self.env["checker"].constraints
            # self.env["checker"] += expr

            # nsols = self.env["checker"].solveAll()
            # assert nsols >= 3, (
            #     f"{nsols}:The {expr} for {A_enc} made model unsat\n\n{T_enc}\n\n{self.env['checker']}\n\n"
            # )

            actual_solutions = self.solutions_checker()

            def show_assignments(As):
                return "\n".join(repr(a) for a in As)

            expected_solutions = self.env["expected_solutions"]
            remaining = without(actual_solutions, expected_solutions)
            assert len(np.unique(actual_solutions, axis=0)) == len(actual_solutions)

            self.env["cuts"][-1]["remain"] = remaining
            self.env["cuts"][-1]["n_sols"] = len(actual_solutions)
            if self.env["verbosity"]:
                self.log(f"Expected ({len(expected_solutions)}) (model w/ table)", verbosity=2)
                self.log(expected_solutions, verbosity=3)
                self.log(f"Actual ({len(actual_solutions)}) (model w/o table but with lazy constraints)", verbosity=2)
                self.log(actual_solutions, verbosity=3)
                self.log(f"Remaining non-solutions to cut ({len(remaining)})", verbosity=2)
                self.log(remaining, verbosity=3)
            if len(self.env["cuts"]) >= 2:
                removed = without(self.env["cuts"][-2]["remain"], remaining)
                if self.env["verbosity"]:
                    self.log(
                        "A",
                        self.user_vars,
                        tuple(x.value() for x in sorted(self.user_vars, key=lambda x: x.name) if x.value() is not None),
                        verbosity=3,
                    )
                    self.log(f"REMOVED {len(removed)}", verbosity=2)
                    self.log(removed, verbosity=3)

            strength = (
                len(self.env["cuts"][-2]["remain"]) - len(self.env["cuts"][-1]["remain"])
                if len(self.env["cuts"]) >= 2
                else len(self.env["remain"]) - len(remaining)
            )
            self.env["cuts"][-1]["strength"] = strength
            if self.env["verbosity"]:
                self.log(self.env["checker"], verbosity=4)
                self.log(f"EXPECTED ({len(expected_solutions)})", verbosity=2)
                self.log(expected_solutions, verbosity=3, indent=2)
                self.log(f"ACTUAL ({len(actual_solutions)})", verbosity=2)
                self.log(actual_solutions, verbosity=3, indent=2)
                self.log(f"TO REMOVE ({len(remaining)})", verbosity=2)
                self.log(remaining, verbosity=3)
                self.log(f"STRENGTH == {strength}", verbosity=2)

            # assert repeated or strength
            assert strength

            assert len(expected_solutions) <= len(actual_solutions), (
                f"Missing sols:\n\n{show_assignments(expected_solutions)}\n\n {show_assignments(actual_solutions)}\n\n{self.env['checker']}"
            )

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

    # @mem_profile
    def solve(self, time_limit=None, solution_callback=None, env=None, **kwargs):
        """
        Call the gurobi solver with cut generation
        """

        if env is not None:
            self.env = {**self.env, **env}

        self.env["cuts"] = []
        self.env["time_cb"] = 0.0

        assert solution_callback is None, "For now, no solution_callback in `CPM_lazy_gurobi`"

        if self.env["verbosity"]:
            self.log("Solving.. ")

        try:
            if self.env["checked"]:
                for iteration in itertools.count():
                    hassol = self.env["checker"].solve(**kwargs)

                    if not hassol:
                        break
                    all_xs = {x_enc_i for x_enc, _, _, _ in self.tables for x_enc_i in x_enc}
                    x_enc_a = {x_enc_i: x_enc_i.value() for x_enc_i in all_xs}

                    self.check_max_iterations(iteration)
                    # self.solution_callback_inner(x_enc_a, "MIPSOL")
                    assert all(x.value() is not None for x in all_xs), f"Has sol but no value {all_xs}"
                    for expr, k in self.solution_callback_inner(x_enc_a, "MIPSOL"):
                        self.env["checker"] += [expr <= k]
                    if self.env["found_feasible"]:
                        break
            else:
                hassol = super().solve(
                    solution_callback=self.get_solution_callback(),
                    time_limit=time_limit,
                    **kwargs,
                )

            if hassol:
                # if self.env["debug"]:
                #     assert self.env["feasible"]
                if not self.env["found_feasible"]:
                    if self.env["verbosity"]:
                        self.log("WARN: not found feas")
                # assert self.env["found_feasible"]

            if hasattr(self.native_model, "_callback_exception"):
                raise self.native_model._callback_exception or Exception(
                    "Gurobi was interrupted (perhaps the solution callback called model.terminate())"
                )

        except Infeasible:
            hassol = False
        except Exception as e:
            if self.env["debug"]:
                with open("/tmp/failed_model.pkl", "wb") as f:
                    pickle.dump(self.env["model"], f)
            self.env["verbosity"] = 4
            try:
                self.stats()
            except Exception as e_:
                if self.env["verbosity"]:
                    self.log("Also exception during stats:", e_)
            raise e

        # TODO recheck https://or.stackexchange.com/questions/12591/ensure-gurobi-uses-callback-on-all-feasible-solutions It looks like if the solution at the end of the root node is integer, gurobi doesn't pass through callbacks for fractional solutions.

        return hassol

    def get_x_encs(self, X):
        return [x_enc_i for x in X for x_enc_i in self.ivarmap[x]._xs]

    def transform_(self, cpm_expr):
        if cpm_expr.name != "table" or len(cpm_expr.args[1]) <= self.env["cutoff"]:
            cons = super().transform(cpm_expr)
        else:
            if len(set(cpm_expr.args[0])) < len(cpm_expr.args[0]):
                cpm_expr = normalize_table(cpm_expr)
            X, T = cpm_expr.args

            # only check after normalize, since normalize may remove all rows
            if not len(T):
                return [cp.BoolVal(False)]
            assert len(set(X)) == len(X), f"Dup. int vars in table for {cpm_expr}"

            X_enc, T_enc, cons = self.encode_table_constraint(X, T)
            X_enc = np.fromiter((x_enc_i for x_enc in X_enc for x_enc_i in x_enc._xs), _BoolVarImpl)

            if self.env["verbosity"]:
                self.log("X =", ", ".join(f"{x} in {x.lb}..{x.ub}" for x in X), verbosity=3)
                self.log("T =", verbosity=3)
                self.log(T, verbosity=3)
                self.log("T_enc =", verbosity=3)
                self.log(np.astype(T_enc, int), verbosity=3)
            assert len(set(X_enc)) == len(X_enc), f"Dup. bool vars in table for {cpm_expr}"

            parts = np.fromiter((i for i, x in enumerate(X) for _ in range(dom_size(x))), dtype=int)
            self.tables.append((X_enc, T_enc, parts, cpm_expr))

        if self.env["checked"]:
            for c in cons:
                self.env["checker"] += c

        return cons

    def transform(self, cpm_expressions):
        return [cpm_con for cpm_expr in cpm_expressions for cpm_con in self.transform_(cpm_expr)]

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
