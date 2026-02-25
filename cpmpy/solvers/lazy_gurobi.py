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
from cpmpy.expressions.utils import is_true_cst, is_false_cst, show_assignment, is_bool
from cpmpy.solvers.gurobi import CPM_gurobi, Feature
from cpmpy.expressions.variables import NegBoolView, _BoolVarImpl
from cpmpy.transformations.linearize import only_positive_bv

CHECKER_TIME_LIMIT = None


def none(A):
    return not A.any()


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


def show_table(T_enc, index=INDEX, full=False):
    result = np.astype(T_enc, int) if T_enc.dtype == bool else T_enc
    if full:
        with np.printoptions(**DEBUG_NP_PRINTOPTIONS):
            return np.array2string(result)
    return result


def show_nz(A, index=INDEX):
    nz = np.atleast_1d(A).nonzero()[0] + 1
    return nz[:10]


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
    INPUT_LAST = "input_last"
    COM_MIN = "com_min"
    COM_MAX = "com_max"

    def __bool__(self):
        return self is not Coverlift.No


class TableData:
    """Data structure for an encoded table constraint."""

    def __init__(self, X_enc, T_enc, parts, cpm_expr, solver):
        self.X_enc = X_enc  # numpy array of encoded boolean variables
        self.T_enc = T_enc  # numpy array of the encoded table (boolean matrix)
        self.parts = parts  # numpy array: which original variable each encoded var corresponds to
        self.cpm_expr = cpm_expr  # the original CPMpy table constraint expression
        self.solver = solver  # reference to the solver instance

        if self.env["negatives"]:
            self.T_enc = np.concatenate([self.T_enc, ~self.T_enc], axis=1, dtype=bool)
            assert np.issubdtype(self.T_enc.dtype, np.bool), self.T_enc.dtype
            # TODO !!
            # self.parts = np.concatenate([self.parts, self.parts])
            assert np.issubdtype(self.parts.dtype, np.integer), self.parts.dtype
            self.parts = np.concatenate([self.parts, -self.parts], dtype=int) if self.parts.size else self.parts
            assert np.issubdtype(self.parts.dtype, np.integer), self.parts.dtype
            # TODO concat
            self.densities = self.T_enc.mean(axis=0)

    @property
    def env(self):
        return self.solver.env

    def is_feasible(self, A_enc):
        T_enc = self.T_enc[:, : len(self.X_enc)] if self.env["negatives"] else self.T_enc
        return (T_enc == A_enc).all(1).any()

    def cols(self):
        return (len(self.T_enc.T) // 2) if self.env["negatives"] else len(self.T_enc.T)

    def get_expr(self, X, C_enc, k):
        return cp.sum(C_enc[X] * self.X_enc[X]) <= k

    def show_cut(self, X, C_enc, k, verbosity=2):
        terms = []
        for i, c in enumerate(C_enc):
            if c:
                if self.solver.env["negatives"] and i >= self.cols():
                    terms.append(f"{c} * (1 - b_{show(i - self.cols())})")
                else:
                    terms.append(f"{c} * b_{show(i)}")
        cut_str = " + ".join(terms)
        self.solver.log(
            f"cut == {cut_str} <= {k}",
            indent=2,
            verbosity=verbosity,
        )

    def choose(self, choices, R, A_enc, heuristic=Heuristic.GREEDY):
        T_enc = self.T_enc
        parts = self.parts

        # if R.sum() <= self.env["negatives"]:
        #     choices[: (len(T_enc.T) // 2)] = False

        # print(show_table(densities), "dens")

        if self.solver.env["verbosity"]:
            self.solver.log(f"Choose from {show_nz(choices)} to allow remaining rows R=\n{show_nz(R)}", verbosity=3)
            self.solver.log(show_table(T_enc[R, :]), "T_enc[R,:]", verbosity=3)
            self.solver.log("", parts, "parts", verbosity=3)
            self.solver.log("", show_table(A_enc), "A_enc", verbosity=3)
            self.solver.log("", choices.astype(int), "choices", verbosity=3)

        if none(choices):
            return None
        match heuristic:
            case Heuristic.INPUT:
                return np.argmax(choices)
            case Heuristic.GREEDY:
                parts_ = np.add.accumulate(np.unique_counts(parts[choices]).counts)
                parts_ -= parts_[0]
                # map back to the right part index
                if self.solver.env["verbosity"]:
                    self.solver.log(show_table(T_enc[np.ix_(R, choices)]), "T_enc[R,choices]", verbosity=3)
                # print(show_table(parts_), "parts_")

                # heur = np.bitwise_or.reduceat(
                #     # get only the relevant rows and columns
                #     T_enc[np.ix_(R, choices)],
                #     # for the columns of each part
                #     parts_,
                #     # see if there is any 1 in the row
                #     axis=1,
                # )

                # how many additional rows will be removed (high is good)
                H = (~T_enc[np.ix_(R, choices)]).sum(0)
                if self.solver.env["negatives"]:
                    densities = self.densities
                    # upper bound on how many additional non-rows will be allowd (high is bad)
                    # B = (1 - densities[choices]) * np.pow(2, len(C_enc) - 1 - (C_enc != 0).sum())

                    # the pos cols will have low density, the neg cols have high density
                    B = densities[choices]
                    if self.solver.env["verbosity"]:
                        self.solver.log(show_table(densities[choices]), "^DDD", verbosity=3)
                        self.solver.log("HHH", H, verbosity=2)
                        self.solver.log("B", B, verbosity=2)
                        self.solver.log("H", H, verbosity=2)
                        self.solver.log("H", H * (B.max() + 1) + B, verbosity=2)
                    # TODO native?
                    h = np.argmax(H * (B.max() + 1) + B)

                    # # upper bound on how many additional non-rows will be allowd (high is bad)
                    # B = (1 - densities[choices]) * np.pow(2, len(C_enc) - 1 - (C_enc != 0).sum())
                    # if self.solver.env["verbosity"]:
                    #     self.solver.log("H", H, verbosity=3)
                    #     self.solver.log("B", B, verbosity=3)

                else:
                    h = np.argmax(H)

                if self.solver.env["verbosity"]:
                    self.solver.log("H", H, verbosity=3)

                return choices.nonzero()[0][h]

            case Heuristic.REDUCE:
                assert False
                # choice = min(
                #     (i for i in A if R.intersection(rows(T_enc, i)) != R),
                #     key=lambda i: len(rows(T_enc, i)),
                #     default=None,  # TODO [?] check this edge-case
                # )
                # return choice if choice is not None else self.solver.choose(A, T_enc, R, heuristic=Heuristic.GREEDY)

    def explain(self, A_enc, frm=None):
        """The `explain_frac2` alg."""
        T_enc = self.T_enc
        parts = self.parts
        X_enc = self.X_enc
        solver = self.solver

        if self.env["negatives"]:
            if np.issubdtype(A_enc.dtype, np.bool):
                A_enc = np.concatenate([A_enc, ~A_enc], dtype=A_enc.dtype)
            else:
                A_enc = np.concatenate([A_enc, 1.0 - A_enc], dtype=A_enc.dtype)

        def assert_example(A, B):
            assert (A.nonzero()[0] == [i - 1 for i in B]).all(), A.nonzero()[0]

        # TODO convert T_enc to set of tuples?

        if self.env["verbosity"]:
            solver.log(f"Explain frm={frm}", end="\n", verbosity=2)
            solver.log("", np.astype(A_enc > 0.5, int) if frm == "MIPSOL" else A_enc, "A_enc", verbosity=3, indent=0)
            solver.log(np.astype(T_enc, int), "T_enc", verbosity=3, indent=0)
            solver.log(f"p{np.astype(parts, int)}", "parts", verbosity=3, indent=0)

        # assert len(A_enc) == len(T_enc.T)


        self.env["cuts"].append({"from": frm})

        m = len(T_enc)  # number of cols
        W = solver.is_ge(A_enc, 1.0)

        A_enc_pos = solver.is_gt(A_enc, 0.0)
        F = (~W) & A_enc_pos

        if self.env["verbosity"]:
            solver.log(f"W = {show_nz(W)}", verbosity=3)
            solver.log(f"F = {show_nz(F)}", verbosity=3)

        if F.any():
            # D are the difficult rows which only contains 1s for each W/F columns
            if self.env["verbosity"]:
                solver.log(show_table((W | F) >= T_enc))
            D = ((W | F) >= T_enc).all(1)

            if self.env["verbosity"]:
                solver.log(f"D = {show_nz(D)}", verbosity=3)
            # then U are rows fractional columns which are not in D
            U = F > ~((~T_enc[D, :]).all(0))  # tricky

            if self.env["verbosity"]:
                solver.log(f"U = {show_nz(U)}", verbosity=3)

            if none(U):
                if self.env["verbosity"]:
                    solver.log("unexplainable, because U is empty", indent=2, verbosity=3)
                return True
            else:
                R = np.ones(m, dtype=bool)
                choice = self.choose(U & A_enc_pos, R, A_enc, heuristic=self.env["heuristic"])

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
                R = T_enc[:, choice]
                k = 0

                if self.env["example_frac"]:
                    assert_example(X, [2])
                    assert k == 0
                    assert_example(R, [1, 5])
                    assert parts[choice] == 1

                if self.env["verbosity"]:
                    solver.log(f"intially chosen from U; {show(choice)}", verbosity=3, indent=solver.indent + 2)

        else:
            R = np.ones(m, dtype=bool)
            X = np.zeros(len(T_enc.T), dtype=bool)
            choices = np.ones(len(T_enc.T), dtype=bool)
            k = -1

        if self.env["verbosity"]:
            solver.log(f"R ({R.sum()})", verbosity=2, indent=solver.indent + 2)
            solver.log(f"X {show_nz(X)}", verbosity=3, indent=solver.indent + 2)
            solver.log(f"choices = {show_nz(choices)}", verbosity=3, indent=solver.indent + 2)

        for iteration in itertools.count():
            if none(R):
                break

            choice = self.choose(choices & A_enc_pos, R, A_enc, heuristic=self.env["heuristic"])
            assert not X[choice]

            if choice is None:
                # this can happen only if assignment is feasible
                raise Exception("No choices left for ", frm)

            is_pos = choice < self.cols()

            part = parts[choice]

            choice_parts = part == parts
            R = R & T_enc[:, choice]
            X[choice] = 1
            k += 1

            # remove from choices
            if self.solver.env["negatives"]:
                if part > 0:
                    chosen = choice_parts | (parts == -part)
                    # self.solver.log("rm from choices", show_nz(chosen))
                    choices[chosen] = False
                else:
                    choices[choice] = False
                    choices[choice - self.cols()] = False
                    # if only one choice remains in this part, remove it too
                    remaining = np.flatnonzero(choices & choice_parts)
                    if len(remaining) == 1:
                        self.log("ONE REMAIN")
                        choices[remaining[0]] = False
                        choices[remaining[0] - self.cols()] = False
            else:
                choices[choice_parts] = False

            solver.check_max_iterations(iteration)

            if self.env["verbosity"]:
                solver.log(f"Ak = {A_enc[X].sum()} < {k}", verbosity=3)
                solver.log(
                    f"chosen {'pos' if is_pos else 'neg'} col. {show_ind(choice)} of part {parts[choice]}",
                    verbosity=3,
                    indent=solver.indent + 2,
                )
                solver.log(f"remaining choices {show_nz(choices)}", verbosity=3, indent=solver.indent + 2)
                if F.any():
                    solver.log("FRAC", frm, verbosity=2, indent=solver.indent + 2)
                solver.log(
                    f"R choice = {'+' if is_pos else '-'}b_{show(choice)} -> ({R.sum()})", verbosity=2, indent=solver.indent + 2
                )
                solver.log(f"= {show_nz(R)}", verbosity=3, indent=solver.indent + 3)
                solver.log(f"X {show_nz(X)}", verbosity=3, indent=solver.indent + 2)

        C_enc = np.zeros(len(X), dtype=int)
        C_enc[X] = 1

        if self.env["verbosity"]:
            solver.log(f"by explanation of size ({sum(X)})", verbosity=2)
            solver.log(show_nz(X), verbosity=3)
            solver.log("C_enc", C_enc, verbosity=3)
            self.show_cut(X, C_enc, k, verbosity=1)

        # if self.env["debug"] and X_enc is not None:
        #     expr = self.get_expr(X, C_enc, k)
        #     solver.show_cut(X, C_enc, k)
        #     solver.check_explanation(expr, X_enc, A_enc, T_enc, frm)

        if self.env["coverlift"]:
            if self.env["verbosity"]:
                Xl = X.sum()
            X, C_enc, k = self.gencoverlift(X, C_enc, k, A_enc, heuristic=self.env["coverlift"], frm=frm)
            if self.env["verbosity"]:
                solver.log("coverlift added ", X.sum() - Xl, "of", Xl, verbosity=2)
                self.show_cut(X, C_enc, k)

        if self.env["shrink"]:
            C_enc, k = solver.shrink(X, C_enc, k, T_enc, A_enc, parts)

        if self.env["verbosity"]:
            self.show_cut(X, C_enc, k, verbosity=1)

        self.env["cuts"][-1]["size"] = len(X)

        return self.revert_cut(X, C_enc, k)

    def revert_cut(self, S, C_enc, k):
        if self.solver.env["negatives"]:
            n = len(S) // 2
            return S[:n] | S[n:], C_enc[:n] - C_enc[n:], k - sum(C_enc[n:])
        else:
            return S, C_enc, k

    def gencoverlift(self, S, C_enc, k, A_enc, heuristic=Coverlift.INPUT, frm=None):
        solver = self.solver
        T_enc, parts = self.T_enc, self.parts

        if self.env["verbosity"]:
            solver.log("gencoverlift", verbosity=2)
            solver.log("T_enc", T_enc.shape, verbosity=2)
            solver.log(show_table(T_enc, full=self.env["verbosity"] == 3), verbosity=3)
            solver.log("", show_table(parts), "parts", verbosity=3)

        R = np.ones(len(T_enc), dtype=bool)

        def tight(R, RS):
            return R & (RS == k)

        # Compute the upper bound for each row
        RS = (C_enc * T_enc).sum(axis=1)

        # All rows where upper bound == k are tight
        R_tight = tight(R, RS)

        # We cannot select a column if it has a 1 in any tight row
        X = T_enc[R_tight, :].any(0)

        if self.env["verbosity"]:
            solver.log(f"Coverlift from {show_nz(~X)}", verbosity=3)
            solver.log(show_table(T_enc[R, :]), verbosity=3)
            solver.log("", show_table(C_enc), "C_enc <=", k, verbosity=3)
            solver.log("", show_table(parts), "parts", verbosity=3)

            solver.log("RS (row slack?)", show_table(RS), verbosity=3)
            solver.log(show_table(T_enc[:, C_enc != 0]), verbosity=3)
            solver.log("", show_table(C_enc[C_enc != 0]), verbosity=3)
            solver.log("", show_table(parts[C_enc != 0]), "parts", verbosity=3)
            solver.log(f"R_tight = {show_nz(R_tight)}", verbosity=3)
            solver.log(f"X = {show_nz(X)}", verbosity=3)
            solver.log(f"S_ = {show_nz(S)}", verbosity=3)
            solver.log(f"k_ = {k}", verbosity=3)

        # centre of mass heuristic
        if heuristic in (Coverlift.COM_MIN, Coverlift.COM_MAX):
            com = T_enc.sum(axis=0) / len(T_enc)
            if self.env["verbosity"]:
                solver.log("COM", T_enc.sum(axis=0), verbosity=3)
                solver.log("COM", com, verbosity=2)

        for iteration in itertools.count(1):
            if self.env["verbosity"]:
                solver.log(f"coverlift #{iteration}", verbosity=3)
            if X.all():
                break

            assert (
                not self.env["example2"]
                or (
                    R_tight
                    == [
                        [False, True, False, False, True],
                        [False, True, True, False, True],
                        [True, True, True, False, True],
                    ][iteration]
                ).all()
            ), f"{iteration}; {R_tight}"

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
                    ][iteration]
                ).all()
            ), f"{iteration}; {[i + 1 for i in X.nonzero()[0]]}"

            # The choices are the unselected columns
            choices = np.where(X, np.nan, X)

            match heuristic:
                case _ if self.env["example2"]:
                    j = [3, 7, 0][iteration]
                case Coverlift.INPUT:
                    j = np.nanargmax(choices)
                case Coverlift.INPUT_LAST:
                    j = len(choices) - 1 - np.nanargmax(choices[::-1])
                case Coverlift.COM_MIN:
                    direction = com - np.where(X, np.nan, A_enc)
                    j = np.nanargmin(direction)
                case Coverlift.COM_MAX:
                    direction = com - np.where(X, np.nan, A_enc)
                    if self.env["verbosity"]:
                        solver.log("direction", direction, verbosity=2)
                    j = np.nanargmax(direction)

            assert not X[j]

            # any pure 0/1 columns have been filtered out, a min always exists
            a_j = np.min(k - RS[(~R_tight) & T_enc[:, j]])

            if self.env["verbosity"]:
                solver.log(f"Lift j={j} : {a_j}*b_{show(j)}", verbosity=2)

            # assert C_enc[j] == 0
            # Calculate new row upper bounds just for the added column
            RS = RS + a_j * T_enc[:, j]

            # Find and update newly tight rows
            N_tight = tight(~R_tight, RS)
            R_tight |= N_tight

            # Add var and coefficient to cut
            S[j] = True
            C_enc[j] = a_j

            X |= T_enc[N_tight, :].any(0)

            if self.env["verbosity"]:
                solver.log(f"S = {show_nz(S)}", verbosity=3)
                solver.log("", show_table(C_enc), "C_enc <=", k, verbosity=3)
                solver.log("a_j", a_j, verbosity=3)
                solver.log("RS (row slack?)", show_table(RS), verbosity=3)
                # solver.log(f"N_tight = {show_nz(N_tight)}", verbosity=3)
                solver.log(f"R_tight = {show_nz(R_tight)}", verbosity=3)
                solver.log(f"X = {show_nz(X)}", verbosity=3)
                solver.log(f"choices = {show_nz(~X)}", verbosity=3)
                # solver.log("A", a_j * T_enc_.T[j], verbosity=3)

            assert not self.env["example2"] or a_j == [2, 1, 1][iteration]

            assert (
                not self.env["example2"] or (RS == [[1.0, 2.0, 0.0, 1.0, 2.0], [1.0, 2.0, 2.0, 1.0, 2.0], RS][iteration]).all()
            ), f"{iteration}; {RS}"

            solver.check_max_iterations(iteration)

            assert X[j], f"{show(j)} not chosen in {X}"
            if self.env["verbosity"]:
                self.show_cut(S, C_enc, k)

        return S, C_enc, k


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
        if self.env["verbosity"] >= 4:
            self.native_model.Params.LogFile = "/tmp/gurobi.log"
            self.native_model.Params.OutputFlag = 1
            self.native_model.write("/tmp/gurobi.lp")

        if self.env["checked"] and cpm_model is not None:
            if self.env["verbosity"]:
                self.log("CHECKED: solve model to determine expected feasibility")
            self.env["user_vars"] = tuple(sorted(self.user_vars, key=lambda x: x.name))
            self.env["feasible"] = cpm_model.solve()
            self.env["model"] = cpm_model
            _, self.env["expected_solutions"] = cp.solvers.utils.solutions(
                cpm_model,
                X=self.env["user_vars"],
                projected_solution_limit=None,
                time_limit=CHECKER_TIME_LIMIT,
            )
            self.env["remain"] = without(self.solutions_checker(), self.env["expected_solutions"])
            if self.env["verbosity"]:
                self.log("SOLS", len(self.env["expected_solutions"]))
                self.log("TO REMOVE\n", self.env["remain"], verbosity=3)

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
            "avg_power": (sum(s["power"] for s in cuts) / len(cuts)) if cuts and self.env["checked"] else None,
        }

    def shrink(self, X, C_enc, k, T_enc, A_enc, parts):
        if self.env["verbosity"]:
            self.log(f"start shrink for X={show_nz(X)}, k={k}", verbosity=2)
            self.log(show_table(T_enc), verbosity=3)
            self.log("", show_table(C_enc), "C_enc", verbosity=3)
            # self.log("", show_table(A_enc), "A_enc", verbosity=3)

        shrunk = 0
        for i in np.unique(parts):
            C_enc_ = C_enc[parts == i]
            C_enc[parts == i] = 0

            if self.env["verbosity"]:
                self.log("i", i, show_table(parts == i), verbosity=3)
                self.log("C", show_table(C_enc), verbosity=3)
                self.log("A", np.sum(C_enc * T_enc, axis=1), verbosity=3)
            if np.all(np.sum(C_enc * T_enc, axis=1) <= k - 1):
                shrunk += 1
                k -= 1
                if self.env["verbosity"]:
                    self.log(f"shrinking {i + INDEX} in {show_nz(X)}", verbosity=3)
            else:
                C_enc[parts == i] = C_enc_
        if self.env["verbosity"] and shrunk:
            self.log(f"shrunk by {shrunk}", verbosity=1)
        return C_enc, k

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
                if self.env["verbosity"]:
                    self.env["cuts"][-1]["cut"] = (expr, k)
                yield expr, k
                # break # TODO could lead to fewer cuts, but then expensive part of callback is repeated often
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

        all_xs = {x_enc_i for table in self.tables for x_enc_i in table.X_enc}

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
                self.log(f"  == {expr}", indent=2, verbosity=1)

        if isinstance(expr, (bool, np.bool_)):
            expr = cp.BoolVal(expr)

        if self.env["debug"]:
            self.env["cuts"][-1]["expr"] = expr

        # if self.env["debug"]:
        #     self.check_explanation(expr, X_enc, A_enc, T_enc, table)

        return expr

    def _vary(self, a=None, b=None):
        if a is None:
            a = lambda: False
        if b is None:
            b = lambda: True
        return a() if self.env.get("variant", 0) == 0 else b()

    def _explain_assignment(self, x_enc_a, frm=None):
        # If fully integer, we can check if the tables are feasible yet
        if self.env["verbosity"]:
            self.log("EXPLAIN", frm, verbosity=2)
            self.log("Full sol", x_enc_a, verbosity=4)

        for i, tbl in enumerate(self.tables, start=INDEX):
            X_enc, T_enc, parts = tbl.X_enc, tbl.T_enc, tbl.parts
            # A_enc = np.array([x_enc_a[x_enc_i] for x_enc_i in X_enc])
            if frm == "MIPSOL":
                A_enc = np.fromiter((x_enc_a[x_enc_i] > 0.5 for x_enc_i in X_enc), dtype=bool)
                is_integer = True
            else:
                A_enc = np.fromiter((x_enc_a[x_enc_i] for x_enc_i in X_enc), dtype=float)
                is_integer = False
                if self.is_integral(A_enc).all():
                    A_enc = A_enc > 0.5
                    assert A_enc.dtype == bool
                    is_integer = True

            # TODO figure out when can be skipped
            if is_integer and tbl.is_feasible(A_enc):
                if self.env["verbosity"]:
                    self.log(
                        f"table {i}/{len(self.tables)} feasible:",
                        verbosity=4,
                    )
                    self.log(
                        f"by {show_table(A_enc)}\n\n{show_table(T_enc)}",
                        verbosity=4,
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
                explanation = tbl.explain(A_enc, frm=frm)

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
                elif is_integer:  # unsat
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
            X=self.env["user_vars"],
            projected_solution_limit=None,
            time_limit=time_limit,
            sorted=True,
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
            return bool(self.is_le(lhs, k))  # np -> python bool

        case = f"The explanation\n\n{expr}\n== {value(expr)}\n\n from assignment {frm}\n\n{show_assignment(X_enc)}\n\nfor A_enc:\n\n{show_table(A_enc)}\n\n for tables:\n\n{show_table(T_enc)}\n\n  "

        if not is_true_cst(expr):
            assert value(expr) is False, f"Did not cut off assignment:\n\n{case}"

        for i, T_enc_i in enumerate(T_enc):
            for x_i, a_i_j in zip(X_enc, T_enc_i):
                x_i._value = a_i_j
            assert value(expr) is True, f"Cut off row {show(i)} for case:\n\n{case}\n\n{show_table(T_enc_i)}"

        if "cut" not in self.env["cuts"][-1]:
            return

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
            self.env["cuts"][-1]["power"] = strength / len(self.env["cuts"][-1]["cut"][0].args)
            self.env["remain"] = remaining
            if self.env["verbosity"]:
                self.log(
                    f"Remaining non-solutions to cut ({len(remaining)}, STR={self.env['cuts'][-1]['strength']}, PWR={self.env['cuts'][-1]['power']})",
                    verbosity=1,
                )
                self.log(remaining, verbosity=3)
                self.log(self.env["checker"], verbosity=4)
                self.log(f"EXPECTED ({len(expected_solutions)})", verbosity=2)
                self.log(expected_solutions, verbosity=3, indent=2)
                self.log(f"ACTUAL ({len(actual_solutions)})", verbosity=2)
                self.log(actual_solutions, verbosity=3, indent=2)
                self.log(f"TO REMOVE ({len(remaining)})", verbosity=2)
                self.log(remaining, verbosity=3)

            # assert repeated or strength
            assert strength or len(self.tables) > 1

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
            dt = time.time()
            if self.env["checked"]:
                expected_solution = min(self.env["expected_solutions"].tolist(), default=None)
                for iteration in itertools.count():
                    done = not len(self.env["remain"])

                    if time_limit is not None and time.time() - dt > time_limit:
                        raise TimeoutError

                    sol = min(self.env["remain"].tolist(), default=expected_solution)
                    hassol = sol is not None

                    if sol is None:
                        break

                    for x, v in zip(self.env["user_vars"], sol):
                        x._value = v

                    all_xs = {x_enc_i for tbl in self.tables for x_enc_i in tbl.X_enc}
                    for expr, lit in self._csemap.items():
                        lit._value = expr.value()
                    x_enc_a = {x_enc_i: x_enc_i.value() for x_enc_i in all_xs}

                    self.check_max_iterations(iteration)
                    assert all(x.value() is not None for x in all_xs), f"Has sol but no value {all_xs}"
                    for expr, k in self.solution_callback_inner(x_enc_a, "MIPSOL"):
                        self.env["checker"] += [expr <= k]

                    if done:
                        break
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
        if is_bool(cpm_expr) or cpm_expr.name != "table" or len(cpm_expr.args[1]) <= self.env["cutoff"]:
            cons = super().transform(cpm_expr)
        else:
            if len(set(cpm_expr.args[0])) < len(cpm_expr.args[0]):
                cpm_expr = normalize_table(cpm_expr)
            X, T = cpm_expr.args

            # only check after normalize, since normalize may remove all rows
            if not len(T):
                return [cp.BoolVal(False)]
            assert len(set(X)) == len(X), f"Dup. int vars in table for {cpm_expr}"

            X_enc, T_enc, cons, parts = self.encode_table_constraint(X, T)

            if self.env["verbosity"]:
                self.log("X =", ", ".join(f"{x} in {x.lb}..{x.ub}" for x in X), verbosity=3)
                self.log("T =", verbosity=3)
                self.log(T, verbosity=3)
                self.log("T_enc =", verbosity=3)
                self.log(show_table(T_enc), verbosity=3)
                self.log("X_enc =", show_table(X_enc), verbosity=3)
            assert len(set(X_enc)) == len(X_enc), f"Dup. bool vars in table for {cpm_expr}"
            self.tables.append(TableData(X_enc, T_enc, parts, cpm_expr, self))

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
