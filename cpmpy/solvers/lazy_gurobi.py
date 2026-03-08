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
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.core import Comparison, Operator
from cpmpy.expressions.utils import is_true_cst, is_false_cst, show_assignment, is_bool
from cpmpy.solvers.gurobi import CPM_gurobi, Feature
from cpmpy.expressions.variables import NegBoolView, _BoolVarImpl
from cpmpy.transformations.linearize import only_positive_bv

from line_profiler import profile as line_profile

CHECKER_TIME_LIMIT = None

STRICT_CUTS = False


def none(A):
    return not A.any()


def lshift(arr, n):
    """Left shift array elements by n positions, filling with False."""
    result = np.zeros(len(arr), dtype=bool)
    if n < len(arr):
        result[:-n] = arr[n:] if n else arr
    return result


def rshift(arr, n):
    """Right shift array elements by n positions, filling with False."""
    result = np.zeros(len(arr), dtype=bool)
    if n < len(arr):
        result[n:] = arr[:-n] if n else arr
    return result


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
        assert cpm_expr.name in ("<=", "<")
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
    "formatter": {"float_kind": "{:.7f}".format},
    "suppress": True,  # avoid scientific notation
}


def show_table(T_enc, index=INDEX, full=False):
    result = np.astype(T_enc, int) if T_enc.dtype == bool else T_enc
    if full:
        with np.printoptions(**DEBUG_NP_PRINTOPTIONS):
            return np.array2string(result)
    return result


def show_nz(A, index=INDEX):
    return np.flatnonzero(np.atleast_1d(A)) + 1


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
        self.X_enc = cp.cpm_array(X_enc)
        self.cpm_expr = cpm_expr
        self.solver = solver
        self.n_parts = parts.max(initial=0)

        # Pre-compute which variables are negated (for efficient value retrieval)
        self.is_negated = np.array([isinstance(x, NegBoolView) for x in X_enc], dtype=bool)

        # var_indices will be set by solver._finalize_tables() before solve
        self.var_indices = None

        if self.env["negatives"]:
            # Pre-allocate T_enc with space for negatives
            n_rows, n_cols = T_enc.shape
            self.T_enc = np.empty((n_rows, n_cols * 2), dtype=bool)
            self.T_enc[:, :n_cols] = T_enc
            self.T_enc[:, n_cols:] = ~T_enc

            # Pre-allocate parts with space for negatives
            if parts.size:
                self.parts = np.empty(n_cols * 2, dtype=int)
                self.parts[:n_cols] = parts
                self.parts[n_cols:] = np.arange(n_cols) + parts.max() + 1
            else:
                self.parts = parts

            assert np.issubdtype(self.parts.dtype, np.integer), self.parts.dtype
            self.densities = self.T_enc.mean(axis=0)
            self.part_counts = np.unique_counts(self.parts)
            if self.parts.size:
                self.part_count_lookup = np.zeros(self.parts.max() + 1, dtype=int)
                self.part_count_lookup[self.part_counts.values] = self.part_counts.counts
            else:
                self.part_count_lookup = np.array([], dtype=int)
        else:
            self.T_enc = T_enc
            self.parts = parts

    @property
    def env(self):
        return self.solver.env

    def is_feasible(self, A_enc):
        T_enc = self.T_enc[:, : len(self.X_enc)] if self.env["negatives"] else self.T_enc
        return (T_enc == A_enc).all(1).any()

    def cols(self):
        return (len(self.T_enc.T) // 2) if self.env["negatives"] else len(self.T_enc.T)

    def get_expr(self, X, C_enc, k):
        if self.solver.env["negatives"] and len(X) > self.cols():
            X, C_enc, k = self.revert_cut(X, C_enc, k)
        return cp.sum(C_enc[X] * self.X_enc[X]) < k if STRICT_CUTS else cp.sum(C_enc[X] * self.X_enc[X]) <= k

    def show_cut(self, X, C_enc, k, frm=None, A_enc=None, verbosity=2, check=2):
        assert self.env["verbosity"]

        terms = []
        for i, c in enumerate(C_enc):
            if c:
                a = "" if A_enc is None else f"[={A_enc[i]}]"
                part = self.parts[i]
                j = i - np.argmax(self.parts == part)
                if self.is_pos(part):
                    terms.append(f"{c} * b_{show(i)}_{part}_{show(j)} {a}")
                else:
                    c_ = self.counterchoice(i)
                    terms.append(f"{c} * (1 - b_{show(i)}_{self.parts[c_]}_{show(j)} {a})")
        cut_str = " + ".join(terms)

        cut_str = f"{cut_str} < {k}" if STRICT_CUTS else f"{cut_str} <= {k}"
        self.solver.log(f"cut {frm if frm is not None else ''} == {cut_str}", indent=2, verbosity=verbosity)
        self.solver.log(f"C_enc = {C_enc}", indent=2, verbosity=verbosity)
        # self.solver.log(self.X_enc.value(), indent=2, verbosity=3)
        expr = self.get_expr(X, C_enc, k)
        self.solver.log(expr, indent=2, verbosity=3)
        # assert C_enc.sum() < 5

        if check > 0 and A_enc is not None:
            expr = self.get_expr(X, C_enc, k)
            if self.env["debug"] or self.env["checked"]:
                self.check_explanation(
                    X,
                    C_enc,
                    k,
                    A_enc,
                    frm,
                    check=check,
                )

    @line_profile
    def choose(self, choices, R, A_enc, single_choice=False, heuristic=Heuristic.GREEDY):
        T_enc = self.T_enc
        parts = self.parts

        if self.solver.env["verbosity"]:
            self.solver.log(
                f"Choose from vars {show_nz(choices)} / parts {np.unique(parts[choices])} to allow remaining rows R=\n{show_nz(R)}",
                verbosity=3,
            )
            # self.solver.log(self.cpm_expr)
            self.solver.log(
                show_table(T_enc[np.ix_(R, choices)], full=self.env["verbosity"] >= 4), "T_enc[R,choices]", verbosity=3
            )
            self.solver.log("", parts[choices], "parts[choices]", verbosity=3)
            self.solver.log("", show_table(A_enc[choices]), "A_enc[choices]", verbosity=3)

        if none(choices):
            return None, None
        match heuristic:
            case Heuristic.INPUT:
                return np.argmax(choices), None
            case Heuristic.GREEDY:
                if single_choice:
                    reindex = np.flatnonzero(choices)
                    indices = np.arange(len(reindex))
                else:
                    # Segment start indices for reduceat: [0, count[0], count[0]+count[1], ...]
                    choice_parts = parts[choices]
                    counts = np.unique_counts(choice_parts)
                    indices = counts.counts.cumsum()
                    indices = np.concatenate([[0], indices[:-1]])
                    reindex = counts.values

                H = np.bitwise_or.reduceat(
                    # get only the relevant rows and columns
                    T_enc[np.ix_(R, choices)],
                    # for the columns of each part
                    indices,
                    # see if there is any 1 in the col
                    axis=1,
                ).sum(0)

                if self.solver.env["verbosity"]:
                    self.solver.log(show_table(H), f"H (of {R.sum()})", verbosity=2)

                # how many rows will be kept (low is good)
                # H = T_enc[np.ix_(R, choices)].sum(0)  # number of 1's
                if self.solver.env["negatives"] and not single_choice:
                    densities = self.densities

                    # TODO never picks negative terms?

                    B = 1 / (1 + self.part_count_lookup[counts.values])
                    # B = np.zeros(len(H))
                    if self.env["debug"]:
                        assert ((0 < B) & (B < 1)).all(), f"B values out of range: {B}"

                    if self.solver.env["verbosity"]:
                        # self.solver.log(show_table(densities[choices]), "^DDD", verbosity=3)
                        self.solver.log(show_table(reindex), "parts", verbosity=2)
                        self.solver.log("B", B, verbosity=2)
                        # self.solver.log("HB", HB, verbosity=2)
                        # self.solver.log("C", show_nz(choices), verbosity=2)

                    # B = 0

                    HB = H - B  # lex obj since 0<B<1 (no constant cols)
                    h = np.argmin(HB)

                    if self.env["debug"]:
                        assert H.min() < R.sum()

                    # if self.solver._vary():
                    #     # log(a) / (2^(C-1) * (1 - d))  == log₂(a) - (C-1) - log₂(1 - d)
                    #     h = np.argmin(np.log2(H) - (len(A_enc) - 1) - np.log2(1 - self.densities[choices]))
                    # else:
                    #     densities = self.densities
                    #     # upper bound on how many additional non-rows will be allowd (high is bad)
                    #     # B = (1 - densities[choices]) * np.pow(2, len(C_enc) - 1 - (C_enc != 0).sum())
                    #     # the pos cols will have low density, the neg cols have high density
                    #     B = densities[choices]
                    #     h = np.argmax(H * (B.max() + 1) + B)
                    #         # a / (2^(C-1) * b)

                    # TODO native?

                    # # upper bound on how many additional non-rows will be allowd (high is bad)
                    # B = (1 - densities[choices]) * np.pow(2, len(C_enc) - 1 - (C_enc != 0).sum())
                    # if self.solver.env["verbosity"]:
                    #     self.solver.log("H", H, verbosity=3)
                    #     self.solver.log("B", B, verbosity=3)
                else:
                    h = np.argmin(H)

                if self.solver.env["verbosity"]:
                    self.solver.log("chooes #h from H", show(h), H, verbosity=3)
                    self.solver.log("reindex arr", reindex, verbosity=3)
                    self.solver.log("reindex", h, reindex[h], choices[parts == reindex[h]], verbosity=3)

                expected_R = H[h]
                # h is index into parts, map back to column index
                # chosen_part = counts.values[h]
                # return np.flatnonzero(choices & (parts == chosen_part))[0]
                # return parts == reindex[h] if not single_choice else (np.arange(len(parts)) == reindex[h]), expected_R
                return choices & (parts == reindex[h]) if not single_choice else reindex[h], expected_R

            case Heuristic.REDUCE:
                assert False
                # choice = min(
                #     (i for i in A if R.intersection(rows(T_enc, i)) != R),
                #     key=lambda i: len(rows(T_enc, i)),
                #     default=None,  # TODO [?] check this edge-case
                # )
                # return choice if choice is not None else self.solver.choose(A, T_enc, R, heuristic=Heuristic.GREEDY)

    def is_pos(self, part):
        return part <= self.n_parts

    def counterpart(self, part):
        assert self.env["negatives"]
        return part + self.n_parts if self.is_pos(part) else part - self.n_parts
        # TODO faster with mod?

    def counterchoice(self, choice):
        assert self.env["negatives"]
        if isinstance(choice, (int, np.integer)):
            is_pos = choice < self.cols()
            return choice + self.cols() if is_pos else choice - self.cols()
        else:
            is_pos = np.argmax(choice) < self.cols()
            return rshift(choice, self.cols()) if is_pos else lshift(choice, self.cols())

    @line_profile
    def initial(self, A_enc, frm=None, is_integer=None):
        """The `explain_frac2` alg."""
        T_enc = self.T_enc
        parts = self.parts
        solver = self.solver
        A_enc = A_enc.astype(float)

        def get_k():
            return len(np.unique(self.parts[X])) - 1

        def assert_example(A, B):
            assert (A.nonzero()[0] == [i - 1 for i in B]).all(), A.nonzero()[0]

        # TODO convert T_enc to set of tuples?

        # ```
        # [[1 0 0 0 1 0 0 0 1 0 0 0 0 1 1 1 0 1 1 1 0 1 1 1]
        #  [0 1 0 0 0 1 0 0 0 1 0 0 1 0 1 1 1 0 1 1 1 0 1 1]
        #  [0 0 1 0 0 0 1 0 0 0 1 0 1 1 0 1 1 1 0 1 1 1 0 1]
        #  [0 0 0 1 0 0 0 1 0 0 0 1 1 1 1 0 1 1 1 0 1 1 1 0]]
        #  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0] C_enc <= 1
        #  [0 1 0 0 _ 0 _ _ _ _ _ _ _ _ _ _ _ 1 _ _ _ _ _ _] C_enc <= 1
        #  [0 1 0 0 _ 0 _ _ _ 1 _ _ _ _ _ _ _ 1 _ _ _ _ _ _] C_enc <= 1
        #  [1 1 1 1 2 2 2 2 3 3 3 3-1-1-1-1-2-2-2-2-3-3-3-3] parts
        # ```
        # cut == 1 * b_2 + 1 * (1 - b_6) <= 1
        # == sum([1, -1] * [BV[x[0] == 2], BV[x[1] == 2]]) <= 0
        # x[0] = 2 -> x[1] = 2
        # cut == 2 * b_2 + 1 * (1 - b_6) + 1 * (1 - b_10) <= 2 ?

        # cut == 1 * b_2 + 1 * b_9 <= 1
        # cut == 1 * b_2 + 1 * b_3 + 1 * b_4 + 1 * b_9 <= 1
        # cut == 1 * b_2 + 1 * b_3 + 1 * b_4 + 1 * b_9 <= 1
        #   == sum([BV[x[0] == 2], BV[x[0] == 3], BV[x[0] == 4], BV[x[2] == 1]]) <= 1
        # diffs ; took x[2] i/o x[1]
        # diffs ; sets x[2]!= 1 (WEAKER???)
        # diffs ; adds all x!=2 \/ x!=3 \/ x!=4
        # (x[0]!=2 /\ x[0]!=3 /\ x[0]!=4) -> x[2] != 1
        # x[0]=1 -> x[2] != 1
        # x[0]=1 + x[0]=2 + x[0]=2 + x[0]=2 == 1

        if self.env["verbosity"]:
            solver.log(f"Explain frm={frm}/is_integer={is_integer}", end="\n", verbosity=1)
            solver.log(
                "",
                np.astype(A_enc > 0.5, int) if frm == "MIPSOL" else A_enc,
                "A_enc ",
                verbosity=3,
                indent=0,
            )
            solver.log(np.astype(T_enc, int), "T_enc", verbosity=3, indent=0)
            solver.log(f"p{np.astype(parts, int)}", "parts", verbosity=3, indent=0)

        # if self.env["debug"]:
        #     for p in np.unique(parts):
        #         assert A_enc[parts == p].any(), f"Zero part {p} in {A_enc}, {parts}, {self.X_enc}"
        #         assert solver.is_ge(A_enc[parts], 0.0).any(), A_enc

        self.env["cuts"].append({"from": frm})

        m = len(T_enc)  # number of rows
        cols = len(T_enc.T)
        W = solver.is_ge(A_enc, 1.0)

        A_enc_pos = solver.is_gt(A_enc, 0.0)
        F = (~W) & A_enc_pos

        if self.env["verbosity"]:
            solver.log(f"W = {show_nz(W)}", verbosity=3)
            solver.log(f"F = {show_nz(F)}", verbosity=3)

        C_enc = np.zeros(cols, dtype=int)

        if F.any():
            # D are the difficult rows which only contain 1s for each W/F columns
            if self.env["verbosity"]:
                solver.log(show_table((W | F) >= T_enc), verbosity=3)
            D = ((W | F) >= T_enc).all(1)

            if self.env["verbosity"]:
                solver.log(f"D = {show_nz(D)}", verbosity=3)
            # then U are fractional columns which are not in D
            U = F > ~((~T_enc[D, :]).all(0))  # tricky

            if self.env["verbosity"]:
                solver.log(f"U = {show_nz(U)}", verbosity=3)
                solver.log(f"p_u = {parts[U]}", verbosity=3)

            if none(U):
                if self.env["verbosity"]:
                    solver.log("unexplainable, because U is empty", indent=2, verbosity=3)
                return True
            else:
                # TODO remove
                R = np.ones(m, dtype=bool)

                # i = choice
                choices = U & A_enc_pos
                choice, expected_R = self.choose(choices, R, A_enc, single_choice=True, heuristic=self.env["heuristic"])
                part = np.unique(parts[choice])
                assert len(part) == 1, f"Expected to pick a unqiue part but got {part}"
                part = part[0]

                # part, expected_R = self.choose(choices, R, A_enc, single_choice=False, heuristic=self.env["heuristic"])

                # part = parts[choice]
                # choice_parts = parts == part

                if self.env["example_frac"]:
                    assert_example(W, [6])
                    assert_example(F, [2, 3, 8, 9])
                    assert_example(D, [2])
                    assert_example(U, [2, 8])
                    choice = 2 - 1

                # V <- {p(i)}
                choices = np.ones(cols, dtype=bool)
                choices[parts == part] = False  # DON'T choose from current part

                # X <- {i}
                X = np.zeros(cols, dtype=bool)
                X[choice] = True
                # C_enc[choice] = 1.0 / A_enc[choice]
                C_enc[choice] = 1

                # R <- T_i
                # R = T_enc[:, choice]
                R = R & T_enc[:, choice]

                if self.solver.env["negatives"]:
                    # Exclude the counterchoices
                    counterchoice = self.counterchoice(choice)
                    if self.env["verbosity"]:
                        self.solver.log("rm counter", show(counterchoice))
                    choices[counterchoice] = False
                    for counterpart in np.unique(parts[counterchoice]):
                        choices[parts == counterpart] = False

                    # choices[parts == self.counterpart(part)] = False

                if self.env["verbosity"]:
                    solver.log(
                        f"intially chosen from U; add/choice {show_nz(choice)}",
                        verbosity=3,
                        indent=solver.indent + 2,
                    )
                    assert R.sum() == expected_R, f"Number of remaining rows is |R|={R.sum()}, but expected {expected_R}"
                    # solver.log(f"part = {part}", verbosity=3, indent=solver.indent + 2)
                    # solver.log(f"choices = {show_nz(choices)}", verbosity=3, indent=solver.indent + 2)
                    solver.log(f"V = {np.unique(parts[X])}")
                    self.show_cut(X, C_enc, get_k(), A_enc=A_enc, verbosity=1, frm=frm, check=1)
                    # solver.log(f"C_enc = {C_enc}", verbosity=3, indent=solver.indent + 2)

                if self.env["example_frac"]:
                    assert_example(X, [2])
                    # assert k == 0
                    assert_example(R, [1, 5])
                    assert parts[choice] == 1

        else:
            R = np.ones(m, dtype=bool)
            X = np.zeros(len(T_enc.T), dtype=bool)
            choices = np.ones(len(T_enc.T), dtype=bool)

        if self.env["verbosity"]:
            solver.log(f"R ({R.sum()})", verbosity=2, indent=solver.indent + 2)
            solver.log(show_nz(R), verbosity=3, indent=solver.indent + 2)
            solver.log(show_table(T_enc[R, :]), verbosity=3, indent=solver.indent + 2)
            solver.log(f"X {show_nz(X)}", verbosity=3, indent=solver.indent + 2)
            solver.log(f"choices = {show_nz(choices)}", verbosity=3, indent=solver.indent + 2)
            solver.log(f"A_enc_pos = {show_table(A_enc_pos)}", verbosity=3, indent=solver.indent + 2)
            solver.log(f"A_enc_pos = {show_nz(A_enc_pos)}", verbosity=3, indent=solver.indent + 2)

        for iteration in itertools.count():
            if none(R):
                break

            add, expected_R = self.choose(choices & A_enc_pos, R, A_enc, heuristic=self.env["heuristic"])
            part = np.unique(parts[add])

            # part = parts[add]
            # if part is None:
            #     assert False, "no part"
            #     return True

            # assert choice is not None and not X[part], choice

            # is_pos = self.is_pos(part)

            # choice_parts = parts == part  # l

            choices[add] = False

            # choices[choice] = False
            # added = choice_parts & A_enc_pos

            # if self.solver.env["negatives"]:
            #     # Exclude the counterchoices
            #     counterchoices = self.counterchoice(add, part)
            #     if self.env["verbosity"]:
            #         self.solver.log("rm counters: ", show_nz(counterchoices))
            #     choices[counterchoices] = False
            #     # choices[parts == self.counterpart(part)] = False

            if self.solver.env["negatives"]:
                # Exclude the counterchoices
                counterchoice = self.counterchoice(add)
                if self.env["verbosity"]:
                    self.solver.log(
                        "rm counter", show_nz(add), show_nz(counterchoice), parts[add], np.unique(parts[counterchoice])
                    )
                choices[counterchoice] = False
                for counterpart in np.unique(parts[counterchoice]):
                    choices[parts == counterpart] = False

            # k += A_enc[added].sum()

            X[add] = True
            # C_enc[choice] = 1.0 / A_enc[choice]
            C_enc[add] = 1

            if self.env["debug"]:
                R_ = R.sum()

            R = R & T_enc[:, add].any(1)
            # R = R & T_enc[:, choice]

            solver.check_max_iterations(iteration)

            if self.env["verbosity"]:
                # self.solver.log("c ==", (choice_parts & A_enc_pos).sum(), verbosity=2)
                solver.log(f"V = {np.unique(parts[X])}")
                solver.log(
                    f"chosen {'POS' if self.is_pos(part) else 'NEG'} part {part}",
                    verbosity=3,
                    indent=solver.indent + 2,
                )
                # solver.log(f"added columns {show_nz(added)} = {A_enc[added]}", verbosity=3, indent=solver.indent + 2)
                # solver.log(f"C_enc = {C_enc}", verbosity=3, indent=solver.indent + 2)

                solver.log(f"remaining choices {show_nz(choices)}", verbosity=3, indent=solver.indent + 2)
                solver.log(show_table(T_enc[R, :]), "T_enc[R,:]", verbosity=3)
                if F.any():
                    solver.log("FRAC", frm, verbosity=2, indent=solver.indent + 2)
                solver.log(f"R ({R.sum()}) ({expected_R})", verbosity=1, indent=solver.indent + 3)
                solver.log(f"  = {show_nz(R)} ", verbosity=3, indent=solver.indent + 3)
                solver.log(f"X {show_nz(X)}", verbosity=3, indent=solver.indent + 2)
                solver.log(f"k = {get_k()}", verbosity=3, indent=solver.indent + 2)

                self.show_cut(X, C_enc, get_k(), A_enc=A_enc, verbosity=1, frm=frm, check=1)

            # assert self.is_pos(part)

            if self.env["debug"]:
                assert self.solver.is_eq(A_enc[add].sum(), 1), (
                    f"Choices {show_nz(add)} added to {A_enc[add].sum()} for counterexample {A_enc}"
                )
                assert R.sum() == expected_R, f"Number of remaining rows is |R|={R.sum()}, but expected {expected_R}"
                assert R.sum() < R_, f"Did not reduce rows, curr={R.sum()}, prev={R_}"

        # C_enc = np.zeros(cols, dtype=int)
        # C_enc[X] = 1

        if self.env["verbosity"]:
            solver.log(f"by explanation of size ({sum(X)})", verbosity=2)
            solver.log(show_nz(X), verbosity=3)
            # solver.log("C_enc", C_enc, verbosity=3)
            self.show_cut(X, C_enc, get_k(), A_enc=A_enc, verbosity=1, frm=frm)
        return X, C_enc, get_k()

    @line_profile
    def explain(self, A_enc, frm=None, is_integer=None):
        T_enc = self.T_enc
        parts = self.parts
        solver = self.solver

        explanation = self.initial(A_enc, frm=frm, is_integer=is_integer)
        if not isinstance(explanation, tuple):
            return explanation
        X, C_enc, k = explanation

        if self.env["shrink"]:
            C_enc, k = solver.shrink(X, C_enc, k, T_enc, A_enc, parts)

        # if self.env["debug"] and X_enc is not None:
        #     expr = self.get_expr(X, C_enc, k)
        #     solver.show_cut(X, C_enc, k)
        #     solver.check_explanation(expr, X_enc, A_enc, T_enc, frm)

        if self.env["coverlift"]:
            if self.env["verbosity"]:
                Xl = X.sum()
            X, C_enc, k = self.gencoverlift(X, C_enc, k, A_enc, heuristic=self.env["coverlift"], frm=frm)
            if self.env["verbosity"]:
                add = X.sum() - Xl
                solver.log("coverlift added ", add, "of", Xl, verbosity=2)
                if add:
                    self.show_cut(X, C_enc, k, frm=frm, A_enc=A_enc)

        if self.env["verbosity"]:
            self.show_cut(X, C_enc, k, frm=frm, A_enc=A_enc, verbosity=1)

        self.env["cuts"][-1]["size"] = len(X)

        return self.revert_cut(X, C_enc, k)

    def revert_cut(self, X, C_enc, k):
        if self.solver.env["negatives"]:
            n = self.cols()
            X_, C_enc_, k_ = X[:n] | X[n:], C_enc[:n] - C_enc[n:], k - sum(C_enc[n:])
            if self.env["debug"]:
                assert len(X[:n]) == len(X[n:])
                duplicates = np.flatnonzero(X[:n] & X[n:])
                assert len(duplicates) == 0, (
                    f"Both b_{show(duplicates)} and (1 - b_{show(duplicates)}) in cut for {show_nz(X)} / {self.cols()}\n\n{self.get_expr(X_, C_enc_, k_)}\n\n{show_nz(X[:n])} \n {show_nz(X[n:]) + self.cols()} "
                )
            if self.env["verbosity"]:
                self.solver.log("pos/neg = ", X[:n].sum(), X[n:].sum(), verbosity=2)
            return X_, C_enc_, k_
        else:
            return X, C_enc, k

    @line_profile
    def gencoverlift(self, S, C_enc, k, A_enc, heuristic=Coverlift.INPUT, frm=None):
        solver = self.solver
        T_enc, parts = self.T_enc, self.parts

        if self.env["verbosity"]:
            solver.log("gencoverlift", verbosity=2)
            solver.log("T_enc", T_enc.shape, verbosity=2)
            solver.log(show_table(T_enc, full=self.env["verbosity"] == 3), verbosity=3)
            solver.log("", show_table(parts), "parts", verbosity=3)
            solver.log(f"k = {k}", verbosity=3)

        # Compute the upper bound for each row
        RS = (C_enc * T_enc).sum(axis=1)

        def tight(RS, k):
            return RS == k
            return solver.is_ge(RS, k)  # TODO if C_enc/k become fractional

        # All rows where upper bound == k are tight
        # TODO account for  self.solver.is_eq(RS, k)
        R_tight = tight(RS, k)

        # TODO earlier return

        # We cannot select a column if it has a 1 in any tight row
        X = T_enc[R_tight, :].any(0)
        if self.env["negatives"]:
            X[self.cols() :] |= S[: self.cols()]
            X[: self.cols()] |= S[self.cols() :]

        if self.env["verbosity"]:
            solver.log(f"Coverlift from {show_nz(~X)}", verbosity=3)
            solver.log("", show_table(C_enc), "C_enc <=", k, verbosity=3)
            solver.log("", show_table(parts), "parts", verbosity=3)

            solver.log("RS (row slack?)", show_table(RS), verbosity=3)
            solver.log(show_table(T_enc[:, C_enc != 0]), verbosity=3)
            solver.log("", show_table(C_enc[C_enc != 0]), verbosity=3)
            solver.log("", show_table(parts[C_enc != 0]), "parts", verbosity=3)
            solver.log(f"R_tight = {show_nz(R_tight)}", verbosity=3)
            solver.log(f"X = {show_nz(X)}", verbosity=3)
            solver.log(f"S_ = {show_nz(S)}", verbosity=3)
            solver.log(f"k_ = {k}", verbosity=2)

        # centre of mass heuristic
        if heuristic in (Coverlift.COM_MIN, Coverlift.COM_MAX):
            com = T_enc.sum(axis=0) / len(T_enc)
            if self.env["verbosity"]:
                solver.log("COM", T_enc.sum(axis=0), verbosity=3)
                solver.log("COM", com, verbosity=3)

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
                case Coverlift.COM_MIN:
                    direction = com - np.where(X, np.nan, A_enc)
                    j = np.nanargmin(direction)
                case Coverlift.COM_MAX:
                    direction = com - np.where(X, np.nan, A_enc)
                    if self.env["verbosity"]:
                        solver.log("direction", direction, verbosity=3)
                    j = np.nanargmax(direction)

            assert not X[j]

            # any pure 0/1 columns have been filtered out, a min always exists
            a_j = np.min(k - RS[(~R_tight) & T_enc[:, j]])

            if self.env["verbosity"]:
                solver.log(f"Lift {a_j}*b_{show(j)}", verbosity=2)

            # assert C_enc[j] == 0
            # Calculate new row upper bounds just for the added column
            RS += a_j * T_enc[:, j]

            # Find and update newly tight rows
            N_tight = ~R_tight & tight(RS, k)
            R_tight |= N_tight

            # Add var and coefficient to cut
            S[j] = True
            C_enc[j] = a_j

            X |= T_enc[N_tight, :].any(0)

            if self.env["negatives"]:
                X[self.counterchoice(j)] = True

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
                self.show_cut(S, C_enc, k, A_enc=A_enc, verbosity=2)

            assert not self.env["example2"] or a_j == [2, 1, 1][iteration]

            assert (
                not self.env["example2"] or (RS == [[1.0, 2.0, 0.0, 1.0, 2.0], [1.0, 2.0, 2.0, 1.0, 2.0], RS][iteration]).all()
            ), f"{iteration}; {RS}"

            solver.check_max_iterations(iteration)

            assert X[j], f"{show(j)} not chosen in {X}"

        return S, C_enc, k

    def check_explanation(self, X, C_enc, k, A_enc, frm, check=0):
        solver = self.solver
        T_enc = self.T_enc
        parts = self.parts
        X_enc = self.X_enc

        # in case of reverted negative
        A_enc = A_enc[: len(C_enc)]
        T_enc = T_enc[:, : len(C_enc)]

        # if self.solver.env["negatives"] and len(X) > self.cols():
        #     X, C_enc, k = self.revert_cut(X, C_enc, k)

        expr = self.get_expr(X, C_enc, k)

        # == {" + ".join(w * x.value() for (ws, xs, k) in terms(expr) for w, x in zip(ws, xs))}
        case = f"""The explanation

{expr}


SUM {C_enc[X]} * {show_nz(X)} < {k}

SUM {C_enc[X]} * {show_nz(X)} <= {k}

{X}

for counterexample

== {show_assignment(get_variables(expr))}

from assignment {frm}


{show_assignment(X_enc)}

for A_enc:

{show_table(A_enc)}

for tables:

{show_table(T_enc)}"""

        def value_(A_enc):
            lhs = (C_enc * A_enc).sum()
            return bool(solver.is_lt(lhs, k) if STRICT_CUTS else solver.is_le(lhs, k))

        if check >= 1:
            if not is_true_cst(expr):
                assert value_(A_enc) is False, f"Did not cut off assignment:\n\n{case}"
                # assert value(expr, {x: x.value() for x in X_enc}) is False, f"Did not cut off assignment:\n\n{case}"

        if check >= 2:
            for i, T_enc_i in enumerate(T_enc):
                # A_enc_row = {x_j: a_i_j for x_j, a_i_j in zip(X_enc, T_enc_i)}
                # self.show_cut(X, k, A_enc=T_enc_i.astype(int), verbosity=1, frm=frm, check=0)

                assert value_(T_enc_i) is True, (
                    f"Cut off row {show(i)}\n\n{show_nz(T_enc[i, :])}\n{parts[T_enc[i, :]]}\n\n\nfor case:\n\n{case}\n\n{show_table(T_enc_i)}\n\n"
                )
                # assert value(expr, A_enc_row) is True, (
                #     f"Cut off row {show(i)}\n\n{show_nz(T_enc[i, :])}\n{parts[T_enc[i, :]]}\n\n\nfor case:\n\n{case}\n\n{show_table(T_enc_i)}\n\n"
                # )

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

            # Save X_enc values before solving checker (solutions_checker overwrites them)
            # For NegBoolView, save the underlying _bv variable
            def get_base_var(x):
                return x._bv if isinstance(x, NegBoolView) else x

            saved_x_enc_values = {get_base_var(x): get_base_var(x).value() for tbl in solver.tables for x in tbl.X_enc}

            actual_solutions = solver.solutions_checker()

            # Restore X_enc values
            for x, v in saved_x_enc_values.items():
                x._value = v

            def show_assignments(As):
                return "\n".join(repr(a) for a in As)

            expected_solutions = self.env["expected_solutions"]
            remaining = without(actual_solutions, expected_solutions)
            assert len(np.unique(actual_solutions, axis=0)) == len(actual_solutions)

            self.env["cuts"][-1]["remain"] = remaining
            self.env["cuts"][-1]["n_sols"] = len(actual_solutions)
            if self.env["verbosity"]:
                solver.log(f"Expected ({len(expected_solutions)}) (model w/ table)", verbosity=2)
                solver.log(expected_solutions, verbosity=3)
                solver.log(f"Actual ({len(actual_solutions)}) (model w/o table but with lazy constraints)", verbosity=2)
                solver.log(actual_solutions, verbosity=3)
            if len(self.env["cuts"]) >= 2:
                removed = without(self.env["cuts"][-2]["remain"], remaining)
                if self.env["verbosity"]:
                    solver.log(
                        "A",
                        solver.user_vars,
                        tuple(x.value() for x in sorted(solver.user_vars, key=lambda x: x.name) if x.value() is not None),
                        verbosity=3,
                    )
                    solver.log(f"REMOVED {len(removed)}", verbosity=2)
                    solver.log(removed, verbosity=3)

            strength = (
                len(self.env["cuts"][-2]["remain"]) - len(self.env["cuts"][-1]["remain"])
                if len(self.env["cuts"]) >= 2
                else len(self.env["remain"]) - len(remaining)
            )
            self.env["cuts"][-1]["strength"] = strength
            self.env["cuts"][-1]["power"] = strength / len(self.env["cuts"][-1]["cut"][0].args)
            self.env["remain"] = remaining
            if self.env["verbosity"]:
                solver.log(
                    f"Remaining non-solutions to cut ({len(remaining)}, STR={self.env['cuts'][-1]['strength']}, PWR={self.env['cuts'][-1]['power']})",
                    verbosity=1,
                )
                solver.log(remaining, verbosity=3)
                solver.log(self.env["checker"], verbosity=4)
                solver.log(f"EXPECTED ({len(expected_solutions)})", verbosity=2)
                solver.log(expected_solutions, verbosity=4, indent=2)
                solver.log(f"ACTUAL ({len(actual_solutions)})", verbosity=2)
                solver.log(actual_solutions, verbosity=4, indent=2)
                solver.log(f"TO REMOVE ({len(remaining)})", verbosity=2)
                solver.log(remaining, verbosity=4)

            # assert repeated or strength
            assert strength or len(solver.tables) > 1

            assert len(expected_solutions) <= len(actual_solutions), (
                f"Missing sols:\n\n{show_assignments(expected_solutions)}\n\n {show_assignments(actual_solutions)}\n\n{self.env['checker']}"
            )

        # m = cp.Model(explanation, [x == a for x, a in zip(X_enc, A_enc)])
        # assert not m.solve(), f"Explanation {explanation} did not remove failure {A_enc}"


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
            "callbacks": 0,
            "max_iterations": None,
            "checked": False,
            "checker": None,
            "tables": [],
            "found_feasible": False,
            "model": None,
            "example2": False,
            "example_frac": False,
            "feasible": None,
            "cbCut": False,
            "short_channel": False,
            "early": False,
            "choices": None,
            **({} if env is None else env),
        }
        self.indent = 0

        if self.env["verbosity"]:
            cp.transformations.int2bool.IntVarEnc.NAMED = True
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

        # TODO temp
        cp.transformations.int2bool.IntVarEnc.SHORT_CHANNEL = self.env["short_channel"]

        super().__init__(
            name="lazy_gurobi",
            cpm_model=cpm_model,
            verbose=self.env["verbosity"] >= 1,
            short_channel=self.env["short_channel"],
            **kwargs,
        )

        if self.tables:
            self.native_model.Params.LazyConstraints = 1
        if self.env["verbosity"] >= 4:
            if self.env["cbCut"]:
                self.native_model.Params.PreCrush = 1
            self.native_model.Params.LogFile = "/tmp/gurobi.log"
            self.native_model.Params.OutputFlag = 1
            self.native_model.write("/tmp/gurobi.lp")

        if self.env["checked"] and cpm_model is not None:
            if self.env["verbosity"]:
                self.log("CHECKED: solve model to determine expected feasibility")
            self.env["user_vars"] = tuple(sorted(self.user_vars, key=lambda x: x.name))
            self.env["feasible"] = cpm_model.solve()
            self.env["model"] = cpm_model
            self.env["X"], self.env["expected_solutions"] = cp.solvers.utils.solutions(
                cpm_model,
                X=self.env["user_vars"],
                projected_solution_limit=None,
                time_limit=CHECKER_TIME_LIMIT,
            )

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
        if self.env["verbosity"] >= 3:
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

    def solution_callback_inner(self, frm):
        self.env["callbacks"] += 1
        feasible = True  # assume feasible
        for expr in self._explain_assignment(frm=frm):
            if isinstance(expr, Comparison) and expr.name in ("<=", "<"):
                feasible = False  # any expr means not feasible
                assert isinstance(expr.args[0], Operator)
                expr, k = expr.args
                if self.env["verbosity"]:
                    self.env["cuts"][-1]["cut"] = (expr, k)
                yield expr, k
                # TODO could lead to fewer cuts, but then expensive part of callback is repeated often
                if self.env["early"]:
                    break
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
                # self.log(X_enc.value(), verbosity=3)
            self.env["found_feasible"] = feasible
            if self.env["debug"]:
                self.check_max_iterations(self.env["callbacks"])

    def _finalize_tables(self):
        """Build global variable index mapping for all tables. Called before solve."""
        # Collect all unique base variables across all tables
        all_base_vars = []
        var_to_idx = {}
        for tbl in self.tables:
            for x in tbl.X_enc:
                base_var = x._bv if isinstance(x, NegBoolView) else x
                if base_var not in var_to_idx:
                    var_to_idx[base_var] = len(all_base_vars)
                    all_base_vars.append(base_var)

        # Store for callback use
        self._all_base_vars = all_base_vars
        self._all_grb_vars = [self._varmap[x] for x in all_base_vars]
        self._all_values = np.empty(len(all_base_vars), dtype=float)

        # Assign var_indices to each table
        for tbl in self.tables:
            tbl.var_indices = np.array([
                var_to_idx[x._bv if isinstance(x, NegBoolView) else x]
                for x in tbl.X_enc
            ], dtype=int)

    def get_solution_callback(self):
        from gurobipy import GRB

        grb_vars = self._all_grb_vars

        def solution_callback(what, where):
            time_cb = time.time()

            try:
                frm = None

                match where:
                    case GRB.Callback.MIPNODE:
                        if not self.env["fractional"]:
                            return
                        # Optimal solution to LP relaxation
                        if what.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                            frm = "MIPNODE-OPT"
                            cbGetVal = what.cbGetNodeRel
                        else:
                            return
                    case GRB.Callback.MIPSOL:  # Integer solution
                        cbGetVal = what.cbGetSolution
                        frm = "MIPSOL"
                    case _:
                        return

                # Batch retrieve all values into shared array
                self._all_values[:] = cbGetVal(grb_vars)

                # if self.env["verbosity"]:
                #     self.log("VHAT", verbosity=3)
                #     for x_enc_i, grb_x in all_xs:
                #         self.log(x_enc_i, x_enc_i.value(), verbosity=3)

                for expr, k in self.solution_callback_inner(frm):
                    # cut = self._make_numexpr(expr) <= k - self.native_model.Params.FeasibilityTol
                    cut = (
                        # expr < k
                        self._make_numexpr(expr) <= k - self.native_model.Params.FeasibilityTol
                        if STRICT_CUTS
                        # expr < k == expr <= k - 1 is returned
                        else self._make_numexpr(expr) <= k
                    )
                    if frm == "MIPSOL" or not self.env["cbCut"]:
                        what.cbLazy(cut)
                    else:
                        what.cbCut(cut)
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
            expr = cp.sum(C_enc[X] * X_enc[X]) < k if STRICT_CUTS else cp.sum(C_enc[X] * X_enc[X]) <= k
            if self.env["verbosity"]:
                self.log(f"EXPR  == {expr}", indent=2, verbosity=1)
                self.log(show_table(X_enc[X]), indent=2, verbosity=3)
                self.log(show_table(X_enc[X].value()), indent=2, verbosity=2)

        if isinstance(expr, (bool, np.bool_)):
            expr = cp.BoolVal(expr)

        if self.env["debug"]:
            self.env["cuts"][-1]["expr"] = expr

        return expr

    def _vary(self, a=None, b=None):
        if a is None:
            a = lambda: False
        if b is None:
            b = lambda: True
        return a() if self.env.get("variant", 0) == 0 else b()

    @line_profile
    def _explain_assignment(self, frm=None):
        # If fully integer, we can check if the tables are feasible yet
        if self.env["verbosity"]:
            self.log("EXPLAIN", frm, verbosity=2)
            self.log("Full sol", verbosity=4)

        for i, tbl in enumerate(self.tables, start=INDEX):
            X_enc, T_enc, parts = tbl.X_enc, tbl.T_enc, tbl.parts
            # Get values for this table using pre-computed indices
            values = self._all_values[tbl.var_indices].copy()
            # Apply negation where needed
            values[tbl.is_negated] = 1.0 - values[tbl.is_negated]

            if frm == "MIPSOL":
                A_enc = values > 0.5
                is_integer = True
            else:
                A_enc = values
                is_integer = False
                if self.is_integral(A_enc).all():
                    A_enc = A_enc > 0.5
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
                    f"by {show_table(A_enc)}",
                    verbosity=2,
                    indent=2,
                )

            if self.env["negatives"]:
                if np.issubdtype(A_enc.dtype, np.bool):
                    A_enc = np.concatenate([A_enc, ~A_enc], dtype=A_enc.dtype)
                else:
                    A_enc = np.concatenate([A_enc, 1.0 - A_enc], dtype=A_enc.dtype)

            try:
                # encode assignment
                if self.env["verbosity"]:
                    self.log(" ", np.array(X_enc), verbosity=3, indent=0)
                explanation = tbl.explain(A_enc, frm=frm, is_integer=is_integer)

                if self.env["verbosity"]:
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
                        X, C_enc, k = explanation
                        tbl.check_explanation(X, C_enc, k, A_enc, frm, check=2)
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

    def add(self, cons, get_user_vars=True):
        if not isinstance(cons, list):
            cons = [cons]
        return super().add(cons, get_user_vars=get_user_vars)

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

    def objective(self, *args, **kwargs):
        super().objective(*args, **kwargs)
        if self.env["checked"]:
            self.env["checker"].objective(self.obj, **kwargs)

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

        # Build global variable index mapping for all tables
        self._finalize_tables()

        if self.env["verbosity"]:
            self.log("Solving.. ")

        try:
            dt = time.time()
            if self.env["checked"]:
                self.env["checker"] += [
                    c for x_enc in self.ivarmap.values() for c in x_enc.encode_channelling_constraint(csemap=self._csemap)
                ]
                self.env["remain"] = without(self.solutions_checker(), self.env["expected_solutions"])
                if self.env["verbosity"]:
                    self.log("SOLS", len(self.env["expected_solutions"]))
                    # self.log("NON_SOLS", len(self.env["remain"]), verbosity=2)
                    self.log("TO REMOVE\n", self.env["remain"], verbosity=4)
                for iteration in itertools.count():
                    if time_limit is not None and time.time() - dt > time_limit:
                        raise TimeoutError

                    # take the first non-solution, once depeleted, take the first solution
                    if len(self.env["remain"]):
                        if self.env["choices"] is None:
                            sol = min(self.env["remain"].tolist())
                        else:
                            assert self.env["choices"], (
                                f"Choose from\n{'\n'.join(f'{i}: {c}' for i, c in enumerate(self.env['remain']))}"
                            )
                            sol = self.env["remain"][self.env["choices"].pop()]
                    else:
                        if self.env["model"].has_objective():
                            hassol = self.env["checker"].solve()
                            if hassol:
                                self.objective_value_ = self.env["model"].objective_value()
                        else:
                            sol = min(self.env["expected_solutions"].tolist(), default=None)
                            hassol = sol is not None

                            if hassol:
                                for x, v in zip(self.env["user_vars"], sol):
                                    x._value = v

                        break
                        sol = min(self.env["expected_solutions"].tolist(), default=None)
                        hassol = sol is not None
                        if hassol:
                            for x, v in zip(self.env["user_vars"], sol):
                                x._value = v

                            objective = self.env["model"].objective_

                            if objective is not None and (
                                self.objective_value_ is None
                                or (
                                    objective.value() < self.objective_value_
                                    if self.env["model"].objective_is_min
                                    else objective.value() > self.objective_value_
                                )
                            ):
                                self.objective_value_ = objective.value()
                        break

                    # if sol is None:
                    #     break

                    for x, v in zip(self.env["user_vars"], sol):
                        x._value = v

                    for expr, lit in self._csemap.items():
                        lit._value = expr.value()

                    # Populate _all_values from base variable values for _explain_assignment
                    for j, x in enumerate(self._all_base_vars):
                        self._all_values[j] = x.value()

                    self.check_max_iterations(iteration)
                    for expr, k in self.solution_callback_inner("MIPSOL"):
                        # self.env["checker"] += [expr <= k]
                        self.env["checker"] += [expr < k if STRICT_CUTS else expr <= k]

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

    def solveAll(self, *args, **kwargs):
        if self.env["checked"]:
            self.solve(*args, **kwargs)
            return self.env["checker"].solveAll(*args, **kwargs)
        return super().solveAll(*args, **kwargs)

    def get_x_encs(self, X):
        return [x_enc_i for x in X for x_enc_i in self.ivarmap[x]._xs]

    def transform_(self, cpm_expr):
        if not is_bool(cpm_expr) and cpm_expr.name == "table" and len(cpm_expr.args[1]) > self.env["cutoff"]:
            if len(set(cpm_expr.args[0])) < len(cpm_expr.args[0]):
                cpm_expr = normalize_table(cpm_expr)
            X, T = cpm_expr.args

            # only check after normalize, since normalize may remove all rows
            if not len(T):
                return [cp.BoolVal(False)]
            assert len(set(X)) == len(X), f"Dup. int vars in table for {cpm_expr}"

            # TODO maybe unneccesary to pass through TF?
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
        else:
            cons = cpm_expr

        cons = super().transform(cons)
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
