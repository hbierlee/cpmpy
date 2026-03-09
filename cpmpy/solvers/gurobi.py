#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
## gurobi.py
##
"""
    Interface to Gurobi Optimizer's Python API.

    Gurobi Optimizer is a highly efficient commercial solver for Integer Linear Programming (and more).

    Always use :func:`cp.SolverLookup.get("gurobi") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires that the 'gurobipy' python package is installed:

    .. code-block:: console

        $ pip install gurobipy

    Gurobi Optimizer requires an active licence (for example a free academic license)
    You can read more about available licences at https://www.gurobi.com/downloads/

    See detailed installation instructions at:
    https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_gurobi

    ==============
    Module details
    ==============
"""

from typing import Optional, List
import time
from enum import Enum
import pathlib
import numpy as np

import cpmpy as cp
from cpmpy.expressions.globalconstraints import Table
from cpmpy.expressions.utils import dom_size
from .solver_interface import SolverInterface, SolverStatus, ExitStatus, Callback
from ..exceptions import NotSupportedError
from ..expressions.core import *
from ..expressions.utils import argvals, argval
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..expressions.globalconstraints import DirectConstraint
from ..transformations.comparison import only_numexpr_equality
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.get_variables import get_variables
from ..transformations.linearize import linearize_constraint, linearize_reified_variables, only_positive_bv, only_positive_bv_wsum_const, decompose_linear, decompose_linear_objective
from ..transformations.normalize import toplevel_list
from ..transformations.reification import only_implies, reify_rewrite, only_bv_reifies
from ..transformations.safening import no_partial_functions, safen_objective
from cpmpy.expressions.globalconstraints import Table
from cpmpy.expressions.utils import dom_size
import numpy as np
from collections import Counter, defaultdict
from itertools import product
import enum

# save this to reset it (for unit tests)
cpmpy_decompose = Table.decompose


class Feature(Enum):
    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)


class Encoding(Feature):
    CPMPY = "cpmpy"
    XCSP3 = "xcsp3"
    GLEB = "gleb"
    BOOL = "bool"
    MDD = "mdd"


class Order(Feature):
    INPUT = "input"
    DOM_INCR = "dom-incr"
    DOM_DECR = "dom-decr"
    FIEDLER = "fiedler"
    GREEDY = "greedy"
    BIDIRECTIONAL = "bidirectional"


def trivial_decomposition(arr, tab):
    if len(tab) == 0:
        return [False], []
    elif len(tab) == 1:
        return [(x == tab[0][i]) for i, x in enumerate(arr)], []
    else:
        assert False, f"non-trivial {arr} {tab}"


def agreement_score(col1, col2):
    pairs = Counter(zip(col1, col2))
    return sum(n - 1 for n in pairs.values() if n > 1)


def agreement_matrix(arr):
    n_cols = arr.shape[1]
    scores = np.zeros((n_cols, n_cols), dtype=int)
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            score = agreement_score(arr[:, i], arr[:, j])
            scores[i, j] = score
            scores[j, i] = score
    return scores


def spectral_order_edges(arr):
    """
       Spectral ordering with high-similarity columns tending to edges
       """
    n_cols = arr.shape[1]
    if n_cols == 1:
        return [0]

    S = agreement_matrix(arr)

    D = np.diag(S.sum(axis=1))
    L = D - S

    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1]

    # Centrality = total similarity
    centrality = S.sum(axis=1)

    # Edge tendency score: deviation from mean weighted by centrality
    mean_f = np.mean(fiedler)
    score = np.abs(fiedler - mean_f) * centrality

    # Sort by score descending → highest score to edges
    column_order = np.argsort(-score)

    return list(column_order)


def add_column(full_table, table, col):
    if table.size == 0:
        return full_table[:,col][:,None]
    else:
        return np.column_stack((table, full_table[:,col]))

def greedy_ordering(arr):
    n_cols = arr.shape[1]
    if n_cols == 1:
        return [0]
    remaining_cols = list(range(n_cols))
    existing_table = np.array([])
    column_order = []
    while len(remaining_cols) > 0:
        partitions_dict = {}
        for col in remaining_cols:
            partitions_dict[col] = np.unique(add_column(arr, existing_table, col), axis=0).shape[0]
        best_col = min(partitions_dict, key=partitions_dict.get)
        if partitions_dict[best_col] == arr.shape[0]:
            column_order.extend(remaining_cols)
            break
        column_order.append(best_col)
        remaining_cols.remove(best_col)
        existing_table = add_column(arr, existing_table, best_col)
    return column_order


def bidirectional_ordering(arr):
    n_cols = arr.shape[1]
    if n_cols == 1:
        return [0]
    remaining_cols = list(range(n_cols))
    existing_table_left = np.array([])
    existing_table_right = np.array([])
    column_order_left = []
    column_order_right = []

    left_break = False
    right_break = False

    while len(remaining_cols) > 1 and not (left_break and right_break):
        if not left_break:
            partitions_dict = {}
            for col in remaining_cols:
                partitions_dict[col] = np.unique(add_column(arr, existing_table_left, col), axis=0).shape[0]
            best_col = min(partitions_dict, key=partitions_dict.get)
            if partitions_dict[best_col] == arr.shape[0]:
                left_break = True
            else:
                column_order_left.append(best_col)
                remaining_cols.remove(best_col)
                existing_table_left = add_column(arr, existing_table_left, best_col)

        if not right_break:
            partitions_dict = {}
            for col in remaining_cols:
                partitions_dict[col] = np.unique(add_column(arr, existing_table_right, col), axis=0).shape[0]

            best_col = min(partitions_dict, key=partitions_dict.get)
            if partitions_dict[best_col] == arr.shape[0]:
                right_break = True
            else:
                column_order_right.insert(0, best_col)
                remaining_cols.remove(best_col)
                existing_table_right = add_column(arr, existing_table_right, best_col)

    return column_order_left + remaining_cols + column_order_right


try:
    import gurobipy as gp

    GRB_ENV = None
except ImportError:
    pass


class CPM_gurobi(SolverInterface):
    """
    Interface to Gurobi's Python API

    Creates the following attributes (see parent constructor for more):

    - ``grb_model``: object, TEMPLATE's model object

    The :class:`~cpmpy.expressions.globalconstraints.DirectConstraint`, when used, calls a function on the ``grb_model`` object.

    Documentation of the solver's own Python API:
    https://docs.gurobi.com/projects/optimizer/en/current/reference/python.html
    """

    supported_global_constraints = frozenset({"min", "max", "abs", "mul", "pow"})
    supported_reified_global_constraints = frozenset()

    @staticmethod
    def supported():
        return CPM_gurobi.installed() and CPM_gurobi.license_ok()

    @staticmethod
    def installed():
        try:
            import gurobipy as _
            return True
        except ModuleNotFoundError:
            return False
        except Exception as e:
            raise e

    @staticmethod
    def license_ok():
        if not CPM_gurobi.installed():
            warnings.warn(
                f"License check failed, python package 'gurobipy' is not installed! Please check 'CPM_gurobi.installed()' before attempting to check license.")
            return False
        try:
            import gurobipy as gp
            global GRB_ENV
            if GRB_ENV is None:
                # initialise the native gurobi model object
                GRB_ENV = gp.Env(params={"OutputFlag": 0})
                GRB_ENV.start()
            return True
        except Exception as e:
            warnings.warn(f"Problem encountered with Gurobi license: {e}")
            return False

    @staticmethod
    def version() -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.
        """
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("gurobipy")
        except PackageNotFoundError:
            return None

    def filter_table(self, X, table):
        lb = np.array([x.lb for x in X])
        ub = np.array([x.ub for x in X])

        mask = np.all((table >= lb) & (table <= ub), axis=1)
        new_table = table[mask]

        return new_table


    def encode_table_constraint(self, X, T):
        dom_sizes = [dom_size(x) for x in X]
        width = sum(dom_sizes)
        T_enc = np.zeros((len(T), width), dtype=np.bool)

        for i, row in enumerate(T):
            offset = 0
            for x, x_width, a in zip(X, dom_sizes, row):
                if x.lb <= a <= x.ub:
                    T_enc[i, offset + a - x.lb] = True
                #else:
                #    np.delete(T_enc, i)
                offset += x_width


        X_encs, cons = self.encode_table_expr(X)

        X_enc = []
        parts = []
        for i, x_enc in enumerate(X_encs, start=1):
            xs = list(x_enc._xs)
            X_enc += xs
            parts += [i] * len(xs)

        X_enc = np.array(X_enc)
        parts = np.array(parts, dtype=int)

        # Remove constant columns (all True or all False)
        for polarity in (True, False):
            cols = T_enc.all(axis=0) if polarity else (~T_enc).all(axis=0)
            if polarity:
                cons += [x >= 1 for x in X_enc[cols]]
                # cols |= parts == parts[cols]
                # for p in np.unique(parts[cols]):
                #     cols |= (parts == p)
            else:
                cons += [x <= 0 for x in X_enc[cols]]
            T_enc = T_enc[:, ~cols]
            parts = parts[~cols]
            X_enc = X_enc[~cols]

        return X_enc, T_enc, cons, parts

    def encode_int_var(self, x):
        x_enc, exactly_one_con = cp.transformations.int2bool._encode_int_var(self.ivarmap, x, "direct", csemap=self._csemap)
        return x_enc, exactly_one_con + (x_enc.encode_channelling_constraint(csemap=self._csemap) if x._occurs else [])


    def boolvar(self, name=None, **kwargs):
        """Wrap cp.boolvar with safe debug names if enabled"""
        if self.named and name is not None:
            b = cp.boolvar(**kwargs, name=f"_BV{_BoolVarImpl.counter}_{name}")
            _BoolVarImpl.counter += 1
            return b
        else:
            return cp.boolvar(**kwargs)

    def encode_table_expr(self, X):
        x_encs = []
        cons = []
        for i, x in enumerate(X, start=1):
            x_enc, exactly_one_con = cp.transformations.int2bool._encode_int_var(self.ivarmap, x, "direct", csemap=self._csemap)
            x_encs.append(x_enc)
            cons += exactly_one_con
            if x._occurs:
                cons += x_enc.encode_channelling_constraint(csemap=self._csemap)
        return x_encs, cons


        cons = []
        for x in X:
            x_enc, exactly_one_con = cp.transformations.int2bool._encode_int_var(
                self.ivarmap, x, "direct", csemap=self._csemap
            )
            expr, k = x_enc.encode_term()

            # Note: do not use self += [..] to avoid poluting user_vars (and transformation is not really necesary either)
            cons += exactly_one_con
            cons += [cp.sum(c * b for c, b in expr) - x == -k]

        return [self.ivarmap[x.name] for x in X], cons

    def __init__(self, name="gurobi", cpm_model=None, subsolver=None, verbose=False, encoding=Encoding.CPMPY,
                 reduce=False, order=Order.INPUT, combined=False, named=False, output_stats=False, short_channel=False, encode_obj=False,
                 **kwargs):
        """
        Constructor of the native solver object

        Arguments:
            cpm_model: a CPMpy Model()
            subsolver: None, not used
        """
        if not self.installed():
            raise ModuleNotFoundError(
                "CPM_gurobi: Install the python package 'cpmpy[gurobi]' to use this solver interface.")
        elif not self.license_ok():
            raise ModuleNotFoundError(
                "CPM_gurobi: No license found or a problem occured during license check. Make sure your license is activated!")
        import gurobipy as gp

        # TODO: subsolver could be a GRB_ENV if a user would want to hand one over
        self.grb_model = gp.Model(env=GRB_ENV)
        self.output_stats = output_stats
        self.verbose = verbose
        self.ivarmap = dict()
        self.named = named
        self.short_channel = short_channel
        self.encode_obj = encode_obj

        # these should be save for integer objectives
        # the objective for an optimal result has to be strictly within one integer
        self.native_model.Params.MIPGapAbs = 1 - self.native_model.Params.IntFeasTol
        # MIPGap is set to 0, since it is a relative gap: even a small MIPGap will, for a large enough objective, lead to a sub-optimal integer objective declared as optimal
        # e.g., for the default MIPGap of 1e-4 and for an instance with a true optimal objective of 1000000, an objective of 1000010 may be declared optimal
        self.native_model.Params.MIPGap = 0  # no relative MIPGap, just use MIPGapAbs

        match encoding:
            case Encoding.CPMPY:
                Table.decompose = cpmpy_decompose
            case Encoding.XCSP3:
                def xcsp3_decompose(self_):
                    arr, tab = self_.args
                    tab = np.array(tab)
                    tab = self.filter_table(arr, tab)

                    if len(tab) < 2:
                        return trivial_decomposition(arr, tab)

                    row_selected = self.boolvar("r", shape=len(tab))

                    cons = []
                    for i, row in enumerate(tab):
                        subexpr = Operator("and", [x == v for x, v in zip(arr, row)])
                        cons.append(Operator("->", [row_selected[i], subexpr]))

                    return [Operator("or", row_selected)] + cons, []

                Table.decompose = xcsp3_decompose

            case Encoding.GLEB:
                def gleb_decompose(self_):
                    arr, tab = self_.args
                    tab = np.array(tab)
                    tab = self.filter_table(arr, tab)
                    if len(tab) < 2:
                        return trivial_decomposition(arr, tab)

                    cons = []

                    row_selected = self.boolvar(shape=len(tab), name="r")

                    cons += [x == cp.sum(row_selected * tab[:, i]) for i, x in enumerate(arr)]
                    return cons, [cp.sum(row_selected) == 1]

                Table.decompose = gleb_decompose

            case Encoding.BOOL:
                def bool_decompose(self_):
                    X, T = self_.args
                    T = np.array(T)
                    T = self.filter_table(X, T)

                    if len(T) < 2:
                        return trivial_decomposition(X, T)

                    # Encode table T to 01 table `T_enc` and X variables to encodings `X_enc`
                    X_enc, T_enc, cons, _ = self.encode_table_constraint(X, T)
                    # assert len(X_enc)

                    row_selected = self.boolvar(shape=len(T_enc), name="r")

                    cons += [b_i == cp.sum(row_selected * col) for b_i, col in zip(X_enc, T_enc.T)]
                    cons += [cp.sum(row_selected) == 1]

                    return cons, []

                Table.decompose = bool_decompose
            case Encoding.MDD | Encoding.NOREDUCEMDD:

                class MDD:
                    def __init__(self):
                        self.MDD_cache = {}
                        self.MDD_cache_reverse = defaultdict(set)
                        self.repeated_keys = set()

                class TerminatingState(enum.Enum):
                    SRC = 'src'
                    SNK = 'snk'

                class Prefix:

                    def __repr__(self):
                        return f"Prefix(node_id={self.prefix}, counter={self.counter})"

                    def __init__(self, prefix, counter=1):
                        self.prefix = prefix
                        self.counter = counter



                    def __eq__(self, other):
                        if not isinstance(other, Prefix):
                            return False
                        else:
                            return self.prefix == other.prefix and self.counter == other.counter

                    def __hash__(self):
                        return hash((self.prefix, self.counter))

                    def __deepcopy__(self):
                        return Prefix(self.prefix, self.counter)

                    def incr(self):
                        return Prefix(self.prefix, self.counter+1)


                class MDD_node:

                    def __repr__(self):
                        return f"Mdd node(node_id={self.prefix}, level={self.level}, transition={self.transition})"

                    def __init__(self, prefix, level, transition):
                        self.prefix = prefix
                        self.level = level
                        self.transition = transition

                    def __eq__(self, other):
                        if not isinstance(other, MDD_node):
                            return False

                        if self.level != other.level:
                            return False

                        if self.transition.keys() != other.transition.keys():
                            return False

                        for key in self.transition:
                            if self.transition[key] != other.transition[key]:
                                return False

                        return True

                    def deepcopy(self, memo=None):
                        if memo is None:
                            memo = {}

                        if self in memo:
                            return memo[self]

                        new_node = MDD_node(self.prefix, self.level, {})
                        memo[self] = new_node

                        for key, value in self.transition.items():
                            if isinstance(value, MDD_node):
                                new_node.transition[key] = value.deepcopy(memo)
                            elif isinstance(value, Prefix):
                                new_node.transition[key] = value.__deepcopy__()
                            else:
                                new_node.transition[key] = value

                        return new_node

                    __hash__ = object.__hash__


                class MDD_node_key:

                    def __repr__(self):
                        return f'level: {self.level}, transition: {self.transition}'

                    def __init__(self, node: "MDD_node"):
                        self.level = node.level

                        def key_value_repr(v):
                            if isinstance(v, Prefix):
                                return (v.prefix, v.counter)
                            elif isinstance(v, TerminatingState):
                                return ("TerminatingState")
                            else:
                                raise TypeError(f"Unsupported transition value type: {type(v)}")

                        # Freeze transition into tuple of sorted (key, value_repr)
                        self.transition = tuple(
                            sorted((k, key_value_repr(v)) for k, v in node.transition.items())
                        )

                        # Precompute hash
                        self._hash = hash((self.level, self.transition))

                    def __eq__(self, other):
                        if not isinstance(other, MDD_node_key):
                            return False

                        return (self.level == other.level) and (self.transition == other.transition)

                    def __hash__(self):
                        return self._hash




                def lookup_mdd(mdd_node, cache):
                    while isinstance(mdd_node, Prefix):
                        mdd_node = cache[mdd_node]
                    return mdd_node

                def update_mdd_cache(mdd, node_prefix, new_node, combined=True):

                    if combined:
                        if node_prefix in mdd.MDD_cache.keys():
                            old_node = mdd.MDD_cache[node_prefix]
                            mdd.MDD_cache_reverse[MDD_node_key(old_node)].discard(node_prefix)

                    mdd.MDD_cache[node_prefix] = new_node

                    if combined:
                        mdd.MDD_cache_reverse[MDD_node_key(new_node)].add(node_prefix)
                        '''mdd.MDD_cache_reverse = defaultdict(set)
                        for lookup, node in mdd.MDD_cache.items():
                            key = MDD_node_key(node)
                            mdd.MDD_cache_reverse[key].add(lookup)'''

                def reduce_mdd(mdd_node, mdd, level):
                    if isinstance(mdd_node, TerminatingState):
                        return mdd_node


                    mdd_node = lookup_mdd(mdd_node, mdd.MDD_cache)

                    B = {}

                    for key in mdd_node.transition.keys():
                        reduced_mdd = reduce_mdd(mdd_node.transition[key], mdd, level + 1)
                        if reduced_mdd != False:
                            B[key] = reduced_mdd

                    if len(B.keys()) == 0:
                        return False

                    G = MDD_node(mdd_node.prefix, level, B)

                    if mdd_node.prefix in mdd.repeated_keys:
                        return Prefix(mdd_node.prefix).__deepcopy__()

                    temp = None

                    key = MDD_node_key(G)
                    lookups = mdd.MDD_cache_reverse.get(key, set())

                    for lookup in lookups:
                        if lookup.prefix in mdd.repeated_keys:
                            return Prefix(lookup.prefix).__deepcopy__()
                        else:
                            temp = lookup.prefix

                    if temp is not None:
                        mdd.repeated_keys.add(temp)
                        return Prefix(temp).__deepcopy__()

                    update_mdd_cache(mdd, Prefix(mdd_node.prefix), G, combined)

                    return Prefix(mdd_node.prefix)


                def add_row_to_mdd(row, mdd_node, mdd, level=0, new_val=False):
                    mdd_node = lookup_mdd(mdd_node, mdd.MDD_cache)


                    if len(row) == level:
                        return TerminatingState.SNK

                    if combined:
                        mdd_node = mdd_node.deepcopy()

                    tuple_key = Prefix(tuple(row[:level]))

                    if combined:
                        if isinstance(mdd_node, MDD_node):
                            if mdd_node.prefix in mdd.repeated_keys:
                                prev_node = mdd.MDD_cache[tuple_key].deepcopy()
                                update_mdd_cache(mdd, tuple_key, prev_node, combined)
                                tuple_key = tuple_key.incr()

                    value = row[level]

                    if value not in mdd_node.transition:

                        mdd_node.transition[value] = add_row_to_mdd(row,
                                                                    MDD_node(tuple(row[:(level + 1)]), level + 1, {}),
                                                                    mdd, level + 1, new_val=True)

                        if reduce and combined:
                            if not new_val:
                                for key in mdd_node.transition:
                                    if key < value:
                                        reduced_mdd = reduce_mdd(mdd_node.transition[key], mdd, level + 1)
                                        mdd_node.transition[key] = reduced_mdd

                    else:
                        mdd_node.transition[value] = add_row_to_mdd(row, mdd_node.transition[value], mdd, level + 1)


                    update_mdd_cache(mdd, tuple_key, mdd_node, combined)

                    return tuple_key

                def construct_mdd(table):
                    if table.size == 0:
                        return

                    mdd = MDD()

                    mdd_node = MDD_node(tuple(), 0, {})

                    for i in range(0, table.shape[0]):
                        row = table[i]
                        mdd_node = add_row_to_mdd(row, mdd_node, mdd, 0)

                    if combined:
                        mdd_node = lookup_mdd(mdd_node, mdd.MDD_cache)
                        max_key = max(mdd_node.transition.keys())
                        reduced_mdd = reduce_mdd(mdd_node.transition[max_key], mdd, 1)
                        mdd_node.transition[max_key] = reduced_mdd


                    if reduce and not combined:
                        for lookup, node in mdd.MDD_cache.items():
                            key = MDD_node_key(node)
                            mdd.MDD_cache_reverse[key].add(lookup)
                        mdd_node = lookup_mdd(mdd_node, mdd.MDD_cache)
                        for key in mdd_node.transition.keys():
                            reduced_mdd = reduce_mdd(mdd_node.transition[key], mdd, 1)
                            mdd_node.transition[key] = reduced_mdd


                    return mdd.MDD_cache

                class Flow:
                    def __init__(self):
                        self.flow_in = []
                        self.flow_out = []

                    def add_flow_in(self, value):
                        self.flow_in.append(value)

                    def add_flow_out(self, value):
                        self.flow_out.append(value)

                def get_correct_bv(column_number, X, X_enc):

                    cumulative = 0

                    for i, x in enumerate(X):
                        d_size = dom_size(x)

                        if column_number < cumulative + d_size:
                            offset = column_number - cumulative
                            return X_enc[i]._xs[offset]

                        cumulative += d_size

                    return None

                def mdd_to_flow(mdd_cache, X, X_enc):
                    """Convert MDD cache to flow constraints, accounting for column reordering"""
                    domains = np.array([dom_size(x) for x in X])
                    lb = np.array([x.lb for x in X])

                    no_columns = sum(domains)

                    column_counter = {k: 0 for k in range(no_columns)}
                    flow = {k: Flow() for k in mdd_cache.keys()}
                    flow['snk'] = Flow()

                    for key in mdd_cache.keys():

                        val = lookup_mdd(mdd_cache[key], mdd_cache)

                        for (k, v) in val.transition.items():
                            column = sum(domains[:val.level]) + k - lb[val.level]

                            if column < 0 or column >= len(column_counter):
                                continue

                            column_counter[column] += 1
                            flow[key].add_flow_out((column, column_counter[column]))
                            if isinstance(v, Prefix):
                                flow[v].add_flow_in((column, column_counter[column]))
                            if isinstance(v, TerminatingState):
                                flow['snk'].add_flow_in((column, column_counter[column]))

                    cons = []
                    substitution = {}
                    for key in column_counter.keys():
                        if column_counter[key] == 0:
                            continue
                        elif column_counter[key] == 1:
                            substitution[(key, 1)] = get_correct_bv(key, X, X_enc)
                        else:
                            bvs = self.boolvar(shape=column_counter[key], name=f"e_{key}")

                            for n in range(1, column_counter[key] + 1):
                                substitution[(key, n)] = bvs[n - 1]

                    excluded = {Prefix(tuple()), "snk"}
                    for key in sorted(
                            (k for k in flow if k not in excluded),
                            key=lambda k: (len(k.prefix), k.prefix, k.counter)
                    ):
                        if (len(flow[key].flow_in) == 0) or (len(flow[key].flow_out) == 0):
                            continue
                        elif (len(flow[key].flow_in) == 1 and len(flow[key].flow_out) == 1
                                and substitution[flow[key].flow_out[0]] == substitution[flow[key].flow_in[0]]):
                            continue
                        elif (len(flow[key].flow_in) == 1 and len(flow[key].flow_out) == 1
                                and column_counter[flow[key].flow_in[0][0]] > 1 and column_counter[
                                    flow[key].flow_out[0][0]] > 1):
                            substitution[flow[key].flow_out[0]] = substitution[flow[key].flow_in[0]]

                        else:
                            cons += [cp.sum([substitution[(c, m)] for (c, m) in flow[key].flow_in]) == cp.sum(
                                [substitution[(c, m)] for (c, m) in flow[key].flow_out])]


                    cons += [cp.sum([substitution[(c, m)] for (c, m) in flow['snk'].flow_in]) == 1]
                    cons += [cp.sum([substitution[(c, m)] for (c, m) in flow[Prefix(tuple())].flow_out]) == 1]

                    for key in column_counter.keys():
                        if column_counter[key] > 1:
                            cons += [
                                cp.sum([substitution[(key, n)] for n in
                                        range(1, column_counter[key] + 1)]) == get_correct_bv(
                                    key, X, X_enc)]

                    return cons

                def mdd_decompose(self_):
                    X, Tb = self_.args

                    Tb = np.array(Tb)

                    Tb = self.filter_table(X, Tb)

                    if len(Tb) < 2:
                        return trivial_decomposition(X, Tb)


                    match order:
                        case Order.INPUT:
                            ordering = np.array(range(len(X)))
                        case Order.DOM_INCR:
                            ordering = np.argsort([dom_size(x) for x in X])
                        case Order.DOM_DECR:
                            ordering = np.argsort([-dom_size(x) for x in X])
                        case Order.FIEDLER:
                            ordering = np.array(spectral_order_edges(Tb))
                        case Order.GREEDY:
                            ordering = np.array(greedy_ordering(Tb))
                        case Order.BIDIRECTIONAL:
                            ordering = np.array(bidirectional_ordering(Tb))
                        case _:
                            raise ValueError(order)

                    reordered_Tb = Tb[:, ordering]

                    X_reordered = [X[i] for i in ordering]



                    if combined:
                        sorted_T = reordered_Tb[np.lexsort(reordered_Tb.T[::-1])]
                        mdd_cache = construct_mdd(sorted_T)
                    else:
                        mdd_cache = construct_mdd(reordered_Tb)



                    X_enc, cons = self.encode_table_expr(X_reordered)

                    flow_cons = mdd_to_flow(mdd_cache, X_reordered, X_enc)


                    return cons + flow_cons, []

                Table.decompose = mdd_decompose

            case _:
                raise Exception(f"TODO: {encoding}")

        if verbose:
            self.encoding_path = pathlib.Path(f"/tmp/encoding_{name}.txt")
            self.encoding_path.unlink(missing_ok=True)
        else:
            self.encoding_path = None

        # initialise everything else and post the constraints/objective
        # it is sufficient to implement add() and minimize/maximize() below
        super().__init__(name=name, cpm_model=cpm_model, **kwargs)

        if self.encoding_path:
            self.grb_model.Params.LogFile = "/tmp/gurobi.log"
            # self.grb_model.Params.OutputFlag = 1
            print("W", self.encoding_path.with_suffix(".lp"))
            self.grb_model.write(str(self.encoding_path.with_suffix(".lp")))

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.grb_model

    # Floating point comparison methods using Gurobi's tolerance parameters
    # Based on https://github.com/ed-lam/cpaior2025-master-class/blob/5c727db2a103ded7971bb89693fe5bb69d509c76/common.py#L9
    # https://www.gurobi.com/documentation/current/refman/parameters.html#sec:Parameters

    def is_eq(self, x, y):
        """Check if x equals y within FeasibilityTol."""
        return abs(x - y) <= self.native_model.Params.FeasibilityTol

    def is_lt(self, x, y):
        """Check if x is less than y within FeasibilityTol."""
        return x - y < -self.native_model.Params.FeasibilityTol

    def is_le(self, x, y):
        """Check if x is less than or equal to y within FeasibilityTol."""
        return x - y <= self.native_model.Params.FeasibilityTol

    def is_gt(self, x, y):
        """Check if x is greater than y within FeasibilityTol."""
        return x - y > self.native_model.Params.FeasibilityTol

    def is_ge(self, x, y):
        """Check if x is greater than or equal to y within FeasibilityTol."""
        return x - y >= -self.native_model.Params.FeasibilityTol

    def eps_floor(self, x):
        """Floor with IntFeasTol tolerance."""
        return np.floor(x + self.native_model.Params.IntFeasTol)

    def eps_ceil(self, x):
        """Ceiling with IntFeasTol tolerance."""
        return np.ceil(x - self.native_model.Params.IntFeasTol)

    def eps_round(self, x):
        """Round with IntFeasTol tolerance."""
        return np.ceil(x - 0.5 + self.native_model.Params.IntFeasTol)

    def is_integral(self, x):
        """Check if x is integral within IntFeasTol."""
        return np.abs(x - np.round(x)) <= self.native_model.Params.IntFeasTol

    def solve(self, time_limit: Optional[float] = None, solution_callback=None, **kwargs):
        """
            Call the gurobi solver

            Arguments:
                time_limit (float, optional):  maximum solve time in seconds
                solution_callback:             Gurobi callback function
                **kwargs:                      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
            Examples of gurobi supported arguments include:

            - ``Threads`` : int
            - ``MIPFocus`` : int
            - ``ImproveStartTime`` : bool
            - ``FlowCoverCuts`` : int

            For a full list of gurobi parameters, please visit https://www.gurobi.com/documentation/9.5/refman/parameters.html#sec:Parameters
        """
        from gurobipy import GRB

        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            self.time_limit = time_limit
            # self.time_limit = time_limit - (time.time() - self.time)

        if self.time_limit is not None:
            self.time_limit -= time.time() - self.time
            if self.time_limit <= 0:
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
                return
            self.grb_model.setParam("TimeLimit", self.time_limit)

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # edge case, empty model, ensure the solver has something to solve
        if not len(self.user_vars):
            self.add(intvar(1, 1) == 1)

        # call the solver, with parameters
        for param, val in ({"Threads": 1} | kwargs).items():
            self.grb_model.setParam(param, val)

        assert self.native_model.Params.Threads == 1
        _ = self.grb_model.optimize(callback=solution_callback)
        grb_objective = self.grb_model.getObjective()

        grb_status = self.grb_model.Status

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.grb_model.runtime

        # translate exit status
        if grb_status == GRB.OPTIMAL:
            # COP
            if self.has_objective():
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            # CSP
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif grb_status == GRB.INFEASIBLE:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif grb_status == GRB.TIME_LIMIT:
            if self.grb_model.SolCount == 0:
                # can be sat or unsat
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif grb_status == GRB.INTERRUPTED:
            raise getattr(self.native_model, "_callback_exception", None) or Exception(
                "Gurobi was interrupted (perhaps the solution callback called model.terminate())")
        else:  # another?
            raise NotImplementedError(
                f"Translation of gurobi status {grb_status} to CPMpy status not implemented")  # a new status type was introduced, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                solver_val = self.solver_var(cpm_var).X
                if cpm_var.is_bool():
                    cpm_var._value = solver_val >= 0.5
                else:
                    cpm_var._value = round(solver_val)

            # Overwrite
            if self.ivarmap is not None:
                for enc in self.ivarmap.values():
                    for x_enc_i in enc._xs:
                        if isinstance(x_enc_i, NegBoolView):
                            x_enc_i._value = self.solver_var(~x_enc_i).X <= 0.5
                        else:
                            x_enc_i._value = self.solver_var(x_enc_i).X >= 0.5
                    enc._x._value = enc.decode()

            # set _objective_value
            if self.has_objective():
                # assume integer obj
                self.objective_value_ = self.objective_.value()


        else:  # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var._value = None

        if self.output_stats:
            for field, stat in self.stats().items():
                print(f"c Stat={field}={stat}")

        return has_sol

    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """

        if is_num(cpm_var):  # shortcut, eases posting constraints
            return cpm_var

        # special case, negative-bool-view. Should be eliminated in linearize
        if isinstance(cpm_var, NegBoolView):
            return 1 - self.solver_var(~cpm_var)

        # create if it does not exit
        if cpm_var not in self._varmap:
            from gurobipy import GRB
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.grb_model.addVar(vtype=GRB.BINARY, name=cpm_var.name)
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.grb_model.addVar(cpm_var.lb, cpm_var.ub, vtype=GRB.INTEGER, name=str(cpm_var))
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar

        # return from cache
        return self._varmap[cpm_var]

    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            'objective()' can be called multiple times, only the last one is stored

            .. note::
                technical side note: any constraints created during conversion of the objective
                are premanently posted to the solver
        """
        from gurobipy import GRB

        # save user variables
        get_variables(expr, self.user_vars)

        # transform objective
        cons = []
        obj, safe_cons = safen_objective(expr)
        cons += safe_cons
        obj, decomp_cons = decompose_linear_objective(obj,
                                                      supported=self.supported_global_constraints,
                                                      supported_reified=self.supported_reified_global_constraints,
                                                      csemap=self._csemap)
        cons += decomp_cons
        obj, flat_cons = flatten_objective(obj, csemap=self._csemap)
        cons += flat_cons
        obj, k = only_positive_bv_wsum_const(obj)  # remove negboolviews
        cons += self.handle_channelling(cons)

        if self.encode_obj:
            weights, xs = ([1], [obj]) if isinstance(obj, _IntVarImpl) else ([1] * len(obj.args), obj.args) if obj.name == "sum" else obj.args
            # partition based on whether variables already occur in the model
            occurring = [(w, x) for w, x in zip(weights, xs) if x._occurs]
            non_occurring = [(w, x) for w, x in zip(weights, xs) if not x._occurs]

            # only encode non-occurring terms
            non_ocurring, bool_cons, k_ = cp.transformations.int2bool._encode_lin_expr(
                self.ivarmap,
                [x for w, x in non_occurring],
                [w for w, x in non_occurring],
                "direct",
                csemap=self._csemap,
            )
            assert len(bool_cons) == 0

            # combine: occurring terms stay as-is, non-occurring get encoded
            obj = cp.sum(x * w for x, w in occurring + non_ocurring)
            k += k_
        else:
            cons += self.handle_channelling(obj)


        self.add(cons, get_user_vars=False)


        # make objective function or variable and post
        self.obj = obj + k
        grb_obj = self._make_numexpr(obj)
        if minimize:
            self.grb_model.setObjective(grb_obj, sense=GRB.MINIMIZE)
        else:
            self.grb_model.setObjective(grb_obj, sense=GRB.MAXIMIZE)
        self.grb_model.update()

    def has_objective(self):
        return self.grb_model.getObjective().size() != 0  # TODO: check if better way to do this...

    def _make_numexpr(self, cpm_expr):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression

            Used especially to post an expression as objective function
        """
        import gurobipy as gp

        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # sum
        if cpm_expr.name == "sum":
            return gp.quicksum(self.solver_vars(cpm_expr.args))
        if cpm_expr.name == "sub":
            a, b = self.solver_vars(cpm_expr.args)
            return a - b
        # wsum
        if cpm_expr.name == "wsum":
            return gp.quicksum(w * self.solver_var(var) for w, var in zip(*cpm_expr.args))

        raise NotImplementedError("gurobi: Not a known supported numexpr {}".format(cpm_expr))

    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the :ref:`Adding a new solver` docs on readthedocs for more information.

            :param cpm_expr: CPMpy expression, or list thereof
            :type cpm_expr: Expression or list of Expression

            :return: list of Expression
        """
        # apply transformations, then post internally
        # expressions have to be linearized to fit in MIP model. See /transformations/linearize
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"mod", "div",
                                                                  "element"})  # linearize and decompose expect safe exprs
        cpm_cons = decompose_linear(cpm_cons,
                                    supported=self.supported_global_constraints,
                                    supported_reified=self.supported_reified_global_constraints,
                                    csemap=self._csemap)
        cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']),
                                 csemap=self._csemap)  # constraints that support reification
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]),
                                         csemap=self._csemap)  # supports >, <, !=
        cpm_cons = linearize_reified_variables(
            cpm_cons,
            min_values=2,
            csemap=self._csemap,
            ivarmap=self.ivarmap
        )
        cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
        cpm_cons = only_implies(cpm_cons, csemap=self._csemap)  # anything that can create full reif should go above...
        # gurobi does not round towards zero, so no 'div' in supported set: https://github.com/CPMpy/cpmpy/pull/593#issuecomment-2786707188
        cpm_cons = linearize_constraint(cpm_cons, supported=frozenset(
            {"sum", "wsum", "->", "sub", "min", "max", "mul", "abs", "pow"}),
                                        csemap=self._csemap)  # the core of the MIP-linearization
        cpm_cons = only_positive_bv(cpm_cons, csemap=self._csemap)  # after linearization, rewrite ~bv into 1-bv
        cpm_cons += self.handle_channelling(cpm_cons)
        return cpm_cons

    def handle_channelling(self, cpm_expr):
        cpm_cons = []
        if self.ivarmap is not None:
            for x in get_variables(cpm_expr):
                # TODO is int var
                x_enc = self.ivarmap.get(x.name, None)
                if x_enc and x._occurs is False:
                    if self.short_channel:
                        # If transformation is needed
                        x._occurs = True
                        self += x_enc.encode_channelling_constraint(csemap=self._csemap)
                    else:
                        cpm_cons += x_enc.encode_channelling_constraint(csemap=self._csemap)
                x._occurs = True
        return cpm_cons

    def add(self, cpm_expr_orig, get_user_vars=True):
      """
            Eagerly add a constraint to the underlying solver.

              Any CPMpy expression given is immediately transformed (through `transform()`)
              and then posted to the solver in this function.

              This can raise 'NotImplementedError' for any constraint not supported after transformation

              The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
              the user knows and cares about (and will be populated with a value after solve). All other variables
              are auxiliary variables created by transformations.

          :param cpm_expr: CPMpy expression, or list thereof
          :type cpm_expr: Expression or list of Expression

          :return: self
        """
      from gurobipy import GRB

      # add new user vars to the set
      if get_user_vars:
          get_variables(cpm_expr_orig, collect=self.user_vars)

      if self.encoding_path:
          cp.transformations.int2bool.IntVarEnc.NAMED = True
          with open(self.encoding_path, "a") as f:
              print(f"C", cpm_expr_orig, file=f)
              print("X", ", ".join(f"{x} in {x.lb}..{x.ub}" for x in get_variables(cpm_expr_orig)), file=f)

        # transform and post the constraints
      for cpm_expr in self.transform(cpm_expr_orig):
          if self.encoding_path:

              with open(self.encoding_path, "a") as f:
                  print("  ", cpm_expr, file=f)
          if self.time_limit is not None:
              runtime = time.time() - self.time
              if runtime > self.time_limit:
                  self.cpm_status.exitstatus = ExitStatus.UNKNOWN
                  self.cpm_status.runtime = runtime
                  break

          # Comparisons: only numeric ones as 'only_implies()' has removed the '==' reification for Boolean expressions
          # numexpr `comp` bvar|const
          if isinstance(cpm_expr, Comparison):
              lhs, rhs = cpm_expr.args
              grbrhs = self.solver_var(rhs)

              # Thanks to `only_numexpr_equality()` only supported comparisons should remain
              if cpm_expr.name == '<=':
                  grblhs = self._make_numexpr(lhs)
                  self.grb_model.addLConstr(grblhs, GRB.LESS_EQUAL, grbrhs)
              elif cpm_expr.name == '>=':
                  grblhs = self._make_numexpr(lhs)
                  self.grb_model.addLConstr(grblhs, GRB.GREATER_EQUAL, grbrhs)
              elif cpm_expr.name == '==':
                  if isinstance(lhs, _NumVarImpl) \
                          or (isinstance(lhs, Operator) and (
                          lhs.name == 'sum' or lhs.name == 'wsum' or lhs.name == "sub")):
                      # a BoundedLinearExpression LHS, special case, like in objective
                      grblhs = self._make_numexpr(lhs)
                      self.grb_model.addLConstr(grblhs, GRB.EQUAL, grbrhs)

                  elif lhs.name == 'mul':
                      assert len(lhs.args) == 2, "Gurobi only supports multiplication with 2 variables"
                      a, b = self.solver_vars(lhs.args)
                      self.grb_model.setParam("NonConvex", 2)
                      self.grb_model.addConstr(a * b == grbrhs)

                  elif lhs.name == 'div':
                      if not is_num(lhs.args[1]):
                          raise NotSupportedError(
                              f"Gurobi only supports division by constants, but got {lhs.args[1]}")
                      a, b = self.solver_vars(lhs.args)
                      self.grb_model.addLConstr(a / b, GRB.EQUAL, grbrhs)

                  else:
                      # General constraints
                      # grbrhs should be a variable for gurobi in the subsequent, fake it
                      if is_num(grbrhs):
                          grbrhs = self.solver_var(intvar(lb=grbrhs, ub=grbrhs))

                      if lhs.name == 'min':
                          self.grb_model.addGenConstrMin(grbrhs, self.solver_vars(lhs.args))
                      elif lhs.name == 'max':
                          self.grb_model.addGenConstrMax(grbrhs, self.solver_vars(lhs.args))
                      elif lhs.name == 'abs':
                          self.grb_model.addGenConstrAbs(grbrhs, self.solver_var(lhs.args[0]))
                      elif lhs.name == 'pow':
                          x, a = self.solver_vars(lhs.args)
                          self.grb_model.addGenConstrPow(x, grbrhs, a)
                      else:
                          raise NotImplementedError(
                              "Not a known supported gurobi comparison '{}' {}".format(lhs.name, cpm_expr))
              else:
                  raise NotImplementedError(
                      "Not a known supported gurobi comparison '{}' {}".format(lhs.name, cpm_expr))

          elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
              # Indicator constraints
              # Take form bvar -> sum(x,y,z) >= rvar
              cond, sub_expr = cpm_expr.args
              assert isinstance(cond, _BoolVarImpl), f"Implication constraint {cpm_expr} must have BoolVar as lhs"
              assert isinstance(sub_expr, Comparison), "Implication must have linear constraints on right hand side"
              if isinstance(cond, NegBoolView):
                  cond, bool_val = self.solver_var(cond._bv), False
              else:
                  cond, bool_val = self.solver_var(cond), True

              lhs, rhs = sub_expr.args
              if isinstance(lhs, _NumVarImpl) or lhs.name == "sum" or lhs.name == "wsum":
                  lin_expr = self._make_numexpr(lhs)
              else:
                  raise Exception(
                      f"Unknown linear expression {lhs} on right side of indicator constraint: {cpm_expr}")
              if sub_expr.name == "<=":
                  self.grb_model.addGenConstrIndicator(cond, bool_val, lin_expr, GRB.LESS_EQUAL, self.solver_var(rhs))
              elif sub_expr.name == ">=":
                  self.grb_model.addGenConstrIndicator(cond, bool_val, lin_expr, GRB.GREATER_EQUAL,
                                                       self.solver_var(rhs))
              elif sub_expr.name == "==":
                  self.grb_model.addGenConstrIndicator(cond, bool_val, lin_expr, GRB.EQUAL, self.solver_var(rhs))
              else:
                  raise Exception(f"Unknown linear expression {sub_expr} name")

          # True or False
          elif isinstance(cpm_expr, BoolVal):
              self.grb_model.addConstr(cpm_expr.args[0])

          # a direct constraint, pass to solver
          elif isinstance(cpm_expr, DirectConstraint):
              cpm_expr.callSolver(self, self.grb_model)

          else:
              raise NotImplementedError(cpm_expr)  # if you reach this... please report on github

      return self

    __add__ = add  # avoid redirect in superclass

    def solution_hint(self, cpm_vars: List[_NumVarImpl], vals: List[int | bool]):
        """
        Gurobi supports warmstarting the solver with a (in)feasible solution.
        The provided value will affect branching heurstics during solving, making it more likely the final solution will contain the provided assignment.

        To learn more about solution hinting in gurobi, see:
        https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/variable.html#varhintval

        Optionally, you can also set the relative priority of the hint, using:

        .. code-block:: python

            solver.solver_var(cpm_var).setAttr("VarHintPri", <priority>)

        :param cpm_vars: list of CPMpy variables
        :param vals: list of (corresponding) values for the variables
        """
        for cpm_var, val in zip(cpm_vars, vals):
            self.solver_var(cpm_var).setAttr("VarHintVal", val)

    def solveAll(self, display: Optional[Callback] = None, time_limit: Optional[float] = None,
                 solution_limit: Optional[int] = None, call_from_model=False, **kwargs):
        """
            Compute all solutions and optionally display the solutions.

            This is the generic implementation, solvers can overwrite this with
            a more efficient native implementation

            Arguments:
                display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                time_limit: stop after this many seconds (default: None)
                solution_limit: stop after this many solutions (default: None)
                call_from_model: whether the method is called from a CPMpy Model instance or not
                any other keyword argument

            Returns: number of solutions found
        """
        from gurobipy import GRB

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # edge case, empty model, ensure the solver has something to solve
        if not len(self.user_vars):
            self.add(intvar(1, 1) == 1)

        if time_limit is not None:
            self.grb_model.setParam("TimeLimit", time_limit)

        if solution_limit is None:
            raise Exception(
                "Gurobi does not support searching for all solutions. If you really need all solutions, "
                "try setting solution limit to a large number")

        # Force gurobi to keep searching in the tree for optimal solutions
        sa_kwargs = {"PoolSearchMode": 2, "PoolSolutions": solution_limit}

        # solve the model
        self.solve(time_limit=time_limit, **sa_kwargs, **kwargs)

        optimal_val = None
        solution_count = self.grb_model.SolCount
        opt_sol_count = 0

        # clear user vars if no solution found
        if solution_count == 0:
            self.objective_value_ = None
            for var in self.user_vars:
                var._value = None

        for i in range(solution_count):
            # Specify which solution to query
            self.grb_model.setParam("SolutionNumber", i)
            sol_obj_val = self.grb_model.PoolObjVal
            if optimal_val is None:
                optimal_val = sol_obj_val
            if optimal_val is not None:
                # sub-optimal solutions
                if sol_obj_val != optimal_val:
                    break
            opt_sol_count += 1

            # Translate solution to variables
            for cpm_var in self.user_vars:
                solver_val = self.solver_var(cpm_var).Xn
                if cpm_var.is_bool():
                    cpm_var._value = solver_val >= 0.5
                else:
                    cpm_var._value = round(solver_val)

            # Overwrite
            if self.ivarmap is not None:
                for enc in self.ivarmap.values():
                    for x_enc_i in enc._xs:
                        if isinstance(x_enc_i, NegBoolView):
                            x_enc_i._value = self.solver_var(~x_enc_i).Xn <= 0.5
                        else:
                            x_enc_i._value = self.solver_var(x_enc_i).Xn >= 0.5
                    enc._x._value = enc.decode()

            # Translate objective
            if self.has_objective():
                self.objective_value_ = self.grb_model.PoolObjVal

            if display is not None:
                if isinstance(display, Expression):
                    print(argval(display))
                elif isinstance(display, list):
                    print(argvals(display))
                else:
                    display()  # callback

        # Reset pool search mode to default
        self.grb_model.setParam("PoolSearchMode", 0)

        if opt_sol_count:
            if opt_sol_count == solution_limit:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            else:
                grb_status = self.grb_model.Status
                if grb_status == GRB.TIME_LIMIT:  # reached time limit
                    self.cpm_status.exitstatus = ExitStatus.FEASIBLE
                else:  # found all solutions
                    self.cpm_status.exitstatus = ExitStatus.OPTIMAL
        # if unsat or timout with no solution, .solve() will have already set the state accordingly (so nothing to update)

        return opt_sol_count

    def stats(self):
        self.native_model.update()  # Ensure NumConstrs reflects latest state
        return {
            "constraints": self.native_model.NumConstrs,
        }

