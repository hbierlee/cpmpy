#!/usr/bin/env python
#-*- coding:utf-8 -*-
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
from ..transformations.decompose_global import decompose_in_tree, decompose_objective
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.get_variables import get_variables
from ..transformations.linearize import linearize_constraint, only_positive_bv, only_positive_bv_wsum
from ..transformations.normalize import toplevel_list
from ..transformations.reification import only_implies, reify_rewrite, only_bv_reifies
from ..transformations.safening import no_partial_functions, safen_objective
from cpmpy.expressions.globalconstraints import Table
from cpmpy.expressions.utils import dom_size

class Feature(Enum):
    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return self.value

class Encoding(Feature):
    DEFAULT = "default"
    XCSP3 = "xcsp3"
    GLEB = "gleb"
    BOOL_GLEB = "bool-gleb"
    # MDD = "mdd" # TODO removed for now so all available encodings are tested


def trivial_decomposition(arr, tab):
    if len(tab) == 0:
        return [False], []
    elif len(tab) == 1:
        return [(x == tab[0][i]) for i, x in enumerate(arr)], []
    else:
        assert False, f"non-trivial {arr} {tabl}"

def encode_intvar_as_bool(arr):
    shape = sum([dom_size(x) for x in arr])
    bv = cp.boolvar(shape=shape)
    cons = []
    offset = 0
    for x in arr:
        cons += [cp.sum(bv[offset:offset+(dom_size(x))]) == 1]
        cons += [x == cp.sum(bv[offset:offset+dom_size(x)] * range(x.lb, x.lb + dom_size(x)+1))]
        offset += dom_size(x)
    return bv, cons

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

    supported_global_constraints = frozenset({"min", "max", "abs", "pow"})
    supported_reified_global_constraints = frozenset()

    @staticmethod
    def supported():
        return CPM_gurobi.installed() and CPM_gurobi.license_ok()

    @staticmethod
    def installed():
        try:
            import gurobipy as gp
            return True
        except ModuleNotFoundError:
            return False
        except Exception as e:
            raise e

    @staticmethod
    def license_ok():
        if not CPM_gurobi.installed():
            warnings.warn(f"License check failed, python package 'gurobipy' is not installed! Please check 'CPM_gurobi.installed()' before attempting to check license.")
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

    def encode_table_constraint(self, X, T):

        def encode(X, T):
            dom_sizes = [dom_size(x) for x in X]
            width = sum(dom_sizes)
            T_enc = np.zeros((len(T), width), dtype=np.bool)

            for i, row in enumerate(T):
                offset = 0
                for x, x_width, a in zip(X, dom_sizes, row):
                    try:
                        T_enc[i, offset + a - x.lb] = True
                    except IndexError:
                        np.delete(T_enc, i)
                    offset += x_width
            return T_enc

        T_enc = encode(X, T)

        cons = []
        for x in X:
            x_enc, exactly_one_con = cp.transformations.int2bool._encode_int_var(
                self.ivarmap, x, "direct", csemap=self._csemap
            )
            expr, k = x_enc.encode_term()
            # TODO if only BV, then need to assign (but no need to assign if decoding constraint present)
            # Note: do not use self += [..] to avoid poluting user_vars (and transformation is not really necesary either)
            cons += exactly_one_con
            cons += [cp.sum(c * b for c, b in expr) - x == -k]

        return [self.ivarmap[x.name] for x in X], T_enc, cons

    def __init__(self, name="gurobi", cpm_model=None, subsolver=None, verbose=False, encoding=Encoding.DEFAULT, output_stats=False, **kwargs):
        """
        Constructor of the native solver object

        Arguments:
            cpm_model: a CPMpy Model()
            subsolver: None, not used
        """
        if not self.installed():
            raise ModuleNotFoundError("CPM_gurobi: Install the python package 'cpmpy[gurobi]' to use this solver interface.") 
        elif not self.license_ok():
            raise ModuleNotFoundError("CPM_gurobi: No license found or a problem occured during license check. Make sure your license is activated!")
        import gurobipy as gp

        # TODO: subsolver could be a GRB_ENV if a user would want to hand one over
        self.grb_model = gp.Model(env=GRB_ENV)
        self.output_stats = output_stats
        self.verbose = verbose
        self.ivarmap = dict()

        if encoding == Encoding.XCSP3:
            def xcsp3_decompose(self):
                arr, tab = self.args

                if len(tab) < 2:
                    return trivial_decomposition(arr, tab)

                row_selected = cp.boolvar(shape=len(tab))

                cons = []
                for i, row in enumerate(tab):
                    subexpr = Operator("and", [x == v for x, v in zip(arr, row)])
                    cons.append(Operator("->", [row_selected[i], subexpr]))

                return [Operator("or", row_selected)] + cons, []

            Table.decompose = xcsp3_decompose

        if encoding == Encoding.GLEB:
            def gleb_decompose(self):
                arr, tab = self.args
                if len(tab) < 2:
                    return trivial_decomposition(arr, tab)

                cons = []
                row_selected = cp.boolvar(shape=len(tab))
                nptab = np.array(tab)

                cons += [x == cp.sum(row_selected * nptab[:, i]) for i, x in enumerate(arr)]
                cons += [cp.sum(row_selected) == 1]
                return cons, []

            Table.decompose = gleb_decompose

        elif encoding == Encoding.BOOL_GLEB:
            def bool_decompose(self_):
                X, T = self_.args

                if len(T) < 2:
                    return trivial_decomposition(X, T)

                # Encode table T to 01 table `T_enc` and X variables to encodings `X_enc`
                X_enc, T_enc, cons = self.encode_table_constraint(X, T)

                row_selected = cp.boolvar(shape=len(T_enc))
                # For an encoding `x_enc`, the encoding variables are in `x_enc.xs_`, so we can iter over each integer variable, then over each of its encoding variables
                # Then enforce for each column `i` and its associated encoding variable `b_i`, enforce it to be equal to whatever row is selected
                cons += [b_i == cp.sum(row_selected * col) for b_i, col in zip((b_i for x_enc in X_enc for b_i in x_enc._xs), T_enc.T)]
                cons += [cp.sum(row_selected) == 1]

                return cons, []


            Table.decompose = bool_decompose
        # elif encoding == Encoding.MDD:
        #     pass

        if verbose:
            pathlib.Path("/tmp/encoding.txt").unlink(missing_ok=True)


        # initialise everything else and post the constraints/objective
        # it is sufficient to implement add() and minimize/maximize() below
        super().__init__(name=name, cpm_model=cpm_model, **kwargs)

        if verbose:
            self.grb_model.Params.LogFile = "/tmp/gurobi.log"
            self.grb_model.Params.OutputFlag = 1
            self.grb_model.write("/tmp/gurobi.lp")


    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.grb_model


    def solve(self, time_limit:Optional[float]=None, solution_callback=None, **kwargs):
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
            raise getattr(self.native_model, "_callback_exception", None) or Exception("Gurobi was interrupted (perhaps the solution callback called model.terminate())")
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
                    cpm_var._value = int(solver_val)
            # set _objective_value
            if self.has_objective():
                grb_obj_val = grb_objective.getValue()
                if round(grb_obj_val) == grb_obj_val: # it is an integer?:
                    self.objective_value_ = int(grb_obj_val)
                else: #  can happen with DirectVar or when using floats as coefficients
                    self.objective_value_ =  float(grb_obj_val)

        else: # clear values of variables
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

        if is_num(cpm_var): # shortcut, eases posting constraints
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
        obj, safe_cons = safen_objective(expr)
        obj, decomp_cons = decompose_objective(obj,
                                               supported=self.supported_global_constraints,
                                               supported_reified=self.supported_reified_global_constraints,
                                               csemap=self._csemap)
        obj, flat_cons = flatten_objective(obj, csemap=self._csemap)
        obj = only_positive_bv_wsum(obj)  # remove negboolviews

        self.add(safe_cons + decomp_cons + flat_cons)

        # make objective function or variable and post
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
            a,b = self.solver_vars(cpm_expr.args)
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
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"mod", "div", "element"})  # linearize and decompose expect safe exprs
        cpm_cons = decompose_in_tree(cpm_cons,
                                     supported=self.supported_global_constraints | {"alldifferent"}, # alldiff has a specialized MIP decomp in linearize
                                     supported_reified=self.supported_reified_global_constraints,
                                     csemap=self._csemap)
        cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']), csemap=self._csemap)  # constraints that support reification
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]), csemap=self._csemap)  # supports >, <, !=
        cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
        cpm_cons = only_implies(cpm_cons, csemap=self._csemap)  # anything that can create full reif should go above...
        # gurobi does not round towards zero, so no 'div' in supported set: https://github.com/CPMpy/cpmpy/pull/593#issuecomment-2786707188
        cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum","->","sub","min","max","mul","abs","pow"}), csemap=self._csemap)  # the core of the MIP-linearization
        cpm_cons = only_positive_bv(cpm_cons, csemap=self._csemap)  # after linearization, rewrite ~bv into 1-bv
        return cpm_cons

    def add(self, cpm_expr_orig):
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
      get_variables(cpm_expr_orig, collect=self.user_vars)

      if self.verbose:
        with open("/tmp/encoding.txt", "a") as f:
          print(f"C", cpm_expr_orig, file=f)
          print("X", ", ".join(f"{x} in {x.lb}..{x.ub}" for x in get_variables(cpm_expr_orig)), file=f)


      # transform and post the constraints
      for cpm_expr in self.transform(cpm_expr_orig):
        if self.verbose:
          print("  ", cpm_expr, file=open("/tmp/encoding.txt", "a"))
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
                        or (isinstance(lhs, Operator) and (lhs.name == 'sum' or lhs.name == 'wsum' or lhs.name == "sub")):
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
                        raise NotSupportedError(f"Gurobi only supports division by constants, but got {lhs.args[1]}")
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
                raise Exception(f"Unknown linear expression {lhs} on right side of indicator constraint: {cpm_expr}")
            if sub_expr.name == "<=":
                self.grb_model.addGenConstrIndicator(cond, bool_val, lin_expr, GRB.LESS_EQUAL, self.solver_var(rhs))
            elif sub_expr.name == ">=":
                self.grb_model.addGenConstrIndicator(cond, bool_val, lin_expr, GRB.GREATER_EQUAL, self.solver_var(rhs))
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

    def solution_hint(self, cpm_vars:List[_NumVarImpl], vals:List[int|bool]):
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

    def solveAll(self, display:Optional[Callback]=None, time_limit:Optional[float]=None, solution_limit:Optional[int]=None, call_from_model=False, **kwargs):
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
        sa_kwargs = {"PoolSearchMode":2, "PoolSolutions":solution_limit}

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
                    cpm_var._value = int(solver_val)

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
                if grb_status == GRB.TIME_LIMIT: # reached time limit
                    self.cpm_status.exitstatus = ExitStatus.FEASIBLE
                else: # found all solutions   
                    self.cpm_status.exitstatus = ExitStatus.OPTIMAL
        # if unsat or timout with no solution, .solve() will have already set the state accordingly (so nothing to update)


        return opt_sol_count

    def stats(self):
        return {
            "constraints": self.native_model.NumConstrs,
        }

