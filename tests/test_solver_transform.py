"""
Test that solver transformations preserve model equivalence.

Similar to test_tocnf.py, but tests the full solver transformation pipeline
rather than just to_cnf. For each test case:
1. Apply the solver's transform() method to get the transformed constraints
2. Pretty-print what each constraint becomes after transformation
3. Solve both original and transformed models with OR-Tools to get all solutions
4. Assert that the solution sets are equal
"""

import warnings
import pytest
import cpmpy as cp
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl
from cpmpy.expressions.utils import argvals
from cpmpy.tools.xcsp3.globals import NotInDomain
from cpmpy.transformations.get_variables import get_variables
import itertools
from cpmpy.solvers.ortools import CPM_ortools
from cpmpy.solvers.gurobi import CPM_gurobi, Encoding


def generate_test_cases():
    """
    Generate test cases for solver transformation testing.

    Yields tuples of (list of constraints, test_id).
    """
    # Test variables
    a, b, c = cp.boolvar(shape=3, name=["a", "b", "c"])
    x = cp.intvar(1, 3, name="x")
    y, z = cp.intvar(0, 2, shape=2, name=["y", "z"])

    # Boolean logic
    yield [a], "single_boolvar"
    yield [a | b], "or"
    yield [a & b], "and"
    yield [a != b], "neq_bool"
    yield [a == b], "eq_bool"
    yield [a.implies(b)], "implies"
    yield [a.implies(b | c)], "implies_or"
    yield [a.implies(b & c)], "implies_and"
    yield [cp.Xor([a, b])], "xor"
    yield [cp.Xor([a, b, c])], "xor3"

    # Reification
    yield [(a == b) == c], "reify_eq"
    yield [(a | b) == c], "reify_or"
    yield [(a & b).implies(c)], "and_implies"
    yield [c.implies(a & b)], "implies_and2"

    # Linear constraints
    yield [x + y == 3], "sum_eq"
    yield [x + y <= 3], "sum_leq"
    yield [x + y >= 3], "sum_geq"
    yield [x + y != 3], "sum_neq"
    yield [x - y == 1], "sub_eq"
    yield [2 * x + 3 * y <= 10], "wsum_leq"

    # Comparisons with boolean sums
    yield [a + b + c == 1], "bool_sum_eq"
    yield [a + b + c >= 2], "bool_sum_geq"
    yield [a + b + c <= 1], "bool_sum_leq"
    yield [a + b + c != 1], "bool_sum_neq"

    # Reified linear constraints
    yield [a.implies(x + y <= 3)], "implies_linear"
    yield [a.implies(x == y)], "implies_eq"
    yield [(x > y).implies(a)], "linear_implies"
    yield [(x == y) == a], "linear_reify"

    # Global constraints that get decomposed
    yield [cp.AllDifferent([x, y, z])], "alldiff"
    yield [cp.AllDifferent([a, b, c])], "alldiff_bool"  # trivial but valid

    # Min/Max
    yield [cp.min([x, y]) == z], "min_eq"
    yield [cp.max([x, y]) == z], "max_eq"
    yield [cp.min([x, y]) <= 1], "min_leq"

    # Absolute value
    yield [cp.abs(x - 2) == y], "abs_eq"

    # Table constraints (get decomposed for MIP)
    yield [cp.Table([x, y], [[1, 0], [2, 1], [3, 2]])], "table"

    # Multiple constraints
    yield [x != y, y != z, x != z], "multi_neq"
    yield [a.implies(b), b.implies(c)], "chain_implies"
    yield [x + y == 3, a == (x > y)], "mixed"

    # Edge cases
    yield [cp.sum([a, b]) <= 4], "trivial_sum"  # always true
    yield [x >= 1, x <= 3], "bounds"  # redundant with domain

    # NoOverlap constraint (scheduling)
    # Minimal case: two tasks that must not overlap
    s1, s2 = cp.intvar(0, 10, shape=2, name=["s1", "s2"])
    yield [cp.NoOverlap([s1, s2], [3, 3])], "no_overlap_simple"

    yield [cp.NoOverlap([s1, s2], [3, 4])], "no_overlap_with_end"

    # NoOverlap with duration 8 (like reported failure) but smaller domains
    x = cp.intvar(0, 20, name="start1")
    y = cp.intvar(0, 20, name="start2")
    yield [cp.NoOverlap([x, y], [8, 8])], "no_overlap_duration8"

    # AssertionError: Constraint NoOverlap([x[0], x[8]], [15, 15], [IV14, IV15]) failed for assignment
    # x[0] in 129..559 = 164
    # x[8] in 135..591 = 150
    # IV14 in 144..574 = 179
    # IV15 in 150..606 = 165

    # # NoOverlap reproducing the reported failure pattern (large domains)
    # x = cp.intvar(129, 559, name="x_s")
    # y_s = cp.intvar(135, 591, name="y_s")
    # yield [cp.NoOverlap([x, y_s], [15, 15], [x + 15, y_s + 15])], "no_overlap_large_domain"

    x = cp.intvar(1, 4, name="x")
    y = cp.intvar(1, 3, name="y")
    z = cp.intvar(1, 3, name="z")
    X = (x, y, z)
    T = [[2, 1, 1], [3, 2, 2], [4, 3, 3], [1, 2, 3], [2, 1, 2]]
    yield [cp.Table(X, T)], "table_2_root"

    yield [cp.boolvar(name="p").implies(cp.Table(X, T))], "table_2_reif"
    # yield [~(cp.Table(X, T))], "table_2_negated"

    # bug found in xcsp: TankAllocation2-1200_c25
    yield [NotInDomain(cp.intvar(-1, 24), [0, 1, 2, 5, 6, 10, 11, 15, 16, 18, 19, 20, 21, 22, 24])], "not_in_domain"


def generate_solver_classes():
    """Return list of (name, solver_class) tuples for available solvers to test."""
    solvers = []
    if CPM_gurobi.supported():
        solvers.append(("gurobi", CPM_gurobi))

    for encoding in Encoding:
    # for encoding in [Encoding.CPMPY]:
        yield (
            f"base_gurobi-{encoding}",
            CPM_gurobi,
            {
                "verbose": 0,
                "encoding": encoding,
            },
        )

    # Add more solvers here as needed
    # return solvers


def idfn(val):
    """Generate test ID from test case."""
    if isinstance(val, tuple):
        if len(val) == 2 and isinstance(val[1], str):
            return val[1]  # model name
        return str(val[0])  # solver name
    return str(val)


# @pytest.fixture(params=get_solver_classes(), ids=lambda x: x[0])
# def solver_class(request):
#     """Fixture providing solver classes to test."""
#     return request.param


class TestSolverTransform:
    """Test that solver transformations preserve solution equivalence."""

    @pytest.mark.parametrize(("solver", "case"), itertools.product(generate_solver_classes(), generate_test_cases()), ids=idfn)
    def test_transform_equivalence(self, solver, case, capsys):
        """Test that transformed model has same solutions as original."""
        constraints, name = case
        constraints = [c.deepcopy() for c in constraints]
        solver_name, solver_class, solver_kwargs = solver
        solver = solver_class(**solver_kwargs)

        # # Skip if solver not available
        # if not SolverClass.supported():
        #     pytest.skip(f"{solver_name} not available")

        # Get original variables
        vs = cp.cpm_array(list(get_variables(constraints)))

        # Get solutions and validate - may be partial for large domains
        original_sols, original_complete = self.allsols(constraints, vs)

        # Create solver instance (without adding constraints yet)
        # We need to manually call transform to see the transformation
        # solver = SolverClass()

        # Print transformation for each constraint
        print(f"\n{'=' * 60}")
        print(f"Test: {name} (solver: {solver_name})")
        print(f"{'=' * 60}")

        all_transformed = []
        print(f"\nVariables: {', '.join(f'{v} ∈ [{v.lb}, {v.ub}]' for v in vs)}")
        for con in constraints:
            print(f"Original: {con}")
            transformed = solver.transform(con)
            all_transformed.extend(transformed)
            print(f"Transformed ({len(transformed)} constraint(s)):")
            for i, t in enumerate(transformed, 1):
                print(f"  {i}. {t}")


        transformed_sols, transformed_complete = self.allsols(all_transformed, vs, original_cons=constraints, ivarmap=solver.ivarmap)

        print(f"Original solutions: {len(original_sols)}{'' if original_complete else ' (partial)'}")
        print(f"Transformed solutions: {len(transformed_sols)}{'' if transformed_complete else ' (partial)'}")

        # If both are complete, assert full equivalence
        if original_complete and transformed_complete:
            assert original_sols == transformed_sols, (
                f"Solution mismatch for {name}!\n"
                f"Original constraints: {constraints}\n"
                f"Transformed constraints: {all_transformed}\n"
                f"Only in original: {original_sols - transformed_sols}\n"
                f"Only in transformed: {transformed_sols - original_sols}"
            )
        else:
            # For large domains, just verify the sampled solutions are valid
            # (validation already happened in allsols via original_cons)
            print(f"Large domain case - validated {len(transformed_sols)} solutions against original constraints")

    def allsols(self, cons, vs, original_cons=None, solution_limit=10000, timeout=60, ivarmap=None):
        """Get all solutions for the given constraints projected onto variables vs.

        Args:
            cons: List of constraints to solve
            vs: Variables to project solutions onto
            original_cons: Optional list of original constraints to validate each solution against
            solution_limit: Maximum number of solutions to enumerate
            timeout: Maximum time in seconds (default 60). Raises TimeoutError if exceeded.
            ivarmap: Optional dict mapping variable names to their int2bool encodings.
                     Used to decode integer variable values from Boolean encoding.

        Returns:
            Tuple of (set of solutions, bool indicating if enumeration was complete)
        """
        if not CPM_ortools.supported():
            pytest.skip("OR-Tools not available for solution enumeration")

        # Ensure we include a constraint that references all target variables,
        # otherwise an empty constraint list won't enumerate the variable domains.
        # We add a trivially true constraint that references all variables.
        all_cons = list(cons) if cons else []
        if len(vs) > 0:
            # Add dummy constraint that is always true but references all vars
            # sum(vs) >= lower_bound is always true
            lb = sum(v.lb for v in vs)
            all_cons.append(cp.sum(vs) >= lb)

        m = cp.Model(all_cons)
        sols = set()

        def collect():
            # Decode integer variable values from Boolean encoding if ivarmap provided
            if ivarmap:
                for v in vs:
                    if v.name in ivarmap:
                        enc = ivarmap[v.name]
                        decoded_val = enc.decode()
                        if decoded_val is not None:
                            v._value = decoded_val

            sol = tuple(argvals(vs))
            sols.add(sol)

            # Validate solution against original constraints if provided
            if original_cons is not None:
                for con in original_cons:
                    val = con.value()
                    if val is False:
                        # Build variable assignment string for error message
                        all_vars = get_variables(con)
                        var_assignments = "\n".join(f"{v.name} in {v.lb}..{v.ub} = {v.value()}" for v in all_vars)
                        raise AssertionError(f"Constraint {con} failed for assignment\n\n{var_assignments}")

        n = m.solveAll(solver="ortools", display=collect, solution_limit=solution_limit, time_limit=timeout)

        # Check if we hit the timeout
        if m.status().runtime >= timeout:
            raise TimeoutError(f"Solution enumeration timed out after {timeout} seconds")

        # Return solutions and whether enumeration was complete
        is_complete = n < solution_limit
        if not is_complete:
            warnings.warn(
                f"Solution enumeration incomplete: hit limit of {solution_limit} solutions. Model may have more solutions.",
                stacklevel=2,
            )
        return sols, is_complete


@pytest.mark.skipif(not CPM_gurobi.supported(), reason="Gurobi not available")
class TestGurobiTransformDetails:
    """More detailed tests specific to Gurobi transformations."""

    def test_transform_indicator_constraint(self):
        """Test that indicator constraints are correctly transformed."""
        a = cp.boolvar(name="a")
        x, y = cp.intvar(0, 5, shape=2, name=["x", "y"])

        con = a.implies(x + y <= 5)

        solver = CPM_gurobi()
        transformed = solver.transform(con)

        print(f"\nOriginal: {con}")
        print(f"Transformed: {transformed}")

        # Should result in an indicator constraint (->)
        assert any("->" in str(t) for t in transformed), f"Expected indicator constraint in transformation, got: {transformed}"

    def test_no_overlap_rejects_invalid_solution(self):
        """Test that the reported invalid NoOverlap solution is rejected by transformed constraints."""
        # Create NoOverlap constraint matching the failure case
        start1 = cp.intvar(492, 1231, name="x35")
        start2 = cp.intvar(491, 1098, name="x39")
        end1 = cp.intvar(500, 1239, name="end2246")
        end2 = cp.intvar(499, 1106, name="end2247")

        # AssertionError: Constraint NoOverlap([x[35], x[39]], [8, 8], [IV2246, IV2247]) failed for assignment
        # x[35] in 492..1231 = 717
        # x[39] in 491..1098 = 724
        # IV2246 in 500..1239 = 725
        # IV2247 in 499..1106 = 732

        con = cp.NoOverlap([start1, start2], [8, 8], [end1, end2])

        # Transform the constraint
        solver = CPM_gurobi()
        transformed = solver.transform([con])

        print(f"\nOriginal constraint: {con}")
        print(f"Transformed to {len(transformed)} constraints:")
        for i, t in enumerate(transformed, 1):
            print(f"  {i}. {t}")

        # Test invalid assignment values
        invalid_start1 = 717
        invalid_start2 = 724
        invalid_end1 = 725
        invalid_end2 = 732

        print(f"\nTesting that transformed constraints reject invalid assignment:")
        print(f"  {start1.name} = {invalid_start1}")
        print(f"  {start2.name} = {invalid_start2}")
        print(f"  {end1.name} = {invalid_end1}")
        print(f"  {end2.name} = {invalid_end2}")

        # Verify original constraint rejects this by setting values and checking
        start1._value = invalid_start1
        start2._value = invalid_start2
        end1._value = invalid_end1
        end2._value = invalid_end2
        original_value = con.value()
        print(f"\nOriginal constraint evaluates to: {original_value}")
        assert original_value is False, "Original NoOverlap constraint should reject overlapping tasks"

        # Clear values for the solving test
        start1._value = None
        start2._value = None
        end1._value = None
        end2._value = None

        # Check that transformed model with fixed values is unsatisfiable
        # Add constraints to fix the variables to the invalid values
        fixed_model = cp.Model(
            transformed
            + [
                start1 == invalid_start1,
                start2 == invalid_start2,
                end1 == invalid_end1,
                end2 == invalid_end2,
            ]
        )

        print(f"\nSolving transformed model with fixed invalid assignment...")
        is_sat = fixed_model.solve(solver="ortools")

        assert not is_sat, (
            f"Transformed constraints should reject the invalid assignment!\n"
            f"The transformed model with the fixed invalid assignment is satisfiable, "
            f"but the original constraint rejects it.\n"
            f"This means the transformation allows an invalid solution."
        )

        print("✓ Transformed model correctly rejects the invalid assignment (UNSAT)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
