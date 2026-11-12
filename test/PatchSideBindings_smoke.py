# Binding smoke test for boundaryEdges(side) and boundaryElementDarts(side): data shape, selected exact values and
# exception translation.  Their correctness is tested in C++ (PatchSideEdges_test).
import sys

try:
    import numpy as np
except ImportError:
    print("numpy is not available; not run")
    sys.exit(77)

import splines

problems = []


def check(condition, message):
    if not condition:
        problems.append(message)


def open_knots(n_cells, degree):
    return [0.0] * degree + [float(i) for i in range(n_cells + 1)] + [float(n_cells)] * degree


def greville(knots, degree):
    return [sum(knots[k + 1:k + degree + 1]) / degree for k in range(len(knots) - degree - 1)]


def outcome(call):
    try:
        call()
    except Exception as e:
        return type(e).__name__ + ": " + str(e)
    return "returned"


degree = 2


def discretization(n_s, n_t, marks=None):
    knots_s, knots_t = open_knots(n_s, degree), open_knots(n_t, degree)
    g_s, g_t = greville(knots_s, degree), greville(knots_t, degree)
    cpts = np.array([[x for _ in g_t for x in g_s], [y for y in g_t for _ in g_s]])
    kv_s, kv_t = splines.KnotVector(knots_s, 1e-10), splines.KnotVector(knots_t, 1e-10)
    if marks is None:
        return splines.NavierStokesTPDiscretization(kv_s, kv_t, degree, degree, cpts)
    return splines.NavierStokesHierarchicalDiscretization(kv_s, kv_t, degree, degree, cpts, marks)


# n_s != n_t in both orders, since T1 on a square mesh does not distinguish the source map from the line map.
nsds = {"tensor product 2 x 4": discretization(2, 4),
        "hierarchical 3 x 2": discretization(3, 2, [[(2, 1), (0, 0)], [(5, 3)]])}
sides = (splines.PatchSide.S0, splines.PatchSide.S1, splines.PatchSide.T0, splines.PatchSide.T1)

results = {}
for name, nsd in nsds.items():
    for side in sides:
        for method in ("boundaryEdges", "boundaryElementDarts"):
            where = name + " " + method + "(" + str(side) + ")"
            try:
                results[(name, side, method)] = getattr(nsd, method)(side)
            except Exception as e:
                problems.append(where + " raised " + type(e).__name__ + ": " + str(e))
                continue
        edges = results.get((name, side, "boundaryEdges"))
        darts = results.get((name, side, "boundaryElementDarts"))
        where = name + " " + str(side)
        if edges is not None:
            check(isinstance(edges, list) and len(edges) > 0 and all(isinstance(e, splines.Edge) for e in edges),
                  where + ": boundaryEdges is not a nonempty list of Edge")
        if darts is not None:
            check(isinstance(darts, list) and all(isinstance(d, int) for d in darts),
                  where + ": boundaryElementDarts is not a list of int")
        if edges is not None and darts is not None:
            check(len(edges) == len(darts), where + ": " + str(len(edges)) + " edges but " + str(len(darts)) +
                  " element darts")

# Hand derived T1 values.  Tensor product: dart pos + 4 (i + n_s j), T1 at pos 2 and j = n_t - 1, in order along
# the side.  Hierarchical: global darts number level by level; no order is claimed, so they are compared sorted.
exact = {("tensor product 2 x 4", "boundaryEdges"): ([splines.Edge(d) for d in (26, 30)], False),
         ("tensor product 2 x 4", "boundaryElementDarts"): ([24, 28], False),
         ("hierarchical 3 x 2", "boundaryEdges"): ([splines.Edge(d) for d in (14, 18, 114, 498, 502)], True),
         ("hierarchical 3 x 2", "boundaryElementDarts"): ([13, 16, 112, 496, 500], True)}
for (name, method), (expected, unordered) in exact.items():
    actual = results.get((name, splines.PatchSide.T1, method))
    if actual is None:
        continue
    check((sorted(actual) if unordered else actual) == expected,
          name + " " + method + "(T1) is " + repr(actual) + ", not " + repr(expected))

unknown = splines.PatchSide(7)
for name, nsd in nsds.items():
    for method in ("boundaryEdges", "boundaryElementDarts"):
        got = outcome(lambda: getattr(nsd, method)(unknown))
        check(got == "ValueError: Unknown patch side.",
              name + " " + method + "(PatchSide(7)): " + got + ", not ValueError: Unknown patch side.")

for problem in problems:
    print(problem)
print(str(len(problems)) + " problems")
sys.exit(1 if problems else 0)
