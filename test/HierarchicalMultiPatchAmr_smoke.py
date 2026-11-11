# Binding smoke test for the multipatch AMR layer: availability, input records, data shape, selected exact values,
# the interiorEdgeSegments refusal and exception translation.  Correctness is tested in C++
# (HierarchicalMultiPatchAmr_test).
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


def raises(kind, message, call):
    try:
        call()
    except kind as e:
        return str(e) == message or message is None
    except Exception:
        return False
    return False


def outcome(call):
    try:
        value = call()
    except Exception as e:
        return type(e).__name__ + ": " + str(e)
    return "returned " + type(value).__name__ + " of " + str(len(value))


for name in ("PatchGeometry", "PatchSideRef", "PatchElement", "NavierStokesHierarchicalMultiPatchDiscretization"):
    if not hasattr(splines, name):
        print("missing binding: " + name)
        sys.exit(1)

# Patch 0 is [0, 2] x [0, 2] in 2 x 2 cells; patch 1 is [2, 5] x [0, 2] in 3 x 2 cells; patch 0's S1 glued to patch
# 1's S0.  Different cell counts and marks on the two patches, so a swapped patch or (i, j) changes the results.
degree = 2
patch_cells = [(2, 2), (3, 2)]
x_offsets = [0.0, 2.0]
marks = [[(0, 0, 1), (1, 2, 0)], [(1, 5, 1)]]


def patch_geometry(p, x_shift=0.0):
    n_s, n_t = patch_cells[p]
    knots_s, knots_t = open_knots(n_s, degree), open_knots(n_t, degree)
    g_s, g_t = greville(knots_s, degree), greville(knots_t, degree)
    cpts = np.array([[x_offsets[p] + x for _ in g_t for x in g_s], [y for y in g_t for _ in g_s]])
    cpts[0, 0] += x_shift
    return splines.PatchGeometry(knot_vec_s=splines.KnotVector(knots_s, 1e-10),
                                 knot_vec_t=splines.KnotVector(knots_t, 1e-10),
                                 degree_s=degree, degree_t=degree, control_points=cpts)


patches = [patch_geometry(0), patch_geometry(1)]
gluing = [(splines.PatchSideRef(patch=0, side=splines.PatchSide.S1),
           splines.PatchSideRef(patch=1, side=splines.PatchSide.S0))]
elems = [[splines.PatchElement(patch=p, i=i, j=j) for p, i, j in level] for level in marks]
build = splines.NavierStokesHierarchicalMultiPatchDiscretization
nsd = build(patches, gluing, elems)

for name in ("activeLeafElementInfo", "h1PatchLevelActiveIds", "l2PatchLevelActiveIds", "l2PatchLevelSupports",
             "interiorEdgeSegments"):
    if not hasattr(nsd, name):
        print("missing binding: " + name)
        sys.exit(1)

element = splines.PatchElement(patch=1, i=2, j=0)
check((element.patch, element.i, element.j) == (1, 2, 0), "PatchElement fields")
side = splines.PatchSideRef(patch=1, side=splines.PatchSide.S0)
check((side.patch, side.side) == (1, splines.PatchSide.S0), "PatchSideRef fields")
check((patches[1].degree_s, patches[1].degree_t, patches[1].control_points.shape) == (2, 2, (2, 20)),
      "PatchGeometry fields")

expected_leaves = []
for p, (n_s, n_t) in enumerate(patch_cells):
    cells = [(i, j) for j in range(n_t) for i in range(n_s)]
    for level in range(len(marks) + 1):
        refined = {(i, j) for q, i, j in marks[level] if q == p} if level < len(marks) else set()
        expected_leaves += [(p, level, cell) for cell in cells if cell not in refined]
        cells = [(2 * i + a, 2 * j + b) for i, j in refined for b in range(2) for a in range(2)]

infos = nsd.activeLeafElementInfo()
check(isinstance(infos, list), "activeLeafElementInfo: not a list")
leaves = []
darts = []
for info in infos:
    if not (isinstance(info, dict) and set(info) == {"dart", "patch", "level", "indices"}):
        problems.append("activeLeafElementInfo: record " + repr(info) + " lacks exactly dart, patch, level, indices")
        continue
    check(all(isinstance(info[key], int) for key in ("dart", "patch", "level")),
          "dart, patch, level not ints: " + repr(info))
    check(isinstance(info["indices"], tuple) and len(info["indices"]) == 2 and
          all(isinstance(i, int) for i in info["indices"]), "indices not a 2-tuple of ints: " + repr(info))
    leaves.append((info["patch"], info["level"], tuple(info["indices"])))
    darts.append(info["dart"])
check(sorted(leaves) == sorted(expected_leaves),
      "leaves " + repr(sorted(leaves)) + " are not the leaves of the marks " + repr(sorted(expected_leaves)))
element_darts = [e.dart for e in nsd.elements()]
check(sorted(darts) == sorted(element_darts) and len(set(darts)) == len(darts),
      "leaf darts are not the darts of elements(), once each")

# Hand derived, by support containment on each patch; no L2 function and none of these H1 functions spans the
# interface.  H1 has degree 2 and L2 degree 1 on each level's open knots, function k_s + (functions in s) k_t.
expected_active = {
    ("H1", 0, 0): sorted(set(range(16)) - {12}),
    ("H1", 0, 1): [24, 25, 30, 31],
    ("H1", 0, 2): [],
    ("H1", 1, 0): sorted(set(range(20)) - {4}),
    ("H1", 1, 1): [6, 7, 14, 15],
    ("H1", 1, 2): [],
    ("L2", 0, 0): sorted(set(range(9)) - {6}),
    ("L2", 0, 1): [15, 16, 20, 21],
    ("L2", 0, 2): [],
    ("L2", 1, 0): sorted(set(range(12)) - {3}),
    ("L2", 1, 1): [5, 6, 12, 13],
    ("L2", 1, 2): [50, 51],
}
active_query = {"H1": nsd.h1PatchLevelActiveIds, "L2": nsd.l2PatchLevelActiveIds}
for (space, p, level), expected in expected_active.items():
    actual = active_query[space](p, level)
    check(isinstance(actual, list) and all(isinstance(fid, int) for fid in actual),
          space + " active ids of patch " + str(p) + " level " + str(level) + " not a list of ints")
    check(actual == expected, space + " active ids of patch " + str(p) + " level " + str(level) + " are " +
          repr(actual) + ", not " + repr(expected))

for p, (n_s, n_t) in enumerate(patch_cells):
    for level in range(len(marks) + 1):
        where = "l2PatchLevelSupports(" + str(p) + ", " + str(level) + ")"
        supports = nsd.l2PatchLevelSupports(p, level)
        if not isinstance(supports, dict):
            problems.append(where + ": not a dict")
            continue
        n_functions = (n_s * 2**level + 1) * (n_t * 2**level + 1)
        check(set(supports) == set(range(n_functions)),
              where + ": keys are not the ids of the patch level's " + str(n_functions) + " functions")
        cells = set()
        for fid, support in supports.items():
            check(isinstance(support, list) and len(support) > 0, where + ": support of " + str(fid) + " is empty")
            for cell in support:
                check(isinstance(cell, tuple) and len(cell) == 2 and all(isinstance(i, int) for i in cell),
                      where + ": support cell of " + str(fid) + " not a 2-tuple of ints: " + repr(cell))
                cells.add(tuple(cell))
        level_cells = {(i, j) for j in range(n_t * 2**level) for i in range(n_s * 2**level)}
        check(cells == level_cells, where + ": supports do not cover the patch level's cells")

# L2 function k covers cells k - 1 and k of its direction.  Each changes if patches 0 and 1 swap.
exact_supports = {
    (0, 0, 4): {(0, 0), (1, 0), (0, 1), (1, 1)},
    (1, 0, 4): {(0, 0), (0, 1)},
    (1, 0, 11): {(2, 1)},
    (1, 1, 12): {(4, 0), (5, 0), (4, 1), (5, 1)},
    (0, 2, 46): {(0, 4), (1, 4), (0, 5), (1, 5)},
}
for (p, level, fid), expected in exact_supports.items():
    actual = set(nsd.l2PatchLevelSupports(p, level).get(fid, []))
    check(actual == expected, "l2PatchLevelSupports(" + str(p) + ", " + str(level) + ")[" + str(fid) + "] is " +
          repr(sorted(actual)) + ", not " + repr(sorted(expected)))

refusal = ("NotImplementedError: interiorEdgeSegments compares parametric coordinates as if all patches shared one "
           "chart; it is not implemented for multipatch discretizations.")
for form, call in (("nsd.interiorEdgeSegments()", nsd.interiorEdgeSegments),
                   ("NavierStokesDiscretization.interiorEdgeSegments(nsd)",
                    lambda: splines.NavierStokesDiscretization.interiorEdgeSegments(nsd))):
    got = outcome(call)
    check(got == refusal, form + ": " + got + ", not the NotImplementedError refusal")

# Patch 1 alone, marks (2, 0) then (5, 1): 20 pairs of leaves share an edge (by hand), one segment each.
single = splines.NavierStokesHierarchicalDiscretization(patches[1].knot_vec_s, patches[1].knot_vec_t, degree, degree,
                                                        patches[1].control_points, [[(2, 0)], [(5, 1)]])
got = outcome(single.interiorEdgeSegments)
check(got == "returned list of 20", "single-patch interiorEdgeSegments: " + got + ", not a list of 20")

n_levels = len(marks) + 1
for query in (nsd.h1PatchLevelActiveIds, nsd.l2PatchLevelActiveIds, nsd.l2PatchLevelSupports):
    check(raises(IndexError, "Patch is outside the hierarchical multipatch space.", lambda: query(2, 0)),
          query.__name__ + ": patch 2 did not raise IndexError with its message")
    check(raises(IndexError, "Level is outside the hierarchical multipatch space.", lambda: query(0, n_levels)),
          query.__name__ + ": level " + str(n_levels) + " did not raise IndexError with its message")
check(raises(IndexError, "A refinement names an element outside its patch level.",
             lambda: build(patches, gluing, [[splines.PatchElement(patch=0, i=2, j=0)]])),
      "a mark outside its patch did not raise IndexError with its message")
for where, bad in (("a repeated mark", [[splines.PatchElement(patch=0, i=0, j=1)] * 2]),
                   ("a mark that is not a leaf", [[splines.PatchElement(patch=1, i=2, j=0)],
                                                  [splines.PatchElement(patch=0, i=0, j=0)]])):
    check(raises(RuntimeError, "Asked to refine an element that isn't active", lambda: build(patches, gluing, bad)),
          where + " did not raise RuntimeError with its message")
check(raises(ValueError, "Patch geometries disagree on a shared H1 coefficient.",
             lambda: build([patches[0], patch_geometry(1, 1e-6)], gluing, elems)),
      "disagreeing geometry did not raise ValueError with its message")
check(raises(TypeError, None, lambda: build(patches, gluing, [[(1, 2, 0)]])),
      "a tuple in place of a PatchElement did not raise TypeError")

for problem in problems:
    print(problem)
print(str(len(problems)) + " problems")
sys.exit(1 if problems else 0)
