# Binding smoke test for the AMR queries: availability, data shape and exception translation.
# Their correctness is tested in C++ (HierarchicalAmrQueries_test).
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


# 4 x 3 level 0 cells; (2, 0) refined, then its child (5, 1), so swapping (i, j) changes the leaves.
degree = 2
n_s, n_t = 4, 3
marks = [[(2, 0)], [(5, 1)]]
knots_s, knots_t = open_knots(n_s, degree), open_knots(n_t, degree)
g_s, g_t = greville(knots_s, degree), greville(knots_t, degree)
cpts = np.array([[x for _ in g_t for x in g_s], [y for y in g_t for _ in g_s]])
nsd = splines.NavierStokesHierarchicalDiscretization(
    splines.KnotVector(knots_s, 1e-10), splines.KnotVector(knots_t, 1e-10), degree, degree, cpts, marks)

for name in ("activeLeafElementInfo", "l2LevelSupports"):
    if not hasattr(nsd, name):
        print("missing binding: " + name)
        sys.exit(1)

expected_leaves = []
cells = [(i, j) for j in range(n_t) for i in range(n_s)]
for level in range(len(marks) + 1):
    refined = set(marks[level]) if level < len(marks) else set()
    expected_leaves += [(level, cell) for cell in cells if cell not in refined]
    cells = [(2 * i + a, 2 * j + b) for i, j in refined for b in range(2) for a in range(2)]

infos = nsd.activeLeafElementInfo()
check(isinstance(infos, list), "activeLeafElementInfo: not a list")
leaves = []
for info in infos:
    if not (isinstance(info, dict) and set(info) == {"dart", "level", "indices"}):
        problems.append("activeLeafElementInfo: record " + repr(info) + " lacks exactly dart, level, indices")
        continue
    check(isinstance(info["dart"], int) and isinstance(info["level"], int), "dart, level not ints: " + repr(info))
    check(isinstance(info["indices"], tuple) and len(info["indices"]) == 2 and
          all(isinstance(i, int) for i in info["indices"]), "indices not a 2-tuple of ints: " + repr(info))
    leaves.append((info["level"], tuple(info["indices"])))
check(sorted(leaves) == sorted(expected_leaves),
      "leaves " + repr(sorted(leaves)) + " are not the leaves of the marks " + repr(sorted(expected_leaves)))

l2_levels = nsd.hierarchicalRefinementDiagnostics()["L2"]["levels"]
check(len(l2_levels) == len(marks) + 1, "L2 has " + str(len(l2_levels)) + " levels")
for level, summary in enumerate(l2_levels):
    supports = nsd.l2LevelSupports(level)
    if not isinstance(supports, dict):
        problems.append("l2LevelSupports(" + str(level) + "): not a dict")
        continue
    check(set(supports) == set(range(summary["full_count"])),
          "l2LevelSupports(" + str(level) + "): keys are not the ids of the level's " +
          str(summary["full_count"]) + " functions")
    cells = set()
    for fid, support in supports.items():
        check(isinstance(support, list) and len(support) > 0, "support of " + str(fid) + " is not a nonempty list")
        for cell in support:
            check(isinstance(cell, tuple) and len(cell) == 2 and all(isinstance(i, int) for i in cell),
                  "support cell of " + str(fid) + " not a 2-tuple of ints: " + repr(cell))
            cells.add(tuple(cell))
    level_cells = {(i, j) for j in range(n_t * 2**level) for i in range(n_s * 2**level)}
    check(cells == level_cells, "l2LevelSupports(" + str(level) + "): supports do not cover the level's cells")

try:
    nsd.l2LevelSupports(len(l2_levels))
    problems.append("l2LevelSupports past the last level did not raise")
except IndexError:
    pass

for problem in problems:
    print(problem)
print(str(len(problems)) + " problems")
sys.exit(1 if problems else 0)
