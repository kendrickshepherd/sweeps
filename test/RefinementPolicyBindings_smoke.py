# Binding smoke test for refinement policies: availability, record shape, read-only records and exception
# translation.  The policy is tested in C++ (NFormSupportPolicy_test, SharedInterfaceRefinement_test).
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


def attempt(message, call):
    try:
        check(call(), message)
    except Exception as e:
        problems.append(message + " (" + type(e).__name__ + ": " + str(e) + ")")


def open_knots(n_cells, degree):
    return [0.0] * degree + [float(i) for i in range(n_cells + 1)] + [float(n_cells)] * degree


def greville(knots, degree):
    return [sum(knots[k + 1:k + degree + 1]) / degree for k in range(len(knots) - degree - 1)]


def raised(call):
    try:
        call()
    except Exception as e:
        return e
    return None


for name in ("RefinementPolicy", "PolicyFailure", "PolicyCheck", "PolicyViolation"):
    if not hasattr(splines, name):
        print("missing binding: " + name)
        sys.exit(1)
for cls in (splines.NavierStokesHierarchicalDiscretization, splines.NavierStokesHierarchicalMultiPatchDiscretization):
    for name in ("checkRefinementPolicy", "requireRefinementPolicy"):
        if not hasattr(cls, name):
            print("missing binding: " + cls.__name__ + "." + name)
            sys.exit(1)

Policy = splines.RefinementPolicy
check(set(Policy.__members__) == {"NFormSupport", "ZeroFormSupport", "EM", "MHD"},
      "RefinementPolicy members are " + str(sorted(Policy.__members__)))
check(splines.PolicyViolation.__bases__ == (Exception,),
      "PolicyViolation bases are " + str(splines.PolicyViolation.__bases__))
check(not issubclass(splines.PolicyViolation, RuntimeError), "PolicyViolation is a RuntimeError")

# Degree 2 on 8 x 8 unit cells: the L2 space has degree 1, interior supports of 2 x 2 cells and corner supports of
# one cell.  single_bad's level 0 block {3, 4, 5}^2 is a union of supports; its level 1 cell (9, 8) is in none.
degree = 2
n_cells = 8
knots = open_knots(n_cells, degree)
g = greville(knots, degree)
n_cpts = len(g) ** 2


def cpts(x_offset=0.0):
    return np.array([[x_offset + x for _ in g for x in g], [y for y in g for _ in g]])


def single(cls, elems, rational=False):
    kv = splines.KnotVector(knots, 1e-10)
    if rational:
        return cls(kv, kv, degree, degree, cpts(), np.ones(n_cpts), elems)
    return cls(kv, kv, degree, degree, cpts(), elems)


def multi(marks):
    def geometry(x_offset):
        return splines.PatchGeometry(knot_vec_s=splines.KnotVector(knots, 1e-10),
                                     knot_vec_t=splines.KnotVector(knots, 1e-10),
                                     degree_s=degree, degree_t=degree, control_points=cpts(x_offset))

    gluing = [(splines.PatchSideRef(patch=0, side=splines.PatchSide.S1),
               splines.PatchSideRef(patch=1, side=splines.PatchSide.S0))]
    elems = [[splines.PatchElement(patch=p, i=i, j=j) for p, i, j in level] for level in marks]
    return splines.NavierStokesHierarchicalMultiPatchDiscretization([geometry(0.0), geometry(float(n_cells))],
                                                                     gluing, elems)


block = [(i, j) for j in (3, 4, 5) for i in (3, 4, 5)]
single_ok_marks = [[(0, 0)]]
single_bad_marks = [block, [(9, 8)]]
base = splines.NavierStokesHierarchicalDiscretization
builders = {
    "single_ok": lambda: single(base, single_ok_marks),
    "single_bad": lambda: single(base, single_bad_marks),
    "multi_ok": lambda: multi([[(1, 7, 7)]]),
    "multi_bad": lambda: multi([[(1, 5, 2)]]),
}
expected = {"single_bad": (Policy.NFormSupport, 0, 1, (9, 8)), "multi_bad": (Policy.NFormSupport, 1, 0, (5, 2))}

nsds = {}
for name, build in builders.items():
    try:
        nsds[name] = build()
    except Exception as e:
        problems.append(name + ": construction raised " + type(e).__name__ + ": " + str(e))


def fields(failure):
    return (failure.policy, failure.patch, failure.level, failure.cell)


def message(fixture):
    policy, patch, level, cell = expected[fixture]
    return ("Refinement policy NFormSupport is violated on patch " + str(patch) + ", level " + str(level) +
            ", cell (" + ", ".join(str(c) for c in cell) + ").")


checks = {}
for name in ("single_ok", "multi_ok"):
    if name not in nsds:
        continue
    nsd = nsds[name]
    attempt(name + ": check is not a PolicyCheck",
            lambda: isinstance(nsd.checkRefinementPolicy(Policy.NFormSupport), splines.PolicyCheck))
    attempt(name + ": check is not satisfied", lambda: nsd.checkRefinementPolicy(Policy.NFormSupport).satisfied is True)
    attempt(name + ": check failure is not None", lambda: nsd.checkRefinementPolicy(Policy.NFormSupport).failure is None)
    attempt(name + ": require did not return None", lambda: nsd.requireRefinementPolicy(Policy.NFormSupport) is None)

for name in ("single_bad", "multi_bad"):
    if name not in nsds:
        continue
    nsd = nsds[name]
    try:
        checks[name] = nsd.checkRefinementPolicy(Policy.NFormSupport)
    except Exception as e:
        problems.append(name + ": check raised " + type(e).__name__ + ": " + str(e))
        continue
    c = checks[name]
    attempt(name + ": check is satisfied", lambda: c.satisfied is False)
    attempt(name + ": check failure is not a PolicyFailure", lambda: isinstance(c.failure, splines.PolicyFailure))
    attempt(name + ": check failure fields are not " + str(expected[name]), lambda: fields(c.failure) == expected[name])
    attempt(name + ": check failure cell is not a tuple of ints",
            lambda: type(c.failure.cell) is tuple and all(type(x) is int for x in c.failure.cell))

    e = raised(lambda: nsd.requireRefinementPolicy(Policy.NFormSupport))
    attempt(name + ": require raised " + repr(e) + ", not PolicyViolation",
            lambda: isinstance(e, splines.PolicyViolation))
    attempt(name + ": require message is not '" + message(name) + "'", lambda: str(e) == message(name))
    attempt(name + ": require failure != check failure", lambda: e.failure == c.failure)
    attempt(name + ": require failure fields are not " + str(expected[name]), lambda: fields(e.failure) == expected[name])

if "single_bad" in checks:
    c = checks["single_bad"]
    for obj, attr, value in ((c.failure, "patch", 7), (c.failure, "cell", (0, 0)), (c, "satisfied", True),
                             (c, "failure", None)):
        attempt(type(obj).__name__ + "." + attr + " is writable",
                lambda: isinstance(raised(lambda: setattr(obj, attr, value)), AttributeError))

for name, nsd in nsds.items():
    for policy in (Policy.ZeroFormSupport, Policy.EM, Policy.MHD):
        text = "Refinement policy " + policy.name + " is not implemented."
        for form in ("checkRefinementPolicy", "requireRefinementPolicy"):
            e = raised(lambda: getattr(nsd, form)(policy))
            check(isinstance(e, NotImplementedError) and str(e) == text,
                  name + ": " + form + "(" + policy.name + ") raised " + repr(e) + ", not NotImplementedError('" +
                  text + "')")

unknown = Policy(4)
for name in ("single_bad", "multi_bad"):
    if name not in nsds:
        continue
    for form in ("checkRefinementPolicy", "requireRefinementPolicy"):
        e = raised(lambda: getattr(nsds[name], form)(unknown))
        check(isinstance(e, ValueError) and str(e) == "Unknown refinement policy.",
              name + ": " + form + "(RefinementPolicy(4)) raised " + repr(e) + ", not ValueError")

derived = (("NURBSNavierStokesHierarchicalDiscretization", True),
           ("AdaptiveLocalNavierStokesHierarchicalDiscretization", False),
           ("NURBSAdaptiveLocalNavierStokesHierarchicalDiscretization", True))
for fixture, marks in (("single_ok", single_ok_marks), ("single_bad", single_bad_marks)):
    if fixture not in nsds:
        continue
    reference = nsds[fixture].checkRefinementPolicy(Policy.NFormSupport)
    for cls_name, rational in derived:
        try:
            nsd = single(getattr(splines, cls_name), marks, rational)
        except Exception as e:
            problems.append(cls_name + " " + fixture + ": construction raised " + type(e).__name__ + ": " + str(e))
            continue
        attempt(cls_name + " " + fixture + ": check differs from the base's",
                lambda: (nsd.checkRefinementPolicy(Policy.NFormSupport).satisfied == reference.satisfied and
                         nsd.checkRefinementPolicy(Policy.NFormSupport).failure == reference.failure))
        e = raised(lambda: nsd.requireRefinementPolicy(Policy.NFormSupport))
        if reference.satisfied:
            check(e is None, cls_name + " " + fixture + ": require raised " + repr(e))
        else:
            attempt(cls_name + " " + fixture + ": require did not raise PolicyViolation with the base's failure",
                    lambda: isinstance(e, splines.PolicyViolation) and e.failure == reference.failure)

if "single_bad" in checks and "multi_bad" in checks:
    attempt("PolicyFailure == does not distinguish single_bad from multi_bad",
            lambda: checks["single_bad"].failure != checks["multi_bad"].failure)
    attempt("PolicyFailure == is not reflexive across copies",
            lambda: checks["single_bad"].failure ==
            nsds["single_bad"].checkRefinementPolicy(Policy.NFormSupport).failure)

for problem in problems:
    print(problem)
print(str(len(problems)) + " problems")
sys.exit(1 if problems else 0)
