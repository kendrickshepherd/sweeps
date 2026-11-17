# THB activity reference and benchmark

These are the equivalence gates and repeatable baseline for optimization C.
They do **not** implement the leaf-local activity algorithm.

## Routine correctness tests

```sh
cmake --build build --target ThbActivityReference_test -j 8
build/test/ThbActivityReference_test
ctest --test-dir build -R python_thb_benchmark_runner_test --output-on-failure
```

`test/support/ThbActivityReference.hpp` preserves the full-domain algorithm
from `b99440c`. It must remain independent of `ThbActiveFunctions.hpp` when
production changes. It is an equivalence reference, not a proof of THB
correctness. The new single-patch tests also use an independent knot-support
oracle; the existing SharedInterfaceRefinement tests remain mathematical
regressions, especially for level jumps greater than one.

Every compared level checks **exact function-ID sets**, not just counts:

1. Frozen reference versus the direct production helper.
2. Frozen reference versus the constructor's stored active functions.

The matrix covers 2D and 3D; H1, H(div), H(curl), and L2; degrees 1--3;
unrefined and unused levels; uniform and asymmetric refinement; level jumps
of two; nonuniform knots; anisotropic and non-dyadic ratios; C0 knots and
their discontinuous reduced spaces; fixed-seed histories; reversed/swapped
two-patch interfaces; and four-patch junctions. These are fixture tests,
not an exhaustive refinement-class or exactness guarantee. Global
multipatch checks use the existing function maps; they do not independently
re-derive merging or orientation signs.

## Opt-in measurements

Use an already configured **Release** build with its existing dependencies.
The runner builds the native target through CMake before measuring, so it
uses the project's configured compiler and rebuilds changed dependencies.
It refuses a build directory configured from another source checkout, and
rejects sources that change during the build or measurements.
It needs only Python's standard library, not NumPy or StokesFlow.

```sh
# Small verification run: four fixtures, depths 0 and 2, one fresh process each.
python3 benchmarks/run_thb_activity.py --build-dir build --quick > /tmp/thb-quick.json

# Historical deep baseline, three independent processes per fixture.
python3 benchmarks/run_thb_activity.py --build-dir build \
    --fixtures F1 F1n F1w F2 --depths 7 --repeats 3 > /tmp/thb-depth7-before-c.json

# Full depth sweep: 96 fresh serial processes; potentially several minutes.
python3 benchmarks/run_thb_activity.py --build-dir build --full > /tmp/thb-full.json
```

The binary is `build/test/thb_activity_benchmark`. Its target is
`EXCLUDE_FROM_ALL` and is **not registered with ctest**. Full sweeps do not
run during routine testing. Run timings serially on an otherwise idle
machine; on macOS, `caffeinate -i python3 ...` can inhibit idle sleep.

Fixtures retain the original profiling definitions, with degree 3:

| Fixture | Background | Marked cells on each transition |
|---|---|---|
| F1 | One 7x7 patch | `[0,4)^2` at every level |
| F1n | One 6x6 NURBS patch, weights one | `[0,4)^2` at every level |
| F1w | One 7x7 patch | `[0,6)^2`, then `[0,10)^2`, then `[0,16)^2` |
| F2 | Two glued 7x7 patches | Rightmost 4x4 interface block on patch 0; leftmost on patch 1 |

All background levels refine dyadically; leaves are independent by patch.
Depth is limited to 0--7 because full-domain extraction storage makes larger
runs expensive in memory. Level 7 can already use several GB. This benchmark
does not address the storage changes proposed under D.

## Results and failure semantics

Stdout is one JSON report; build logs and progress go to stderr. Each
fixture/repeat gets a fresh process, so peak memory is not contaminated by
earlier runs. Every process enforces NFormSupportPolicy and exact reference,
direct-helper, and constructor comparisons before returning a successful
record. A mismatch **throws and produces a nonzero exit**, never just a
printed `laf_matches` flag. The runner also fails on changed deterministic
results between repeats, invalid/nonfinite timings, child-process failure,
or changes to sources, build artifacts, or git state during measurement.
Failed runs produce no successful JSON report.

The report includes:

- Constructor and policy-check seconds, separate from verification.
- Frozen-reference and production-helper seconds measured after construction
  (warm, reference first). These are diagnostic timings, not an isolated
  cold-start experiment. Verification time is not constructor time.
- Exact-checked level count, leaf/function counts, and extraction nonzeros.
- A stable activity fingerprint for repeat summaries. Exact set comparisons
  are the correctness gate; the fingerprint is not a cryptographic proof.
- Peak RSS immediately after construction and after checks. macOS reports
  bytes; Linux's KiB is normalized to bytes. Unsupported systems report
  null. Peak RSS is **not retained allocation or current resident memory**.
- Actual compiler version from the binary, configured compiler path/flags,
  platform, commit and dirty state, timestamps, and source/artifact SHA-256.

Dirty trees are permitted for development and explicitly labelled. For a
shareable before/after baseline, commit each tree and retain the reports
outside the checkout. Do not change files while a run is in progress.
Compare equivalent Release builds on the same machine. No timing threshold
is a correctness gate, and percentage speedups are observations rather than
guarantees.

The benchmark checks H1, H(div) and L2 to match the historical Stokes wrapper.
H(curl) is covered by the routine reference matrix, not the timing fixture.
