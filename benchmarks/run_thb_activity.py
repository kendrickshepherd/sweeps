#!/usr/bin/env python3
"""Build, verify and measure THB activity in fresh serial processes; stdout is one JSON report."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
TIMINGS = ("constructor_s", "policy_s", "reference_s", "production_activity_s")
STABLE = ("schema", "fixture", "depth", "degree", "verified", "leaves", "h1_functions",
          "hdiv_functions", "l2_functions", "extraction_nnz", "compared_levels", "activity_fingerprint",
          "compiler", "compiler_path", "build_type", "pointer_bits")


def validate_record(record, fixture, depth):
    if any(key not in record for key in STABLE + TIMINGS):
        raise ValueError("Incomplete native benchmark record")
    if (record["schema"] != 1 or record["verified"] is not True or record["fixture"] != fixture
            or record["depth"] != depth or record["degree"] != 3):
        raise ValueError("Native benchmark did not verify the requested fixture")
    expected = (3 if fixture == "F2" else 4) * (depth + 1)
    if record["compared_levels"] != expected:
        raise ValueError("Not every activity level was compared")
    for key in TIMINGS:
        value = record[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid timing: " + key)
    for key in ("leaves", "h1_functions", "hdiv_functions", "l2_functions", "extraction_nnz", "compared_levels"):
        if type(record[key]) is not int or record[key] < 0:
            raise ValueError("Invalid count: " + key)
    if not isinstance(record["activity_fingerprint"], str) or not record["activity_fingerprint"].isdigit():
        raise ValueError("Invalid activity fingerprint")


def require_repeat_equivalence(first, later):
    for key in STABLE:
        if first[key] != later[key]:
            raise ValueError("Repeat changed " + key + ": " + repr(first[key]) + " -> " + repr(later[key]))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_hashes():
    files = {ROOT / "CMakeLists.txt", ROOT / "test/CMakeLists.txt", ROOT / "test/ThbActivityReference_test.cpp"}
    files.update((ROOT / "src").rglob("*.cpp"))
    files.update((ROOT / "src").rglob("*.hpp"))
    files.update((ROOT / "src").rglob("CMakeLists.txt"))
    files.update((ROOT / "test/support").rglob("*.hpp"))
    files.update((ROOT / "benchmarks").glob("*.cpp"))
    files.update((ROOT / "benchmarks").glob("*.py"))
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(files)}


def artifact_hashes(build, binary):
    files = {binary}
    for suffix in ("*.dylib", "*.so", "*.dll"):
        files.update((build / "src").rglob(suffix))
    return {str(path.relative_to(build)): sha256(path) for path in sorted(files)}


def cache_values(build):
    wanted = {"CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE",
              "CMAKE_OSX_ARCHITECTURES", "CMAKE_OSX_SYSROOT", "CMAKE_GENERATOR", "CMAKE_HOME_DIRECTORY"}
    values = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if line.startswith(("#", "//")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        name = line.split(":", 1)[0]
        if name in wanted:
            values[name] = line.split("=", 1)[1]
    return values


def require_compatible_build(cache):
    home = cache.get("CMAKE_HOME_DIRECTORY")
    if not home or Path(home).resolve() != ROOT:
        raise ValueError("Build directory belongs to a different source checkout")
    if cache.get("CMAKE_BUILD_TYPE") != "Release":
        raise ValueError("Use a configured Release build; this runner does not change compiler settings")


def git_output(*args):
    return subprocess.check_output(["git", "-C", str(ROOT)] + list(args), text=True).rstrip("\r\n")


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="depths 0 and 2, one repeat (default)")
    mode.add_argument("--full", action="store_true", help="depths 0 through 7, three repeats; potentially expensive")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--fixtures", nargs="+", choices=("F1", "F1n", "F1w", "F2"))
    parser.add_argument("--depths", nargs="+", type=int, choices=range(8))
    parser.add_argument("--repeats", type=positive)
    parser.add_argument("--build-jobs", type=positive, default=2)
    args = parser.parse_args(argv)
    build = args.build_dir.resolve()
    cache = cache_values(build)
    require_compatible_build(cache)
    source_before = input_hashes()
    status_before, commit_before = git_output("status", "--porcelain"), git_output("rev-parse", "HEAD")
    subprocess.run(["cmake", "--build", str(build), "--target", "thb_activity_benchmark",
                    "--parallel", str(args.build_jobs)], check=True, stdout=sys.stderr)
    cache = cache_values(build)
    require_compatible_build(cache)
    if input_hashes() != source_before:
        raise ValueError("Sources changed during the build; discard the run")
    binary = build / "test/thb_activity_benchmark"
    if sys.platform == "win32":
        binary = binary.with_suffix(".exe")
    fixtures = args.fixtures or ["F1", "F1n", "F1w", "F2"]
    depths = args.depths if args.depths is not None else (list(range(8)) if args.full else [0, 2])
    repeats = args.repeats if args.repeats is not None else (3 if args.full else 1)
    artifacts_before = artifact_hashes(build, binary)
    provenance = {"utc_started": datetime.now(timezone.utc).isoformat(), "git_commit": commit_before,
                  "git_status": status_before.splitlines(), "dirty": bool(status_before),
                  "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
                  "cmake_cache": cache, "sources_sha256": source_before,
                  "artifacts_sha256": artifacts_before}
    records, summaries = [], []
    for fixture in fixtures:
        for depth in depths:
            group = []
            for repeat in range(repeats):
                print("Measuring {} depth {} repeat {}/{}".format(fixture, depth, repeat + 1, repeats), file=sys.stderr)
                completed = subprocess.run([str(binary), fixture, str(depth)], check=True, text=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                record = json.loads(completed.stdout)
                validate_record(record, fixture, depth)
                if group:
                    require_repeat_equivalence(group[0], record)
                record["repeat"] = repeat + 1
                group.append(record)
                records.append(record)
            summaries.append({"fixture": fixture, "depth": depth, "repeats": repeats,
                              "median_s": {key: statistics.median(row[key] for row in group) for key in TIMINGS}})
    if input_hashes() != source_before or artifact_hashes(build, binary) != artifacts_before:
        raise ValueError("Sources or build artifacts changed during measurement; discard the run")
    if git_output("rev-parse", "HEAD") != commit_before or git_output("status", "--porcelain") != status_before:
        raise ValueError("Git state changed during measurement; discard the run")
    provenance["utc_finished"] = datetime.now(timezone.utc).isoformat()
    json.dump({"schema": 1, "provenance": provenance, "runs": records, "summaries": summaries},
              sys.stdout, indent=2, sort_keys=True, allow_nan=False)
    print()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print("Benchmark command failed with exit {}: {}".format(error.returncode, error.cmd), file=sys.stderr)
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError) as error:
        print("Benchmark failed: " + str(error), file=sys.stderr)
        sys.exit(1)
