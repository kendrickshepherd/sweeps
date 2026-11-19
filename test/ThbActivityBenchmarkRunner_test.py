"""Fast standard-library checks of the benchmark runner; no spline solve or numpy dependency."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    "thb_benchmark_runner", Path(__file__).resolve().parents[1] / "benchmarks/run_thb_activity.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def record():
    return dict(schema=1, fixture="F1", depth=2, degree=3, verified=True, leaves=145,
                h1_functions=196, hdiv_functions=392, l2_functions=196, extraction_nnz=28787,
                compared_levels=12, activity_fingerprint="123456", compiler="test", compiler_path="/test/c++",
                build_type="Release", pointer_bits=64,
                constructor_s=0.01, policy_s=0.001, reference_s=0.001, production_activity_s=0.001)


class RunnerChecks(unittest.TestCase):
    def test_release_build_of_this_checkout_is_accepted(self):
        runner.require_compatible_build({"CMAKE_HOME_DIRECTORY": str(runner.ROOT), "CMAKE_BUILD_TYPE": "Release"})

    def test_other_checkout_or_build_type_is_refused(self):
        for change in ({"CMAKE_HOME_DIRECTORY": str(runner.ROOT.parent)}, {"CMAKE_BUILD_TYPE": "Debug"},
                       {"CMAKE_HOME_DIRECTORY": ""}):
            cache = {"CMAKE_HOME_DIRECTORY": str(runner.ROOT), "CMAKE_BUILD_TYPE": "Release"}
            cache.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.require_compatible_build(cache)

    def test_main_refuses_other_checkout_before_building(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "CMakeCache.txt"
            cache.write_text("CMAKE_HOME_DIRECTORY:INTERNAL=" + str(runner.ROOT.parent)
                             + "\nCMAKE_BUILD_TYPE:STRING=Release\n")
            with mock.patch.object(runner.subprocess, "run") as build:
                with self.assertRaisesRegex(ValueError, "different source checkout"):
                    runner.main(["--build-dir", folder, "--quick"])
                build.assert_not_called()

    def test_main_rejects_source_changes_during_build_before_measuring(self):
        cache = {"CMAKE_HOME_DIRECTORY": str(runner.ROOT), "CMAKE_BUILD_TYPE": "Release"}
        with mock.patch.object(runner, "cache_values", return_value=cache), \
                mock.patch.object(runner, "require_configured_compiler", return_value={}), \
                mock.patch.object(runner, "input_hashes", side_effect=[{"input": "before"}, {"input": "after"}]), \
                mock.patch.object(runner, "git_output", side_effect=["", "commit"]), \
                mock.patch.object(runner.subprocess, "run") as build, \
                mock.patch.object(runner, "artifact_hashes") as artifacts:
            with self.assertRaisesRegex(ValueError, "Sources changed during the build"):
                runner.main(["--quick"])
            build.assert_called_once()
            artifacts.assert_not_called()

    def test_macro_identity_matches_cmake_naming(self):
        apple = {"__clang__": "1", "__GNUC__": "4", "__GNUC_MINOR__": "2", "__GNUC_PATCHLEVEL__": "1",
                 "__clang_major__": "17", "__clang_minor__": "0", "__clang_patchlevel__": "0",
                 "__apple_build_version__": "17000013"}
        self.assertEqual(runner.macro_identity(apple), ("AppleClang", "17.0.0.17000013"))
        llvm = {key: value for key, value in apple.items() if key != "__apple_build_version__"}
        self.assertEqual(runner.macro_identity(dict(llvm, __clang_major__="12")), ("Clang", "12.0.0"))
        self.assertEqual(runner.macro_identity({"__GNUC__": "13", "__GNUC_MINOR__": "2", "__GNUC_PATCHLEVEL__": "0"}),
                         ("GNU", "13.2.0"))
        for macros in ({}, {"__clang__": "1", "__clang_major__": "17"}):
            with self.subTest(macros=macros):
                self.assertIsNone(runner.macro_identity(macros))

    def write_compiler_file(self, folder, text):
        path = Path(folder) / "CMakeFiles/3.30.3"
        path.mkdir(parents=True)
        (path / "CMakeCXXCompiler.cmake").write_text(text)
        return {"CMAKE_CACHE_MAJOR_VERSION": "3", "CMAKE_CACHE_MINOR_VERSION": "30", "CMAKE_CACHE_PATCH_VERSION": "3",
                "CMAKE_CXX_COMPILER": "/test/mpic++"}

    def test_configured_identity_is_read_from_the_cache_version(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = self.write_compiler_file(folder, 'set(CMAKE_CXX_COMPILER_ID "AppleClang")\n'
                                                     'set(CMAKE_CXX_COMPILER_VERSION "17.0.0.17000013")\n')
            self.assertEqual(runner.configured_identity(Path(folder), cache), ("AppleClang", "17.0.0.17000013"))
            with self.assertRaisesRegex(ValueError, "Cannot find"):
                runner.configured_identity(Path(folder), dict(cache, CMAKE_CACHE_PATCH_VERSION="4"))

    def test_incomplete_configured_identity_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = self.write_compiler_file(folder, 'set(CMAKE_CXX_COMPILER_ID "AppleClang")\n')
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                runner.configured_identity(Path(folder), cache)

    def test_environment_compiler_must_match_configured_compiler(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = self.write_compiler_file(folder, 'set(CMAKE_CXX_COMPILER_ID "AppleClang")\n'
                                                     'set(CMAKE_CXX_COMPILER_VERSION "17.0.0.17000013")\n')
            with mock.patch.object(runner, "environment_identity", return_value=("AppleClang", "17.0.0.17000013")):
                self.assertEqual(runner.require_configured_compiler(Path(folder), cache),
                                 {"id": "AppleClang", "version": "17.0.0.17000013"})
            for current in (("Clang", "12.0.0"), ("AppleClang", "16.0.0.16000026"), None):
                with self.subTest(current=current), \
                        mock.patch.object(runner, "environment_identity", return_value=current), \
                        self.assertRaisesRegex(ValueError, "configure environment"):
                    runner.require_configured_compiler(Path(folder), cache)

    def test_main_refuses_other_compiler_before_building(self):
        cache = {"CMAKE_HOME_DIRECTORY": str(runner.ROOT), "CMAKE_BUILD_TYPE": "Release",
                 "CMAKE_CXX_COMPILER": "/test/mpic++"}
        with mock.patch.object(runner, "cache_values", return_value=cache), \
                mock.patch.object(runner, "configured_identity", return_value=("AppleClang", "17.0.0.17000013")), \
                mock.patch.object(runner, "environment_identity", return_value=("Clang", "12.0.0")), \
                mock.patch.object(runner.subprocess, "run") as build:
            with self.assertRaisesRegex(ValueError, "configure environment"):
                runner.main(["--quick"])
            build.assert_not_called()

    def test_valid_record(self):
        runner.validate_record(record(), "F1", 2)

    def test_nonfinite_and_negative_timings_are_rejected(self):
        for key in runner.TIMINGS:
            for value in (float("nan"), float("inf"), -0.1, True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    runner.validate_record(dict(record(), **{key: value}), "F1", 2)

    def test_unverified_wrong_fixture_and_missing_level_are_rejected(self):
        for change in ({"verified": False}, {"fixture": "F2"}, {"depth": 3}, {"compared_levels": 11}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.validate_record(dict(record(), **change), "F1", 2)

    def test_incomplete_records_are_rejected(self):
        row = record()
        del row["activity_fingerprint"]
        with self.assertRaises(ValueError):
            runner.validate_record(row, "F1", 2)

    def test_invalid_counts_are_rejected(self):
        for value in (-1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.validate_record(dict(record(), leaves=value), "F1", 2)

    def test_repeat_timings_may_change(self):
        runner.require_repeat_equivalence(record(), dict(record(), constructor_s=2.0))

    def test_repeat_counts_and_activity_must_not_change(self):
        for change in ({"h1_functions": 197}, {"activity_fingerprint": "654321"}, {"compiler": "different"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.require_repeat_equivalence(record(), dict(record(), **change))


if __name__ == "__main__":
    unittest.main()
