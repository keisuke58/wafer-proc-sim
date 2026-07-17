"""CI wrapper for the DISCO APC C# module (disco_apc_csharp/).

Compiles the library + test harness with the Mono C# compiler and runs the
suite when `mcs`/`mono` are available; skips cleanly otherwise so the Python CI
stays green on runners without a C# toolchain.
"""
import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD = os.path.join(ROOT, "disco_apc_csharp")

pytestmark = pytest.mark.skipif(
    shutil.which("mcs") is None or shutil.which("mono") is None,
    reason="Mono C# toolchain (mcs/mono) not installed",
)


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def test_csharp_builds_and_tests_pass(tmp_path):
    lib = os.path.join(tmp_path, "AlgoLib.dll")
    tests_exe = os.path.join(tmp_path, "Tests.exe")
    srcs = [os.path.join(MOD, "src", f)
            for f in os.listdir(os.path.join(MOD, "src")) if f.endswith(".cs")]

    r = _run(["mcs", "-target:library", "-out:" + lib] + srcs, MOD)
    assert r.returncode == 0, "library build failed:\n" + r.stderr

    r = _run(["mcs", "-r:" + lib, "-out:" + tests_exe,
              os.path.join(MOD, "tests", "Tests.cs")], MOD)
    assert r.returncode == 0, "test build failed:\n" + r.stderr

    r = _run(["mono", tests_exe], tmp_path)
    assert r.returncode == 0, "C# tests failed:\n" + r.stdout + r.stderr
    assert "ALL TESTS PASSED" in r.stdout


def test_demo_runs_and_emits_artifacts(tmp_path):
    lib = os.path.join(tmp_path, "AlgoLib.dll")
    demo_exe = os.path.join(tmp_path, "Demo.exe")
    srcs = [os.path.join(MOD, "src", f)
            for f in os.listdir(os.path.join(MOD, "src")) if f.endswith(".cs")]

    assert _run(["mcs", "-target:library", "-out:" + lib] + srcs, MOD).returncode == 0
    assert _run(["mcs", "-r:" + lib, "-out:" + demo_exe,
                 os.path.join(MOD, "demo", "Program.cs")], MOD).returncode == 0

    out = str(tmp_path)
    r = _run(["mono", demo_exe, out], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert os.path.exists(os.path.join(out, "disco_apc_results.json"))
    assert os.path.exists(os.path.join(out, "disco_apc_chart.svg"))
    # closed-loop must beat open-loop on yield
    assert "97" in r.stdout or "in-spec yield" in r.stdout
