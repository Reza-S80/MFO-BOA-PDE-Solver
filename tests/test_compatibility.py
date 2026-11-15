"""Gate 3 — compatibility tests for the legacy entry points.

- Importing the notebook_code.py launcher must be safe (no execution).
- An invalid NB_MODE must still fail (the assert in config.py).
- The FULL configuration must be the thesis configuration, read from the
  package (not from the launcher).

The DEMO and diag CLI commands themselves are covered by TestDemoRun and
TestDiagEansatz2 in test_solver.py, which run the launchers in temp dirs.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _stage_tmpdir(tmpdir):
    """Copy the launchers and the src/ tree into a fresh temp dir."""
    for name in ("notebook_code.py", "diag_eansatz2.py"):
        shutil.copy(REPO_ROOT / name, Path(tmpdir) / name)
    shutil.copytree(REPO_ROOT / "src", Path(tmpdir) / "src")


def _run(args, tmpdir, extra_env=None):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    if extra_env:
        env.update(extra_env)
    return subprocess.run([sys.executable, *args], cwd=tmpdir, env=env,
                          capture_output=True, text=True, timeout=600)


class TestCompatibility(unittest.TestCase):
    """Legacy entry points keep their CLI contract."""

    def test_launcher_import_is_safe(self):
        # Importing the launcher must NOT run the experiment. (Python's
        # bytecode cache __pycache__ is a normal import artifact, not an
        # experiment side effect, so it is excluded from the check.)
        with tempfile.TemporaryDirectory(prefix="compat-") as tmpdir:
            _stage_tmpdir(tmpdir)
            proc = _run(["-c", "import notebook_code"], tmpdir)
            leftovers = sorted(p.name for p in Path(tmpdir).iterdir()
                               if p.name != "__pycache__")
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(proc.stdout, "", "importing the launcher printed output")
        self.assertEqual(proc.stderr, "", "importing the launcher printed to stderr")
        self.assertEqual(leftovers, ["diag_eansatz2.py", "notebook_code.py", "src"],
                         f"importing the launcher created files: {leftovers}")

    def test_invalid_nb_mode_fails(self):
        with tempfile.TemporaryDirectory(prefix="compat-") as tmpdir:
            _stage_tmpdir(tmpdir)
            proc = _run(["notebook_code.py"], tmpdir,
                        extra_env={"NB_MODE": "BOGUS"})
        self.assertNotEqual(proc.returncode, 0,
                            "invalid NB_MODE must fail, but the launcher exited 0")
        self.assertIn("MODE", proc.stderr,
                      "invalid NB_MODE must fail on the MODE assert")

    def test_full_config_via_package(self):
        # FULL is the thesis configuration: 30 runs, n_pop=50, T=200,
        # max_evals = 50 * (1 + 200) = 10050. Read from the package config.
        env = dict(os.environ, MPLBACKEND="Agg",
                   PYTHONPATH=str(REPO_ROOT / "src"))
        proc = subprocess.run(
            [sys.executable, "-c",
             "from mfo_boa_pde.config import CONFIGS\n"
             "f = CONFIGS['FULL']\n"
             "assert f['n_runs'] == 30, f\n"
             "assert f['n_pop'] == 50, f\n"
             "assert f['T_updates'] == 200, f\n"
             "assert f['n_pop'] * (1 + f['T_updates']) == 10050, f\n"
             "print('FULL-OK')"],
            capture_output=True, text=True, timeout=300, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("FULL-OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
