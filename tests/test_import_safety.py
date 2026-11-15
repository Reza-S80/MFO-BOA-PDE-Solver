"""Gate 2 — import-safety tests for the mfo_boa_pde package.

Importing the package (or any of its modules) must:
- exit 0 with no output,
- create no files in the working directory (no checkpoints, plots, cache),
- not consume the global NumPy RNG state,
- expose the public API (main, run_one, run_seed) and __version__ == "1.1.5".

All imports run in a subprocess in a fresh temp directory, so a side effect
cannot be masked by files already present in the repo.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
MODULES = ("config", "problems", "basis", "solver", "optimizers",
           "diagnostics", "experiment")


def _run_import(code, cwd):
    """Run `python -c <code>` with the repo src/ on the path; return proc."""
    env = dict(os.environ, MPLBACKEND="Agg",
               PYTHONPATH=str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)


class TestImportSafety(unittest.TestCase):
    """Importing the package must be silent and side-effect free."""

    def test_package_import_is_silent(self):
        with tempfile.TemporaryDirectory(prefix="import-safety-") as tmpdir:
            proc = _run_import("import mfo_boa_pde", tmpdir)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(proc.stdout, "", "import printed to stdout")
        self.assertEqual(proc.stderr, "", "import printed to stderr")

    def test_each_module_imports_silently(self):
        for mod in MODULES:
            with self.subTest(module=mod):
                with tempfile.TemporaryDirectory(prefix="import-safety-") as tmpdir:
                    proc = _run_import(f"import mfo_boa_pde.{mod}", tmpdir)
                self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
                self.assertEqual(proc.stdout, "")
                self.assertEqual(proc.stderr, "")

    def test_import_creates_no_files(self):
        with tempfile.TemporaryDirectory(prefix="import-safety-") as tmpdir:
            proc = _run_import(
                "import mfo_boa_pde; "
                "from mfo_boa_pde import experiment, diagnostics", tmpdir)
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            leftovers = [p.name for p in Path(tmpdir).iterdir()]
        self.assertEqual(leftovers, [],
                         f"import created files in the working dir: {leftovers}")

    def test_public_api(self):
        with tempfile.TemporaryDirectory(prefix="import-safety-") as tmpdir:
            proc = _run_import(
                "import mfo_boa_pde\n"
                "from mfo_boa_pde import main, run_one, run_seed\n"
                "assert mfo_boa_pde.__version__ == '1.1.5'\n"
                "assert callable(main) and callable(run_one) and callable(run_seed)\n"
                "assert mfo_boa_pde.main is main\n"
                "print('API-OK')", tmpdir)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("API-OK", proc.stdout)

    def test_import_does_not_consume_global_rng(self):
        # The legacy global RNG stream must be identical with and without
        # the import in between: importing must not draw from it.
        code_with_import = (
            "import numpy as np\n"
            "np.random.seed(42)\n"
            "_ = np.random.rand()\n"
            "import mfo_boa_pde\n"
            "print(repr(np.random.rand()))\n")
        code_without_import = (
            "import numpy as np\n"
            "np.random.seed(42)\n"
            "_ = np.random.rand()\n"
            "print(repr(np.random.rand()))\n")
        with tempfile.TemporaryDirectory(prefix="import-safety-") as tmpdir:
            p1 = _run_import(code_with_import, tmpdir)
            p2 = _run_import(code_without_import, tmpdir)
        self.assertEqual(p1.returncode, 0, p1.stderr[-2000:])
        self.assertEqual(p2.returncode, 0, p2.stderr[-2000:])
        self.assertEqual(p1.stdout.strip(), p2.stdout.strip(),
                         "import consumed global NumPy RNG state")


if __name__ == "__main__":
    unittest.main()
