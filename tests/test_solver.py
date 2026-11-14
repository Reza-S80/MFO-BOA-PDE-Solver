"""Test suite for MFO-BOA-PDE-Solver.

Run:  python -m unittest discover -s tests -v
Requires MPLBACKEND=Agg (set by CI and by the subprocess environment here).
Measured DEMO wall time in the clean env: 159 s (2026-11-14, this VM).

Design notes
------------
notebook_code.py has no main guard: importing it runs the whole experiment.
Tests therefore run it as a subprocess in a fresh temp directory, so the
checkpoint files (ckpt_DEMO_*.pkl) cannot mask a failed execution, and assert
on the exit code, stdout, and the artifacts it writes (results_DEMO.csv,
ofat_DEMO.csv).

What the DEMO run itself already gates (hard checks, exit non-zero on
failure): B-spline partition of unity and polynomial reproduction (check 1),
analytic vs finite-difference derivatives (check 2), lstsq fits of the
reference solutions (check 3), manufactured-solution residual F(theta*) < 1e-24
with perturbed F > 1e-8 (check 4 — the PDE residual, BC and IC test), and the
scipy L-BFGS-B / least_squares baselines with sum(r^2) == F (check 5).
Check 6 (Sphere/Rastrigin convergence ratios) is a soft, reported-only
threshold by design — in the recorded DEMO run Rastrigin fails for all three
algorithms (ratios 0.64-0.94 vs threshold 0.5) and Sphere passes; the suite
records this but does not gate on it.

HONEST LIMITATION — the DEMO does not reproduce thesis Table 3-2, and the
tests below deliberately do NOT assert that it does:
- Thesis Table 3-2 (30 runs, FULL config: n_pop=50, T=200, max_evals=10050):
  Burgers hybrid mean RMSE 2.8e-4, MFO 7.3e-4, BOA 9.1e-4; 62%/69%;
  Wilcoxon p=0.003/0.001.
- The DEMO (3 runs, n_pop=20, T=99, max_evals=2000) measures Burgers means
  of ~0.61 (hybrid), ~0.62 (MFO), ~0.59 (BOA) — the hybrid does NOT beat
  BOA in the DEMO, and all three are ~2000x above Table 3-2.
- This is structural, not a matter of more evaluations: the code's own
  check-3 lstsq approximation floor for Burgers at the default 5x5 basis
  (25 coefficients) is 6.801e-02, and even the largest swept basis (144
  coefficients) has floor 1.52e-02 — still 54x above the claimed 2.8e-4.
  No optimizer in the repo's configuration space can beat the lstsq floor,
  so Table 3-2 is not reproducible by this code as configured (FULL uses
  the same 5x5 default). Heat and diffusion are different: their floors
  reach the thesis values at n=100-144. See FIXES.md for the floor table
  and what would be needed to close the gap.
- The tests therefore assert (a) the machinery is correct (checks 1-5),
  (b) the budget contract and finiteness, (c) the DEMO results match the
  recorded clean-env DEMO values within 5% (regression guard), and
  (d) every DEMO RMSE is at or above the run's own lstsq floor (the
  mathematical lower bound — guards against impossibly-good numbers).
  The FULL configuration is checked statically (it is the thesis
  configuration: n_runs=30, n_pop=50, T_updates=200).

Measured DEMO values below come from the clean-env DEMO run recorded in
phase1/demo-run-log-2026-11-14.txt (exit 0, 159 s). If the code changes,
re-run the DEMO and update the constants from the new log — never from memory.
"""

import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_TIMEOUT_S = 1800  # DEMO takes ~3 min on this VM; CI runners vary.
DIAG_TIMEOUT_S = 600

# --- Measured in the clean-env DEMO run 2026-11-14 (see the run log) ---
# Mean RMSE per (problem, algorithm) over the 3 DEMO runs.
DEMO_MEANS = {
    ("heat", "MFO"): 3.502933e-01,
    ("heat", "BOA"): 4.520589e-01,
    ("heat", "MFO-BOA"): 4.228320e-01,
    ("burgers", "MFO"): 6.179112e-01,
    ("burgers", "BOA"): 5.949733e-01,
    ("burgers", "MFO-BOA"): 6.125108e-01,
    ("diffusion", "MFO"): 3.955176e-01,
    ("diffusion", "BOA"): 3.850181e-01,
    ("diffusion", "MFO-BOA"): 3.255845e-01,
}
# diag_eansatz2.py: A2' PDE-constrained RMSE (heat), measured 2026-11-14.
DIAG_A2_RMSE = 1.4846972673422742e-05

# Thesis Table 3-2 (Burgers, 30 runs, FULL) — reference only; the DEMO does
# not reproduce these (see the module docstring). Do not assert them here.
THESIS_BURGERS_HYBRID_MEAN = 2.8e-4
THESIS_BURGERS_MFO_MEAN = 7.3e-4
THESIS_BURGERS_BOA_MEAN = 9.1e-4


def _run_demo(tmpdir):
    """Run the DEMO once in a fresh dir; return (returncode, stdout)."""
    for name in ("notebook_code.py", "diag_eansatz2.py"):
        shutil.copy(REPO_ROOT / name, Path(tmpdir) / name)
    env = dict(os.environ, MPLBACKEND="Agg", NB_MODE="DEMO",
               PYTHONUNBUFFERED="1")
    proc = subprocess.run(
        [sys.executable, "notebook_code.py"],
        cwd=tmpdir, env=env, capture_output=True, text=True,
        timeout=DEMO_TIMEOUT_S)
    return proc.returncode, proc.stdout + proc.stderr


def _read_results_csv(tmpdir):
    with open(Path(tmpdir) / "results_DEMO.csv", newline="") as f:
        return list(csv.DictReader(f))


def _parse_lstsq_floors(output):
    """Check-3 lstsq approximation floors, {(problem): floor RMSE}."""
    floors = {}
    for m in re.finditer(
            r"برازش lstsq \((\w+)\) RMSE_approx=([0-9.]+e[+-][0-9]+)",
            output):
        floors[m.group(1)] = float(m.group(2))
    return floors


class TestDemoRun(unittest.TestCase):
    """One DEMO execution, shared by all assertions in this class."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="solver-demo-")
        cls.returncode, cls.output = _run_demo(cls.tmpdir)
        cls.rows = None
        if Path(cls.tmpdir, "results_DEMO.csv").exists():
            cls.rows = _read_results_csv(cls.tmpdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_demo_exits_zero(self):
        self.assertEqual(
            self.returncode, 0,
            "DEMO must exit 0; output tail:\n" + self.output[-3000:])

    def test_demo_final_budget_assert_printed(self):
        self.assertIn("assert نهایی", self.output)
        self.assertIn("2000 ارزیابی مصرف کردند", self.output)
        self.assertIn("خلاصه‌ی اعتبارسنجی", self.output)

    def test_validation_checks_1_to_5_pass(self):
        # Hard gates inside the script: partition of unity, derivatives,
        # lstsq fits, manufactured-solution residual, scipy baselines.
        for marker in ("partition of unity", "مشتق اول", "مشتق دوم",
                       "برازش lstsq", "جواب ساختگی",
                       "Σr² = F", "L-BFGS-B ≤ مبنای lstsq"):
            self.assertIn(marker, self.output, f"missing check line: {marker}")
        # Check 6 (Sphere/Rastrigin) is soft by design and prints after the
        # first "Sphere" line, so only the part before it is gated here.
        self.assertNotIn("[FAIL]", self.output.split("Sphere", 1)[0],
                         "a hard validation check FAILED (checks 1-5 run "
                         "before the Sphere benchmark)")

    def test_results_csv_shape_and_budget(self):
        self.assertIsNotNone(self.rows, "results_DEMO.csv was not written")
        # 3 problems x 3 algorithms x 3 runs = 27 rows.
        self.assertEqual(len(self.rows), 27)
        problems = {r["problem"] for r in self.rows}
        self.assertEqual(problems, {"heat", "burgers", "diffusion"})
        algos = {r["algorithm"] for r in self.rows}
        self.assertEqual(algos, {"MFO", "BOA", "MFO-BOA"})
        for r in self.rows:
            self.assertEqual(int(r["n_evals"]), 2000,
                             "budget contract: every run must use 2000 evals")
            self.assertTrue(float(r["RMSE"]) > 0.0)
            self.assertTrue(float(r["RMSE"]) < 1.0)

    def test_results_csv_finite(self):
        for r in self.rows:
            for col in ("RMSE", "MaxAE", "best_F", "time_s"):
                self.assertTrue(math.isfinite(float(r[col])),
                                f"non-finite {col} in row {r}")

    def test_demo_means_match_recorded(self):
        # Guards against silent drift of the DEMO itself. Update the
        # constants from a fresh DEMO log if the code changes — never from
        # memory. These are the recorded DEMO values (~0.2-0.6), NOT the
        # thesis Table 3-2 values (see the module docstring).
        for (problem, alg), expected in DEMO_MEANS.items():
            vals = [float(r["RMSE"]) for r in self.rows
                    if r["problem"] == problem and r["algorithm"] == alg]
            self.assertEqual(len(vals), 3, f"{problem}/{alg}: expected 3 runs")
            mean = sum(vals) / len(vals)
            self.assertAlmostEqual(mean, expected, delta=0.05 * expected,
                                   msg=f"{problem}/{alg} mean drifted")

    def test_demo_rmse_above_lstsq_floor(self):
        # Mathematical lower bound: no theta in the basis space can beat
        # the least-squares fit on the same validation grid. Guards against
        # impossibly-good (buggy/fabricated) RMSE values.
        floors = _parse_lstsq_floors(self.output)
        self.assertEqual(set(floors), {"heat", "burgers", "diffusion"},
                         f"could not parse check-3 floors from stdout: {floors}")
        for r in self.rows:
            floor = floors[r["problem"]]
            self.assertGreaterEqual(float(r["RMSE"]), 0.9 * floor,
                                    f"RMSE below the lstsq floor in row {r}")

    def test_ofat_csv_written(self):
        path = Path(self.tmpdir) / "ofat_DEMO.csv"
        self.assertTrue(path.exists(), "ofat_DEMO.csv was not written")
        with open(path, newline="") as f:
            ofat = list(csv.DictReader(f))
        # DEMO OFAT: 9 levels x 3 runs x 3 problems = 81 rows.
        self.assertEqual(len(ofat), 81)


class TestDiagEansatz2(unittest.TestCase):
    """diag_eansatz2.py: E-ansatz diagnostic on the heat equation."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="solver-diag-")
        for name in ("notebook_code.py", "diag_eansatz2.py"):
            shutil.copy(REPO_ROOT / name, Path(cls.tmpdir) / name)
        env = dict(os.environ, MPLBACKEND="Agg", NB_MODE="DEMO",
                   PYTHONUNBUFFERED="1")
        cls.proc = subprocess.run(
            [sys.executable, "diag_eansatz2.py", "notebook_code.py"],
            cwd=cls.tmpdir, env=env, capture_output=True, text=True,
            timeout=DIAG_TIMEOUT_S)
        cls.json_path = Path(cls.tmpdir) / "diag_eansatz2.json"
        cls.result = None
        if cls.json_path.exists():
            cls.result = json.loads(cls.json_path.read_text())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_diag_exits_zero(self):
        self.assertEqual(
            self.proc.returncode, 0,
            "diag_eansatz2.py must exit 0; output tail:\n"
            + (self.proc.stdout + self.proc.stderr)[-3000:])

    def test_diag_json_written_and_ok(self):
        self.assertIsNotNone(self.result, "diag_eansatz2.json was not written")
        self.assertTrue(self.result["ok"],
                        f"A2' must be <= 2 * TARGET (3.2e-5): {self.result}")
        for key in ("A1", "A2", "rank_ratio"):
            self.assertTrue(math.isfinite(self.result[key]),
                            f"non-finite {key}: {self.result}")

    def test_diag_a2_matches_recorded_value(self):
        self.assertAlmostEqual(self.result["A2"], DIAG_A2_RMSE,
                               delta=0.05 * DIAG_A2_RMSE)


class TestStaticConfig(unittest.TestCase):
    """Static checks that need no execution."""

    def test_full_config_is_the_thesis_config(self):
        # The FULL configuration is what produced thesis Table 3-2
        # (30 runs, n_pop=50, T=200, max_evals=10050). Checked from the
        # source so the suite does not have to run FULL (hours). Note: FULL
        # uses the same default basis as DEMO, so even FULL cannot beat the
        # check-3 lstsq floor (see the module docstring).
        src = (REPO_ROOT / "notebook_code.py").read_text()
        m = re.search(r'"FULL":\s*dict\((.*?)\)\s*,?\n\}', src, re.S)
        self.assertIsNotNone(m, "FULL config block not found in source")
        block = m.group(1)
        self.assertIn("n_runs=30", block)
        self.assertIn("n_pop=50", block)
        self.assertIn("T_updates=200", block)

    def test_requirements_lists_core_deps(self):
        reqs = (REPO_ROOT / "requirements.txt").read_text()
        for dep in ("numpy", "scipy", "pandas", "matplotlib", "ipython"):
            self.assertIn(dep, reqs)

    def test_readme_thesis_numbers_unchanged(self):
        # Guardrail: the README headline numbers are thesis Table 3-2 and
        # must not drift (round-10 fabricated numbers must never land here).
        readme = (REPO_ROOT / "README.md").read_text()
        self.assertIn("62% versus standalone MFO", readme)
        self.assertIn("69% versus standalone BOA", readme)
        self.assertIn("p = 0.003 versus MFO", readme)
        self.assertIn("p = 0.001 versus BOA", readme)
        self.assertIn("2.8e-4", readme)
        self.assertIn("7.3e-4", readme)
        self.assertIn("9.1e-4", readme)
        for bad in ("59.2%", "68.1%", "1.806e-03", "4.426e-03",
                    "5.670e-03", "0.0012", "0.0004"):
            self.assertNotIn(bad, readme)


if __name__ == "__main__":
    unittest.main()
