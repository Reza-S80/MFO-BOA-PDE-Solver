"""Gate 5 (automated) — FULL numerical equivalence against the recorded baseline.

The migration's FULL comparison (270 runs, 1620 OFAT rows, bit-identical) was
verified once by hand (see docs/migration/migration-notes.md, Gate 5). This
test makes it repeatable: it runs a fresh FULL subprocess and compares the
canonical numerical content against docs/migration/baseline-full-results.json
at rtol 1e-12 / atol 1e-14 — the same tolerances as the DEMO equivalence test.

OPT-IN: a FULL run takes ~3 minutes, so this test is SKIPPED by default.
Run it with:

    MFO_BOA_RUN_FULL_EQUIVALENCE=1 python -m unittest tests.test_full_equivalence -v

(or set the variable in CI / a nightly workflow). The diag is not compared:
diag_eansatz2.py is DEMO-only by design, and the FULL baseline records
diag: null.

Added after the post-merge panel review (Felo + Manus, 2026-11-15): the
DEMO-only suite did not reproduce the FULL comparison in CI.
"""
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_numerical_equivalence import (  # noqa: E402
    _assert_close, _canonicalize, ATOL, RTOL)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_JSON = REPO_ROOT / "docs" / "migration" / "baseline-full-results.json"
FULL_TIMEOUT_S = 1800

RUN_FULL = os.environ.get("MFO_BOA_RUN_FULL_EQUIVALENCE") == "1"


@unittest.skipUnless(RUN_FULL,
                     "FULL equivalence is opt-in: set "
                     "MFO_BOA_RUN_FULL_EQUIVALENCE=1 to run it (~3 min)")
class TestFullEquivalence(unittest.TestCase):
    """One fresh FULL run, compared against the committed baseline."""

    @classmethod
    def setUpClass(cls):
        assert BASELINE_JSON.exists(), (
            f"baseline file missing: {BASELINE_JSON} (Gate 5 evidence)")
        cls.baseline = json.loads(BASELINE_JSON.read_text())
        cls.tmpdir = tempfile.mkdtemp(prefix="solver-full-equiv-")
        shutil.copy(REPO_ROOT / "notebook_code.py",
                    Path(cls.tmpdir) / "notebook_code.py")
        shutil.copytree(REPO_ROOT / "src", Path(cls.tmpdir) / "src")
        env = dict(os.environ, MPLBACKEND="Agg", NB_MODE="FULL",
                   PYTHONUNBUFFERED="1")
        cls.full = subprocess.run(
            [sys.executable, "notebook_code.py"],
            cwd=cls.tmpdir, env=env, capture_output=True, text=True,
            timeout=FULL_TIMEOUT_S)
        cls.new = _canonicalize(cls.tmpdir, mode="FULL")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_full_exits_zero(self):
        self.assertEqual(self.full.returncode, 0,
                         "equivalence FULL must exit 0; output tail:\n"
                         + (self.full.stdout + self.full.stderr)[-3000:])

    def test_checkpoint_runs_match_baseline(self):
        base = self.baseline["runs"]
        self.assertEqual(len(self.new["runs"]), 270,
                         f"expected 270 runs, got {len(self.new['runs'])}")
        self.assertEqual(set(self.new["runs"]), set(base),
                         "run key set differs from baseline")
        for key, b in base.items():
            with self.subTest(run=key):
                n = self.new["runs"][key]
                self.assertEqual(n["seed"], b["seed"], f"{key}: seed")
                self.assertEqual(n["n_evals"], b["n_evals"], f"{key}: n_evals")
                self.assertEqual(n["n_coeffs"], b["n_coeffs"], f"{key}: n_coeffs")
                _assert_close(self, n["RMSE"], b["RMSE"], f"{key}: RMSE")
                _assert_close(self, n["MaxAE"], b["MaxAE"], f"{key}: MaxAE")
                _assert_close(self, n["best_F"], b["best_F"], f"{key}: best_F")
                self.assertTrue(
                    np.allclose(n["theta"], np.asarray(b["theta"], float),
                                rtol=RTOL, atol=ATOL, equal_nan=True),
                    f"{key}: theta differs beyond rtol={RTOL}/atol={ATOL}")
                self.assertTrue(
                    np.allclose(n["curve"], np.asarray(b["curve"], float),
                                rtol=RTOL, atol=ATOL, equal_nan=True),
                    f"{key}: curve differs beyond rtol={RTOL}/atol={ATOL}")

    def test_results_csv_matches_baseline(self):
        rows = self.new["results_rows"]
        self.assertEqual(len(rows), 270)
        self.assertEqual(self.baseline["results_csv_n_rows"], 270)
        base = self.baseline["runs"]
        for r in rows:
            key = f"{r['problem']}/{r['algorithm']}/{r['run']}"
            with self.subTest(run=key):
                b = base[key]
                _assert_close(self, r["RMSE"], b["RMSE"], f"{key}: CSV RMSE")
                _assert_close(self, r["MaxAE"], b["MaxAE"], f"{key}: CSV MaxAE")
                _assert_close(self, r["best_F"], b["best_F"], f"{key}: CSV best_F")
                self.assertEqual(int(r["n_evals"]), b["n_evals"])

    def test_ofat_rows_match_baseline(self):
        base_rows = self.baseline["ofat"]["rows"]
        self.assertEqual(self.baseline["ofat"]["n_rows"], 1620)
        self.assertEqual(len(self.new["ofat_rows"]), 1620)

        def _key(r):
            return (r["problem"], r["algorithm"], r["run"],
                    r["factor"], r["level"])

        base_by_key = {_key(r): r for r in base_rows}
        new_by_key = {_key(r): r for r in self.new["ofat_rows"]}
        self.assertEqual(set(new_by_key), set(base_by_key),
                         "OFAT row key set differs from baseline")
        for k, b in base_by_key.items():
            with self.subTest(ofat=k):
                n = new_by_key[k]
                _assert_close(self, n["RMSE"], b["RMSE"], f"{k}: OFAT RMSE")
                _assert_close(self, n["MaxAE"], b["MaxAE"], f"{k}: OFAT MaxAE")
                _assert_close(self, n["best_F"], b["best_F"], f"{k}: OFAT best_F")
                self.assertEqual(int(n["n_evals"]), int(b["n_evals"]))
                self.assertEqual(int(n["n_coeffs"]), int(b["n_coeffs"]))
                # lstsq_approx_RMSE is present only for factor "n"; it is
                # empty (NaN) for the other factors in both CSVs.
                if str(b.get("lstsq_approx_RMSE", "")) not in ("", "nan"):
                    _assert_close(self, n["lstsq_approx_RMSE"],
                                  b["lstsq_approx_RMSE"],
                                  f"{k}: OFAT lstsq_approx_RMSE")
                else:
                    self.assertIn(str(n.get("lstsq_approx_RMSE", "")),
                                  ("", "nan"),
                                  f"{k}: OFAT lstsq_approx_RMSE should be NaN")


if __name__ == "__main__":
    unittest.main()
