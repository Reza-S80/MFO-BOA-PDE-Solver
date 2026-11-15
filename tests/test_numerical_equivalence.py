"""Gate 4 — DEMO numerical equivalence against the recorded baseline.

Runs its own DEMO subprocess (launcher + src/ tree in a fresh temp dir),
canonicalizes the checkpoints + CSVs + diag output the same way
docs/migration/baseline-demo-results.json was built from the baseline
(e358e80) run, and compares canonical numerical content at rtol 1e-12 /
atol 1e-14. Checkpoint *filenames* are never compared (CFG_HASH changes
when source bytes change); time_s is never compared (wall time).

The baseline file is committed in docs/migration/ (Gate 0 evidence).
"""

import csv
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_JSON = REPO_ROOT / "docs" / "migration" / "baseline-demo-results.json"
DEMO_TIMEOUT_S = 1800
DIAG_TIMEOUT_S = 600
RTOL = 1e-12
ATOL = 1e-14


def _canonicalize(tmpdir):
    """Canonical per-run / OFAT / CSV / diag data from a DEMO temp dir."""
    tmpdir = Path(tmpdir)
    out = {"runs": {}, "ofat_rows": [], "results_rows": [], "diag": None}

    for key in ("heat", "burgers", "diffusion"):
        for pkl in sorted(tmpdir.glob(f"ckpt_DEMO_*_{key}.pkl")):
            if not re.fullmatch(r"ckpt_DEMO_[0-9a-f]{12}_" + key + r"\.pkl", pkl.name):
                continue  # OFAT checkpoints (bare DataFrames) are not runs
            with open(pkl, "rb") as f:
                R = pickle.load(f)
            for _, row in R["df"].iterrows():
                k = f"{row['problem']}/{row['algorithm']}/{int(row['run'])}"
                out["runs"][k] = {
                    "seed": int(row["seed"]),
                    "n_evals": int(row["n_evals"]),
                    "n_coeffs": int(row["n_coeffs"]),
                    "RMSE": float(row["RMSE"]),
                    "MaxAE": float(row["MaxAE"]),
                    "best_F": float(row["best_F"]),
                    "theta": np.asarray(
                        R["thetas"][(row["algorithm"], int(row["run"]))], float),
                    "curve": np.asarray(
                        R["curves"][(row["algorithm"], int(row["run"]))], float),
                }

    with open(tmpdir / "ofat_DEMO.csv", newline="") as f:
        out["ofat_rows"] = list(csv.DictReader(f))
    with open(tmpdir / "results_DEMO.csv", newline="") as f:
        out["results_rows"] = list(csv.DictReader(f))
    diag_path = tmpdir / "diag_eansatz2.json"
    if diag_path.exists():
        out["diag"] = json.loads(diag_path.read_text())
    return out


def _assert_close(testcase, actual, expected, label):
    testcase.assertTrue(
        math.isclose(float(actual), float(expected), rel_tol=RTOL, abs_tol=ATOL),
        f"{label}: {actual!r} != {expected!r} (rtol={RTOL}, atol={ATOL})")


class TestNumericalEquivalence(unittest.TestCase):
    """One fresh DEMO + diag, compared against the committed baseline."""

    @classmethod
    def setUpClass(cls):
        assert BASELINE_JSON.exists(), (
            f"baseline file missing: {BASELINE_JSON} (Gate 0 evidence)")
        cls.baseline = json.loads(BASELINE_JSON.read_text())
        cls.tmpdir = tempfile.mkdtemp(prefix="solver-equiv-")
        for name in ("notebook_code.py", "diag_eansatz2.py"):
            shutil.copy(REPO_ROOT / name, Path(cls.tmpdir) / name)
        shutil.copytree(REPO_ROOT / "src", Path(cls.tmpdir) / "src")
        env = dict(os.environ, MPLBACKEND="Agg", NB_MODE="DEMO",
                   PYTHONUNBUFFERED="1")
        cls.demo = subprocess.run(
            [sys.executable, "notebook_code.py"],
            cwd=cls.tmpdir, env=env, capture_output=True, text=True,
            timeout=DEMO_TIMEOUT_S)
        cls.diag_proc = None
        if cls.demo.returncode == 0:
            cls.diag_proc = subprocess.run(
                [sys.executable, "diag_eansatz2.py", "notebook_code.py"],
                cwd=cls.tmpdir, env=env, capture_output=True, text=True,
                timeout=DIAG_TIMEOUT_S)
        cls.new = _canonicalize(cls.tmpdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_demo_exits_zero(self):
        self.assertEqual(self.demo.returncode, 0,
                         "equivalence DEMO must exit 0; output tail:\n"
                         + (self.demo.stdout + self.demo.stderr)[-3000:])

    def test_checkpoint_runs_match_baseline(self):
        base = self.baseline["runs"]
        self.assertEqual(len(self.new["runs"]), 27,
                         f"expected 27 runs, got {len(self.new['runs'])}")
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
        self.assertEqual(len(rows), 27)
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
        self.assertEqual(self.baseline["ofat"]["n_rows"], 81)
        self.assertEqual(len(self.new["ofat_rows"]), 81)

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

    def test_diag_matches_baseline(self):
        self.assertIsNotNone(self.diag_proc, "diag was not run (DEMO failed)")
        self.assertEqual(self.diag_proc.returncode, 0,
                         "diag_eansatz2.py must exit 0; output tail:\n"
                         + (self.diag_proc.stdout
                            + self.diag_proc.stderr)[-3000:])
        self.assertIsNotNone(self.new["diag"], "diag_eansatz2.json missing")
        base = self.baseline["diag"]
        for key in ("A1", "A2", "rank_ratio"):
            _assert_close(self, self.new["diag"][key], base[key],
                          f"diag {key}")
        self.assertTrue(self.new["diag"]["ok"], "diag ok must be true")
        self.assertTrue(base["ok"], "baseline diag ok must be true")


if __name__ == "__main__":
    unittest.main()
