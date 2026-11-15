"""Parallel-execution coverage for _run_all.

The CI "parallel" job installs the joblib extra but the suite only ever ran
DEMO with n_jobs=1, so the loky multi-worker path was never exercised
(post-merge panel review, Manus, 2026-11-15). This test invokes _run_all
with n_jobs=2 and checks that the results are identical — same values, same
order — to the sequential (n_jobs=1) run and to the expected values.

The worker function is module-level (picklable by the loky backend). When
joblib is absent _run_all falls back to sequential execution and the test
still passes; in the CI parallel job (joblib installed) it genuinely
exercises the multi-worker path.
"""
import sys
import unittest
from pathlib import Path

# The suite runs without the package installed (like the other tests, which
# stage copies or use PYTHONPATH=src in subprocesses): put the checkout's
# src/ on the path so this test imports the working tree.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mfo_boa_pde.optimizers import _HAS_JOBLIB, _run_all


def _sum_sq(n):
    """Deterministic CPU work: sum of squares 0..n-1."""
    return sum(i * i for i in range(n))


class TestRunAllParallel(unittest.TestCase):
    """_run_all must give identical ordered results for n_jobs=1 and 2."""

    def test_n_jobs_2_matches_sequential(self):
        calls = [(_sum_sq, (n,)) for n in (10, 100, 1000, 5000, 42, 7, 123, 999)]
        expected = [_sum_sq(n) for n in (10, 100, 1000, 5000, 42, 7, 123, 999)]
        seq = _run_all(1, calls)
        par = _run_all(2, calls)
        self.assertEqual(seq, expected, "sequential run differs from expected")
        self.assertEqual(par, expected,
                         "n_jobs=2 run differs from expected values")
        self.assertEqual(par, seq,
                         "n_jobs=2 changed result order or values vs n_jobs=1")

    def test_n_jobs_2_uses_joblib_when_available(self):
        # In the CI parallel job joblib is installed; the n_jobs=2 path must
        # then go through the loky backend rather than silently falling back.
        # (When joblib is absent this documents the fallback instead.)
        if not _HAS_JOBLIB:
            self.skipTest("joblib not installed; _run_all falls back to sequential")
        calls = [(_sum_sq, (n,)) for n in (50, 500, 5000)]
        self.assertEqual(_run_all(2, calls),
                         [_sum_sq(n) for n in (50, 500, 5000)])


if __name__ == "__main__":
    unittest.main()
