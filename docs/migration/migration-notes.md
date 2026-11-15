# Migration notes — src-layout package (branch `refactor/src-package-migration`)

Behavior-preserving refactor of the flat `notebook_code.py` script into the
installable src-layout package `mfo_boa_pde`. No numerical behavior changed
on purpose; Gate 4 verifies that at rtol 1e-12 / atol 1e-14 against the
baseline recorded in this directory (`baseline.json`,
`baseline-demo-results.json`).

## Module map

| Module | From | Contents |
|---|---|---|
| `src/mfo_boa_pde/config.py` | CELL 2 (config block) | MODE, MASTER_SEED, ALG_PARAMS, CONFIGS, CFG, N_POP, T_UPD, MAX_EVALS, N_RUNS, W_FIT, NXNT, COEF_BOUNDS, NCX/NCT, N_IC/N_BC, CACHE_DIR, `_config_hash`, CFG_HASH |
| `src/mfo_boa_pde/problems.py` | CELL 8 | PROBLEMS, PROB_IDS, ALG_NAMES |
| `src/mfo_boa_pde/basis.py` | CELL 4, CELL 8, CELL 10 | DEG, clamped_knots, basis, greville, val_grid, heat_reference, burgers_cole_hopf, diffusion_mol, _ref_path, load_reference |
| `src/mfo_boa_pde/solver.py` | CELL 4, CELL 8, CELL 10, CELL 11 | pde_residual, Machine, Fitness, CountedFunc, SHOCK_*, shock_region, make_points, make_machine, check, lstsq_fit |
| `src/mfo_boa_pde/optimizers.py` | CELL 2, CELL 6 | _run_all (joblib-optional), _rnd, _clip, mfo_move, boa_move, _prep, _generations, optimize_mfo, optimize_boa, optimize_hybrid, ALGOS |
| `src/mfo_boa_pde/diagnostics.py` | CELL 16, CELL 22, CELL 26 | holm, summary_table, friedman_table, wilcoxon_pairs, report_problem, plot_problem, shock_ratio_sensitivity, ofat_problem |
| `src/mfo_boa_pde/experiment.py` | CELL 16 | run_seed, run_one, RESULTS, run_problem, main() (+ _validation_checks, _problem_sections, _ofat_section, _finalize) |
| `src/mfo_boa_pde/__init__.py` | — | Public API: `main`, `run_one`, `run_seed`; `__version__ = "1.1.5"` |

## Design decisions (deviations from the GPT plan, with reasons)

1. **`config.py` owns the run configuration** (the plan had experiment.py own
   it). Rationale: `Machine`/`Fitness` take `w=W_FIT` as a default argument
   and several modules need MODE/CFG/CFG_HASH; a dedicated config module
   avoids circular imports and avoids changing those signatures.
2. **`problems.py` is a dedicated module** (the plan left ownership open).
   PROBLEMS is needed by basis, solver, experiment and diagnostics; owning it
   in one place with only math/numpy imports breaks every cycle.
3. **`_run_all` lives in `optimizers.py`**, not experiment.py (per §10). It is
   a generic parallel/sequential helper; the only experiment dependency in
   diagnostics (ofat_problem) receives `run_one` as an injected argument.
4. **`_config_hash` hashes the whole package.** The original hashed the
   single script file; the package hashes every `.py` file in its own
   directory (sorted) plus the config JSON. Checkpoint filenames therefore
   differ from the baseline by design (e.g. `ckpt_DEMO_92a880bc9b23_heat.pkl`
   vs `ckpt_DEMO_a4dbf673b7b8_heat.pkl`); Gate 4 compares canonicalized
   content, never raw filenames or pickle hashes.
5. **`CACHE_DIR = "cache"` stays CWD-relative** (per §10). The directory is
   created lazily inside `load_reference` (before `np.savez`) instead of at
   import time, so importing the package has no side effects.
6. **No prints at import.** The joblib-missing message, the MODE prints, the
   `pd.set_option` calls and the `warnings.filterwarnings` call all moved
   into `main()`, which reproduces the original top-level stdout sequence
   and execution order.
7. **`check()` lives in `solver.py`** (per plan); **`lstsq_fit` lives in
   `solver.py`** (per plan). **DEG stays in `basis.py`** (B-spline degree,
   not run config). **SHOCK_DENSITY_RATIO / SHOCK_SAMPLING_ENABLED stay in
   `solver.py`** next to `shock_region` / `make_points`.
8. **`report_problem(key, R)` / `plot_problem(key, R)`** take the result dict
   as an argument instead of reading `experiment.RESULTS` (avoids a circular
   import). `plot_problem` computes `x_eval` from `MAX_EVALS` internally.
9. **`diag_eansatz2.py` imports from the package** (no line-number
   extraction): `basis`, `load_reference` from `.basis`, `PROBLEMS` from
   `.problems`, `make_points` from `.solver`. It sets `NB_MODE=DEMO` and
   `MPLBACKEND=Agg` before importing, keeps the positional script argument
   as a compatibility argument (existence-checked only), and writes
   `diag_eansatz2.json` next to itself. `lsq_linear` is hoisted to a top
   import (was an inline import). All computation is verbatim.
10. **`notebook_code.py` is a thin launcher**: it bootstraps `src/` next to
    itself (if present) and falls back to the installed package, then calls
    `main()` only under `if __name__ == "__main__"`. `python notebook_code.py`
    (DEMO default) and `NB_MODE=FULL python notebook_code.py` keep working;
    an invalid NB_MODE still fails on the config assert.
11. **Tests**: `test_solver.py` still invokes the solver via subprocess
    (per §10 — it never imported notebook_code.py). `_run_demo` and
    `TestDiagEansatz2.setUpClass` now also copy the `src/` tree into the
    temp dir (the launchers bootstrap it from their own directory), and
    `test_full_config_is_the_thesis_config` reads
    `src/mfo_boa_pde/config.py` (the CONFIGS literal moved verbatim).
    New suites: `test_import_safety.py` (Gate 2), `test_compatibility.py`
    (Gate 3), `test_numerical_equivalence.py` (Gate 4).
12. **CI** installs the package with `pip install -e .` (not
    `pip install -r requirements.txt`) and gains a second job with
    `pip install -e ".[parallel]"`. `requirements.txt` stays synchronized
    with `pyproject.toml`. No version bump (1.1.5), no new release.

## Verification (gates) — run 2026-11-15 on this VM

- **Gate 0 — baseline** (commit e358e80, worktree
  `MFO-BOA-PDE-Solver-baseline`): DEMO exit 0 (~25 s), diag exit 0,
  27 runs / 81 OFAT rows / 27 results rows, CFG_HASH a4dbf673b7b8,
  diag A1 1.267958409687594e-06, A2 1.4846972673422742e-05,
  rank_ratio 1.043654172098714e-16, ok true. FULL baseline also ran to
  completion (~3 min): 270 runs / 1620 OFAT rows / 270 results rows, all
  n_evals = 10050, «assert نهایی» printed. Canonical data:
  `baseline-demo-results.json`, `baseline-full-results.json`, `baseline.json`.
- **Gate 1 — test suite: 27 tests, all pass** (`Ran 27 tests in 80.876s —
  OK` in the migration venv; also green in the clean venv, see Gate 6).
  14 in test_solver.py (DEMO, diag, static config) + 5 import-safety +
  3 compatibility + 5 numerical-equivalence.
- **Gate 2 — import safety: pass.** Package and every module import
  silently (exit 0, no stdout/stderr), create no files, do not consume the
  global NumPy RNG state; public API `main` / `run_one` / `run_seed` and
  `__version__ == "1.1.5"`.
- **Gate 3 — compatibility: pass.** Launcher import is safe (no execution,
  no files besides `__pycache__`); invalid NB_MODE fails on the config
  assert; FULL config via the package is n_runs=30, n_pop=50,
  T_updates=200, max_evals=10050. DEMO and diag CLI commands are covered
  by TestDemoRun / TestDiagEansatz2.
- **Gate 4 — DEMO numerical equivalence: pass** at rtol 1e-12 / atol 1e-14
  (curves compared with equal_nan=True — the first two curve points are
  NaN by construction, `left=np.nan` in `np.interp`). All 27 runs match on
  seed / n_evals / n_coeffs / RMSE / MaxAE / best_F / theta / curve; all
  81 OFAT rows and 27 results-CSV rows match; diag A1/A2/rank_ratio match.
- **Gate 5 — FULL config and FULL numerical equivalence: pass.** Static
  config verified (see Gate 3), and the migrated FULL run was executed and
  compared against `baseline-full-results.json`: **270/270 runs and
  1620/1620 OFAT rows match with worst relative difference 0.0**
  (bit-identical). Migrated FULL also prints «assert نهایی … 10050».
- **Gate 6 — clean venv install + CI: pass (local part).** Fresh venv:
  `pip install -e .` then `pip install -e ".[parallel]"` succeed;
  `pip show mfo-boa-pde-solver` reports 1.1.5, editable location the repo;
  `mfo_boa_pde.__file__` points at `src/mfo_boa_pde/__init__.py`;
  `_HAS_JOBLIB` is True with the parallel extra. Full suite green in the
  clean venv too (see `clean-venv-suite.log` — same 27 tests).
  `git status --short` clean apart from the branch changes;
  `git diff --check` clean. CI runs on the PR (two jobs: default and
  `[parallel]`).
