# Changelog

## Unreleased

### Added

- Migrated the solver into an installable package, `src/mfo_boa_pde` (PR #7, merge c1541ca).
- Package execution via `python -m mfo_boa_pde`.
- Import-safety coverage of the `__main__` module.
- Opt-in FULL numerical-equivalence testing against `baseline-full-results.json` (270 runs, 270 result-CSV rows, 1,620 OFAT rows).
- Deterministic parallel tests for `_run_all` with `n_jobs=2` versus sequential execution.
- Numerical-equivalence canonicalization supporting DEMO and FULL modes.
- Scheduled weekly FULL numerical-equivalence CI job (`.github/workflows/full-equivalence.yml`).

### Fixed

- `_config_hash` re-raises source-file read errors instead of silently accepting incomplete hashes.
- `diag_eansatz2.py` uses `os.environ.setdefault("NB_MODE", "DEMO")`.
- Obsolete file references removed from `test_solver.py` documentation.
- Loose `0.9 * floor` assertion replaced with a tolerance-based lower-bound assertion.
- README test count corrected from 27 to 33, with the run-count matrix by environment and the FULL opt-in explanation.
- Burgers headline figures now carry the approximation-floor disclosure (default-basis floor 6.801e-02, ~243x the reported 2.8e-4; largest-basis floor 1.52e-02, ~54x).
- joblib pinned to 1.6.0 in the `parallel` extra.

## v1.1.5 (2026-11-14)

- Added a 14-test suite using Python's standard-library `unittest` framework (`tests/test_solver.py`).
- Added continuous integration testing on Python 3.12.
- Pinned the tested dependencies: NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.1, matplotlib 3.11.2, and IPython 9.18.1.
- Fixed the DEMO execution crash caused by `NameError: W_FIT`.
- Fixed segment drift in `diag_eansatz2.py` by correcting the source-code extraction boundaries.
- No changes to solver results; README figures remain the thesis-reported values.

## v1.1.4 (2026-11-13)
- Made joblib truly optional in requirements.txt (commented out; `pip install -r requirements.txt` no longer installs it). No code change from v1.1.3.

## v1.1.3 (2026-11-13)
- Linked the CITATION.cff and CHANGELOG.md references in the README; marked joblib optional in requirements.txt. No code change from v1.1.2.

## v1.1.2 (2026-11-13)
- Strengthened CITATION.cff (author ORCID + repository URL); added a "How to cite" section, an RMSE figure (docs/rmse-burgers.png), and a CHANGELOG link to the README. No code change from v1.1.1.

## v1.1.1 (2026-11-13)
- Spelling only: unified "M.Sc." in the README. Adds CHANGELOG.md and CITATION.cff. No code change from v1.1.

## v1.1 (2026-11-13)
- Added absolute mean-RMSE table to the README (Hybrid MFO–BOA 2.8e-4, MFO 7.3e-4, BOA 9.1e-4; Burgers' equation, 30 runs, thesis Table 3-2). No code change from v1.0.

## v1.0 (2026-11-13)
- First public release: M.Sc. thesis code (hybrid MFO–BOA with B-spline collocation for nonlinear PDEs, Damghan University, 2026).
