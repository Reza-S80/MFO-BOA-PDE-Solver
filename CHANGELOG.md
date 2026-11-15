# Changelog

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
