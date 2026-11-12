# MFO-BOA-PDE-Solver

Hybrid Moth-Flame Optimization (MFO) + Butterfly Optimization Algorithm (BOA) with B-spline collocation for solving nonlinear partial differential equations. MSc thesis code, Damghan University (2026).

## What it does
- Searches PDE-solver configurations with a hybrid MFO–BOA metaheuristic.
- Discretizes with B-spline / cubic-spline collocation (`scipy.interpolate.BSpline`, `CubicSpline`).
- `notebook_code.py`: the solver (functions `run_seed`, `lstsq_fit`, `run_one`); `diag_eansatz2.py`: E-ansatz diagnostic — `(1-e^(-t))` envelope + exact boundary condition `u(x,0) = sin(x)`.

## Requirements
- Python 3, NumPy, SciPy, pandas, matplotlib; joblib optional (falls back to sequential execution); IPython for `display`.

## Usage
- Run `notebook_code.py` for the solver. `diag_eansatz2.py` replays segments of it — pass the path to `notebook_code.py` as a command-line argument (the default path in `SRC` points to the author's machine).

## Thesis
Solving Nonlinear Partial Differential Equations Using the Hybrid Moth-Flame and Butterfly Metaheuristic Algorithm and Its Applications in Engineering Problems — MSc thesis, Damghan University, 2026.

## License
MIT.
