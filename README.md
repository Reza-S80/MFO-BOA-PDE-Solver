# MFO-BOA-PDE-Solver

Hybrid Moth-Flame Optimization (MFO) + Butterfly Optimization Algorithm (BOA) with B-spline collocation for solving nonlinear partial differential equations. MSc thesis code, Damghan University (2026).

## What it does

- Searches PDE-solver configurations with a hybrid MFO–BOA metaheuristic.
- Discretizes with B-spline / cubic-spline collocation (`scipy.interpolate.BSpline`, `CubicSpline`).
- `notebook_code.py`: the solver (functions `run_seed`, `lstsq_fit`, `run_one`); `diag_eansatz2.py`: E-ansatz diagnostic — `(1-e^(-t))` envelope + exact boundary condition `u(x,0) = sin(x)`.

## Results

Burgers' equation: the hybrid MFO–BOA optimizer reduced RMSE by 62% versus standalone MFO and by 69% versus standalone BOA. Wilcoxon signed-rank tests: p = 0.003 versus MFO and p = 0.001 versus BOA. These results are from the MSc thesis; this repository does not include a PINN baseline experiment.

## Requirements

- Python 3, NumPy, SciPy, pandas, matplotlib; joblib optional (falls back to sequential execution); IPython for `display`.

## Installation

```bash
git clone https://github.com/Reza-S80/MFO-BOA-PDE-Solver.git
cd MFO-BOA-PDE-Solver
pip install numpy scipy pandas matplotlib joblib ipython
```

## Usage

Run the demo benchmark (NB_MODE=DEMO by default), or the full experiment with NB_MODE=FULL. The diagnostic replays segments of notebook_code.py — pass its path as an argument, or omit it to use the copy in the same directory:

```bash
python notebook_code.py
NB_MODE=FULL python notebook_code.py
python diag_eansatz2.py notebook_code.py
```

## Thesis

Solving Nonlinear Partial Differential Equations Using the Hybrid Moth-Flame and Butterfly Metaheuristic Algorithm and Its Applications in Engineering Problems — MSc thesis, Damghan University, 2026.

## License

MIT.
