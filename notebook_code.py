"""Thin compatibility launcher for the mfo_boa_pde package.

The solver now lives in src/mfo_boa_pde/ (installed with `pip install -e .`).
This script only bootstraps that package and runs its main(), preserving
the original command-line interface:

    python notebook_code.py               # DEMO (default)
    NB_MODE=FULL python notebook_code.py  # FULL (thesis configuration)

NB_MODE is read from the environment by mfo_boa_pde.config at import time;
an invalid value still fails (assert in config). Importing this module is
safe: the experiment only runs under `if __name__ == "__main__"`.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if os.path.isdir(_SRC):
    # Editable checkout: prefer the src/ tree next to this script.
    sys.path.insert(0, _SRC)
# Otherwise fall back to the installed package (site-packages).

from mfo_boa_pde.experiment import main

if __name__ == "__main__":
    main()
