"""Allow `python -m mfo_boa_pde` — delegates to the experiment entry point.

The supported entry point remains the notebook_code.py compatibility launcher
at the repository root; this module exists so the installed package is also
runnable directly.
"""
from .experiment import main

if __name__ == "__main__":
    main()
