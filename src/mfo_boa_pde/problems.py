"""Problem definitions.

A dedicated module because PROBLEMS / PROB_IDS / ALG_NAMES are needed by
basis.py, solver.py, experiment.py and diagnostics.py; owning them here
avoids circular imports.
"""
import math

import numpy as np

PROBLEMS = {
    "heat": dict(title="Heat", kind="heat", par=dict(alpha=1.0), a=0.0, b=math.pi, T=1.0, tau=1e-4,
                 nxnt=(4, 4), ic=lambda x: np.sin(x)),
    "burgers": dict(title="Burgers", kind="burgers", par=dict(nu=0.01), a=0.0, b=1.0, T=1.0, tau=1e-3,
                    nxnt=(5, 5), ic=lambda x: np.sin(np.pi * x)),
    "diffusion": dict(title="Nonlinear diffusion", kind="diffusion", par=dict(D0=0.1, eps=0.5), a=0.0, b=1.0, T=1.0,
                      tau=1e-3, nxnt=(5, 5), ic=lambda x: np.sin(np.pi * x) + 0.5 * np.sin(2 * np.pi * x)),
}
PROB_IDS = {"heat": 0, "burgers": 1, "diffusion": 2}
ALG_NAMES = ["MFO", "BOA", "MFO-BOA"]
