"""Run configuration for mfo_boa_pde (moved verbatim from notebook_code.py).

Owns the MODE/CONFIGS constants, seeds, weights, basis sizes and the
configuration hash. Importing this module has no side effects (no prints,
no directory creation, no RNG use).
"""
import hashlib
import json
import os

MODE = os.environ.get("NB_MODE", "DEMO")           # "DEMO" یا "FULL"
MASTER_SEED = 20261021

ALG_PARAMS = dict(b=1.0, c=0.01, a=0.1, p=0.8, K=10, HR=0.5)   # جدول پارامترهای پژوهش

CONFIGS = {
    "DEMO": dict(n_runs=3,  n_pop=20, T_updates=99,  # max_evals = 20*(1+99) = 2000
                 ofat_runs=3,
                 ofat_levels=dict(N=[10, 20], HR=[0.2, 0.5, 0.8], n=[16, 25], K=[5, 10]),
                 bench_runs=3, n_jobs=1),
    "FULL": dict(n_runs=30, n_pop=50, T_updates=200, # max_evals = 50*(1+200) = 10050
                 ofat_runs=30,
                 ofat_levels=dict(N=[10, 20, 30, 50, 100], HR=[0.2, 0.4, 0.5, 0.6, 0.8],
                                  n=[16, 20, 25, 30, 36], K=[5, 10, 20]),
                 bench_runs=10, n_jobs=max(1, (os.cpu_count() or 1))),
}
assert MODE in CONFIGS, "MODE باید DEMO یا FULL باشد"
CFG = CONFIGS[MODE]
N_POP, T_UPD = CFG["n_pop"], CFG["T_updates"]
MAX_EVALS = N_POP * (1 + T_UPD)          # شامل N_POP ارزیابی اولیه
N_RUNS = CFG["n_runs"]

W_FIT = dict(w1=1.0, w2=20.0, w3=20.0)   # ASSUMPTION: وزن‌های برازندگی

def _config_hash():
    """Hash of the run configuration AND of the package source files.

    The original hashed a single file (notebook_code.py); the package hashes
    every .py file in this package directory instead, so any module change
    invalidates stale checkpoints. The hash value differs from the baseline
    by design; checkpoint *content* is compared, never the hash.
    """
    _h = hashlib.sha256()
    _h.update(json.dumps(CFG, sort_keys=True, default=str).encode())
    _h.update(json.dumps(ALG_PARAMS, sort_keys=True, default=str).encode())
    _h.update(json.dumps(W_FIT, sort_keys=True, default=str).encode())
    _h.update(f"{MODE}|{MASTER_SEED}|{MAX_EVALS}|{N_RUNS}".encode())
    try:
        _pkg = os.path.dirname(os.path.abspath(__file__))
        for _name in sorted(os.listdir(_pkg)):
            if _name.endswith(".py"):
                with open(os.path.join(_pkg, _name), "rb") as _sf:
                    _h.update(_sf.read())
    except Exception:
        pass
    return _h.hexdigest()[:12]


CFG_HASH = _config_hash()

# ابعاد پایه برای «تعداد کل ضرایب» n (ASSUMPTION: n = nx*nt)
NXNT = {16: (4, 4), 20: (4, 5), 25: (5, 5), 30: (5, 6), 36: (6, 6), 64: (8, 8), 100: (10, 10), 144: (12, 12)}
COEF_BOUNDS = (-2.0, 2.0)
NCX, NCT = 30, 20                        # ۶۰۰ نقطه‌ی کالوکیشن
N_IC, N_BC = 41, 41

CACHE_DIR = "cache"  # CWD-relative (as in the original); created lazily by load_reference.
