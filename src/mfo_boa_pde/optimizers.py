"""MFO / BOA / hybrid optimizers and the parallel-execution helper.

Moved verbatim from notebook_code.py (CELL 2 _run_all, CELL 6 optimizers).
joblib stays optional: the try/except fallback is preserved, but the
"joblib not found" message is NOT printed at import (import must be silent);
main() prints it when the fallback is in effect.
"""
import math

import numpy as np

from .config import ALG_PARAMS

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False
    Parallel = delayed = None

def _run_all(n_jobs, calls):
    """اجرای فهرست فراخوانی‌های (تابع، آرگومان‌ها): موازی با joblib، وگرنه ترتیبی.
    نتیجه‌ی عددی در هر دو حالت عیناً یکسان است — هر اجرا RNG مستقل خودش را از run_seed
    می‌گیرد (مستقل از ترتیب اجرا) — و زمان‌بندی هر اجرا جداگانه در time_s ثبت می‌شود."""
    if _HAS_JOBLIB:
        return Parallel(n_jobs=n_jobs, backend="loky")(delayed(fn)(*args) for fn, args in calls)
    return [fn(*args) for fn, args in calls]

def _rnd(x):                     # گرد کردن «نیم به بالا» (نه banker's rounding)
    return int(math.floor(x + 0.5))

def _clip(P, lb, ub):
    return np.minimum(np.maximum(P, lb), ub)

def mfo_move(pop, fit, k, T, b, lb, ub, rng):
    n, dim = pop.shape
    flames = pop[np.argsort(fit)]
    m = max(1, _rnd(n - k * (n - 1) / T))              # m(k) = round(N - k (N-1)/T)
    F = flames[np.minimum(np.arange(n), m - 1)]
    D = np.abs(F - pop)
    t = rng.uniform(-1.0, 1.0, (n, dim))
    return _clip(D * np.exp(b * t) * np.cos(2 * np.pi * t) + F, lb, ub)

def boa_move(pop, fit, g_best, c, a, p, lb, ub, rng):
    n, dim = pop.shape
    I = 1.0 / (1.0 + np.maximum(fit, 0.0))              # شدت نرمال‌شده برای کمینه‌سازی
    f = (c * I ** a)[:, None]
    r2 = rng.random((n, 1)) ** 2
    glob = rng.random(n) < p
    if n < 3:
        glob[:] = True                                  # ASSUMPTION: فاز محلی نیازمند سه اندیس متمایز
    idx = np.arange(n)
    if n >= 3:
        o1 = rng.integers(1, n, n)                      # افست‌های متمایز در {1..n-1}
        o2 = rng.integers(1, n - 1, n)
        o2 = np.where(o2 >= o1, o2 + 1, o2)
        j, k2 = (idx + o1) % n, (idx + o2) % n
        local = pop + (r2 * pop[j] - pop[k2]) * f
    else:
        local = pop
    gl = pop + (r2 * g_best[None, :] - pop) * f
    return _clip(np.where(glob[:, None], gl, local), lb, ub)

def _prep(bounds, dim):
    dim = dim or int(np.size(bounds[0]))
    lb, ub = (np.broadcast_to(np.asarray(v, float), (dim,)) for v in bounds)
    return lb, ub, dim

def _generations(n_pop, max_evals):
    assert (max_evals - n_pop) % n_pop == 0 and max_evals >= 2 * n_pop, \
        "max_evals باید برابر N*(1+T) با T صحیح باشد"
    return (max_evals - n_pop) // n_pop

def optimize_mfo(fitness, bounds, n_pop, max_evals, rng, dim=None, b=ALG_PARAMS["b"]):
    c0 = fitness.count
    T = _generations(n_pop, max_evals)
    lb, ub, dim = _prep(bounds, dim)
    pop = rng.uniform(lb, ub, (n_pop, dim))
    fit = fitness(pop)
    arch_x, arch_f = pop[np.argmin(fit)].copy(), float(fit.min())
    curve = [float(fit.min())]
    for k in range(1, T + 1):
        pop = mfo_move(pop, fit, k, T, b, lb, ub, rng)
        fit = fitness(pop)
        if fit.min() < arch_f:
            arch_f, arch_x = float(fit.min()), pop[np.argmin(fit)].copy()
        curve.append(float(fit.min()))
    n_ev = fitness.count - c0
    assert n_ev == max_evals, f"MFO: شمارنده {n_ev} != {max_evals}"
    return arch_x, np.array(curve), n_ev

def optimize_boa(fitness, bounds, n_pop, max_evals, rng, dim=None,
                 c=ALG_PARAMS["c"], a=ALG_PARAMS["a"], p=ALG_PARAMS["p"]):
    c0 = fitness.count
    T = _generations(n_pop, max_evals)
    lb, ub, dim = _prep(bounds, dim)
    pop = rng.uniform(lb, ub, (n_pop, dim))
    fit = fitness(pop)
    arch_x, arch_f = pop[np.argmin(fit)].copy(), float(fit.min())
    curve = [float(fit.min())]
    for k in range(1, T + 1):
        pop = boa_move(pop, fit, arch_x, c, a, p, lb, ub, rng)
        fit = fitness(pop)
        if fit.min() < arch_f:
            arch_f, arch_x = float(fit.min()), pop[np.argmin(fit)].copy()
        curve.append(float(fit.min()))
    n_ev = fitness.count - c0
    assert n_ev == max_evals, f"BOA: شمارنده {n_ev} != {max_evals}"
    return arch_x, np.array(curve), n_ev

def optimize_hybrid(fitness, bounds, n_pop, max_evals, rng, dim=None, HR=ALG_PARAMS["HR"], K=ALG_PARAMS["K"],
                    b=ALG_PARAMS["b"], c=ALG_PARAMS["c"], a=ALG_PARAMS["a"], p=ALG_PARAMS["p"]):
    assert n_pop >= 2, "هیبرید به دست‌کم ۲ عضو نیاز دارد (تقسیم به دو زیرجمعیت)"
    c0 = fitness.count
    T = _generations(n_pop, max_evals)
    lb, ub, dim = _prep(bounds, dim)
    n_m = max(1, min(n_pop - 1, _rnd(HR * n_pop)))
    pop0 = rng.uniform(lb, ub, (n_pop, dim))
    fit0 = fitness(pop0)
    M, fM = pop0[:n_m].copy(), fit0[:n_m].copy()
    B, fB = pop0[n_m:].copy(), fit0[n_m:].copy()
    arch_f = float(fit0.min()); arch_x = pop0[np.argmin(fit0)].copy()
    # آرشیو اختصاصی زیرجمعیت BOA — مطابق پایان‌نامه، تبادل اطلاعات فقط هر K تکرار انجام می‌شود؛
    # پس g* زیرجمعیت BOA از آرشیو خودش می‌آید، نه از آرشیو ترکیبی هیبرید.
    b_arch_f = float(fB.min()); b_arch_x = B[np.argmin(fB)].copy()
    curve = [arch_f]
    for k in range(1, T + 1):
        M = mfo_move(M, fM, k, T, b, lb, ub, rng)                 # ابتدا زیرجمعیت MFO
        B = boa_move(B, fB, b_arch_x, c, a, p, lb, ub, rng)       # سپس زیرجمعیت BOA (g* از آرشیو خودش)
        f_all = fitness(np.vstack([M, B]))
        fM, fB = f_all[:n_m].copy(), f_all[n_m:].copy()
        if f_all.min() < arch_f:
            arch_f = float(f_all.min()); arch_x = np.vstack([M, B])[np.argmin(f_all)].copy()
        if k % K == 0:                                            # اشتراک‌گذاری دوره‌ای — تنها کانال تبادل (بدون ارزیابی مجدد)
            iM, iB = np.argmin(fM), np.argmin(fB)
            wM, wB = np.argmax(fM), np.argmax(fB)
            bestM, bestB = M[iM].copy(), B[iB].copy()
            fbestM, fbestB = fM[iM], fB[iB]
            M[wM], fM[wM] = bestB, fbestB
            B[wB], fB[wB] = bestM, fbestM
        if fB.min() < b_arch_f:                                   # به‌روزرسانی آرشیو BOA پس از تبادل احتمالی
            b_arch_f = float(fB.min()); b_arch_x = B[np.argmin(fB)].copy()
        curve.append(arch_f)                                      # بهترینِ آرشیوشده تا این نسل (مطابق «بهترین برازندگی» پایان‌نامه)
    n_ev = fitness.count - c0
    assert n_ev == max_evals, f"MFO-BOA: شمارنده {n_ev} != {max_evals}"
    return arch_x, np.array(curve), n_ev

ALGOS = {"MFO": optimize_mfo, "BOA": optimize_boa, "MFO-BOA": optimize_hybrid}
