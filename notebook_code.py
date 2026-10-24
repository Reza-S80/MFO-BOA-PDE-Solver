
# ===== CELL 2 =====
import os, sys, time, platform, warnings, hashlib, json, math, pickle
from functools import lru_cache
import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats, optimize, integrate, sparse
from scipy.interpolate import BSpline, CubicSpline
try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    # fallback محافظه‌کارانه (try/except بی‌صدا ممنوع): اعلام صریح و اجرای ترتیبی
    _HAS_JOBLIB = False
    Parallel = delayed = None
    print("joblib یافت نشد — اجرای ترتیبی")
from IPython.display import display

def _run_all(n_jobs, calls):
    """اجرای فهرست فراخوانی‌های (تابع، آرگومان‌ها): موازی با joblib، وگرنه ترتیبی.
    نتیجه‌ی عددی در هر دو حالت عیناً یکسان است — هر اجرا RNG مستقل خودش را از run_seed
    می‌گیرد (مستقل از ترتیب اجرا) — و زمان‌بندی هر اجرا جداگانه در time_s ثبت می‌شود."""
    if _HAS_JOBLIB:
        return Parallel(n_jobs=n_jobs, backend="loky")(delayed(fn)(*args) for fn, args in calls)
    return [fn(*args) for fn, args in calls]

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")

# ═══════════════ تنظیمات: DEMO و FULL با مقادیر صریح ═══════════════
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

# هش پیکربندی برای اعتبارسنجی چک‌پوینت (جلوگیری از بارگذاری نتایج کهنه پس از تغییر کد/پارامتر)
def _config_hash():
    _h = hashlib.sha256()
    _h.update(json.dumps(CFG, sort_keys=True, default=str).encode())
    _h.update(json.dumps(ALG_PARAMS, sort_keys=True, default=str).encode())
    _h.update(json.dumps(W_FIT, sort_keys=True, default=str).encode())
    _h.update(f"{MODE}|{MASTER_SEED}|{MAX_EVALS}|{N_RUNS}".encode())
    try:
        with open(__file__, "rb") as _sf:
            _h.update(_sf.read())
    except Exception:
        pass
    return _h.hexdigest()[:12]

CFG_HASH = _config_hash()

# ابعاد پایه برای «تعداد کل ضرایب» n (ASSUMPTION: n = nx*nt)
NXNT = {16: (4, 4), 20: (4, 5), 25: (5, 5), 30: (5, 6), 36: (6, 6), 64: (8, 8), 100: (10, 10), 144: (12, 12)}
W_FIT = dict(w1=1.0, w2=20.0, w3=20.0)   # ASSUMPTION: وزن‌های برازندگی
COEF_BOUNDS = (-2.0, 2.0)
NCX, NCT = 30, 20                        # ۶۰۰ نقطه‌ی کالوکیشن
N_IC, N_BC = 41, 41
CACHE_DIR = "cache"; os.makedirs(CACHE_DIR, exist_ok=True)

print(f"MODE={MODE} | runs={N_RUNS} | N={N_POP} | T(update generations)={T_UPD} | max_evals={MAX_EVALS}")
print(f"n_jobs={CFG['n_jobs']} | collocation={NCX}x{NCT}={NCX*NCT} | weights={W_FIT} | bounds={COEF_BOUNDS}")

# ===== CELL 4 =====
# ═══════════════ ماشین B-spline ═══════════════
DEG = 3

def clamped_knots(a, b, n, k=DEG):
    assert n >= k + 1, "تعداد پایه باید دست‌کم k+1 باشد"
    inner = np.linspace(a, b, n - k - 1 + 2)[1:-1]
    return np.concatenate([np.full(k + 1, a), inner, np.full(k + 1, b)])

def basis(a, b, n, x, nu=0):
    """ماتریس (n, len(x)) از nu-امین مشتق تحلیلی پایه‌های B-spline مکعبی clamped."""
    spl = BSpline(clamped_knots(a, b, n), np.eye(n), DEG, extrapolate=False)
    return spl(np.asarray(x, float), nu=nu).T

def pde_residual(kind, par, U, Ux, Ut, Uxx):
    if kind == "heat":
        return Ut - par["alpha"] * Uxx
    if kind == "burgers":
        return Ut + U * Ux - par["nu"] * Uxx
    if kind == "diffusion":                     # u_t - (D(u) u_x)_x ، با جمله‌ی D'(u) u_x^2
        D = par["D0"] * (1.0 + par["eps"] * U ** 2)
        Dp = 2.0 * par["D0"] * par["eps"] * U
        return Ut - (Dp * Ux ** 2 + D * Uxx)
    raise ValueError(kind)

class Machine:
    """ماشین کالوکیشن تانسوری؛ همه‌ی ماتریس‌ها یک بار ساخته می‌شوند."""
    def __init__(self, prob, nx, nt, x_col, t_col, x_ic, t_bc, targets=None):
        self.prob, self.nx, self.nt = prob, nx, nt
        self.dim = nx * nt
        a, b, T = prob["a"], prob["b"], prob["T"]
        self.x_col, self.t_col, self.x_ic, self.t_bc = x_col, t_col, x_ic, t_bc
        self.Px, self.Pdx, self.Pddx = (basis(a, b, nx, x_col, d) for d in (0, 1, 2))
        self.Pt, self.Pdt = (basis(0.0, T, nt, t_col, d) for d in (0, 1))
        self.Pic = basis(a, b, nx, x_ic)
        self.Pt0 = basis(0.0, T, nt, np.array([0.0]))
        self.PL = basis(a, b, nx, np.array([a]))
        self.PR = basis(a, b, nx, np.array([b]))
        self.Pbc_t = basis(0.0, T, nt, t_bc)
        self.u0 = prob["ic"](x_ic)
        z = lambda s: np.zeros(s)
        tg = targets or {}
        self.src = tg.get("src", z((len(x_col), len(t_col))))
        self.gL = tg.get("gL", z(len(t_bc)))
        self.gR = tg.get("gR", z(len(t_bc)))
        self.u0 = tg.get("ic", self.u0)

    def with_targets(self, targets):
        return Machine(self.prob, self.nx, self.nt, self.x_col, self.t_col, self.x_ic, self.t_bc, targets)

    # --- ارزیابی میدان روی نقاط کالوکیشن (دسته‌ای) ---
    def fields(self, X):
        C = X.reshape(-1, self.nx, self.nt)
        M = C @ self.Pt                    # (m, nx, nct)
        Mt = C @ self.Pdt
        U = np.matmul(self.Px.T, M)
        Ux = np.matmul(self.Pdx.T, M)
        Uxx = np.matmul(self.Pddx.T, M)
        Ut = np.matmul(self.Px.T, Mt)
        return U, Ux, Ut, Uxx

    def parts(self, X):
        """(E_r, E_BC, E_IC) برای دسته‌ای از m نامزد."""
        C = X.reshape(-1, self.nx, self.nt)
        U, Ux, Ut, Uxx = self.fields(X)
        res = pde_residual(self.prob["kind"], self.prob["par"], U, Ux, Ut, Uxx) - self.src
        Er = np.mean(res ** 2, axis=(1, 2))
        Mbc = C @ self.Pbc_t               # (m, nx, n_bc)
        uL = np.einsum("i,mik->mk", self.PL[:, 0], Mbc) - self.gL
        uR = np.einsum("i,mik->mk", self.PR[:, 0], Mbc) - self.gR
        Ebc = 0.5 * (np.mean(uL ** 2, axis=1) + np.mean(uR ** 2, axis=1))
        M0 = C @ self.Pt0                  # (m, nx, 1)
        u_ic = np.einsum("ik,mi->mk", self.Pic, M0[:, :, 0])
        Eic = np.mean((u_ic - self.u0) ** 2, axis=1)
        return Er, Ebc, Eic

    def objective(self, X, w=W_FIT):
        with np.errstate(all="ignore"):
            Er, Ebc, Eic = self.parts(X)
            F = w["w1"] * Er + w["w2"] * Ebc + w["w3"] * Eic
        return np.where(np.isfinite(F), F, 1e10)

    def residual_vector(self, theta, w=W_FIT):
        """بردار r با sum(r^2) == F (برای least_squares)."""
        X = np.asarray(theta, float)[None]
        C = X.reshape(-1, self.nx, self.nt)
        U, Ux, Ut, Uxx = self.fields(X)
        res = (pde_residual(self.prob["kind"], self.prob["par"], U, Ux, Ut, Uxx) - self.src)[0].ravel()
        Mbc = C @ self.Pbc_t
        uL = np.einsum("i,mik->mk", self.PL[:, 0], Mbc)[0] - self.gL
        uR = np.einsum("i,mik->mk", self.PR[:, 0], Mbc)[0] - self.gR
        M0 = C @ self.Pt0
        u_ic = np.einsum("ik,mi->mk", self.Pic, M0[:, :, 0])[0] - self.u0
        return np.concatenate([np.sqrt(w["w1"] / res.size) * res,
                               np.sqrt(w["w2"] / (2 * uL.size)) * uL,
                               np.sqrt(w["w2"] / (2 * uR.size)) * uR,
                               np.sqrt(w["w3"] / u_ic.size) * u_ic])

    # --- میدان روی شبکه‌ی دلخواه (اعتبارسنجی) ---
    def grid(self, theta, xv, tv, dx=0):
        C = np.asarray(theta, float).reshape(self.nx, self.nt)
        a, b, T = self.prob["a"], self.prob["b"], self.prob["T"]
        return basis(a, b, self.nx, xv, dx).T @ C @ basis(0.0, T, self.nt, tv)

    def design(self, xv, tv):
        a, b, T = self.prob["a"], self.prob["b"], self.prob["T"]
        return np.kron(basis(a, b, self.nx, xv).T, basis(0.0, T, self.nt, tv).T)

class Fitness:
    """تابع برازندگی با شمارنده‌ی سراسری؛ فراخوانی برداری روی m نامزد = m ارزیابی."""
    def __init__(self, machine, w=W_FIT):
        self.m, self.w, self.count = machine, w, 0
    def __call__(self, X):
        X = np.asarray(X, float)
        single = X.ndim == 1
        X2 = X[None] if single else X
        self.count += X2.shape[0]
        v = self.m.objective(X2, self.w)
        return float(v[0]) if single else v

class CountedFunc:
    """پوشش شمارنده برای توابع محک (Sphere/Rastrigin)؛ ورودی (m,d)."""
    def __init__(self, f): self.f, self.count = f, 0
    def __call__(self, X):
        X = np.asarray(X, float); single = X.ndim == 1
        X2 = X[None] if single else X
        self.count += X2.shape[0]
        v = self.f(X2)
        return float(v[0]) if single else v
print("ماشین B-spline تعریف شد.")

# ===== CELL 6 =====
# ═══════════════ الگوریتم‌ها ═══════════════
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
print("الگوریتم‌ها:", list(ALGOS))

# ===== CELL 8 =====
# ═══════════════ مسائل ═══════════════
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

def val_grid(key):
    p = PROBLEMS[key]
    return np.linspace(p["a"], p["b"], 2 * NCX - 1), np.linspace(0.0, p["T"], 2 * NCT - 1)

# ---------- گرما: جواب دقیق ----------
def heat_reference(par, xv, tv):
    return np.exp(-par["alpha"] * tv)[None, :] * np.sin(xv)[:, None]

# ---------- برگرز: کول–هاپف به‌صورت انتگرال پایدار ----------
# u = ∫ (x-η)/t · e^{-F} dη / ∫ e^{-F} dη ،  F = (x-η)^2/(4νt) + (1-cos πη)/(2πν)
# (بسط فرد و ۲-تناوبی شرط اولیه ⇒ مسئله‌ی دیریکله‌ی [0,1])؛ ترفند: کم کردن min F برای پایداری.
_GLX, _GLW = np.polynomial.legendre.leggauss(16)
def burgers_cole_hopf(x, t, nu, c_trunc=50.0, n_panels=200):
    x = np.asarray(x, float); t = np.asarray(t, float)
    U = np.empty((x.size, t.size))
    for j, tj in enumerate(t):
        if tj == 0.0:
            U[:, j] = np.sin(np.pi * x); continue
        R = math.sqrt(4 * nu * tj * c_trunc)                       # شعاع برش انتگرال
        e = np.linspace(-R, R, n_panels + 1)
        mid, hf = 0.5 * (e[1:] + e[:-1]), 0.5 * (e[1:] - e[:-1])
        z = (mid[:, None] + hf[:, None] * _GLX[None, :]).ravel()
        w = (hf[:, None] * _GLW[None, :]).ravel()
        eta = x[:, None] + z[None, :]
        F = z ** 2 / (4 * nu * tj) + (1 - np.cos(np.pi * eta)) / (2 * np.pi * nu)
        F -= F.min(axis=1, keepdims=True)
        W = w * np.exp(-F)
        U[:, j] = (W * (-z) / tj).sum(1) / W.sum(1)
    return U

# ---------- انتشار غیرخطی: روش خطوط (BDF) با پالایش شبکه ----------
def diffusion_mol(par, ic, T, nx_ref, tv, xv, rtol=1e-10, atol=1e-12):
    h = 1.0 / nx_ref
    x = np.linspace(0.0, 1.0, nx_ref + 1)
    D0, eps = par["D0"], par["eps"]
    def rhs(t, u):
        full = np.concatenate(([0.0], u, [0.0]))
        um = 0.5 * (full[1:] + full[:-1])
        flux = D0 * (1.0 + eps * um ** 2) * (full[1:] - full[:-1]) / h     # شکل پایستار (D u_x)_x
        return (flux[1:] - flux[:-1]) / h
    n = nx_ref - 1
    sp = sparse.diags([np.ones(n - 1), np.ones(n), np.ones(n - 1)], [-1, 0, 1], format="csr")
    sol = integrate.solve_ivp(rhs, (0.0, T), ic(x[1:-1]), method="BDF", t_eval=tv, rtol=rtol, atol=atol,
                              jac_sparsity=sp)
    assert sol.success, sol.message
    Ufull = np.vstack([np.zeros((1, tv.size)), sol.y, np.zeros((1, tv.size))])
    return CubicSpline(x, Ufull, axis=0)(xv)

def _ref_path(key):
    p = PROBLEMS[key]
    tag = json.dumps([key, p["par"], p["a"], p["b"], p["T"], NCX, NCT, "v3"], sort_keys=True)
    return os.path.join(CACHE_DIR, f"ref_{key}_{hashlib.md5(tag.encode()).hexdigest()[:10]}.npz")

def load_reference(key):
    """(xv, tv, Uref, info) با کش دیسکی."""
    path = _ref_path(key)
    if os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        return d["xv"], d["tv"], d["U"], dict(zip(d["info_k"].tolist(), d["info_v"].tolist()))
    p = PROBLEMS[key]; xv, tv = val_grid(key)
    if key == "heat":
        U, info = heat_reference(p["par"], xv, tv), {"ref_error_est": 0.0}
    elif key == "burgers":
        U1 = burgers_cole_hopf(xv, tv, p["par"]["nu"], 50.0, 200)
        U2 = burgers_cole_hopf(xv, tv, p["par"]["nu"], 80.0, 400)
        U, info = U2, {"ref_error_est": float(np.abs(U1 - U2).max())}   # اختلاف دو truncation
    else:
        Us = {n: diffusion_mol(p["par"], p["ic"], p["T"], n, tv, xv) for n in (400, 800, 1600)}
        U = Us[1600]
        info = {"ref_error_est": float(np.abs(Us[1600] - Us[800]).max() / 3.0),     # برآورد ریچاردسون (مرتبه ۲)
                "diff_400_800": float(np.abs(Us[800] - Us[400]).max()),
                "diff_800_1600": float(np.abs(Us[1600] - Us[800]).max())}
    np.savez(path, xv=xv, tv=tv, U=U, info_k=np.array(list(info)), info_v=np.array(list(info.values())))
    return xv, tv, U, info

SHOCK_DENSITY_RATIO = 3.0  # ASSUMPTION: نسبت چگالی ناحیه‌ی شوک به بیرون، پیش از نرمال‌سازی — انتخاب طراحی مبتنی بر فیزیک گرادیان شوک
SHOCK_SAMPLING_ENABLED = True  # پرچم فعال‌سازی نمونه‌برداری تطبیقی ناحیه‌ی شوک (False = شبکه‌ی یکنواخت)

_shock_region_cache = {}
def shock_region(width=0.15):
    """ASSUMPTION: ناحیه‌ی شوک = [x* - width, b]، x* مکان بیشینه‌ی |u_x| در جواب کول–هاپف."""
    # کش دستی به‌جای lru_cache: wrapperِ lru_cache با loky/cloudpickle پیکِل نمی‌شود (مسیر موازی FULL)
    if width not in _shock_region_cache:
        p = PROBLEMS["burgers"]
        xs = np.linspace(p["a"], p["b"], 401); ts = np.linspace(0.2, p["T"], 17)
        Ux = np.gradient(burgers_cole_hopf(xs, ts, p["par"]["nu"]), xs, axis=0)
        xstar = xs[np.argmax(np.abs(Ux).max(axis=1))]
        _shock_region_cache[width] = (max(p["a"], xstar - width), p["b"], xstar)
    return _shock_region_cache[width]

def make_points(key, density_ratio=None):
    p = PROBLEMS[key]
    xs = np.linspace(p["a"], p["b"], 20001)
    w = np.ones_like(xs)
    if key == "burgers" and SHOCK_SAMPLING_ENABLED:
        r = SHOCK_DENSITY_RATIO if density_ratio is None else density_ratio
        lo, hi, _ = shock_region(); w = np.where((xs >= lo) & (xs <= hi), r, 1.0)
    cdf = np.cumsum(w); cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    x_col = np.interp((np.arange(NCX) + 0.5) / NCX, cdf, xs)
    t_col = (np.arange(NCT) + 0.5) / NCT * p["T"]
    return x_col, t_col, np.linspace(p["a"], p["b"], N_IC), np.linspace(0.0, p["T"], N_BC)

def make_machine(key, nx=None, nt=None, density_ratio=None):
    p = PROBLEMS[key]
    nx, nt = (nx, nt) if nx else p["nxnt"]
    xc, tc, xi, tb = make_points(key, density_ratio)
    return Machine(p, nx, nt, xc, tc, xi, tb)

print("مسائل تعریف شد.")

# ===== CELL 10 =====
def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not ok:
        raise AssertionError(f"اعتبارسنجی «{name}» شکست خورد. {detail}")

VAL_ROWS = []
rng_grid = np.random.default_rng(777)      # rng جداگانه و ثابت (فقط جواب ساختگی)

# ── آزمون ۱: partition of unity و بازتولید چندجمله‌ای ──
def greville(knots, n, k=DEG):
    return np.array([knots[i + 1:i + k + 1].mean() for i in range(n)])

pou_max, poly_max = 0.0, 0.0
for key, p in PROBLEMS.items():
    for (lo, hi, n) in [(p["a"], p["b"], p["nxnt"][0]), (0.0, p["T"], p["nxnt"][1]), (p["a"], p["b"], 7)]:
        xs = np.linspace(lo, hi, 2001)
        B = basis(lo, hi, n, xs)
        pou_max = max(pou_max, np.abs(B.sum(0) - 1).max())
        g = greville(clamped_knots(lo, hi, n), n)
        Bg = basis(lo, hi, n, g)                                  # (n, n): ماتریس درون‌یابی در نقاط گرویل
        for deg in range(4):
            coef = np.linalg.solve(Bg.T, g ** deg)                # ضرایب درون‌یاب چندجمله‌ای (فضای پایه شامل P3 است)
            err = np.abs(B.T @ coef - xs ** deg).max() / max(1.0, np.abs(xs ** deg).max())
            poly_max = max(poly_max, err)
check("۱ partition of unity", pou_max < 1e-12, f"max|ΣB-1|={pou_max:.2e}")
check("۱ بازتولید چندجمله‌ای تا درجه‌ی ۳", poly_max < 1e-12, f"max خطای نسبی={poly_max:.2e}")
VAL_ROWS += [("1 PoU", pou_max, "<1e-12"), ("1 poly", poly_max, "<1e-12")]

# ── آزمون ۲: مشتق تحلیلی در برابر تفاضل مرکزی (فقط تست) ──
d1_max, d2_max = 0.0, 0.0
for key, p in PROBLEMS.items():
    for (lo, hi, n) in [(p["a"], p["b"], 6), (0.0, p["T"], 6)]:
        kn = clamped_knots(lo, hi, n)
        xs = rng_grid.uniform(lo, hi, 400)
        h1, h2 = 1e-6, 1e-3
        far = np.min(np.abs(xs[:, None] - kn[None, :]), axis=1) > 5 * h2        # دور از گره‌ها
        xs = xs[far & (xs > lo + 5 * h2) & (xs < hi - 5 * h2)]
        d1 = (basis(lo, hi, n, xs + h1) - basis(lo, hi, n, xs - h1)) / (2 * h1)
        d2 = (basis(lo, hi, n, xs + h2) - 2 * basis(lo, hi, n, xs) + basis(lo, hi, n, xs - h2)) / h2 ** 2
        d1_max = max(d1_max, np.abs(d1 - basis(lo, hi, n, xs, 1)).max())
        d2_max = max(d2_max, np.abs(d2 - basis(lo, hi, n, xs, 2)).max())
check("۲ مشتق اول", d1_max < 1e-6, f"max|Δ|={d1_max:.2e}")
check("۲ مشتق دوم", d2_max < 1e-5, f"max|Δ|={d2_max:.2e}")
VAL_ROWS += [("2 d/dx", d1_max, "<1e-6"), ("2 d2/dx2", d2_max, "<1e-5")]

# ===== CELL 11 =====
# ── آزمون ۳: برازش lstsq جواب مرجع و «خطای تقریب فضای پایه» ──
def lstsq_fit(M, Uref, xv, tv):
    A = M.design(xv, tv)
    theta, *_ = np.linalg.lstsq(A, Uref.ravel(), rcond=None)
    err = (A @ theta).reshape(Uref.shape) - Uref
    return theta, float(np.sqrt(np.mean(err ** 2))), float(np.abs(err).max()), A

rows = []
for key in PROBLEMS:
    xv, tv, Uref, info = load_reference(key)
    M = make_machine(key)
    th, r, m, A = lstsq_fit(M, Uref, xv, tv)
    s = np.linalg.svd(A, compute_uv=False)
    rows.append(dict(problem=key, n_coeffs=M.dim, rmse_approx=r, maxae_approx=m,
                     rms_of_reference=float(np.sqrt(np.mean(Uref ** 2))), cond_design=s[0] / s[-1],
                     ref_error_est=info["ref_error_est"]))
    check(f"۳ برازش lstsq ({key})", np.isfinite(r) and r < rows[-1]["rms_of_reference"], f"RMSE_approx={r:.3e}")
VAL_ROWS.append(("3 lstsq", max(x["rmse_approx"] for x in rows), "finite & < rms(ref)"))
print("\nخطای تقریب فضای پایه (روی شبکه‌ی اعتبارسنجی) و عدد شرطی ماتریس طرح A:")
display(pd.DataFrame(rows).set_index("problem"))

# جاروب ظرفیت: خطای تقریب برحسب تعداد کل ضرایب (برای تفسیر آستانه‌ی موفقیت)
sweep = []
for key in PROBLEMS:
    xv, tv, Uref, _ = load_reference(key)
    for n_tot, (nx_, nt_) in NXNT.items():
        M = make_machine(key, nx_, nt_)
        _, r, m, _ = lstsq_fit(M, Uref, xv, tv)
        sweep.append(dict(problem=key, n_coeffs=n_tot, rmse_approx=r))
SWEEP = pd.DataFrame(sweep).pivot(index="n_coeffs", columns="problem", values="rmse_approx")
print("خطای تقریب lstsq برحسب تعداد کل ضرایب (کران پایین RMSE برای هر الگوریتم بهینه‌سازی روی همین فضا):")
display(SWEEP)

# ===== CELL 12 =====
# ── آزمون ۴: جواب ساختگی (manufactured solution) ──
rows = []
for key, p in PROBLEMS.items():
    M = make_machine(key)
    th_star = rng_grid.uniform(-1, 1, M.dim)                    # دقیقاً در فضای پایه
    U, Ux, Ut, Uxx = M.fields(th_star[None])
    src = pde_residual(p["kind"], p["par"], U, Ux, Ut, Uxx)[0]  # f = N[u*]  ⇒ باقیمانده صفر
    Mt = M.with_targets(dict(
        src=src,
        gL=(M.PL[:, 0] @ th_star.reshape(M.nx, M.nt) @ M.Pbc_t),
        gR=(M.PR[:, 0] @ th_star.reshape(M.nx, M.nt) @ M.Pbc_t),
        ic=(M.Pic.T @ (th_star.reshape(M.nx, M.nt) @ M.Pt0[:, 0]))))
    F0 = float(Mt.objective(th_star[None])[0])
    Fp = float(Mt.objective((th_star + 0.01 * rng_grid.standard_normal(M.dim))[None])[0])
    rows.append(dict(problem=key, F_at_theta_star=F0, F_perturbed=Fp))
    check(f"۴ جواب ساختگی ({key})", F0 < 1e-24 and Fp > 1e-8, f"F(θ*)={F0:.2e}, F(θ*+δ)={Fp:.2e}")
VAL_ROWS.append(("4 manufactured", max(r["F_at_theta_star"] for r in rows), "<1e-24"))
display(pd.DataFrame(rows).set_index("problem"))

# ===== CELL 13 =====
# ── آزمون ۵: مبناهای scipy روی همان تابع برازندگی و همان کران‌ها ──
lo_b, hi_b = COEF_BOUNDS
rows = []
for key in PROBLEMS:
    xv, tv, Uref, _ = load_reference(key)
    M = make_machine(key)
    th_ls, _, _, _ = lstsq_fit(M, Uref, xv, tv)
    th_ls = np.clip(th_ls, lo_b, hi_b)
    f_obj = lambda th: float(M.objective(th[None])[0])
    # سازگاری بردار باقیمانده با برازندگی
    r_chk = M.residual_vector(th_ls)
    rel = abs(float(r_chk @ r_chk) - f_obj(th_ls)) / max(f_obj(th_ls), 1e-300)
    check(f"۵ Σr² = F ({key})", rel < 1e-10, f"خطای نسبی={rel:.1e}")
    bnds = [(lo_b, hi_b)] * M.dim
    rg = np.random.default_rng(555 + PROB_IDS[key])
    starts = [th_ls] + [rg.uniform(lo_b, hi_b, M.dim) for _ in range(5)]
    lb_runs = [optimize.minimize(f_obj, s0, method="L-BFGS-B", bounds=bnds,
                                 options=dict(maxiter=3000, maxfun=300000)) for s0 in starts]
    ls_runs = [optimize.least_squares(M.residual_vector, s0, bounds=(lo_b, hi_b), method="trf", max_nfev=2000)
               for s0 in starts]
    F_ls = f_obj(th_ls)
    F_lb = min(r.fun for r in lb_runs)
    F_lsq = min(2 * r.cost for r in ls_runs)                  # least_squares: cost = ½ Σ r²
    err = lambda th: float(np.sqrt(np.mean((M.grid(th, xv, tv) - Uref) ** 2)))
    best_lb = min(lb_runs, key=lambda r: r.fun).x
    rows.append(dict(problem=key, F_lstsq_baseline=F_ls, F_LBFGSB_best=F_lb, F_least_squares_best=F_lsq,
                     F_LBFGSB_from_lstsq_start=lb_runs[0].fun,
                     RMSE_LBFGSB=err(best_lb), RMSE_lstsq=err(th_ls)))
    check(f"۵ L-BFGS-B ≤ مبنای lstsq ({key})", F_lb <= F_ls * (1 + 1e-9) + 1e-30,
          f"F_LBFGSB={F_lb:.3e} ≤ F_lstsq={F_ls:.3e}")
SCIPY_BASE = pd.DataFrame(rows).set_index("problem")
VAL_ROWS.append(("5 scipy", float((SCIPY_BASE.F_LBFGSB_best / SCIPY_BASE.F_lstsq_baseline).max()), "<=1"))
print("مبناهای scipy (برازندگی نهایی و RMSE روی شبکه‌ی اعتبارسنجی):")
display(SCIPY_BASE)

# ===== CELL 14 =====
# ── آزمون ۶: Sphere و Rastrigin با بعد، کران و بودجه‌ی مشخص و برابر ──
# نکته‌ی طراحی: برخلاف آزمون‌های ۱–۵، این آزمون از check()ِ خطادهنده استفاده نمی‌کند؛
# نسبت‌ها صادقانه گزارش و PASS/FAIL فقط ثبت می‌شود. رد شدن در آستانه‌ی عملکردی باگ نیست:
# «درستی پیاده‌سازی» (شمارنده‌ی دقیق max_evals) با «کیفیت همگرایی» (نسبت) فرق دارد.
sphere = lambda X: np.sum(X ** 2, axis=1)
rastrigin = lambda X: 10 * X.shape[1] + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=1)
BENCH = {"Sphere": sphere, "Rastrigin": rastrigin}
DIM_B, BND_B = 10, (-5.12, 5.12)
bench_rows = []
bench_ok_all = True
for fname, f in BENCH.items():
    for alg, opt in ALGOS.items():
        init_b, fin_b, evs = [], [], []
        for r in range(CFG["bench_runs"]):
            seed = int(np.random.SeedSequence(MASTER_SEED, spawn_key=(900, r)).generate_state(1, dtype=np.uint64)[0])
            cf = CountedFunc(f)
            x, curve, nev = opt(cf, BND_B, N_POP, MAX_EVALS, np.random.default_rng(seed), dim=DIM_B)
            assert cf.count == MAX_EVALS and nev == MAX_EVALS
            init_b.append(curve[0]); fin_b.append(float(cf(x[None])[0])); evs.append(nev)
            # ارزیابیِ گزارشیِ بهترین جواب از مسیر cf انجام می‌شود تا شمارش شود:
            # یک ارزیابیِ اضافه در هر اجرا، خارج از بودجه‌ی بهینه‌سازیِ هر الگوریتم و برای هر سه الگوریتم یکسان.
            assert cf.count == MAX_EVALS + 1
        ratio = float(np.median(fin_b) / np.median(init_b))
        thr = 0.5
        ok = ratio < thr and all(e == MAX_EVALS for e in evs)
        bench_ok_all = bench_ok_all and ok
        bench_rows.append(dict(function=fname, algorithm=alg, dim=DIM_B, budget=MAX_EVALS,
                               median_initial_best=np.median(init_b), median_final_best=np.median(fin_b),
                               ratio=ratio, thr=thr, verdict="PASS" if ok else "FAIL"))
        print(f"[{'PASS' if ok else 'FAIL'}] ۶ {fname} / {alg}  ratio={ratio:.3g} (آستانه‌ی {thr})")
    # مرجعِ جستجوی تصادفی با تابعِ خام f اجرا می‌شود و عمداً شمارش نمی‌شود (فقط خطِ پایه‌ی گزارشی، خارج از بودجه‌ی الگوریتم‌ها)
    rs = [float(f(np.random.default_rng(MASTER_SEED + r).uniform(*BND_B, (MAX_EVALS, DIM_B))).min())
          for r in range(CFG["bench_runs"])]
    print(f"   مرجع: جستجوی تصادفی هم‌بودجه روی {fname}: میانه‌ی بهترین = {np.median(rs):.4g}")
VAL_ROWS.append(("6 benchmarks", max(r["ratio"] for r in bench_rows), "ratio<0.5"))
display(pd.DataFrame(bench_rows).set_index(["function", "algorithm"]))
if bench_ok_all:
    print("\nخلاصه‌ی اعتبارسنجی: همه‌ی آزمون‌ها PASS شدند.")
else:
    print("\nخلاصه‌ی اعتبارسنجی: برخی ردیف‌های بنچمارک در آستانه‌ی عملکردی رد شدند (جزئیات بالا)؛ رد شدن ≠ باگ.")
display(pd.DataFrame(VAL_ROWS, columns=["test", "worst_value", "criterion"]).set_index("test"))

# ===== CELL 16 =====
def run_seed(key, run, tag=0):
    ss = np.random.SeedSequence(MASTER_SEED, spawn_key=(tag, PROB_IDS[key], run))
    return int(ss.generate_state(1, dtype=np.uint64)[0])

X_EVAL = np.linspace(0, MAX_EVALS, 200)

def run_one(key, alg, run, nx, nt, n_pop, max_evals, alg_kw, tag=0):
    p = PROBLEMS[key]
    M = make_machine(key, nx, nt)
    xv, tv, Uref, _ = load_reference(key)
    seed = run_seed(key, run, tag)
    fit = Fitness(M)
    t0 = time.perf_counter()
    theta, curve, n_ev = ALGOS[alg](fit, COEF_BOUNDS, n_pop, max_evals, np.random.default_rng(seed),
                                    dim=M.dim, **alg_kw)
    dt = time.perf_counter() - t0
    assert fit.count == max_evals == n_ev                           # قرارداد بودجه
    err = M.grid(theta, xv, tv) - Uref
    xs = n_pop * (1 + np.arange(len(curve)))
    cur = np.interp(np.linspace(0, max_evals, 200), xs, np.minimum.accumulate(curve), left=np.nan)
    return dict(problem=key, algorithm=alg, run=run, seed=seed, RMSE=float(np.sqrt(np.mean(err ** 2))),
                MaxAE=float(np.abs(err).max()),
                n_evals=n_ev, time_s=dt, n_coeffs=M.dim, best_F=float(np.min(curve))), cur, theta

RESULTS = {}

def run_problem(key):
    ckpt = f"ckpt_{MODE}_{CFG_HASH}_{key}.pkl"
    _need = CFG["n_runs"] * len(ALG_NAMES)
    if os.path.exists(ckpt):
        try:
            with open(ckpt, "rb") as _f:
                _R = pickle.load(_f)
            if len(_R["df"]) >= _need and set(_R["df"]["algorithm"]) >= set(ALG_NAMES):
                print(f"↺ {key}: چک‌پوینت کامل ({len(_R['df'])} سطر) بازیابی شد — اجرا رد شد.")
                RESULTS[key] = _R
                return _R
            print(f"چک‌پوینت {key} ناقص است — اجرای دوباره.")
        except Exception as _e:
            print(f"چک‌پوینت {key} خوانده نشد ({_e}) — اجرای دوباره.")
    for k in PROBLEMS:                       # کش مرجع پیش از اجرای موازی
        load_reference(k)
    jobs = [(alg, r) for r in range(N_RUNS) for alg in ALG_NAMES]
    out = _run_all(CFG["n_jobs"],
                   [(run_one, (key, alg, r, *PROBLEMS[key]["nxnt"], N_POP, MAX_EVALS, {})) for alg, r in jobs])
    df = pd.DataFrame([o[0] for o in out])
    curves = {(o[0]["algorithm"], o[0]["run"]): o[1] for o in out}
    thetas = {(o[0]["algorithm"], o[0]["run"]): o[2] for o in out}
    RESULTS[key] = dict(df=df, curves=curves, thetas=thetas)
    with open(ckpt, "wb") as _f:
        pickle.dump(RESULTS[key], _f)
    print(f"چک‌پوینت ذخیره شد: {ckpt}")
    return RESULTS[key]

# ---------- آمار ----------
def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj

def summary_table(df, tau):
    # نرخ موفقیت (معیار پایان‌نامه): کسر اجراهایی با RMSE < tau، با بازه‌ی اطمینان ۹۵٪ ویلسون
    def wilson(k, n, z=1.96):
        if n == 0:
            return (np.nan, np.nan)
        p = k / n
        den = 1 + z * z / n
        ctr = p + z * z / (2 * n)
        mgn = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return ((ctr - mgn) / den, (ctr + mgn) / den)
    rows = []
    for alg in ALG_NAMES:
        d = df[df.algorithm == alg]
        n = len(d)
        q1, q3 = np.percentile(d.RMSE, [25, 75])
        k = int((d.RMSE < tau).sum())
        lo, hi = wilson(k, n)
        rows.append(dict(algorithm=alg, mean=d.RMSE.mean(), std=d.RMSE.std(ddof=1) if n > 1 else np.nan,
                         median=d.RMSE.median(), IQR=q3 - q1, MaxAE_mean=d.MaxAE.mean(),
                         success_rate=(k / n if n else np.nan), sr_lo95=lo, sr_hi95=hi,
                         time_mean_s=d.time_s.mean()))
    return pd.DataFrame(rows).set_index("algorithm")

def friedman_table(df):
    P = df.pivot(index="run", columns="algorithm", values="RMSE")[ALG_NAMES]
    n, k = P.shape
    chi, pv = stats.friedmanchisquare(*[P[a].values for a in ALG_NAMES])
    ranks = P.rank(axis=1).mean()
    q = stats.studentized_range.ppf(0.95, k, np.inf) / math.sqrt(2)
    cd = q * math.sqrt(k * (k + 1) / (6.0 * n))
    return dict(chi2=chi, p=pv, n_blocks=n), ranks, cd

def wilcoxon_pairs(df):
    P = df.pivot(index="run", columns="algorithm", values="RMSE")[ALG_NAMES]
    pairs = [("MFO-BOA", "MFO"), ("MFO-BOA", "BOA"), ("MFO", "BOA")]
    rows = []
    for a, b in pairs:
        d = (P[a] - P[b]).values
        nz = d[d != 0]
        try:
            w = stats.wilcoxon(P[a].values, P[b].values, alternative="two-sided")
            pv = float(w.pvalue)
        except ValueError:
            if nz.size == 0:      # تنها حالت مشروع: همه‌ی تفاضل‌های زوجی صفر
                pv = 1.0
            else:
                raise
        if nz.size:
            rk = stats.rankdata(np.abs(nz)); wp, wm = rk[nz > 0].sum(), rk[nz < 0].sum()
            rrb = float((wp - wm) / (wp + wm))
        else:
            rrb = 0.0
        rows.append(dict(pair=f"{a} vs {b}", median_diff=float(np.median(d)), p_value=pv, effect_r_rb=rrb,
                         wins_first=int((d < 0).sum()), n=len(d)))
    out = pd.DataFrame(rows)
    out["p_Holm"] = holm(out.p_value)
    out["p_Bonferroni"] = np.minimum(1.0, out.p_value * len(out))
    out["significant_Holm_0.05"] = out.p_Holm < 0.05
    return out.set_index("pair")

def report_problem(key):
    R = RESULTS[key]; p = PROBLEMS[key]; df = R["df"]
    print(f"══ {p['title']} | runs={df.run.nunique()} | max_evals={MAX_EVALS} ══")
    display(summary_table(df, p["tau"]))
    fr, ranks, cd = friedman_table(df)
    print(f"Friedman: χ²={fr['chi2']:.4g}, p={fr['p']:.4g}, n={fr['n_blocks']} | mean ranks: "
          + ", ".join(f"{a}={v:.2f}" for a, v in ranks.items()) + f" | Nemenyi CD(0.05)={cd:.3f}")
    display(wilcoxon_pairs(df))

def plot_problem(key):
    R = RESULTS[key]; p = PROBLEMS[key]; df = R["df"]
    xv, tv, Uref, _ = load_reference(key); M = make_machine(key)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    cols = {"MFO": "tab:blue", "BOA": "tab:orange", "MFO-BOA": "tab:green"}
    for alg in ALG_NAMES:
        C = np.array([R["curves"][(alg, r)] for r in range(df.run.nunique())])
        med, q1, q3 = np.nanmedian(C, 0), np.nanpercentile(C, 25, 0), np.nanpercentile(C, 75, 0)
        ax[0, 0].plot(X_EVAL, med, color=cols[alg], label=alg)
        ax[0, 0].fill_between(X_EVAL, q1, q3, color=cols[alg], alpha=0.2)
    ax[0, 0].set_yscale("log"); ax[0, 0].set_xlabel("function evaluations"); ax[0, 0].set_ylabel("best-so-far fitness")
    ax[0, 0].set_title("Convergence (median and IQR)"); ax[0, 0].legend()
    best = df[df.algorithm == "MFO-BOA"].sort_values("RMSE").iloc[0]
    th = R["thetas"][("MFO-BOA", int(best.run))]
    Un = M.grid(th, xv, tv)
    for frac, ls in [(0.25, ":"), (0.5, "--"), (1.0, "-")]:
        j = int(round(frac * (len(tv) - 1)))
        ax[0, 1].plot(xv, Uref[:, j], "k" + ("-" if ls == "-" else ls), lw=1.2)
        ax[0, 1].plot(xv, Un[:, j], color="tab:green", ls=ls if ls != "-" else "-", lw=2, alpha=0.8)
    ax[0, 1].set_title(f"Best MFO-BOA run vs reference (black), t = 0.25T, 0.5T, T")
    ax[0, 1].set_xlabel("x"); ax[0, 1].set_ylabel("u")
    im = ax[1, 0].imshow(np.abs(Un - Uref).T, origin="lower", aspect="auto", extent=[xv[0], xv[-1], tv[0], tv[-1]])
    ax[1, 0].set_title("|error| on validation grid (best MFO-BOA run)"); ax[1, 0].set_xlabel("x"); ax[1, 0].set_ylabel("t")
    fig.colorbar(im, ax=ax[1, 0])
    data = [df[df.algorithm == a].RMSE.values for a in ALG_NAMES]
    ax[1, 1].boxplot(data); ax[1, 1].set_xticklabels(ALG_NAMES); ax[1, 1].set_yscale("log")
    ax[1, 1].set_title("RMSE distribution over runs")
    fig.suptitle(f"{p['title']} — {MODE}", y=1.0)
    fig.tight_layout(); plt.show()
print("توابع اجرا و آمار تعریف شد.")

# ===== CELL 18 =====
xv, tv, Uref, info = load_reference("heat")
M = make_machine("heat")
print(f"heat: n_coeffs={M.dim} (nx={M.nx}, nt={M.nt}) | grid validation={Uref.shape} | collocation={len(M.x_col)}x{len(M.t_col)}")
print(f"خطای مرجع (جواب دقیق): {info['ref_error_est']}")
th, r, m, A = lstsq_fit(M, Uref, xv, tv)
print(f"cond(A)={np.linalg.cond(A):.3e} | خطای تقریب فضای پایه: RMSE={r:.3e}, MaxAE={m:.3e}")
run_problem("heat"); report_problem("heat"); plot_problem("heat")

# ===== CELL 20 =====
xv, tv, Uref, info = load_reference("burgers")
p = PROBLEMS["burgers"]
check("برگرز: اختلاف دو truncation مرجع < 1e-6", info["ref_error_est"] < 1e-6, f"max|ΔU|={info['ref_error_est']:.2e}")
# سازگاری مرجع: شرط اولیه، شرایط مرزی و باقیمانده‌ی PDE (نقاط دور از شوک)
check("برگرز: شرط اولیه", np.abs(Uref[:, 0] - np.sin(np.pi * xv)).max() < 1e-14)
check("برگرز: شرایط مرزی", np.abs(Uref[0]).max() < 1e-10 and np.abs(Uref[-1]).max() < 1e-10,
      f"max|u(0,t)|={np.abs(Uref[0]).max():.1e}, max|u(1,t)|={np.abs(Uref[-1]).max():.1e}")
xt = np.array([0.2, 0.35, 0.5, 0.6]); tt = np.array([0.3, 0.5, 0.8]); h = 1e-3
f = lambda x, t: burgers_cole_hopf(np.atleast_1d(x), np.atleast_1d(t), p["par"]["nu"])[:, 0]
res = []
for x0 in xt:
    for t0 in tt:
        ut = (f(x0, t0 + h) - f(x0, t0 - h)) / (2 * h)
        ux = (f(x0 + h, t0) - f(x0 - h, t0)) / (2 * h)
        uxx = (f(x0 + h, t0) - 2 * f(x0, t0) + f(x0 - h, t0)) / h ** 2
        res.append(abs(ut + f(x0, t0) * ux - p["par"]["nu"] * uxx)[0])
check("برگرز: باقیمانده‌ی PDE روی جواب کول–هاپف (نقاط هموار)", max(res) < 1e-4, f"max|R|={max(res):.2e}")
lo, hi, xstar = shock_region()
print(f"x*={xstar:.3f} | ناحیه‌ی شوک با چگالی ۳ برابر: [{lo:.3f}, {hi:.3f}]")
M = make_machine("burgers")
th, r, m, A = lstsq_fit(M, Uref, xv, tv)
print(f"burgers: n_coeffs={M.dim} | cond(A)={np.linalg.cond(A):.3e} | خطای تقریب فضای پایه RMSE={r:.3e}, MaxAE={m:.3e}")
print(f"نقاط x کالوکیشن: min Δx={np.diff(M.x_col).min():.4f}, max Δx={np.diff(M.x_col).max():.4f}")
run_problem("burgers"); report_problem("burgers"); plot_problem("burgers")

# ===== CELL 22 =====
# ── مطالعه‌ی حساسیت SHOCK_DENSITY_RATIO (سبک؛ جاروب کامل بهینه‌سازی اجرا نشده) ──
def shock_ratio_sensitivity(ratios=(1, 2, 3, 5)):
    p = PROBLEMS["burgers"]
    xv_v, tv_v = val_grid("burgers")
    _, _, Uref_v, _ = load_reference("burgers")
    rows = []
    for r in ratios:
        M = make_machine("burgers", density_ratio=r)
        xc, tc = M.x_col, M.t_col
        Uref_c = burgers_cole_hopf(xc, tc, p["par"]["nu"])
        A_c = M.design(xc, tc)
        theta, *_ = np.linalg.lstsq(A_c, Uref_c.ravel(), rcond=None)
        err = (M.design(xv_v, tv_v) @ theta).ravel() - Uref_v.ravel()
        rows.append(dict(density_ratio=r,
                         rmse_validation=float(np.sqrt(np.mean(err ** 2))),
                         maxae_validation=float(np.abs(err).max())))
    return pd.DataFrame(rows).set_index("density_ratio")

SENS_SHOCK = shock_ratio_sensitivity()
print("حساسیت خطای تقریب به SHOCK_DENSITY_RATIO — برازش lstsq روی نقاط کالوکیشن هر نسبت، ارزیابی روی شبکه‌ی اعتبارسنجی ثابت و مستقل:")
display(SENS_SHOCK)
print("یادداشت: این مطالعه‌ی سبک جایگزین جاروب کامل بهینه‌سازی نیست و نتایج آن نتایج پژوهش نیستند.")

# ===== CELL 24 =====
xv, tv, Uref, info = load_reference("diffusion")
print("پالایش شبکه‌ی مرجع:", {k: f"{v:.3e}" for k, v in info.items()})
check("انتشار: همگرایی پالایش شبکه (800→1600 کوچک‌تر از 400→800)", info["diff_800_1600"] < info["diff_400_800"])
check("انتشار: برآورد خطای مرجع < 1e-5", info["ref_error_est"] < 1e-5, f"{info['ref_error_est']:.2e}")
M = make_machine("diffusion")
th, r, m, A = lstsq_fit(M, Uref, xv, tv)
print(f"diffusion: n_coeffs={M.dim} | cond(A)={np.linalg.cond(A):.3e} | خطای تقریب فضای پایه RMSE={r:.3e}, MaxAE={m:.3e}")
run_problem("diffusion"); report_problem("diffusion"); plot_problem("diffusion")

# ===== CELL 26 =====
def ofat_problem(key):
    p = PROBLEMS[key]
    lv = CFG["ofat_levels"]
    plan = [("N", N_, dict(N=N_)) for N_ in lv["N"]] + [("HR", h, dict(HR=h)) for h in lv["HR"]] \
         + [("n", n_, dict(n=n_)) for n_ in lv["n"]] + [("K", k_, dict(K=k_)) for k_ in lv["K"]]
    ckpt = f"ckpt_{MODE}_{CFG_HASH}_ofat_{key}.pkl"
    _need = len(plan) * CFG["ofat_runs"]
    if os.path.exists(ckpt):
        try:
            _df0 = pd.read_pickle(ckpt)
            if len(_df0) >= _need:
                print(f"↺ OFAT {key}: چک‌پوینت کامل ({len(_df0)} سطر) بازیابی شد — اجرا رد شد.")
                return _df0
            print(f"چک‌پوینت OFAT {key} ناقص است — اجرای دوباره.")
        except Exception as _e:
            print(f"چک‌پوینت OFAT {key} خوانده نشد ({_e}) — اجرای دوباره.")
    xv, tv, Uref, _ = load_reference(key)
    rows = []
    # بودجه‌ی OFAT دقیقاً ثابت در همه‌ی سطوح و همه‌ی عامل‌ها: بزرگ‌ترین مضربِ مشترکِ N_POP و همه‌ی سطوح N
    # که از MAX_EVALS بیشتر نشود. (DEMO: n_pop=20 ← 2000؛ FULL: n_pop=50، ‏lcm(50,10,20,30,50,100)=300 ← 9900)
    # بنابراین مضرب هر n_pop است و _generations سالم می‌ماند.
    # توجه: در FULL این یکسان‌سازی، بودجه‌ی عامل‌های HR/n/K را نیز از 10050 به 9900 می‌رساند تا برابریِ بودجه در کل OFAT حفظ شود.
    L = math.lcm(N_POP, *lv["N"])
    mev_const = (MAX_EVALS // L) * L
    assert mev_const > 0
    jobs = []
    for factor, level, ov in plan:
        n_pop = ov.get("N", N_POP)
        nx, nt = NXNT[ov["n"]] if "n" in ov else p["nxnt"]
        kw = {k: v for k, v in ov.items() if k in ("HR", "K")}
        for r in range(CFG["ofat_runs"]):
            mev = mev_const  # بودجه‌ی دقیقاً ثابت؛ چون L مضرب N_POP و همه‌ی سطوح N است، مضرب هر n_pop نیز هست
            assert mev % n_pop == 0
            jobs.append((factor, level, r, nx, nt, n_pop, mev, kw))
    out = _run_all(CFG["n_jobs"],
                   [(run_one, (key, "MFO-BOA", r, nx, nt, n_pop, mev, kw, 100))
                    for (_, _, r, nx, nt, n_pop, mev, kw) in jobs])
    for (factor, level, *_), o in zip(jobs, out):
        d = dict(o[0]); d.update(factor=factor, level=level); rows.append(d)
    df = pd.DataFrame(rows)
    # خطای تقریب lstsq برای هر n
    approx = {}
    for n_, (nx, nt) in NXNT.items():
        if n_ in CFG["ofat_levels"]["n"]:
            approx[n_] = lstsq_fit(make_machine(key, nx, nt), Uref, xv, tv)[1]
    df["lstsq_approx_RMSE"] = [approx.get(l) if f == "n" else np.nan for f, l in zip(df.factor, df.level)]
    df.to_pickle(ckpt)
    print(f"چک‌پوینت OFAT ذخیره شد: {ckpt}")
    return df

OFAT = {}
for key in PROBLEMS:
    OFAT[key] = ofat_problem(key)
    g = OFAT[key].groupby(["factor", "level"], sort=False).agg(
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"), RMSE_median=("RMSE", "median"),
        n_coeffs=("n_coeffs", "first"), time_mean_s=("time_s", "mean"),
        n_evals=("n_evals", "first"), lstsq_approx_RMSE=("lstsq_approx_RMSE", "first"))
    print(f"══ OFAT — {PROBLEMS[key]['title']} (runs per level = {CFG['ofat_runs']}) ══")
    display(g)

fig, axes = plt.subplots(len(PROBLEMS), 4, figsize=(16, 3.4 * len(PROBLEMS)), squeeze=False)
for i, key in enumerate(PROBLEMS):
    for j, factor in enumerate(["N", "HR", "n", "K"]):
        d = OFAT[key][OFAT[key].factor == factor]
        g = d.groupby("level").RMSE
        axes[i, j].errorbar(g.mean().index.astype(float), g.mean(), yerr=g.std().fillna(0), marker="o", capsize=3)
        axes[i, j].set_yscale("log"); axes[i, j].set_title(f"{PROBLEMS[key]['title']} — {factor}")
        axes[i, j].set_xlabel(factor); axes[i, j].set_ylabel("mean RMSE")
fig.tight_layout(); plt.show()

# ===== CELL 28 =====
ALL = pd.concat([RESULTS[k]["df"] for k in PROBLEMS], ignore_index=True)
ALL = ALL[["problem", "algorithm", "run", "seed", "RMSE", "MaxAE", "n_evals", "time_s", "n_coeffs", "best_F"]]
ALL = ALL.sort_values(["problem", "algorithm", "run"]).reset_index(drop=True)
ALL.insert(0, "mode", MODE)
fname = f"results_{MODE}.csv"
ALL.to_csv(fname, index=False)
OF = pd.concat([OFAT[k].assign(problem=k) for k in PROBLEMS], ignore_index=True); OF.insert(0, "mode", MODE)
OF.to_csv(f"ofat_{MODE}.csv", index=False)
print(f"ذخیره شد: {fname} ({len(ALL)} سطر) و ofat_{MODE}.csv ({len(OF)} سطر)")
assert (ALL.n_evals == MAX_EVALS).all(), "قرارداد بودجه نقض شد"
print(f"assert نهایی: همه‌ی اجراهای اصلی دقیقاً {MAX_EVALS} ارزیابی مصرف کردند. ✔")
display(ALL.groupby(["problem", "algorithm"]).agg(RMSE_mean=("RMSE", "mean"),
                                                   time_mean_s=("time_s", "mean"), n_evals=("n_evals", "first")))

# ===== CELL 29 =====
print("Python      :", sys.version.split()[0], "|", platform.platform())
for m in (np, scipy, pd, matplotlib):
    print(f"{m.__name__:11s}:", m.__version__)
if _HAS_JOBLIB:
    import joblib; print("joblib     :", joblib.__version__)
else:
    print("joblib     : نصب نیست — اجرای ترتیبی")
print("MODE:", MODE, "| MASTER_SEED:", MASTER_SEED)
print("ALG_PARAMS:", ALG_PARAMS, "| W_FIT:", W_FIT, "| COEF_BOUNDS:", COEF_BOUNDS)
print("فهرست seedها (اجراهای اصلی؛ برچسب 0 — بین سه الگوریتم مشترک):")
for key in PROBLEMS:
    print(f"  {key:10s}", [run_seed(key, r) for r in range(N_RUNS)])
print("فهرست seedهای OFAT (برچسب 100):")
for key in PROBLEMS:
    print(f"  {key:10s}", [run_seed(key, r, 100) for r in range(CFG["ofat_runs"])])
