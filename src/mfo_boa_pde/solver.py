"""B-spline collocation machine, fitness, and PDE problem helpers."""
import numpy as np

from .basis import basis, burgers_cole_hopf
from .config import N_IC, N_BC, NCX, NCT, W_FIT
from .problems import PROBLEMS

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

def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not ok:
        raise AssertionError(f"اعتبارسنجی «{name}» شکست خورد. {detail}")

def lstsq_fit(M, Uref, xv, tv):
    A = M.design(xv, tv)
    theta, *_ = np.linalg.lstsq(A, Uref.ravel(), rcond=None)
    err = (A @ theta).reshape(Uref.shape) - Uref
    return theta, float(np.sqrt(np.mean(err ** 2))), float(np.abs(err).max()), A
