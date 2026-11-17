"""B-spline basis, reference solutions and validation grids.

The reference cache directory is CWD-relative ("cache", as in the original)
and is created lazily inside load_reference instead of at import time, so
importing this module has no side effects.
"""
import hashlib
import json
import math
import os

import numpy as np
from scipy import integrate, sparse
from scipy.interpolate import BSpline, CubicSpline

from .config import CACHE_DIR, NCX, NCT
from .problems import PROBLEMS

DEG = 3

def clamped_knots(a, b, n, k=DEG):
    assert n >= k + 1, "تعداد پایه باید دست‌کم k+1 باشد"
    inner = np.linspace(a, b, n - k - 1 + 2)[1:-1]
    return np.concatenate([np.full(k + 1, a), inner, np.full(k + 1, b)])

def basis(a, b, n, x, nu=0):
    """ماتریس (n, len(x)) از nu-امین مشتق تحلیلی پایه‌های B-spline مکعبی clamped."""
    spl = BSpline(clamped_knots(a, b, n), np.eye(n), DEG, extrapolate=False)
    return spl(np.asarray(x, float), nu=nu).T

def greville(knots, n, k=DEG):
    return np.array([knots[i + 1:i + k + 1].mean() for i in range(n)])

def val_grid(key):
    p = PROBLEMS[key]
    return np.linspace(p["a"], p["b"], 2 * NCX - 1), np.linspace(0.0, p["T"], 2 * NCT - 1)

def heat_reference(par, xv, tv):
    return np.exp(-par["alpha"] * tv)[None, :] * np.sin(xv)[:, None]


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
        cached = np.load(path, allow_pickle=False)
        return cached["xv"], cached["tv"], cached["U"], dict(zip(cached["info_k"].tolist(), cached["info_v"].tolist()))
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
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, xv=xv, tv=tv, U=U, info_k=np.array(list(info)), info_v=np.array(list(info.values())))
    return xv, tv, U, info
