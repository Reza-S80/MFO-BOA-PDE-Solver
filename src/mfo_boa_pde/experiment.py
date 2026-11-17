"""Experiment orchestration: seeds, single runs, checkpoints, main().

All former top-level execution code (validation checks, problem sections,
OFAT, final aggregation) lives in main() so that importing this module
executes nothing.
"""
import os
import pickle
import platform
import sys
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from IPython.display import display
from scipy import optimize

from .basis import basis, burgers_cole_hopf, clamped_knots, greville, load_reference
from .config import (
    ALG_PARAMS,
    CFG,
    CFG_HASH,
    COEF_BOUNDS,
    MAX_EVALS,
    MODE,
    MASTER_SEED,
    N_POP,
    N_RUNS,
    NCT,
    NCX,
    NXNT,
    T_UPD,
    W_FIT,
)
from .diagnostics import ofat_problem, plot_problem, report_problem, shock_ratio_sensitivity
from .optimizers import _HAS_JOBLIB, _run_all, ALGOS
from .problems import ALG_NAMES, PROB_IDS, PROBLEMS
from .solver import CountedFunc, Fitness, check, lstsq_fit, make_machine, pde_residual, shock_region

def run_seed(key, run, tag=0):
    ss = np.random.SeedSequence(MASTER_SEED, spawn_key=(tag, PROB_IDS[key], run))
    return int(ss.generate_state(1, dtype=np.uint64)[0])

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


def _validation_checks():
    """Validation checks 1-6."""
    VAL_ROWS = []
    rng_grid = np.random.default_rng(777)

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



def _problem_sections():
    """Heat / Burgers / shock-sensitivity / diffusion sections."""
    xv, tv, Uref, info = load_reference("heat")
    M = make_machine("heat")
    print(f"heat: n_coeffs={M.dim} (nx={M.nx}, nt={M.nt}) | grid validation={Uref.shape} | collocation={len(M.x_col)}x{len(M.t_col)}")
    print(f"خطای مرجع (جواب دقیق): {info['ref_error_est']}")
    th, r, m, A = lstsq_fit(M, Uref, xv, tv)
    print(f"cond(A)={np.linalg.cond(A):.3e} | خطای تقریب فضای پایه: RMSE={r:.3e}, MaxAE={m:.3e}")
    R = run_problem("heat"); report_problem("heat", R); plot_problem("heat", R)
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
    R = run_problem("burgers"); report_problem("burgers", R); plot_problem("burgers", R)
    SENS_SHOCK = shock_ratio_sensitivity()
    print("حساسیت خطای تقریب به SHOCK_DENSITY_RATIO — برازش lstsq روی نقاط کالوکیشن هر نسبت، ارزیابی روی شبکه‌ی اعتبارسنجی ثابت و مستقل:")
    display(SENS_SHOCK)
    print("یادداشت: این مطالعه‌ی سبک جایگزین جاروب کامل بهینه‌سازی نیست و نتایج آن نتایج پژوهش نیستند.")
    xv, tv, Uref, info = load_reference("diffusion")
    print("پالایش شبکه‌ی مرجع:", {k: f"{v:.3e}" for k, v in info.items()})
    check("انتشار: همگرایی پالایش شبکه (800→1600 کوچک‌تر از 400→800)", info["diff_800_1600"] < info["diff_400_800"])
    check("انتشار: برآورد خطای مرجع < 1e-5", info["ref_error_est"] < 1e-5, f"{info['ref_error_est']:.2e}")
    M = make_machine("diffusion")
    th, r, m, A = lstsq_fit(M, Uref, xv, tv)
    print(f"diffusion: n_coeffs={M.dim} | cond(A)={np.linalg.cond(A):.3e} | خطای تقریب فضای پایه RMSE={r:.3e}, MaxAE={m:.3e}")
    R = run_problem("diffusion"); report_problem("diffusion", R); plot_problem("diffusion", R)


def _ofat_section():
    """OFAT study and figure."""
    OFAT = {}
    for key in PROBLEMS:
        OFAT[key] = ofat_problem(key, run_one)
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
    return OFAT


def _finalize(ofat):
    """Final aggregation, CSV output, version and seed listing."""
    ALL = pd.concat([RESULTS[k]["df"] for k in PROBLEMS], ignore_index=True)
    ALL = ALL[["problem", "algorithm", "run", "seed", "RMSE", "MaxAE", "n_evals", "time_s", "n_coeffs", "best_F"]]
    ALL = ALL.sort_values(["problem", "algorithm", "run"]).reset_index(drop=True)
    ALL.insert(0, "mode", MODE)
    fname = f"results_{MODE}.csv"
    ALL.to_csv(fname, index=False)
    OF = pd.concat([ofat[k].assign(problem=k) for k in PROBLEMS], ignore_index=True); OF.insert(0, "mode", MODE)
    OF.to_csv(f"ofat_{MODE}.csv", index=False)
    print(f"ذخیره شد: {fname} ({len(ALL)} سطر) و ofat_{MODE}.csv ({len(OF)} سطر)")
    assert (ALL.n_evals == MAX_EVALS).all(), "قرارداد بودجه نقض شد"
    print(f"assert نهایی: همه‌ی اجراهای اصلی دقیقاً {MAX_EVALS} ارزیابی مصرف کردند. ✔")
    display(ALL.groupby(["problem", "algorithm"]).agg(RMSE_mean=("RMSE", "mean"),
                                                       time_mean_s=("time_s", "mean"), n_evals=("n_evals", "first")))
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


def main():
    """Run the full experiment exactly as the original script did.

    Reproduces the original top-level execution order (and its stdout):
    configuration prints, validation checks 1-6, problem sections, OFAT,
    final aggregation and version/seed listing.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    if not _HAS_JOBLIB:
        print("joblib یافت نشد — اجرای ترتیبی")
    print(f"MODE={MODE} | runs={N_RUNS} | N={N_POP} | T(update generations)={T_UPD} | max_evals={MAX_EVALS}")
    print(f"n_jobs={CFG['n_jobs']} | collocation={NCX}x{NCT}={NCX*NCT} | weights={W_FIT} | bounds={COEF_BOUNDS}")
    print("ماشین B-spline تعریف شد.")
    print("الگوریتم‌ها:", list(ALGOS))
    print("مسائل تعریف شد.")
    _validation_checks()
    print("توابع اجرا و آمار تعریف شد.")
    _problem_sections()
    ofat = _ofat_section()
    _finalize(ofat)
