"""Statistics, reporting, plotting, shock sensitivity and OFAT.

Reporting functions take the result dict as an argument instead of reading
experiment.RESULTS (avoids circular imports); ofat_problem takes run_one as
an injected argument for the same reason. Configuration constants are read
from .config.
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats

from .basis import burgers_cole_hopf, load_reference, val_grid
from .config import CFG, CFG_HASH, MAX_EVALS, MODE, N_POP, NXNT
from .optimizers import _run_all
from .problems import ALG_NAMES, PROBLEMS
from .solver import lstsq_fit, make_machine

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

def report_problem(key, R):
    p = PROBLEMS[key]; df = R["df"]
    print(f"══ {p['title']} | runs={df.run.nunique()} | max_evals={MAX_EVALS} ══")
    display(summary_table(df, p["tau"]))
    fr, ranks, cd = friedman_table(df)
    print(f"Friedman: χ²={fr['chi2']:.4g}, p={fr['p']:.4g}, n={fr['n_blocks']} | mean ranks: "
          + ", ".join(f"{a}={v:.2f}" for a, v in ranks.items()) + f" | Nemenyi CD(0.05)={cd:.3f}")
    display(wilcoxon_pairs(df))

def plot_problem(key, R):
    p = PROBLEMS[key]; df = R["df"]
    x_eval = np.linspace(0, MAX_EVALS, 200)
    xv, tv, Uref, _ = load_reference(key); M = make_machine(key)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    cols = {"MFO": "tab:blue", "BOA": "tab:orange", "MFO-BOA": "tab:green"}
    for alg in ALG_NAMES:
        C = np.array([R["curves"][(alg, r)] for r in range(df.run.nunique())])
        med, q1, q3 = np.nanmedian(C, 0), np.nanpercentile(C, 25, 0), np.nanpercentile(C, 75, 0)
        ax[0, 0].plot(x_eval, med, color=cols[alg], label=alg)
        ax[0, 0].fill_between(x_eval, q1, q3, color=cols[alg], alpha=0.2)
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

def ofat_problem(key, run_one):
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
            mev = mev_const
            assert mev % n_pop == 0
            jobs.append((factor, level, r, nx, nt, n_pop, mev, kw))
    out = _run_all(CFG["n_jobs"],
                   [(run_one, (key, "MFO-BOA", r, nx, nt, n_pop, mev, kw, 100))
                    for (_, _, r, nx, nt, n_pop, mev, kw) in jobs])
    for (factor, level, *_), o in zip(jobs, out):
        d = dict(o[0]); d.update(factor=factor, level=level); rows.append(d)
    df = pd.DataFrame(rows)
    approx = {}
    for n_, (nx, nt) in NXNT.items():
        if n_ in CFG["ofat_levels"]["n"]:
            approx[n_] = lstsq_fit(make_machine(key, nx, nt), Uref, xv, tv)[1]
    df["lstsq_approx_RMSE"] = [approx.get(l) if f == "n" else np.nan for f, l in zip(df.factor, df.level)]
    df.to_pickle(ckpt)
    print(f"چک‌پوینت OFAT ذخیره شد: {ckpt}")
    return df
