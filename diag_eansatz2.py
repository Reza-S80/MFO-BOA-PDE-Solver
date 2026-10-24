"""E-ANSATZ variant: (1-e^(-t)) envelope + exact u_BC — answers DeepSeek's objection.

u(x,t;c) = u_BC(x,t) + (1-e^(-t)) * sum_{i=1}^{15} c_i B_i(x)
u_BC(x,t) = sin(x)  [exact IC: u(x,0)=sin(x); exact BC: sin(0)=sin(pi)=0]

At t=0: (1-e^0)=0 -> u(x,0) = sin(x) EXACTLY. c is NOT forced to 0.
DeepSeek's objection resolved IF this ansatz reaches ~3.2e-5.

(A1') pure projection; (A2') PDE-constrained exact LS; (A3') rank-1 structural check.
"""
import os, sys, json
os.environ["MPLBACKEND"] = "Agg"
os.environ["NB_MODE"] = "DEMO"

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/hatch/workspace/thesis-code/debug-bundle/notebook_code.py"
SEGMENTS = [(1, 460), (496, 500), (628, 630), (632, 632), (634, 650)]
# sanity: each segment must start at an expected marker, else the line numbers drifted
_MARKERS = ["", "def lstsq_fit", "def run_seed", "X_EVAL", "def run_one"]
ns = {"__name__": "eansatz2"}
lines = open(SRC, encoding="utf-8").readlines()
for (a, b), _m in zip(SEGMENTS, _MARKERS):
    _first = lines[a - 1].strip()
    assert _first.startswith(_m), f"segment {a}-{b} starts with unexpected line: {_first[:60]}"
exec(compile("".join("".join(lines[a - 1:b]) for a, b in SEGMENTS), SRC, "exec"), ns)
np = ns["np"]

p = ns["PROBLEMS"]["heat"]
a, b, T = p["a"], p["b"], p["T"]
alpha = p["par"]["alpha"]
W = {"w1": 1.0, "w2": 20.0, "w3": 20.0}
TARGET = 3.2e-5

xv, tv, Uref, _ = ns["load_reference"]("heat")
x_col, t_col, x_ic, t_bc = ns["make_points"]("heat", None)
nA = 15

# envelope g(t) = 1 - e^{-t}
def g(t): return 1.0 - np.exp(-t)
def gp(t): return np.exp(-t)  # g'(t)

# basis: phi_i(x,t) = g(t) * B_i(x)
# u_t = g'(t)*B_i ; u_xx = g(t)*B_i''
# PDE residual: g'*B_i - alpha*g*B_dd_i
Bx_v = ns["basis"](a, b, nA, xv, nu=0)

# (A1') pure projection of (Uref - sin(x)) onto g(t)*B_i(x)
# Uref - sin(x) = sin(x)*(e^{-t} - 1) = -sin(x)*g(t)
# So we project -sin(x)*g(t) onto g(t)*B_i(x) -> spatial projection of -sin(x)
rhs_sp = -np.sin(xv)
c1, *_ = np.linalg.lstsq(Bx_v.T, rhs_sp, rcond=None)
sp_rmse = float(np.sqrt(np.mean((Bx_v.T @ c1 - rhs_sp) ** 2)))
# full field RMSE: E = g(t)*(Bx@c1 + sin(x)) = g(t)*r(x)
r_x = Bx_v.T @ c1 - rhs_sp
t_w = float(np.sqrt(np.mean(g(tv) ** 2)))
A1_rmse = sp_rmse * t_w
print(f"A1' pure projection: spatial RMSE={sp_rmse:.3e} -> full RMSE={A1_rmse:.3e}", flush=True)

# (A3') structural: E(x,t) = g(t)*r(x) -> rank-1
E = g(tv)[None, :] * r_x[:, None]
_, s, _ = np.linalg.svd(E, full_matrices=False)
rank_ratio = float(s[1] / s[0])
print(f"A3' structural: sigma2/sigma1 = {rank_ratio:.2e}", flush=True)

# (A2') PDE-constrained: u = sin(x) + g(t)*sum c_i B_i(x)
# residual for basis i: g'(t)*B_i(x) - alpha*g(t)*B_dd_i(x)
Bc = ns["basis"](a, b, nA, x_col, nu=0)
Bddc = ns["basis"](a, b, nA, x_col, nu=2)
Gt = g(t_col); Gp = gp(t_col)
# L[i,j,k] = Gp[k]*Bc[i,j] - alpha*Gt[k]*Bddc[i,j]
L = (Gp[None, None, :] * Bc[:, :, None] - alpha * Gt[None, None, :] * Bddc[:, :, None])
L = L.reshape(nA, -1).T  # (600, 15)
# The PDE residual of u_BC=sin(x): u_t=0, u_xx=-sin(x) -> L(u_BC) = 0 - alpha*(-sin) = alpha*sin
# So total residual: L(u) = alpha*sin(x_col) + sum_i c_i * L_i
# We solve for c to minimize ||alpha*sin + L@c||^2 (weighted) + BC + IC
# BC: u(0,t) = sin(0) + g(t)*sum c_i B_i(0) = 0 + g(t)*c_1*B_1(0)... target 0
#   Actually u_BC(0,t)=sin(0)=0, so BC residual = g(t)*sum c_i B_i(0)
# IC: u(x,0) = sin(x) + 0 = sin(x) EXACT -> E_IC = 0 identically! (no IC term needed)
sin_col = np.sin(x_col)
src = alpha * sin_col  # (30,) source from u_BC; broadcast over t
# Build weighted system: minimize w1*mean((src + L@c)^2) + w2*mean((g*B_bc@c)^2)
# Note: src is (30,) over x, needs to be (600,) over (x,t) grid
src_full = np.tile(src[:, None], (1, len(t_col))).ravel()  # (600,)
Nr = 600
S_pde = np.sqrt(W["w1"] / Nr) * L
t_pde = -np.sqrt(W["w1"] / Nr) * src_full
# BC
Ba = ns["basis"](a, b, nA, np.array([a]), nu=0)[:, 0]
Bb = ns["basis"](a, b, nA, np.array([b]), nu=0)[:, 0]
Gtb = g(t_bc)
Bbc = np.vstack([np.outer(Gtb, Ba), np.outer(Gtb, Bb)])  # (82, 15)
Nbc = Bbc.shape[0]
S = np.vstack([S_pde, np.sqrt(W["w2"] / Nbc) * Bbc])
tvec = np.concatenate([t_pde, np.zeros(Nbc)])
c2, *_ = np.linalg.lstsq(S, tvec, rcond=None)
# bounded-feasibility check: notebook constrains c to [-2, 2]
from scipy.optimize import lsq_linear
c2b = lsq_linear(S, tvec, bounds=(-2, 2)).x
print(f"A2' unbounded max|c| = {np.abs(c2).max():.3f} (bound 2.0)", flush=True)
c2 = c2b  # use the bounded solution for the reported RMSE
# RMSE on validation grid
U = np.sin(xv)[:, None] + g(tv)[None, :] * (Bx_v.T @ c2)[:, None]
A2_rmse = float(np.sqrt(np.mean((U - Uref) ** 2)))
print(f"A2' PDE-constrained attainable RMSE = {A2_rmse:.3e}", flush=True)
print(f"IC exact: yes (g(0)=0). max|c| = {np.abs(c2).max():.3f}", flush=True)

print("---- RESULT ----", flush=True)
# NOTE: A3' rank-1 test is vacuous (E=g(t)r(x) is rank-1 by construction) — reported, not gated.
ok = (A2_rmse <= 2 * TARGET)
print(f"A2'={A2_rmse:.3e} (thesis 3.2e-5, 2x = {2*TARGET:.1e}), rank1={rank_ratio:.1e}", flush=True)
with open("/home/hatch/workspace/thesis-code/diag_eansatz2.json", "w") as f:
    json.dump(dict(A1=A1_rmse, A2=A2_rmse, rank_ratio=rank_ratio, ok=bool(ok)), f, indent=1)
