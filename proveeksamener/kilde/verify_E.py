"""Verification of all numbers in practice exam E."""
import numpy as np
import sympy as sp

x, y, z, t, s = sp.symbols("x y z t s", real=True)
ok = lambda c, msg: print(("OK  " if c else "FAIL"), msg)

print("=== E1: linear least squares")
A = sp.Matrix([[1, 0], [1, 1], [1, 2], [1, 3]]); b = sp.Matrix([1, 2, 2, 4])
ATA, ATb = A.T * A, A.T * b; print("    A^TA =", ATA.tolist(), " A^Tb =", list(ATb), " det =", ATA.det())
xs = ATA.solve(ATb); ok(xs == sp.Matrix([9, 9]) / 10, f"x* = {list(xs)}")
r = A * xs - b; print("    r =", list(r), " ||r||^2 =", r.dot(r))
q1 = sp.Matrix([1, 1, 1, 1]) / 2; q2 = sp.Matrix([-3, -1, 1, 3]) / (2 * sp.sqrt(5))
Q = sp.Matrix.hstack(q1, q2); R = sp.Matrix([[2, 3], [0, sp.sqrt(5)]])
ok(sp.simplify(Q * R - A) == sp.zeros(4, 2), "A = Q R (reduced)")
ok(sp.simplify(Q.T * Q - sp.eye(2)) == sp.zeros(2, 2), "Q^T Q = I")
d = sp.simplify(Q.T * b); print("    d = Q^T b =", list(d))
xq = R.solve(d); ok(sp.simplify(xq - xs) == sp.zeros(2, 1), "QR solution equals normal eq solution")
print("    ||b||^2 - ||d||^2 =", sp.simplify(b.dot(b) - d.dot(d)))
xr = (ATA + sp.eye(2)).solve(ATb); print("    Tikhonov mu=1:", list(xr), " norm:", sp.sqrt(xr.dot(xr)), " LS norm:", sp.sqrt(xs.dot(xs)), float(sp.sqrt(xs.dot(xs))))
rr = A * xr - b; print("    Tikhonov residual^2:", rr.dot(rr), float(rr.dot(rr)))
print("    cond(A^TA) vs cond(A)^2:", np.linalg.cond(np.array(ATA, dtype=float)), np.linalg.cond(np.array(A, dtype=float))**2)

print("=== E2: ALS rank-1 update")
X = np.zeros((2, 2, 2)); X[:, :, 0] = [[1, 2], [3, 4]]; X[:, :, 1] = [[0, 1], [1, 0]]
def unfold(X, mode):
    return np.reshape(np.moveaxis(X, mode, 0), (X.shape[mode], -1), order="F")
for m in range(3): print(f"    X({m+1}) =", unfold(X, m).tolist())
bv, cv = np.array([1.0, 1.0]), np.array([1.0, 0.0])
kr = np.kron(cv, bv); a_new = unfold(X, 0) @ np.linalg.pinv(kr.reshape(1, -1))
print("    c (x) b =", kr, " a =", a_new.ravel(), " via k-mode:", np.einsum("ijk,j,k->i", X, bv, cv) / (bv @ bv) / (cv @ cv))
# full ALS update formula with pinv, check equals
a2 = unfold(X, 0) @ kr / (kr @ kr); ok(np.allclose(a_new.ravel(), a2), "pinv((c(x)b)^T) = (c(x)b)/||.||^2")
# one more check: best rank-1 a for fixed b, c minimises ||X(1) - a (c(x)b)^T||_F
from scipy.optimize import minimize
res = minimize(lambda aa: np.linalg.norm(unfold(X, 0) - np.outer(aa, kr))**2, np.zeros(2)); print("    numerical argmin:", res.x)

print("=== E3: sphere")
u, v = sp.symbols("u v", real=True)
S = sp.Matrix([sp.cos(u) * sp.sin(v), sp.sin(u) * sp.sin(v), sp.cos(v)])
Su, Sv = S.diff(u), S.diff(v); n = Su.cross(Sv)
ok(sp.simplify(n + sp.sin(v) * S) == sp.zeros(3, 1), "s_u x s_v = -sin(v) s(u,v)")
p = S.subs({u: sp.pi / 4, v: sp.pi / 4}); print("    s(pi/4,pi/4) =", list(p), " n =", [sp.simplify(e.subs({u: sp.pi / 4, v: sp.pi / 4})) for e in n])
print("    plane: p.p =", sp.simplify(p.dot(p)))
print("    s_u, s_v at v=0:", list(Su.subs(v, 0)), list(Sv.subs(v, 0)))
gam = S.subs({u: t, v: t}); gp = gam.diff(t).subs(t, sp.pi / 2)
cr = (Su + Sv).subs({u: sp.pi / 2, v: sp.pi / 2}); print("    gamma'(pi/2) direct =", list(gp), " chain rule =", list(cr))
print("    s_u, s_v at (pi/2,pi/2):", list(Su.subs({u: sp.pi / 2, v: sp.pi / 2})), list(Sv.subs({u: sp.pi / 2, v: sp.pi / 2})), " gamma(pi/2) =", list(gam.subs(t, sp.pi / 2)))

print("=== E4: Gauss-Newton for x1 exp(x2 t)")
x1, x2 = sp.symbols("x1 x2", real=True)
ts = [0, 1, 2]; ys = [1, 2, 4]
Rv = sp.Matrix([x1 * sp.exp(x2 * ti) - yi for ti, yi in zip(ts, ys)]); J = Rv.jacobian([x1, x2])
p0 = {x1: 1, x2: 0}; R0 = Rv.subs(p0); J0 = J.subs(p0)
print("    R0 =", list(R0), " J0 =", J0.tolist(), " J^TJ =", (J0.T * J0).tolist(), " J^TR =", list(J0.T * R0))
D = (J0.T * J0).solve(-J0.T * R0); print("    GN Delta =", list(D), " x1 =", [1 + D[0], D[1]])
E = lambda pp: float(sum(e**2 for e in Rv.subs(pp)) / 2)
print("    E(x0) =", E(p0), " E(x_GN) =", E({x1: 1 + D[0], x2: D[1]}))
for lamv in [1, 10]:
    DL = (J0.T * J0 + lamv * sp.eye(2)).solve(-J0.T * R0)
    print(f"    LM lambda={lamv}: Delta = {list(DL)} = {[float(e) for e in DL]}, E = {E({x1: 1 + DL[0], x2: DL[1]}):.4f}")
print("    true minimiser (1, ln 2) =", (1, float(sp.log(2))))

print("=== E5: f = x^2 + 5y^2")
f = x**2 + 5 * y**2; g = sp.Matrix([f.diff(x), f.diff(y)])
g0 = g.subs({x: 1, y: 1}); print("    grad(1,1) =", list(g0), " f(1,1) =", f.subs({x: 1, y: 1}))
phi = f.subs({x: 1 - t * g0[0], y: 1 - t * g0[1]}); ts_ = sp.solve(sp.diff(phi, t), t); print("    t* =", ts_)
p1 = sp.Matrix([1, 1]) - ts_[0] * g0; print("    x1 =", list(p1), " f(x1) =", f.subs({x: p1[0], y: p1[1]}), " grad(x1) =", list(g.subs({x: p1[0], y: p1[1]})), " dot =", g.subs({x: p1[0], y: p1[1]}).dot(g0))
print("    phi'(t) =", sp.expand(sp.diff(phi, t)))
for ss in [sp.Rational(1, 10), sp.Rational(1, 4), sp.Rational(1, 5)]:
    print(f"    s={ss}: factors (1-2s, 1-10s) = {1 - 2 * ss}, {1 - 10 * ss}")

print("=== E6: rectangular pulse")
k = sp.symbols("k", integer=True, nonzero=True)
ck = sp.simplify(sp.integrate(sp.exp(-sp.I * k * x), (x, -sp.pi / 2, sp.pi / 2)) / (2 * sp.pi)); print("    c_k =", sp.simplify(ck.rewrite(sp.sin)))
for kk in [1, 2, 3, 5]:
    val = sp.integrate(sp.exp(-sp.I * kk * x), (x, -sp.pi / 2, sp.pi / 2)) / (2 * sp.pi)
    ok(sp.simplify(val - sp.sin(kk * sp.pi / 2) / (kk * sp.pi)) == 0, f"c_{kk} = sin(k pi/2)/(k pi) = {sp.simplify(val)}")
n_ = sp.symbols("n", integer=True, nonnegative=True)
print("    sum_{odd} 1/k^2 =", sp.summation(1 / (2 * n_ + 1)**2, (n_, 0, sp.oo)))
err = sp.Rational(1, 2) - (sp.Rational(1, 4) + 2 / sp.pi**2); print("    best-approx error (normalised) =", err, float(err))
xs_ = np.linspace(-np.pi, np.pi, 400001); fv = (np.abs(xs_) < np.pi / 2).astype(float)
S2 = 0.5 + 2 / np.pi * np.cos(xs_); print("    numeric (1/2pi)||f-S2||^2 =", np.trapezoid((fv - S2)**2, xs_) / (2 * np.pi))
# Gibbs: max of S_N near jump
for NN in [11, 51, 201]:
    SN = 0.5 + sum(2 * np.sin(m * np.pi / 2) / (m * np.pi) * np.cos(m * xs_) for m in range(1, NN + 1))
    print(f"    max S_{NN} =", SN.max())

print("=== E7: DFT/FFT of (1,2,3,4)")
f = np.array([1, 2, 3, 4], dtype=complex); fh = np.fft.fft(f) / 4; print("    fhat =", fh)
gv = f[:2] + f[2:]; hv = (f[:2] - f[2:]) * np.exp(-2j * np.pi * np.arange(2) / 4)
gh = np.fft.fft(gv) / 2; hh = np.fft.fft(hv) / 2; print("    g =", gv, " h =", hv, " ghat =", gh, " hhat =", hh)
ok(np.allclose(fh[0::2], gh / 2) and np.allclose(fh[1::2], hh / 2), "radix-2 split (first/second half) correct")
fh2 = fh.copy(); fh2[1] = fh2[3] = 0; print("    filtered signal =", np.fft.ifft(fh2 * 4).real)

print("=== E8: d'Alembert & bounded wave")
uu = sp.exp(-(x - 2 * t)**2)
ok(sp.simplify(uu.diff(t, 2) - 4 * uu.diff(x, 2)) == 0, "PDE"); ok(sp.simplify(uu.subs(t, 0) - sp.exp(-x**2)) == 0, "u(x,0)")
ok(sp.simplify(uu.diff(t).subs(t, 0) - 4 * x * sp.exp(-x**2)) == 0, "u_t(x,0)")
fd = sp.exp(-x**2); gd = 4 * x * sp.exp(-x**2); sv_ = sp.symbols("sv")
dal = (fd.subs(x, x + 2 * t) + fd.subs(x, x - 2 * t)) / 2 + sp.integrate(gd.subs(x, sv_), (sv_, x - 2 * t, x + 2 * t)) / 4
ok(sp.simplify(dal - uu) == 0, "d'Alembert formula gives exp(-(x-2t)^2)")
ub = sp.Rational(3, 4) * sp.sin(x) * sp.cos(2 * t) - sp.Rational(1, 4) * sp.sin(3 * x) * sp.cos(6 * t)
ok(sp.simplify(ub.diff(t, 2) - 4 * ub.diff(x, 2)) == 0, "bounded PDE"); ok(sp.simplify(ub.subs(t, 0) - sp.sin(x)**3) == 0, "u(x,0)=sin^3")
ok(sp.simplify(ub.diff(t).subs(t, 0)) == 0, "u_t(x,0)=0"); ok(sp.simplify(ub.subs(x, sp.pi)) == 0 and ub.subs(x, 0) == 0, "BC")
ok(sp.simplify(ub - (sp.sin(x + 2 * t)**3 + sp.sin(x - 2 * t)**3) / 2) == 0, "equals d'Alembert with odd extension")
An = [sp.simplify(2 / sp.pi * sp.integrate(sp.sin(x)**3 * sp.sin(m * x), (x, 0, sp.pi))) for m in range(1, 5)]; print("    A_n =", An)

print("=== E9: Crank-Nicolson, u(0)=0, u_x(1)=0, M=2, alpha=1")
Am = sp.Matrix([[-2, 1], [2, -2]]); al = 1
U0 = sp.Matrix([sp.Rational(3, 4), 1]); rhs = (sp.eye(2) + sp.Rational(al, 2) * Am) * U0
U1 = (sp.eye(2) - sp.Rational(al, 2) * Am).solve(rhs); print("    rhs =", list(rhs), " U1 =", list(U1), [float(e) for e in U1])
print("    lhs matrix =", (sp.eye(2) - sp.Rational(al, 2) * Am).tolist(), " rhs matrix =", (sp.eye(2) + sp.Rational(al, 2) * Am).tolist())
# reference: fine-grid CN solution and exact series at t = 1/4
lam_n = [((n - 0.5) * np.pi)**2 for n in range(1, 200)]
Bn = [2 * float(sp.integrate((2 * x - x**2) * sp.sin((n - sp.Rational(1, 2)) * sp.pi * x), (x, 0, 1))) for n in range(1, 30)]
ex = lambda xx, tt: sum(Bn[n] * np.exp(-lam_n[n] * tt) * np.sin((n + 0.5) * np.pi * xx) for n in range(29))
print("    exact u(1/2,1/4), u(1,1/4):", ex(0.5, 0.25), ex(1.0, 0.25))
print("    explicit alpha=1 one step:", list((sp.eye(2) + Am) * U0))

print("=== E10: Bezier")
P = [sp.Matrix([0, 0]), sp.Matrix([1, 2]), sp.Matrix([3, 2]), sp.Matrix([4, 0])]
bt = sum((sp.binomial(3, i) * t**i * (1 - t)**(3 - i) * P[i] for i in range(4)), sp.zeros(2, 1))
print("    b(t) =", [sp.factor(e) for e in bt], " b(1/2) =", list(bt.subs(t, sp.Rational(1, 2))))
lev = [P]
for kk in range(3):
    lev.append([(lev[-1][i] + lev[-1][i + 1]) / 2 for i in range(len(lev[-1]) - 1)])
for L in lev: print("      ", [list(q) for q in L])
print("    b'(0) =", list(bt.diff(t).subs(t, 0)), " b'(1) =", list(bt.diff(t).subs(t, 1)), " b''(0) =", list(bt.diff(t, 2).subs(t, 0)), " b''(1) =", list(bt.diff(t, 2).subs(t, 1)))
Q = [sp.Matrix([4, 0]), sp.Matrix([5, -2]), sp.Matrix([7, -2]), sp.Matrix([8, 0])]
bq = sum((sp.binomial(3, i) * t**i * (1 - t)**(3 - i) * Q[i] for i in range(4)), sp.zeros(2, 1))
print("    q-curve: b(0) =", list(bq.subs(t, 0)), " b'(0) =", list(bq.diff(t).subs(t, 0)), " b''(0) =", list(bq.diff(t, 2).subs(t, 0)))
Q2 = [sp.Matrix([4, 0]), sp.Matrix([5, -2]), sp.Matrix([5, -6]), sp.Matrix([8, 0])]
bq2 = sum((sp.binomial(3, i) * t**i * (1 - t)**(3 - i) * Q2[i] for i in range(4)), sp.zeros(2, 1))
print("    with q2=(5,-6): b''(0) =", list(bq2.diff(t, 2).subs(t, 0)))
