"""Verification of all numbers in practice exam C."""
import numpy as np
import sympy as sp

x, y, z, t, s, w = sp.symbols("x y z t s omega", real=True)
k = sp.symbols("k", integer=True, nonzero=True)
ok = lambda c, msg: print(("OK  " if c else "FAIL"), msg)

print("=== C1: SVD / Moore-Penrose (underdetermined)")
A = sp.Matrix([[1, 0, 1], [0, 1, 1]])
AAT = A * A.T
ok(AAT == sp.Matrix([[2, 1], [1, 2]]), f"AA^T = {AAT.tolist()}")
ok(sorted(AAT.eigenvals().keys()) == [1, 3], "eigenvalues 3, 1 -> sigma = sqrt3, 1")
u1 = sp.Matrix([1, 1]) / sp.sqrt(2); u2 = sp.Matrix([1, -1]) / sp.sqrt(2)
v1 = A.T * u1 / sp.sqrt(3); v2 = A.T * u2 / 1
ok(sp.simplify(v1 - sp.Matrix([1, 1, 2]) / sp.sqrt(6)) == sp.zeros(3, 1), f"v1 = {list(v1)}")
ok(sp.simplify(v2 - sp.Matrix([1, -1, 0]) / sp.sqrt(2)) == sp.zeros(3, 1), f"v2 = {list(v2)}")
U = sp.Matrix.hstack(u1, u2); S = sp.diag(sp.sqrt(3), 1); V = sp.Matrix.hstack(v1, v2)
ok(sp.simplify(U * S * V.T - A) == sp.zeros(2, 3), "U S V^T = A")
Ad = sp.simplify(V * S.inv() * U.T)
ok(Ad == sp.Matrix([[2, -1], [-1, 2], [1, 1]]) / 3, f"A^+ = {Ad.tolist()}")
ok(Ad == A.pinv(), "matches sympy pinv")
b = sp.Matrix([1, 2]); xd = Ad * b
ok(xd == sp.Matrix([0, 1, 1]), f"x+ = {list(xd)}")
xt = sp.Matrix([1, 2, 0]); ok(A * xt == b, "x~=(1,2,0) solves Ax=b")
ok(A * sp.Matrix([1, 1, -1]) == sp.zeros(2, 1), "ker A = span(1,1,-1)")
ok(xd.dot(sp.Matrix([1, 1, -1])) == 0, "x+ orthogonal to ker A")
ok(xt - xd == sp.Matrix([1, 1, -1]), "x~ = x+ + 1*(1,1,-1)")
print("    norms^2:", xd.dot(xd), xt.dot(xt))
A1 = sp.simplify(sp.sqrt(3) * u1 * v1.T)
ok(A1 == sp.Matrix([[1, 1, 2], [1, 1, 2]]) / 2, f"A1 = {A1.tolist()}")
E = A - A1
print("    A - A1 =", E.tolist(), " 2-norm:", sp.sqrt(max((E * E.T).eigenvals())), " F-norm:", sp.sqrt(sum(e**2 for e in E)))

print("=== C2: tensor with slices M, 2M")
M = np.array([[1, 2], [0, 1]]); X = np.zeros((2, 2, 2)); X[:, :, 0] = M; X[:, :, 1] = 2 * M
def unfold(X, mode):  # Kolda convention: first remaining index fastest
    return np.reshape(np.moveaxis(X, mode, 0), (X.shape[mode], -1), order="F")
X1, X2, X3 = (unfold(X, m) for m in range(3))
print("    X(1) =", X1.tolist()); print("    X(2) =", X2.tolist()); print("    X(3) =", X3.tolist())
N = np.array([[1, -2], [0, 1]])
Y = np.einsum("ji,ikl->jkl", N, X)  # X x_1 N
ok(np.allclose(unfold(Y, 0), N @ X1), "Y(1) = N X(1)")
print("    Y slices:", Y[:, :, 0].tolist(), Y[:, :, 1].tolist())
c = np.array([1, 2]); e1, e2 = np.eye(2)
Xcp = np.einsum("i,j,k->ijk", e1, [1, 2], c) + np.einsum("i,j,k->ijk", e2, [0, 1], c)
ok(np.allclose(Xcp, X), "X = e1 o (1,2) o c + e2 o (0,1) o c")
Xcp2 = np.einsum("i,j,k->ijk", [1, 0], e1, c) + np.einsum("i,j,k->ijk", [2, 1], e2, c)
ok(np.allclose(Xcp2, X), "X = (1,0) o e1 o c + (2,1) o e2 o c")
print("    ranks of X(1),X(2),X(3):", [np.linalg.matrix_rank(m) for m in (X1, X2, X3)])
Uc, Sc, _ = np.linalg.svd(X3); print("    X(3) left sing. vec:", Uc[:, 0], "sigma:", Sc, "sqrt30 =", np.sqrt(30))
C = (c / np.linalg.norm(c)).reshape(2, 1)
G = np.einsum("ijk,kl->ijl", X, C)  # X x_3 C^T  (A=B=I)
print("    core (A=B=I):", G[:, :, 0].tolist(), " sqrt5*M =", (np.sqrt(5) * M).tolist())
ok(np.allclose(np.einsum("ijl,kl->ijk", G, C), X), "G x_3 C = X (exact)")

print("=== C3: helix")
cc = sp.Matrix([2 * sp.cos(t), 2 * sp.sin(t), sp.sqrt(5) * t])
v = cc.diff(t); a = v.diff(t)
ok(sp.simplify(v.norm()) == 3, f"speed = {sp.simplify(v.norm())}")
print("    c(pi/2) =", list(cc.subs(t, sp.pi / 2)), " c'(pi/2) =", list(v.subs(t, sp.pi / 2)), " a =", list(a))
ok(sp.simplify(v.dot(a)) == 0, "a orthogonal to v")
ct = cc.subs(t, s / 3); ok(sp.simplify(ct.diff(s).norm()) == 1, "arc-length param unit speed")
d = sp.Matrix([t**3, t**6]); print("    d'(t) =", list(d.diff(t)))

print("=== C4: Newton for (x^3 - y, y^3 - x)")
F = sp.Matrix([x**3 - y, y**3 - x]); J = F.jacobian([x, y])
print("    J =", J.tolist(), " det =", sp.factor(J.det()))
sols = sp.solve(list(F), [x, y], dict=True); print("    real zeros:", [s_ for s_ in sols if all(v_.is_real for v_ in s_.values())])
x0 = sp.Matrix([2, 1]); D = -J.subs({x: 2, y: 1}).inv() * F.subs({x: 2, y: 1})
ok(D == sp.Matrix([sp.Rational(-4, 7), sp.Rational(1, 7)]), f"Delta = {list(D)} -> x1 = {list(x0 + D)}")
print("    F(2,1) =", list(F.subs({x: 2, y: 1})), " J(2,1) =", J.subs({x: 2, y: 1}).tolist())
print("    diagonal singular: 9x^4-1=0 ->", sp.solve(9 * x**4 - 1, x))
print("    det J(1,1) =", J.subs({x: 1, y: 1}).det())
# iterate a bit to see convergence
xx = np.array([2.0, 1.0])
for it in range(5):
    Fv = np.array([xx[0]**3 - xx[1], xx[1]**3 - xx[0]]); Jv = np.array([[3 * xx[0]**2, -1], [-1, 3 * xx[1]**2]])
    xx = xx - np.linalg.solve(Jv, Fv); print("      it", it + 1, xx, np.linalg.norm(xx - 1))

print("=== C5: f = x^3 - 3xy + y^3")
f = x**3 - 3 * x * y + y**3
g = sp.Matrix([f.diff(x), f.diff(y)]); H = sp.hessian(f, (x, y))
print("    grad =", list(g), " stationary:", sp.solve(list(g), [x, y], dict=True))
print("    H(0,0) =", H.subs({x: 0, y: 0}).tolist(), "det", H.subs({x: 0, y: 0}).det(), " H(1,1) =", H.subs({x: 1, y: 1}).tolist(), "det", H.subs({x: 1, y: 1}).det())
print("    f(1,1) =", f.subs({x: 1, y: 1}), " f(-2,0) =", f.subs({x: -2, y: 0}))
p0 = sp.Matrix([1, 0]); g0 = g.subs({x: 1, y: 0}); print("    grad(1,0) =", list(g0), " f(1,0) =", f.subs({x: 1, y: 0}))
for ss in [sp.Rational(1, 3), sp.Rational(1, 6)]:
    p1 = p0 - ss * g0; print(f"    s={ss}: x1 = {list(p1)}, f = {f.subs({x: p1[0], y: p1[1]})}")
q0 = sp.Matrix([2, 2]); D = -H.subs({x: 2, y: 2}).inv() * g.subs({x: 2, y: 2})
q1 = q0 + D; print("    Newton from (2,2): grad =", list(g.subs({x: 2, y: 2})), " H =", H.subs({x: 2, y: 2}).tolist(), " x1 =", list(q1))
D2 = -H.subs({x: q1[0], y: q1[1]}).inv() * g.subs({x: q1[0], y: q1[1]}); print("    second step x2 =", list(q1 + D2))

print("=== C6: Fourier series of x^2")
c0 = sp.integrate(x**2, (x, -sp.pi, sp.pi)) / (2 * sp.pi)
ck = sp.simplify(sp.integrate(x**2 * sp.exp(-sp.I * k * x), (x, -sp.pi, sp.pi)) / (2 * sp.pi))
print("    c0 =", c0, " ck =", sp.simplify(ck.rewrite(sp.cos)))
for kk in [1, 2, 3, -2]:
    val = sp.integrate(x**2 * sp.exp(-sp.I * kk * x), (x, -sp.pi, sp.pi)) / (2 * sp.pi)
    ok(sp.simplify(val - 2 * (-1)**kk / sp.Integer(kk)**2) == 0, f"c_{kk} = 2(-1)^k/k^2")
ok(sp.summation(1 / sp.Symbol("n", integer=True, positive=True)**2, (sp.Symbol("n", integer=True, positive=True), 1, sp.oo)) == sp.pi**2 / 6, "sum 1/n^2 = pi^2/6")
lhs = sp.integrate(x**4, (x, -sp.pi, sp.pi)) / (2 * sp.pi); print("    (1/2pi) int x^4 =", lhs)
S4 = sp.simplify((lhs - sp.pi**4 / 9) / 8); ok(sp.simplify(S4 - sp.pi**4 / 90) == 0, f"sum 1/k^4 = {S4}")
for kk in [1, 2, 3]:
    val = sp.integrate(x * sp.exp(-sp.I * kk * x), (x, -sp.pi, sp.pi)) / (2 * sp.pi)
    ok(sp.simplify(val - sp.I * (-1)**kk / kk) == 0, f"c_{kk}(x) = i(-1)^k/k")
h = sp.integrate((x - z)**2 * sp.cos(2 * z), (z, -sp.pi, sp.pi))  # (f*g)(x) with g=cos2 at shifted arg: int f(y) g(x-y) dy, here use f(x-z) g(z)
# careful: periodic f; (g*f)(x) = int g(z) f(x-z) dz needs periodic f; check numerically instead
def fper(u): u = (u + np.pi) % (2 * np.pi) - np.pi; return u**2
xs = np.linspace(-np.pi, np.pi, 200001); xm = 0.7
conv = np.trapezoid(fper(xs) * np.cos(2 * (xm - xs)), xs)
ok(abs(conv - np.pi * np.cos(2 * xm)) < 1e-6, f"(f*cos2)(0.7) = {conv:.6f} vs pi cos(1.4) = {np.pi*np.cos(1.4):.6f}")

print("=== C7: DFT of (1,2,0,-1)")
f = np.array([1, 2, 0, -1], dtype=complex); N = 4
F4 = np.array([[np.exp(-2j * np.pi * j * kk / N) for j in range(N)] for kk in range(N)])
fh = F4 @ f / N; print("    fhat =", np.round(fh, 10))
xj = 2 * np.pi * np.arange(4) / 4; p = 0.5 + 0.5 * np.cos(xj) + 1.5 * np.sin(xj)
ok(np.allclose(p, f.real), f"p(x_j) = {np.round(p, 10)}")
print("    N^2 =", 1024**2, " N log2 N =", 1024 * 10, " ratio =", 1024**2 / 10240)

print("=== C8: heat on R")
u = 1 / sp.sqrt(1 + 8 * t) * sp.exp(-x**2 / (1 + 8 * t))
ok(sp.simplify(u.diff(t) - 2 * u.diff(x, 2)) == 0, "u_t = 2 u_xx")
ok(sp.simplify(u.subs(t, 0) - sp.exp(-x**2)) == 0, "IC")
ok(sp.simplify(sp.integrate(u, (x, -sp.oo, sp.oo)) - sp.sqrt(sp.pi)) == 0, "mass sqrt(pi) conserved")
FT = lambda g: sp.simplify(sp.integrate(g * sp.exp(-sp.I * w * x), (x, -sp.oo, sp.oo)) / sp.sqrt(2 * sp.pi))
uh = FT(u.subs(t, sp.Rational(1, 2))); target = sp.exp(-w**2 * (1 + 8 * sp.Rational(1, 2)) / 4) / sp.sqrt(2)
ok(sp.simplify(uh - target) == 0, f"uhat(w,1/2) = {uh}")

print("=== C9: heat FD, h=1/4")
U0 = np.array([1.0, 2.0, 1.0])
def explicit(U, al):
    Ue = np.concatenate([[0], U, [0]]); return U + al * (Ue[:-2] - 2 * U + Ue[2:])
for al in [0.5, 1.0]:
    U = U0.copy(); out = []
    for n in range(3): U = explicit(U, al); out.append(U.copy())
    print(f"    alpha={al}:", [o.tolist() for o in out])
Bm = np.array([[-2, 1, 0], [1, -2, 1], [0, 1, -2]]); U1 = np.linalg.solve(np.eye(3) - 1.0 * Bm, U0)
print("    implicit alpha=1:", U1, " *7 =", U1 * 7)
print("    f(x)=2(1-|2x-1|) at 1/4,1/2,3/4:", [2 * (1 - abs(2 * q - 1)) for q in (0.25, 0.5, 0.75)])

print("=== C10: natural cubic spline through (0,0),(1,1),(2,0),(3,1)")
from scipy.interpolate import CubicSpline
cs = CubicSpline([0, 1, 2, 3], [0, 1, 0, 1], bc_type="natural")
print("    M =", cs(np.array([0, 1, 2, 3.0]), 2), " s(1/2) =", cs(0.5), " s'(1) =", cs(1, 1))
xs_ = sp.symbols("xs")
s01 = sp.Rational(5, 3) * xs_ - sp.Rational(2, 3) * xs_**3
ok(abs(float(s01.subs(xs_, 0.37)) - cs(0.37)) < 1e-12, "piece on [0,1] = 5/3 x - 2/3 x^3")
s12 = 1 - sp.Rational(1, 3) * (xs_ - 1) - 2 * (xs_ - 1)**2 + sp.Rational(4, 3) * (xs_ - 1)**3
ok(abs(float(s12.subs(xs_, 1.61)) - cs(1.61)) < 1e-12, "piece on [1,2]")
s23 = 1 - sp.Rational(5, 3) * (3 - xs_) + sp.Rational(2, 3) * (3 - xs_)**3
ok(abs(float(s23.subs(xs_, 2.42)) - cs(2.42)) < 1e-12, "piece on [2,3]")
# membership checks
f2 = sp.Abs(xs_ - 2)**3
print("    |x-2|^3 derivs left/right at 2:", [sp.limit(sp.diff((2 - xs_)**3, xs_, m), xs_, 2) for m in range(3)], [sp.limit(sp.diff((xs_ - 2)**3, xs_, m), xs_, 2) for m in range(3)])
print("    x^3+(x-1)_+^2: second derivative jump at 1 =", 2)
