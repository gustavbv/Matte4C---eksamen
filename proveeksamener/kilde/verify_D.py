"""Verification of all numbers in practice exam D."""
import numpy as np
import sympy as sp

x, y, z, t, s = sp.symbols("x y z t s", real=True)
w = sp.symbols("omega", real=True)
ok = lambda c, msg: print(("OK  " if c else "FAIL"), msg)

print("=== D1: rank-deficient least squares")
A = sp.Matrix([[1, 1], [2, 2], [2, 2]]); b = sp.Matrix([3, 0, 3])
ATA = A.T * A; print("    A^T A =", ATA.tolist(), " eig:", ATA.eigenvals(), " rank:", A.rank())
u1 = sp.Matrix([1, 2, 2]) / 3; v1 = sp.Matrix([1, 1]) / sp.sqrt(2); s1 = 3 * sp.sqrt(2)
ok(sp.simplify(s1 * u1 * v1.T - A) == sp.zeros(3, 2), "A = sigma1 u1 v1^T, sigma1 = 3 sqrt2")
Ad = sp.simplify(v1 * u1.T / s1); ok(Ad == sp.Matrix([[1, 2, 2], [1, 2, 2]]) / 18, f"A^+ = {Ad.tolist()}")
ok(Ad == A.pinv(), "matches pinv")
print("    A^T b =", list(A.T * b))
xd = Ad * b; ok(xd == sp.Matrix([1, 1]) / 2, f"x+ = {list(xd)}")
r = A * xd - b; print("    residual =", list(r), " norm =", sp.sqrt(r.dot(r)))
for xx in [sp.Matrix([1, 0]), sp.Matrix([2, -1])]:
    rr = A * xx - b; print("    other LS sol", list(xx), "residual norm", sp.sqrt(rr.dot(rr)), "norm^2", xx.dot(xx))
ok(xd.dot(sp.Matrix([1, -1])) == 0, "x+ orthogonal to ker A = span(1,-1)")

print("=== D2: X = a o a o a + b o b o b")
a = np.array([1, 1]); bb = np.array([1, -1])
X = np.einsum("i,j,k->ijk", a, a, a) + np.einsum("i,j,k->ijk", bb, bb, bb)
print("    slices:", X[:, :, 0].tolist(), X[:, :, 1].tolist())
def unfold(X, mode):
    return np.reshape(np.moveaxis(X, mode, 0), (X.shape[mode], -1), order="F")
X1 = unfold(X, 0); print("    X(1) =", X1.tolist(), " rank", np.linalg.matrix_rank(X1))
Am = np.column_stack([a, bb]); KR = np.column_stack([np.kron(a, a), np.kron(bb, bb)])
ok(np.allclose(Am @ KR.T, X1), "X(1) = A (C kr B)^T")
print("    KR =", KR.T.tolist())
print("    ||X||_F =", np.linalg.norm(X), " <aaa,bbb> =", (a @ bb)**3)
for wv in ([1, 1], [1, -1], [1, 0]):
    print("    X x3", wv, "=", np.einsum("ijk,k->ij", X, wv).tolist())

print("=== D3: cone")
u, v = sp.symbols("u v", real=True)
sv = sp.Matrix([u * sp.cos(v), u * sp.sin(v), u])
su, svv = sv.diff(u), sv.diff(v); n = su.cross(svv)
print("    s_u =", list(su), " s_v =", list(svv), " n =", [sp.simplify(e) for e in n])
ok(sp.simplify(sv[0]**2 + sv[1]**2 - sv[2]**2) == 0, "on cone x^2+y^2=z^2")
sub = {u: 5, v: sp.atan2(4, 3)}
print("    at p: s_u =", [sp.simplify(e.subs(sub)) for e in su], " s_v =", [sp.simplify(e.subs(sub)) for e in svv], " n =", [sp.simplify(e.subs(sub)) for e in n])
print("    plane check n.(p) =", sp.Matrix([-3, -4, 5]).dot(sp.Matrix([3, 4, 5])))

print("=== D4: Jacobian & Newton")
G = sp.Matrix([x * y**2 - z, sp.sin(x * z) + y]); JG = G.jacobian([x, y, z])
pt = {x: 0, y: 1, z: 2}; print("    G' =", JG.tolist()); print("    G(a) =", list(G.subs(pt)), " G'(a) =", JG.subs(pt).tolist())
L = G.subs(pt) + JG.subs(pt) * sp.Matrix([x - 0, y - 1, z - 2]); print("    L =", [sp.expand(e) for e in L])
F = sp.Matrix([x**2 - y**2 - 3, x * y - 2]); JF = F.jacobian([x, y]); print("    J_F =", JF.tolist(), " det", sp.factor(JF.det()))
print("    zeros:", sp.solve(list(F), [x, y], dict=True))
D = -JF.subs({x: 1, y: 1}).inv() * F.subs({x: 1, y: 1}); print("    F(1,1) =", list(F.subs({x: 1, y: 1})), " Delta =", list(D), " x1 =", list(sp.Matrix([1, 1]) + D))
xx = np.array([1.0, 1.0])
for it in range(5):
    Fv = np.array([xx[0]**2 - xx[1]**2 - 3, xx[0] * xx[1] - 2]); Jv = np.array([[2 * xx[0], -2 * xx[1]], [xx[1], xx[0]]])
    xx = xx - np.linalg.solve(Jv, Fv); print("      it", it + 1, xx)

print("=== D5: Lagrange xy on x^2+4y^2=8")
lam = sp.symbols("lam", real=True)
f = x * y; g = x**2 + 4 * y**2 - 8
sols = sp.solve([f.diff(x) - lam * g.diff(x), f.diff(y) - lam * g.diff(y), g], [x, y, lam], dict=True)
for so in sols: print("    ", so, " f =", f.subs(so))
tang = sp.Matrix([2, -1]) / sp.sqrt(5); grad = sp.Matrix([f.diff(x), f.diff(y)]).subs({x: 2, y: 1})
print("    grad f(2,1) =", list(grad), " grad g(2,1) =", [g.diff(x).subs({x: 2, y: 1}), g.diff(y).subs({x: 2, y: 1})], " D_tangent f =", grad.dot(tang))

print("=== D6: Fourier transform")
FT = lambda gg, lo, hi: sp.simplify(sp.integrate(gg * sp.exp(-sp.I * w * x), (x, lo, hi)) / sp.sqrt(2 * sp.pi))
fh = FT(1, -1, 1); print("    F(chi) =", sp.simplify(fh.rewrite(sp.sin)))
ok(sp.simplify(fh.rewrite(sp.sin) - sp.sqrt(2 / sp.pi) * sp.sin(w) / w) == 0, "table value")
# convolution numerically
def chi(q): return (np.abs(q) <= 1).astype(float)
ys = np.linspace(-3, 3, 600001)
for xv in [0.0, 0.5, 1.3, 2.5]:
    val = np.trapezoid(chi(ys) * chi(xv - ys), ys); print(f"    (f*f)({xv}) = {val:.5f} vs {max(0, 2 - abs(xv))}")
gh = FT(2 - sp.Abs(x), -2, 2); gh = sp.simplify(gh.rewrite(sp.sin))
target = 2 * sp.sqrt(2 / sp.pi) * sp.sin(w)**2 / w**2
print("    F(triangle) numeric at w=0.7:", sp.N(gh.subs(w, 0.7)), " vs ", sp.N(target.subs(w, 0.7)))
gh2 = sp.integrate((2 - x) * sp.cos(w * x), (x, 0, 2)) * 2 / sp.sqrt(2 * sp.pi)
ok(abs(complex(sp.N(gh2.subs(w, 0.7) - target.subs(w, 0.7)))) < 1e-12, "F(triangle) = 2 sqrt(2/pi) sin^2 w / w^2")
fx = sp.integrate(x * sp.exp(-x**2) * sp.exp(-sp.I * w * x), (x, -sp.oo, sp.oo)) / sp.sqrt(2 * sp.pi)
ok(sp.simplify(fx - (-sp.I * w / (2 * sp.sqrt(2)) * sp.exp(-w**2 / 4))) == 0, f"F(x e^-x^2) = {sp.simplify(fx)}")

print("=== D7: DFT of 2cos x + e^{3ix}")
N = 4; xj = 2 * np.pi * np.arange(N) / N; fj = 2 * np.cos(xj) + np.exp(3j * xj)
print("    samples:", np.round(fj, 10))
fh = np.fft.fft(fj) / N; print("    fhat:", np.round(fh, 10))
p = np.exp(1j * xj) + 2 * np.exp(-1j * xj); ok(np.allclose(p, fj), "p = e^{ix} + 2 e^{-ix} interpolates")
for NN in [6, 7, 8]:
    xx_ = 2 * np.pi * np.arange(NN) / NN; fh_ = np.fft.fft(2 * np.cos(xx_) + np.exp(3j * xx_)) / NN
    print(f"    N={NN}: fhat =", np.round(fh_, 10))

print("=== D8: heat with u(0)=0, u_x(pi)=0")
uu = 3 * sp.exp(-t / 4) * sp.sin(x / 2) - sp.exp(-25 * t / 4) * sp.sin(5 * x / 2)
ok(sp.simplify(uu.diff(t) - uu.diff(x, 2)) == 0, "PDE")
ok(sp.simplify(uu.subs(x, 0)) == 0, "u(0,t)=0"); ok(sp.simplify(uu.diff(x).subs(x, sp.pi)) == 0, "u_x(pi,t)=0")
print("    u(pi,t) =", sp.simplify(uu.subs(x, sp.pi)), " u(x,0) =", uu.subs(t, 0))

print("=== D9: -u''=2, u'(0)=0, u(1)=0, h=1/3")
h = sp.Rational(1, 3); Am = sp.Matrix([[2, -2, 0], [-1, 2, -1], [0, -1, 2]]) / h**2
U = Am.solve(sp.Matrix([2, 2, 2])); print("    U =", list(U), " exact:", [1 - (i * h)**2 for i in range(3)])

print("=== D10: B-splines, uniform knots")
def B(i, kord, xx, tk):
    if kord == 1: return 1.0 if tk[i] <= xx < tk[i + 1] else 0.0
    a1 = (xx - tk[i]) / (tk[i + kord - 1] - tk[i]) * B(i, kord - 1, xx, tk)
    a2 = (tk[i + kord] - xx) / (tk[i + kord] - tk[i + 1]) * B(i + 1, kord - 1, xx, tk)
    return a1 + a2
tk = {i: float(i) for i in range(-5, 10)}
def B03(xx):
    if 0 <= xx < 1: return xx**2 / 2
    if 1 <= xx < 2: return (-2 * xx**2 + 6 * xx - 3) / 2
    if 2 <= xx < 3: return (3 - xx)**2 / 2
    return 0.0
ok(all(abs(B(0, 3, q, tk) - B03(q)) < 1e-14 for q in np.linspace(-0.5, 3.5, 801)), "B_{0,3} pieces correct")
ok(all(abs(B(-2, 3, q, tk) + B(-1, 3, q, tk) + B(0, 3, q, tk) - 1) < 1e-14 for q in np.linspace(0, 0.999, 300)), "partition of unity on [0,1)")
print("    B03(1.5) =", B03(1.5))
al, be, ga = sp.symbols("alpha beta gamma")
p2 = -x**3 + al * x**2 + be * x + ga
sol = sp.solve([p2.subs(x, 1) - 1, p2.diff(x).subs(x, 1) - 3, p2.diff(x, 2).subs(x, 1) - 6], [al, be, ga]); print("    alpha,beta,gamma =", sol)
