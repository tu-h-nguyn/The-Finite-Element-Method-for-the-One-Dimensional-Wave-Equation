"""
FEM 1D cho phuong trinh song:  u_tt - c^2 u_xx = 0 tren (0,1)x(0,T),
Dirichlet thuan nhat, phan tu P1, leapfrog (sai phan trung tam).

Khu bac tu do bien -> he doi xung xac dinh duong (giu duoc tinh doi xung).
Khoi tao buoc dau bac hai. Giai he 3 duong cheo bang thuat toan Thomas O(N).
"""
import numpy as np


# ----------------------------------------------------------------------
# 1. Lap ma tran (chi cac bac tu do trong: x_2,...,x_N  ->  n = N-1)
# ----------------------------------------------------------------------
def assemble(N, lumped=False):
    h = 1.0 / N
    n = N - 1
    if lumped:
        Td, Tl = np.full(n, h), np.zeros(n - 1)
    else:
        Td, Tl = np.full(n, 2.0 * h / 3.0), np.full(n - 1, h / 6.0)
    Sd, Sl = np.full(n, 2.0 / h), np.full(n - 1, -1.0 / h)
    return (Td, Tl), (Sd, Sl), h


def tri_mul(diag, off, v):
    """Nhan ma tran 3 duong cheo doi xung voi vector."""
    w = diag * v
    w[:-1] += off * v[1:]
    w[1:] += off * v[:-1]
    return w


# ----------------------------------------------------------------------
# 2. Thuat toan Thomas (O(N)) cho he 3 duong cheo doi xung
# ----------------------------------------------------------------------
def thomas(diag, off, rhs):
    n = len(diag)
    if n == 0:
        return rhs.copy()
    if np.allclose(off, 0.0):
        return rhs / diag
    cp = np.empty(n - 1)
    dp = np.empty(n)
    cp[0] = off[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - off[i - 1] * cp[i - 1]
        if i < n - 1:
            cp[i] = off[i] / m
        dp[i] = (rhs[i] - off[i - 1] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ----------------------------------------------------------------------
# 3. Sai so L2 va nua chuan H1 (Gauss-Legendre 5 diem moi phan tu)
# ----------------------------------------------------------------------
_GP, _GW = np.polynomial.legendre.leggauss(5)


def errors(Uint, N, t, c=1.0):
    """Uint: gia tri tai cac nut trong. Nghiem chinh xac cos(2 pi c t) sin(2 pi x)."""
    h = 1.0 / N
    U = np.concatenate(([0.0], Uint, [0.0]))          # them 2 nut bien
    xl = np.arange(N) * h
    s = 0.5 * (_GP + 1.0)                              # toa do tham chieu [0,1]
    xq = xl[:, None] + h * s[None, :]
    w = 0.5 * h * _GW[None, :]

    Uq = U[:-1, None] * (1 - s)[None, :] + U[1:, None] * s[None, :]
    dUq = ((U[1:] - U[:-1]) / h)[:, None]

    ue = np.cos(2 * np.pi * c * t) * np.sin(2 * np.pi * xq)
    due = np.cos(2 * np.pi * c * t) * 2 * np.pi * np.cos(2 * np.pi * xq)

    l2 = np.sqrt(np.sum(w * (Uq - ue) ** 2))
    h1 = np.sqrt(np.sum(w * (dUq - due) ** 2))
    return l2, h1


# ----------------------------------------------------------------------
# 4. Vong lap thoi gian
# ----------------------------------------------------------------------
def solve(N, theta=0.5, T=1.0, c=1.0, lumped=False):
    """theta = dt / h. Tra ve (l2, h1) tai thoi diem T."""
    (Td, Tl), (Sd, Sl), h = assemble(N, lumped)
    dt = theta * h
    M = max(1, int(round(T / dt)))
    dt = T / M                                        # chia deu de dung dung T

    x = np.linspace(0, 1, N + 1)[1:-1]
    U0 = np.sin(2 * np.pi * x)                        # u(x,0)
    V0 = np.zeros_like(U0)                            # u_t(x,0) = 0

    # Khoi tao bac hai: U1 = U0 + dt V0 + dt^2/2 * Udd0,  Udd0 = -c^2 T^{-1} S U0
    A0 = thomas(Td, Tl, -(c ** 2) * tri_mul(Sd, Sl, U0))
    Um, Uc = U0, U0 + dt * V0 + 0.5 * dt ** 2 * A0

    for _ in range(1, M):
        rhs = -(c ** 2) * tri_mul(Sd, Sl, Uc)
        Un = 2 * Uc - Um + dt ** 2 * thomas(Td, Tl, rhs)
        Um, Uc = Uc, Un
        if not np.isfinite(Uc).all():
            return np.inf, np.inf
    return errors(Uc, N, T, c)


def lam_max(N, lumped=False):
    """Tri rieng suy rong lon nhat cua (S, T) - cong thuc giai tich."""
    h = 1.0 / N
    th = np.arange(1, N) * np.pi / N
    if lumped:
        return np.max((2 - 2 * np.cos(th)) / h ** 2)
    return np.max((6 / h ** 2) * (1 - np.cos(th)) / (2 + np.cos(th)))


# ----------------------------------------------------------------------
# 5. Quet on dinh: giu dt = theta*h CHINH XAC, do he so tang bien do
# ----------------------------------------------------------------------
def growth(N, theta, T=1.0, c=1.0, lumped=False, seed=1e-8):
    """Tra ve max|U| sau M = floor(T/dt) buoc, voi dt = theta*h chinh xac.

    seed: bien do kich thich mode rieng cao nhat (m = N-1). Mode nay moi la
    mode mat on dinh dau tien khi dt vuot nguong; du lieu ban dau sin(2 pi x)
    hau nhu khong chua no, nen neu khong kich thich thi phai cho lam tron may
    khuech dai va bang 3 (Muc 7.3) khong tai lap duoc. Dat seed=0.0 de tat.
    """
    (Td, Tl), (Sd, Sl), h = assemble(N, lumped)
    dt = theta * h
    M = max(1, int(T / dt))
    x = np.linspace(0, 1, N + 1)[1:-1]
    U0 = np.sin(2 * np.pi * x) + seed * np.sin((N - 1) * np.pi * x)
    A0 = thomas(Td, Tl, -(c ** 2) * tri_mul(Sd, Sl, U0))
    Um, Uc = U0, U0 + 0.5 * dt ** 2 * A0
    for _ in range(1, M):
        Un = 2 * Uc - Um + dt ** 2 * thomas(Td, Tl, -(c ** 2) * tri_mul(Sd, Sl, Uc))
        Um, Uc = Uc, Un
        if not np.isfinite(Uc).all():
            return np.inf
    return np.max(np.abs(Uc))
