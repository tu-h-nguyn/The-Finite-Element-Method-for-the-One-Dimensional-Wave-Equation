"""Kiem tra cac khoi co ban cua fem_wave.py doi chieu voi toan hoc dong kin.

Nguyen tac: moi khang dinh duoi day duoc doi chieu voi mot gia tri tinh DOC
LAP -- cong thuc giai tich, mot bo giai dac day, hay mot dang thuc rut ra bang
tay -- chu khong doi chieu voi chinh dau ra cua ma nguon.
"""
from __future__ import annotations

import numpy as np
import pytest

from fem_wave import assemble, errors, growth, lam_max, solve, thomas, tri_mul


# ----------------------------------------------------------------------
# Lap ma tran
# ----------------------------------------------------------------------
@pytest.mark.parametrize("N", [4, 8, 17])
def test_mass_and_stiffness_match_the_textbook_stencils(N):
    """P1 tren luoi deu: T = (h/6)[1 4 1], S = (1/h)[-1 2 -1]."""
    (Td, Tl), (Sd, Sl), h = assemble(N)
    assert h == pytest.approx(1.0 / N)
    assert Td == pytest.approx(np.full(N - 1, 4 * h / 6))
    assert Tl == pytest.approx(np.full(N - 2, h / 6))
    assert Sd == pytest.approx(np.full(N - 1, 2 / h))
    assert Sl == pytest.approx(np.full(N - 2, -1 / h))


@pytest.mark.parametrize("N", [4, 8, 17])
def test_lumping_sums_each_row_onto_its_diagonal(N):
    """Gom khoi luong dung nghia la don tung HANG len duong cheo. Voi mot hang
    trong (co du ca hai lang gieng) tong hang la h/6 + 4h/6 + h/6 = h.

    Luu y: KHONG the doi hoi sum(T_full) == sum(T_lumped) tren khoi cac bac tu
    do TRONG, vi cac hang bien da bi khu -- tong hai ve khac nhau (voi N = 4 la
    0,667 so voi 0,750). Dang thuc chi dung theo TUNG HANG."""
    (Td, Tl), _, h = assemble(N, lumped=False)
    (Ld, Ll), _, _ = assemble(N, lumped=True)
    assert Ll == pytest.approx(np.zeros(N - 2))
    interior = slice(1, -1)          # cac hang co ca hai lang gieng
    assert Td[interior] + 2 * Tl[0] == pytest.approx(Ld[interior])
    assert Ld[interior] == pytest.approx(np.full(max(N - 3, 0), h))


@pytest.mark.parametrize("N", [5, 12])
def test_stiffness_annihilates_nothing_and_is_positive_definite(N):
    """S phai xac dinh duong: bai toan Dirichlet khong co mode khong."""
    (_, _), (Sd, Sl), _ = assemble(N)
    A = np.diag(Sd) + np.diag(Sl, 1) + np.diag(Sl, -1)
    assert np.all(np.linalg.eigvalsh(A) > 0)


# ----------------------------------------------------------------------
# Thuat toan Thomas
# ----------------------------------------------------------------------
@pytest.mark.parametrize("N", [4, 9, 25])
@pytest.mark.parametrize("lumped", [False, True])
def test_thomas_agrees_with_a_dense_solve(N, lumped):
    """O(N) phai cho dung ket qua nhu np.linalg.solve tren ma tran day."""
    (Td, Tl), _, _ = assemble(N, lumped)
    rng = np.random.default_rng(0)
    rhs = rng.standard_normal(N - 1)
    A = np.diag(Td) + np.diag(Tl, 1) + np.diag(Tl, -1)
    assert thomas(Td, Tl, rhs) == pytest.approx(np.linalg.solve(A, rhs), rel=1e-12)


@pytest.mark.parametrize("N", [6, 20])
def test_thomas_inverts_tri_mul(N):
    """Hai ham nay phai nghich dao nhau -- kiem tra chung doi chieu voi nhau."""
    (Td, Tl), _, _ = assemble(N)
    rng = np.random.default_rng(1)
    v = rng.standard_normal(N - 1)
    assert thomas(Td, Tl, tri_mul(Td, Tl, v)) == pytest.approx(v, rel=1e-11)


# ----------------------------------------------------------------------
# Pho cua (S, T): cong thuc dong kin la trung tam cua ca bai toan
# ----------------------------------------------------------------------
@pytest.mark.parametrize("N", [4, 7, 16, 33])
@pytest.mark.parametrize("lumped", [False, True])
def test_lam_max_matches_a_dense_generalised_eigensolve(N, lumped):
    """lam_max dung cong thuc giai tich; doi chieu voi scipy-free dense solve."""
    (Td, Tl), (Sd, Sl), _ = assemble(N, lumped)
    T = np.diag(Td) + np.diag(Tl, 1) + np.diag(Tl, -1)
    S = np.diag(Sd) + np.diag(Sl, 1) + np.diag(Sl, -1)
    # T doi xung xac dinh duong -> dua ve bai toan tri rieng doi xung chuan.
    L = np.linalg.cholesky(T)
    Li = np.linalg.inv(L)
    dense = np.max(np.linalg.eigvalsh(Li @ S @ Li.T))
    assert lam_max(N, lumped) == pytest.approx(dense, rel=1e-10)


@pytest.mark.parametrize("N", [32, 128, 512])
def test_full_mass_spectrum_approaches_three_times_the_lumped_one(N):
    """Nguon goc cua he so sqrt(3) trong dieu kien CFL. Ty so tien toi 3 TU
    DUOI, va chi cham: N = 8 moi dat 2,79, nen phep kiem nay bat dau tu N = 32."""
    assert lam_max(N, lumped=False) / lam_max(N, lumped=True) == pytest.approx(
        3.0, rel=0.02
    )


def test_the_spectral_ratio_increases_monotonically_towards_three():
    r = [lam_max(N, False) / lam_max(N, True) for N in (4, 8, 16, 32, 128, 512)]
    assert all(a < b for a, b in zip(r, r[1:], strict=False)), r
    assert all(x < 3.0 for x in r), r


# ----------------------------------------------------------------------
# Dieu kien CFL sac net -- khang dinh trung tam cua bao cao
# ----------------------------------------------------------------------
@pytest.mark.parametrize("N", [32, 64, 128])
@pytest.mark.parametrize("lumped", [False, True])
def test_the_stability_threshold_is_exactly_where_the_spectrum_says(N, lumped):
    """Nguong SAC NET la theta* = 2/(h c sqrt(lam_max(N))) -- nguong roi rac,
    khong phai gia tri tiem can.

    Kiem tra HAI phia: ngay duoi nguong phai on dinh, ngay tren phai no. Mot
    phep kiem chi mot phia khong phan biet duoc chan sac net voi bat ky chan
    nao chat hon no.

    Hai nguong duoi day do duoc chu khong dat bua: tren ca sau to hop
    (N, lumped), phia on dinh khong bao gio vuot 1,000 -- luoc do BAO TOAN bien
    do, dung nhu mot luoc do leapfrog bao toan nang luong phai the -- con phia
    mat on dinh thap nhat cung dat 283. Khoang cach giua hai phia la 283 lan.

    Luu y: vuot nguong thi bien do no theo cap so nhan nhung van la so HUU HAN
    (cho N = 64, theta = 0,9 cho 3,4e116), nen khong dung duoc np.isfinite.
    """
    theta_star = 2.0 / ((1.0 / N) * np.sqrt(lam_max(N, lumped)))
    assert growth(N, 0.98 * theta_star, T=2.0, lumped=lumped) < 1.1
    assert growth(N, 1.02 * theta_star, T=2.0, lumped=lumped) > 50.0


@pytest.mark.parametrize("N", [16, 32, 64, 128])
@pytest.mark.parametrize("lumped", [False, True])
def test_the_discrete_threshold_converges_to_the_asymptotic_one_from_above(N, lumped):
    """theta* tien ve 1/sqrt(3) (day du) hoac 1 (gom khoi luong) TU TREN."""
    theta_star = 2.0 / ((1.0 / N) * np.sqrt(lam_max(N, lumped)))
    limit = 1.0 if lumped else 1.0 / np.sqrt(3.0)
    assert theta_star > limit
    assert theta_star == pytest.approx(limit, rel=0.02)


@pytest.mark.parametrize("N", [16, 32, 64, 128])
@pytest.mark.parametrize("lumped", [False, True])
def test_the_analytic_threshold_matches_the_spectral_one(N, lumped):
    """2/(c sqrt(lam_max)) phai bang h/sqrt(3) (day du) hoac h (gom khoi luong),
    den bac O(h^2)."""
    h = 1.0 / N
    dt_max = 2.0 / np.sqrt(lam_max(N, lumped))
    expected = h if lumped else h / np.sqrt(3.0)
    assert dt_max == pytest.approx(expected, rel=0.02)


def test_the_familiar_cfl_is_not_sufficient_for_the_full_mass_matrix():
    """Khang dinh trung tam cua bao cao: dt = 0,9 h/c thoa man dieu kien quen
    thuoc dt < h/c, nhung van NO khi dung ma tran khoi luong day du -- trong
    khi cung dt do lai on dinh neu gom khoi luong.

    Day la ly do ton tai cua ca bai toan, nen no duoc kiem ca hai chieu."""
    assert growth(64, 0.9, T=2.0, lumped=False) > 1e100
    assert growth(64, 0.9, T=2.0, lumped=True) < 10.0


# ----------------------------------------------------------------------
# Bac hoi tu
# ----------------------------------------------------------------------
def test_second_order_convergence_in_l2():
    """P1 + leapfrog: bac 2 theo chuan L2."""
    theta = 0.5 / np.sqrt(3.0)
    e = [solve(N, theta=theta, T=0.5)[0] for N in (16, 32, 64, 128)]
    rates = [np.log2(a / b) for a, b in zip(e, e[1:], strict=False)]
    assert all(1.8 < r < 2.2 for r in rates), rates


def test_first_order_convergence_in_h1_seminorm():
    """Nua chuan H1 mat mot bac, dung nhu ly thuyet noi."""
    theta = 0.5 / np.sqrt(3.0)
    e = [solve(N, theta=theta, T=0.5)[1] for N in (16, 32, 64, 128)]
    rates = [np.log2(a / b) for a, b in zip(e, e[1:], strict=False)]
    assert all(0.85 < r < 1.15 for r in rates), rates


# ----------------------------------------------------------------------
# Ham tinh sai so
# ----------------------------------------------------------------------
def test_errors_vanish_on_the_exact_nodal_interpolant_at_t_zero():
    """Tai t = 0 nghiem chinh xac la sin(2 pi x); noi suy nut cua no phai cho
    sai so L2 co O(h^2), khong phai O(1)."""
    prev = None
    for N in (16, 32, 64):
        x = np.linspace(0, 1, N + 1)[1:-1]
        l2, _ = errors(np.sin(2 * np.pi * x), N, 0.0)
        if prev is not None:
            assert prev / l2 == pytest.approx(4.0, rel=0.1)
        prev = l2


def test_errors_is_exactly_zero_for_the_zero_solution_at_a_quarter_period():
    """Tai c t = 1/4 nghiem chinh xac trieu tieu; U = 0 phai cho sai so 0."""
    N = 32
    l2, h1 = errors(np.zeros(N - 1), N, 0.25)
    assert l2 == pytest.approx(0.0, abs=1e-14)
    assert h1 == pytest.approx(0.0, abs=1e-14)


# ----------------------------------------------------------------------
# Khoi tao buoc dau bac hai
# ----------------------------------------------------------------------
def test_refining_dt_at_fixed_mesh_leaves_the_error_at_the_spatial_floor():
    """Voi luoi khong gian co dinh, lam min dt PHAI khong lam sai so xau di:
    sai so da bi thanh phan khong gian O(h^2) chi phoi.

    Phep kiem nay co mat vi mot ly do cu the. Kiem tra dot bien cho thay viec
    doi he so khoi tao buoc dau tu dt^2/2 thanh dt^2 -- tuc bo di chinh cai lam
    cho buoc khoi dong dat bac hai -- KHONG bi bat boi bat ky phep kiem nao
    khac trong tep nay: o N <= 128 sai so khong gian che mat no. O N = 256 thi
    no lo ra ngay, va khong phai lo ra mot chut: nghiem no len 1e20.
    """
    N = 256
    theta0 = 1.0 / np.sqrt(3.0)
    e = [solve(N, theta=f * theta0, T=0.5)[0] for f in (0.5, 0.25, 0.125, 0.0625)]
    assert all(np.isfinite(x) for x in e), e
    assert all(x < 1e-3 for x in e), e
    # Khong doi qua 1% -- da cham san khong gian.
    assert max(e) / min(e) < 1.01, e
