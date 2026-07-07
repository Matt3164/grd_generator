import numpy as np
import pytest

from grd_generator.reflector import farfield
from grd_generator.reflector.spec import ReflectorSpec
from grd_generator.schemas import UVGrid


def test_uniform_aperture_peak_directivity_matches_airy() -> None:
    # Uniform circular aperture (q=0, centered feed), symmetric (no offset to
    # keep it scalar/co-pol): peak directivity ≈ (πD/λ)².
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=1e9,  # ~flat: ψ≈0, n̂≈ẑ
                         offset_clearance_m=0.0, freq_hz=20e9)
    # center the disk on the axis by overriding aperture: build a centered disk.
    n_aperture, pad = 96, 6
    dx = spec.diameter_m / n_aperture
    n = n_aperture * pad
    coords = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(coords, coords)
    inside = (spec.diameter_m / 2.0) ** 2 >= (X**2 + Y**2)
    aperture = inside.astype(np.complex128)

    F, L, M = farfield.far_field_fft(aperture, dx, spec.wavelength_m)
    dL = L[0, 1] - L[0, 0]
    dM = M[1, 0] - M[0, 0]
    norm = farfield.normalize_to_directivity(np.abs(F) ** 2, L, M, dL, dM)
    peak_dbi = 10.0 * np.log10((np.abs(F) * norm).max() ** 2)
    expected = 10.0 * np.log10((np.pi * spec.diameter_m / spec.wavelength_m) ** 2)
    assert peak_dbi == pytest.approx(expected, abs=0.3)


def test_ludwig3_identity_at_boresight() -> None:
    L = np.array([[0.0]])
    M = np.array([[0.0]])
    Fx = np.array([[3.0 + 1j]])
    Fy = np.array([[1.0 - 2j]])
    co, cross = farfield.ludwig3_co_cross(Fx, Fy, L, M)
    assert co[0, 0] == pytest.approx(3.0 + 1j)
    assert cross[0, 0] == pytest.approx(1.0 - 2j)


def test_ludwig3_pure_copol_has_negligible_cross_near_axis() -> None:
    lin = np.linspace(-0.05, 0.05, 11)
    L, M = np.meshgrid(lin, lin)
    Fx = np.ones_like(L, dtype=complex)
    Fy = np.zeros_like(L, dtype=complex)
    _, cross = farfield.ludwig3_co_cross(Fx, Fy, L, M)
    assert np.max(np.abs(cross)) < 1e-2


def test_resample_picks_nearest_grid_values() -> None:
    lin = np.linspace(-0.2, 0.2, 41)
    field = np.tile(lin, (41, 1)).astype(complex)  # value == l
    grid = UVGrid(u_min=-0.15, u_max=0.15, v_min=-0.15, v_max=0.15, n_u=11, n_v=11)
    out = farfield.resample_to_uvgrid(field, lin, grid)
    u, _ = grid.axes()
    # Grille reflector déjà en cosinus directeurs : pas de `sin`, échantillonnage direct.
    np.testing.assert_allclose(out[5, :].real, u, atol=1e-3)
