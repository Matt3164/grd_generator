import numpy as np
import pytest

from grd_generator.reflector import optics
from grd_generator.reflector.spec import ReflectorSpec


def _spec(offset: float = 0.0) -> ReflectorSpec:
    return ReflectorSpec(diameter_m=2.0, focal_length_m=2.0,
                         offset_clearance_m=offset, freq_hz=20e9)


def test_aperture_grid_disk_and_spacing() -> None:
    X, Y, inside, dx = optics.aperture_grid(_spec(), n_aperture=64, pad_factor=2)
    assert X.shape == (128, 128)
    assert dx == pytest.approx(2.0 / 64)
    # disk centered on (0, D/2)=(0,1): a point at (0,1) is inside, (0,3) outside.
    assert inside.sum() > 0
    r2 = X**2 + (Y - 1.0) ** 2
    np.testing.assert_array_equal(inside, r2 <= 1.0**2)


def test_illumination_angle_zero_on_axis() -> None:
    X = np.array([[0.0]])
    Y = np.array([[0.0]])
    assert optics.illumination_angle(X, Y, 2.0)[0, 0] == pytest.approx(0.0)


def test_aperture_field_uniform_when_q0_and_centered_feed() -> None:
    spec = _spec()
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture=32, pad_factor=2)
    field = optics.aperture_field(spec, (0.0, 0.0), X, Y, inside, q=0.0)
    # q=0 → cos^0 = 1 inside; centered feed → zero phase tilt → real & flat.
    assert np.allclose(field[inside], 1.0)
    assert np.allclose(field[~inside], 0.0)


def test_pol_vectors_pure_copol_on_axis() -> None:
    spec = _spec()
    X = np.array([[0.0]])
    Y = np.array([[0.0]])
    ex, ey = optics.aperture_pol_vectors(spec, X, Y)
    assert abs(ey[0, 0]) < 1e-12          # no cross-pol on axis
    assert abs(abs(ex[0, 0]) - 1.0) < 1e-12


def test_pol_vectors_cross_in_diagonal() -> None:
    spec = _spec()
    X = np.array([[0.4]])
    Y = np.array([[0.4]])
    ex, ey = optics.aperture_pol_vectors(spec, X, Y)
    assert abs(ey[0, 0]) > 1e-6           # diagonal region → cross-pol present


def test_aperture_field_defocus_zero_is_identical_to_default() -> None:
    spec = _spec()
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture=32, pad_factor=2)
    field_default = optics.aperture_field(spec, (0.01, 0.0), X, Y, inside, q=2.0)
    field_explicit_zero = optics.aperture_field(
        spec, (0.01, 0.0), X, Y, inside, q=2.0, defocus_m=0.0
    )
    np.testing.assert_array_equal(field_default, field_explicit_zero)


def test_aperture_field_defocus_adds_expected_quadratic_phase() -> None:
    spec = _spec()
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture=32, pad_factor=2)
    defocus_m = 0.2
    for feed_xy in [(0.0, 0.0), (0.02, -0.01)]:
        field_nodefocus = optics.aperture_field(spec, feed_xy, X, Y, inside, q=2.0)
        field_defocus = optics.aperture_field(
            spec, feed_xy, X, Y, inside, q=2.0, defocus_m=defocus_m
        )
        psi = optics.illumination_angle(X, Y, spec.focal_length_m)
        k = 2.0 * np.pi / spec.wavelength_m
        expected_extra_phase = k * defocus_m * (1.0 - np.cos(psi))
        # Ratio plutôt que différence brute pour éviter le wrap de phase ; on
        # ramène aussi l'attendu dans (-pi, pi] par la même transformation,
        # car la phase attendue peut dépasser un tour complet sur l'ouverture.
        extra_phase = np.angle(field_defocus[inside] / field_nodefocus[inside])
        expected_wrapped = np.angle(np.exp(1j * expected_extra_phase[inside]))
        np.testing.assert_allclose(extra_phase, expected_wrapped, atol=1e-9)


def test_aperture_field_defocus_sign_flips_extra_phase() -> None:
    spec = _spec()
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture=32, pad_factor=2)
    feed_xy = (0.0, 0.0)
    field_nodefocus = optics.aperture_field(spec, feed_xy, X, Y, inside, q=2.0)
    field_pos = optics.aperture_field(spec, feed_xy, X, Y, inside, q=2.0, defocus_m=0.2)
    field_neg = optics.aperture_field(spec, feed_xy, X, Y, inside, q=2.0, defocus_m=-0.2)
    extra_phase_pos = np.angle(field_pos[inside] / field_nodefocus[inside])
    extra_phase_neg = np.angle(field_neg[inside] / field_nodefocus[inside])
    np.testing.assert_allclose(extra_phase_pos, -extra_phase_neg, atol=1e-9)
