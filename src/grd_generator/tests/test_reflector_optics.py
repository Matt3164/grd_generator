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
