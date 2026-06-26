import numpy as np
import pytest
from pydantic import ValidationError

from grd_generator.reflector.zone import ServiceZone
from grd_generator.schemas import UVGrid


def _grid() -> UVGrid:
    return UVGrid(u_min=-14, u_max=14, v_min=-14, v_max=14, n_u=29, n_v=29)


def test_zone_radius_bounds() -> None:
    with pytest.raises(ValidationError):
        ServiceZone(radius_deg=5.0)
    with pytest.raises(ValidationError):
        ServiceZone(radius_deg=14.1)
    ServiceZone(radius_deg=6.0)
    ServiceZone(radius_deg=14.0)


def test_zone_mask_is_disk() -> None:
    zone = ServiceZone(radius_deg=8.0)
    grid = _grid()
    mask = zone.mask(grid)
    gu, gv = grid.meshgrid()
    expected = (gu**2 + gv**2) <= 8.0**2
    assert mask.shape == (29, 29)
    np.testing.assert_array_equal(mask, expected)


def test_zone_boundary_on_circle() -> None:
    zone = ServiceZone(radius_deg=10.0, center_uv=(1.0, -2.0))
    u, v = zone.boundary_uv(n=180)
    r = np.hypot(u - 1.0, v + 2.0)
    np.testing.assert_allclose(r, 10.0, atol=1e-9)
    assert u[0] == pytest.approx(u[-1]) and v[0] == pytest.approx(v[-1])  # closed
