import pytest
from pydantic import ValidationError

from grd_generator.schemas import GaussianSpec, UVGrid


def test_uvgrid_axes_lengths() -> None:
    grid = UVGrid(u_min=-1.0, u_max=1.0, v_min=-2.0, v_max=2.0, n_u=5, n_v=9)
    u, v = grid.axes()
    assert u.shape == (5,) and v.shape == (9,)
    assert u[0] == -1.0 and u[-1] == 1.0


def test_uvgrid_meshgrid_shape_is_nv_by_nu() -> None:
    grid = UVGrid(u_min=0.0, u_max=1.0, v_min=0.0, v_max=1.0, n_u=4, n_v=7)
    gu, gv = grid.meshgrid()
    assert gu.shape == (7, 4) and gv.shape == (7, 4)


def test_uvgrid_rejects_degenerate_axis() -> None:
    with pytest.raises(ValidationError):
        UVGrid(u_min=0.0, u_max=1.0, v_min=0.0, v_max=1.0, n_u=1, n_v=5)


def test_gaussian_spec_requires_positive_sigma() -> None:
    with pytest.raises(ValidationError):
        GaussianSpec(center_uv=(0.0, 0.0), sigma=0.0, peak_gain_dbi=40.0)


def test_gaussian_spec_defaults_phase_slope_zero() -> None:
    spec = GaussianSpec(center_uv=(1.0, 2.0), sigma=0.5, peak_gain_dbi=40.0)
    assert spec.phase_slope_radial == 0.0
