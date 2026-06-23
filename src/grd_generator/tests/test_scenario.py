import math

import pytest

from grd_generator.geometry import earth_limb_angle_deg
from grd_generator.scenario import FRANCE, Scenario, sat_frame_geometry


def test_france_defaults() -> None:
    assert FRANCE.name == "france"
    assert FRANCE.sat_lon_deg == 3.0
    assert FRANCE.antenna_latlon == (46.6, 2.5)
    assert FRANCE.zone_radius_deg == 6.0
    assert FRANCE.limb_margin_deg == 0.2


def test_scenario_rejects_non_positive_radius() -> None:
    with pytest.raises(ValueError):
        Scenario(name="x", sat_lon_deg=0.0, antenna_latlon=(0.0, 0.0), zone_radius_deg=0.0)


def test_scenario_accepts_small_positive_radius() -> None:
    # plus de plancher réglementaire de 6° : un petit rayon physique est valide
    sc = Scenario(name="x", sat_lon_deg=0.0, antenna_latlon=(0.0, 0.0), zone_radius_deg=2.0)
    assert sc.zone_radius_deg == 2.0


def test_sat_frame_geometry_zone_under_limb() -> None:
    geo = sat_frame_geometry(FRANCE)
    # The whole service disk stays under the limb (with margin).
    center_dist = math.hypot(*geo.zone_center)
    limb = earth_limb_angle_deg()
    assert center_dist + geo.zone_radius_deg + FRANCE.limb_margin_deg <= limb + 1e-9
    assert geo.grid.n_u == 161 and geo.grid.n_v == 161


def test_sat_frame_geometry_grid_spans_zone() -> None:
    geo = sat_frame_geometry(FRANCE)
    half_span = geo.zone_radius_deg + 0.5  # _GRID_MARGIN_DEG
    assert geo.grid.u_min == pytest.approx(geo.zone_center[0] - half_span)
    assert geo.grid.u_max == pytest.approx(geo.zone_center[0] + half_span)


def test_zone_too_wide_raises() -> None:
    # radius ≥ limb − margin cannot fit under the limb.
    big = Scenario(name="big", sat_lon_deg=0.0, antenna_latlon=(0.0, 0.0), zone_radius_deg=9.0)
    with pytest.raises(ValueError):
        sat_frame_geometry(big)


def test_render_kwargs_and_roundtrip_helpers() -> None:
    geo = sat_frame_geometry(FRANCE)
    kw = geo.render_kwargs()
    assert kw["boresight_lat_deg"] == 0.0
    assert kw["sat_lon_deg"] == FRANCE.sat_lon_deg
    # ground_latlon of the zone center is on Earth; a far off-disk point is None.
    assert geo.ground_latlon(*geo.zone_center) is not None
    assert geo.ground_latlon(45.0, 0.0) is None
    # project_latlon of the antenna aim lands near the zone center direction.
    u, v = geo.project_latlon(*FRANCE.antenna_latlon)
    assert isinstance(u, float) and isinstance(v, float)
