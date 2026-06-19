import numpy as np

from grd_generator.geometry import (
    earth_intersection_latlon,
    earth_limb_angle_deg,
    project_point,
    project_to_angular_masked,
)


def test_limb_angle_about_8_7_degrees() -> None:
    assert abs(earth_limb_angle_deg() - 8.7) < 0.1


def test_boresight_point_projects_to_origin() -> None:
    # A point at the boresight (sat sub-point) projects to (u, v) ≈ (0, 0).
    u, v = project_point(0.0, 3.0, 3.0, 0.0, 3.0)
    assert abs(u) < 1e-6 and abs(v) < 1e-6


def test_project_then_raytrace_roundtrip() -> None:
    # Project a ground point to (u, v), ray-trace back, recover lat/lon.
    sat_lon = 3.0
    lat0, lon0 = 46.6, 2.5
    u, v = project_point(lat0, lon0, sat_lon, 0.0, sat_lon)
    lat, lon, hit = earth_intersection_latlon(
        np.array([u]), np.array([v]), sat_lon, 0.0, sat_lon
    )
    assert bool(hit[0])
    assert abs(float(lat[0]) - lat0) < 1e-3
    assert abs(float(lon[0]) - lon0) < 1e-3


def test_backface_point_is_nan() -> None:
    # A point on the far side of Earth (opposite the sat) is hidden → NaN.
    u, v = project_to_angular_masked(
        np.array([0.0]), np.array([183.0]), 3.0, 0.0, 3.0
    )
    assert np.isnan(u[0]) and np.isnan(v[0])


def test_raytrace_miss_off_disk() -> None:
    # A direction well outside the limb misses Earth → hit False, NaN lat/lon.
    lat, lon, hit = earth_intersection_latlon(
        np.array([45.0]), np.array([0.0]), 3.0, 0.0, 3.0
    )
    assert not bool(hit[0])
    assert np.isnan(lat[0])
