import io
import math
import zipfile

import numpy as np
import pytest

from grd_generator.reflector.grd_export import (
    pattern_to_grd,
    patterns_to_zip_bytes,
    simulation_params_dict,
)
from grd_generator.schemas import UVGrid


def _grid() -> UVGrid:
    return UVGrid(u_min=-10, u_max=10, v_min=-8, v_max=8, n_u=5, n_v=4)


def _fields() -> tuple[np.ndarray, np.ndarray]:
    grid = _grid()
    co = np.arange(grid.n_v * grid.n_u, dtype=np.complex128).reshape(grid.n_v, grid.n_u)
    cross = co * 0.1j
    return co, cross


def test_grd_header_and_row_count() -> None:
    grid = _grid()
    co, cross = _fields()
    text = pattern_to_grd(co, cross, grid, freq_hz=20e9)
    lines = text.splitlines()
    sep = lines.index("++++")
    assert int(lines[sep + 1]) == 1  # KTYPE
    nset, icomp, ncomp, igrid = (int(x) for x in lines[sep + 2].split())
    assert (nset, icomp, ncomp, igrid) == (1, 3, 2, 1)  # co/cross Ludwig-3, uv grid
    nx, ny, klimit = (int(x) for x in lines[sep + 5].split())
    assert (nx, ny, klimit) == (grid.n_u, grid.n_v, 0)
    data_rows = lines[sep + 6 :]
    assert len(data_rows) == grid.n_u * grid.n_v


def test_grd_grid_limits_in_direction_cosines() -> None:
    grid = _grid()
    co, cross = _fields()
    text = pattern_to_grd(co, cross, grid, freq_hz=20e9)
    lines = text.splitlines()
    sep = lines.index("++++")
    xs, ys, xe, ye = (float(x) for x in lines[sep + 4].split())
    assert math.isclose(xs, math.sin(math.radians(-10)), rel_tol=1e-9)
    assert math.isclose(ys, math.sin(math.radians(-8)), rel_tol=1e-9)
    assert math.isclose(xe, math.sin(math.radians(10)), rel_tol=1e-9)
    assert math.isclose(ye, math.sin(math.radians(8)), rel_tol=1e-9)


def test_grd_data_values_roundtrip() -> None:
    grid = _grid()
    co, cross = _fields()
    text = pattern_to_grd(co, cross, grid, freq_hz=20e9)
    lines = text.splitlines()
    sep = lines.index("++++")
    data = lines[sep + 6 :]
    # Première ligne = point (iv=0, iu=0) : co=0, cross=0
    r0 = [float(x) for x in data[0].split()]
    assert r0 == [0.0, 0.0, 0.0, 0.0]
    # Point (iv=0, iu=1) = co=1 → Re=1, Im=0 ; cross=0.1j → Re=0, Im=0.1
    r1 = [float(x) for x in data[1].split()]
    assert math.isclose(r1[0], 1.0) and math.isclose(r1[1], 0.0)
    assert math.isclose(r1[2], 0.0) and math.isclose(r1[3], 0.1)


def test_zip_has_one_grd_per_feed() -> None:
    grid = _grid()
    co, cross = _fields()
    co_fields = [co, co * 2.0, co * 3.0]
    cross_fields = [cross, cross, cross]
    blob = patterns_to_zip_bytes(co_fields, cross_fields, grid, freq_hz=20e9)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = zf.namelist()
        assert len(names) == 3
        assert names[0].endswith(".grd")
        # chaque membre est un GRD parseable non vide
        for n in names:
            body = zf.read(n).decode()
            assert "++++" in body


def test_simulation_params_dict_computes_focal_length() -> None:
    params = simulation_params_dict(
        diameter_m=2.0,
        f_over_d=1.2,
        offset_clearance_m=0.0,
        freq_ghz=20.0,
        q=2.0,
        pitch_m=0.03,
        n_feeds=80,
        zone_radius_deg=6.0,
    )
    assert params["diameter_m"] == 2.0
    assert params["n_feeds"] == 80
    assert math.isclose(params["focal_length_m"], 2.4)  # f_over_d * diameter
    assert params["freq_ghz"] == 20.0
    assert params["defocus_m"] == 0.0  # défaut quand non fourni
    assert params["phase_error_rms_rad"] == 0.0  # défaut quand non fourni
    assert params["phase_corr_length_m"] == pytest.approx(0.05)  # défaut quand non fourni
    assert params["phase_error_seed"] == 0  # défaut quand non fourni
    assert params["phase_error_shared_rms_rad"] == 0.0  # défaut quand non fourni
    assert params["footprint_m"] == 0.0  # défaut quand non fourni
    assert params["footprint_magnification"] == 0.0  # défaut quand non fourni


def test_simulation_params_dict_includes_defocus_m() -> None:
    params = simulation_params_dict(
        diameter_m=2.0,
        f_over_d=1.2,
        offset_clearance_m=0.0,
        freq_ghz=20.0,
        q=2.0,
        pitch_m=0.03,
        n_feeds=80,
        zone_radius_deg=6.0,
        defocus_m=0.15,
    )
    assert params["defocus_m"] == pytest.approx(0.15)


def test_simulation_params_dict_includes_phase_error_fields() -> None:
    params = simulation_params_dict(
        diameter_m=2.0,
        f_over_d=1.2,
        offset_clearance_m=0.0,
        freq_ghz=20.0,
        q=2.0,
        pitch_m=0.03,
        n_feeds=80,
        zone_radius_deg=6.0,
        phase_error_rms_rad=1.0,
        phase_corr_length_m=0.03,
        phase_error_seed=7,
    )
    assert params["phase_error_rms_rad"] == pytest.approx(1.0)
    assert params["phase_corr_length_m"] == pytest.approx(0.03)
    assert params["phase_error_seed"] == 7


def test_simulation_params_dict_includes_phase_error_shared_rms_rad() -> None:
    params = simulation_params_dict(
        diameter_m=2.0,
        f_over_d=1.2,
        offset_clearance_m=0.0,
        freq_ghz=20.0,
        q=2.0,
        pitch_m=0.03,
        n_feeds=80,
        zone_radius_deg=6.0,
        phase_error_shared_rms_rad=0.7,
    )
    assert params["phase_error_shared_rms_rad"] == pytest.approx(0.7)


def test_simulation_params_dict_includes_footprint_fields() -> None:
    params = simulation_params_dict(
        diameter_m=2.2,
        f_over_d=1.2,
        offset_clearance_m=0.0,
        freq_ghz=20.0,
        q=2.0,
        pitch_m=0.004,
        n_feeds=19,
        zone_radius_deg=6.0,
        footprint_m=0.28,
        footprint_magnification=48.0,
    )
    assert params["footprint_m"] == pytest.approx(0.28)
    assert params["footprint_magnification"] == pytest.approx(48.0)
