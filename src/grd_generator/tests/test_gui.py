from pathlib import Path

import numpy as np

from grd_generator.gui import CalibrationResult, build_result

REPORTS = Path(__file__).resolve().parents[3] / "data" / "reference_reports"


def test_build_result_shapes_and_mode() -> None:
    res = build_result(
        antenna_lat=46.6,
        antenna_lon=2.5,
        sat_lon=3.0,
        zone_radius_deg=6.0,
        report_path=REPORTS / "0001",
        n_elements=12,
        seed=0,
        coverage_margin_db=3.0,
        n_u=41,
        n_v=41,
    )
    assert isinstance(res, CalibrationResult)
    assert res.fields.shape == (12, 41, 41)
    assert res.fields.dtype == np.complex128
    assert res.centers_uv.shape == (12, 2)
    assert res.peaks_dbi.shape == (12,)
    assert res.mode == "gaussian"
    assert len(res.specs) == 12


def test_build_result_is_reproducible() -> None:
    kw = dict(
        antenna_lat=46.6, antenna_lon=2.5, sat_lon=3.0, zone_radius_deg=6.0,
        report_path=REPORTS / "0000", n_elements=10, seed=5, coverage_margin_db=3.0,
        n_u=41, n_v=41,
    )
    a = build_result(**kw)  # type: ignore[arg-type]
    b = build_result(**kw)  # type: ignore[arg-type]
    assert np.allclose(a.fields, b.fields)
