from pathlib import Path

import numpy as np

from grd_generator.gui import (
    CalibrationResult,
    ReflectorResult,
    build_reflector_result,
    build_result,
)
from grd_generator.schemas import UVGrid

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


def test_build_reflector_result_pure() -> None:
    grid = UVGrid(u_min=-14, u_max=14, v_min=-14, v_max=14, n_u=41, n_v=41)
    res = build_reflector_result(
        diameter_m=2.0, f_over_d=1.2, offset_clearance_m=0.0, freq_ghz=20.0,
        q=2.0, pitch_m=0.03, n_feeds=7, zone_radius_deg=8.0, grid=grid,
    )
    assert isinstance(res, ReflectorResult)
    assert len(res.co_fields) == 7
    assert res.zone.radius_deg == 8.0
    assert res.co_fields[0].shape == (41, 41)


def test_build_reflector_result_with_defocus() -> None:
    grid = UVGrid(u_min=-14, u_max=14, v_min=-14, v_max=14, n_u=41, n_v=41)
    res = build_reflector_result(
        diameter_m=2.0, f_over_d=1.2, offset_clearance_m=0.0, freq_ghz=20.0,
        q=2.0, pitch_m=0.03, n_feeds=7, zone_radius_deg=8.0, grid=grid,
        defocus_m=0.2,
    )
    assert isinstance(res, ReflectorResult)
    assert len(res.co_fields) == 7
    assert res.co_fields[0].shape == (41, 41)


def test_build_reflector_result_n_aperture_pad_factor_defaults_unchanged() -> None:
    # Les défauts n_aperture=128/pad_factor=4 restent ceux de synth_afr :
    # exposer ces kwargs ne doit rien changer sans les fournir explicitement.
    grid = UVGrid(u_min=-14, u_max=14, v_min=-14, v_max=14, n_u=41, n_v=41)
    res_default = build_reflector_result(
        diameter_m=2.0, f_over_d=1.2, offset_clearance_m=0.0, freq_ghz=20.0,
        q=2.0, pitch_m=0.03, n_feeds=7, zone_radius_deg=8.0, grid=grid,
    )
    res_explicit = build_reflector_result(
        diameter_m=2.0, f_over_d=1.2, offset_clearance_m=0.0, freq_ghz=20.0,
        q=2.0, pitch_m=0.03, n_feeds=7, zone_radius_deg=8.0, grid=grid,
        n_aperture=128, pad_factor=4,
    )
    for a, b in zip(res_default.co_fields, res_explicit.co_fields, strict=True):
        np.testing.assert_array_equal(a, b)


def test_build_result_is_reproducible() -> None:
    kw = dict(
        antenna_lat=46.6, antenna_lon=2.5, sat_lon=3.0, zone_radius_deg=6.0,
        report_path=REPORTS / "0000", n_elements=10, seed=5, coverage_margin_db=3.0,
        n_u=41, n_v=41,
    )
    a = build_result(**kw)  # type: ignore[arg-type]
    b = build_result(**kw)  # type: ignore[arg-type]
    assert np.allclose(a.fields, b.fields)


def test_reflector_studio_pitch_spinbox_accepts_millimeters() -> None:
    """Régression : le spinbox pitch (décimales=3) doit accepter 0.004 m sans arrondi."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    from PyQt6.QtWidgets import QApplication

    from grd_generator.gui import ReflectorStudio

    _app = QApplication.instance() or QApplication([])
    studio = ReflectorStudio(n_u=21, n_v=21)
    studio._pitch.setValue(0.004)
    assert studio._pitch.value() == 0.004
    studio.close()


def test_reflector_studio_click_updates_target_and_zoom_panel() -> None:
    """Un clic sur une carte large recale la cible et alimente le panneau zoom (axe 5)."""
    import os
    from types import SimpleNamespace

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    from PyQt6.QtWidgets import QApplication

    from grd_generator.gui import ReflectorStudio

    _app = QApplication.instance() or QApplication([])
    studio = ReflectorStudio(n_u=41, n_v=41)
    studio._diameter.setValue(2.2)
    studio._n_feeds.setValue(7)
    studio.generate()
    assert len(studio._axes) == 6

    event = SimpleNamespace(inaxes=studio._axes[2], xdata=0.0, ydata=2.0)
    studio._on_click(event)

    assert studio._target_uv == (0.0, 2.0)
    assert len(studio._axes) == 6
    assert studio._zoom_cache_key is not None
    assert studio._zoom_cache_result is not None
    studio.close()


def test_reflector_studio_click_outside_axes_is_noop() -> None:
    """Un événement hors des axes (inaxes=None) ne modifie pas la cible ni ne lève."""
    import os
    from types import SimpleNamespace

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    from PyQt6.QtWidgets import QApplication

    from grd_generator.gui import ReflectorStudio

    _app = QApplication.instance() or QApplication([])
    studio = ReflectorStudio(n_u=21, n_v=21)
    target_before = studio._target_uv

    event = SimpleNamespace(inaxes=None, xdata=1.0, ydata=1.0)
    studio._on_click(event)

    assert studio._target_uv == target_before
    studio.close()
