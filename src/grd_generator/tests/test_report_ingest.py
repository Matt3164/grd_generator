import json
from pathlib import Path

import pytest

from grd_generator.report_ingest import MeasuredStats, read_report

REPORTS = Path(__file__).resolve().parents[3] / "data" / "reference_reports"


def test_read_report_0000_airy() -> None:
    stats = read_report(REPORTS / "0000")  # directory form
    assert isinstance(stats, MeasuredStats)
    assert stats.mode == "airy"
    assert stats.n_elements == 82
    assert stats.spacing_deg == pytest.approx(0.009391, abs=1e-5)
    assert stats.peak_dbi.mean == pytest.approx(27.87512, abs=1e-4)
    assert stats.peak_dbi.std == pytest.approx(0.32509, abs=1e-4)
    assert stats.lobe_width_major_deg.mean == pytest.approx(0.24326, abs=1e-4)
    assert stats.orientation_deg.std == pytest.approx(53.23991, abs=1e-3)
    assert stats.first_null_radius_deg is not None
    assert stats.first_null_radius_deg.mean == pytest.approx(0.02974, abs=1e-4)


def test_read_report_0001_gaussian_no_null() -> None:
    stats = read_report(REPORTS / "0001" / "report.json")  # file form
    assert stats.mode == "gaussian"
    assert stats.n_elements == 86
    assert stats.peak_dbi.mean == pytest.approx(28.33369, abs=1e-4)
    assert stats.first_null_radius_deg is None  # gaussian set has no nulls


def test_read_report_rejects_empty_elements(tmp_path: Path) -> None:
    p = tmp_path / "report.json"
    p.write_text(
        json.dumps({"elements": [], "estimated_params": {"mode": "gaussian", "spacing_deg": 1.0}})
    )
    with pytest.raises(ValueError):
        read_report(p)


def test_read_report_rejects_bad_mode(tmp_path: Path) -> None:
    el = {"peak_dbi": 1.0, "lobe_width_major_deg": 1.0, "lobe_width_minor_deg": 1.0,
          "lobe_orientation_deg": 0.0, "phase_slope_est_rad_per_deg": 1.0, "truncated": False}
    p = tmp_path / "report.json"
    p.write_text(json.dumps({"n_elements": 1, "elements": [el],
                             "estimated_params": {"mode": "weird", "spacing_deg": 1.0}}))
    with pytest.raises(ValueError):
        read_report(p)


def test_read_report_rejects_bad_spacing(tmp_path: Path) -> None:
    el = {"peak_dbi": 1.0, "lobe_width_major_deg": 1.0, "lobe_width_minor_deg": 1.0,
          "lobe_orientation_deg": 0.0, "phase_slope_est_rad_per_deg": 1.0, "truncated": False}
    p = tmp_path / "report.json"
    p.write_text(json.dumps({"n_elements": 1, "elements": [el],
                             "estimated_params": {"mode": "gaussian", "spacing_deg": 0.0}}))
    with pytest.raises(ValueError):
        read_report(p)


def test_read_report_rejects_missing_required_field(tmp_path: Path) -> None:
    el = {"peak_dbi": 1.0, "truncated": False}  # missing widths/orientation/phase
    p = tmp_path / "report.json"
    p.write_text(json.dumps({"n_elements": 1, "elements": [el],
                             "estimated_params": {"mode": "gaussian", "spacing_deg": 1.0}}))
    with pytest.raises(ValueError):
        read_report(p)


def test_read_report_rejects_all_truncated(tmp_path: Path) -> None:
    el = {"peak_dbi": 1.0, "lobe_width_major_deg": 1.0, "lobe_width_minor_deg": 1.0,
          "lobe_orientation_deg": 0.0, "phase_slope_est_rad_per_deg": 1.0, "truncated": True}
    p = tmp_path / "report.json"
    p.write_text(json.dumps({"n_elements": 1, "elements": [el],
                             "estimated_params": {"mode": "gaussian", "spacing_deg": 1.0}}))
    with pytest.raises(ValueError):
        read_report(p)
