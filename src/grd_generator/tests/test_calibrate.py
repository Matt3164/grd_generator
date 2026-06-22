from pathlib import Path

import numpy as np

from grd_generator.calibrate import (
    calibrate,
    generate_calibrated,
    validate_against,
    write_calibrated,
)
from grd_generator.report_ingest import read_report
from grd_generator.schemas import EllipticalSpec

REPORTS = Path(__file__).resolve().parents[3] / "data" / "reference_reports"


def test_calibrate_returns_n_specs_and_is_reproducible() -> None:
    stats = read_report(REPORTS / "0000")
    specs_a, grid_a = calibrate(stats, seed=7)
    specs_b, grid_b = calibrate(stats, seed=7)
    assert len(specs_a) == stats.n_elements
    assert all(isinstance(s, EllipticalSpec) for s in specs_a)
    # reproducible: same seed -> identical first/last spec params
    assert specs_a[0].peak_gain_dbi == specs_b[0].peak_gain_dbi
    assert specs_a[-1].sigma_major == specs_b[-1].sigma_major
    assert grid_a.n_u == 161 and grid_a.n_v == 161


def test_calibrate_orientations_folded() -> None:
    stats = read_report(REPORTS / "0000")
    specs, _ = calibrate(stats, seed=1)
    assert all(-90.0 < s.orientation_deg <= 90.0 for s in specs)


def test_write_calibrated_roundtrip(tmp_path: Path) -> None:
    stats = read_report(REPORTS / "0001")
    specs, grid = calibrate(stats, seed=0)
    out = write_calibrated(tmp_path / "cal.npz", grid, specs, mode=stats.mode)
    data = np.load(out, allow_pickle=False)
    n = stats.n_elements
    assert data["fields"].shape == (n, grid.n_v, grid.n_u)
    assert data["fields"].dtype == np.complex128
    assert data["sigmas_major"].shape == (n,)
    assert data["sigmas_minor"].shape == (n,)
    assert data["orientations_deg"].shape == (n,)
    assert str(data["mode"]) == "gaussian"


def test_validate_against_flags_off_specs() -> None:
    stats = read_report(REPORTS / "0001")
    specs, _ = calibrate(stats, seed=0)
    # conforming specs -> no deviation
    assert validate_against(stats, specs) == []
    # corrupt the peaks far out of tolerance -> deviation reported
    bad = [s.model_copy(update={"peak_gain_dbi": s.peak_gain_dbi + 50.0}) for s in specs]
    devs = validate_against(stats, bad)
    assert any(d["field"] == "peak_dbi" for d in devs)


def test_validate_against_flags_circular_ellipticity() -> None:
    # 0001 has elliptical lobes (measured ellipticity ~1.7); forcing the specs
    # circular (sigma_major == sigma_minor -> ellipticity 1.0) must be flagged.
    stats = read_report(REPORTS / "0001")
    specs, _ = calibrate(stats, seed=0)
    circular = [s.model_copy(update={"sigma_minor": s.sigma_major}) for s in specs]
    devs = validate_against(stats, circular)
    assert any(d["field"] == "ellipticity" for d in devs)


def test_generate_calibrated_end_to_end(tmp_path: Path) -> None:
    out = generate_calibrated(REPORTS / "0000", tmp_path / "cal0000.npz", seed=0)
    assert out.exists()
    data = np.load(out, allow_pickle=False)
    assert data["fields"].shape[0] == 82
    assert str(data["mode"]) == "airy"
