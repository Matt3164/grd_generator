from pathlib import Path
from typing import TypedDict

import numpy as np

from grd_generator.calibrate import (
    DEFAULT_COVERAGE_MARGIN_DB,
    calibrate,
    generate_calibrated,
    validate_against,
    write_calibrated,
)
from grd_generator.report_ingest import read_report
from grd_generator.scenario import FRANCE, sat_frame_geometry
from grd_generator.schemas import EllipticalSpec
from grd_generator.synth import pattern_envelope_min_in_zone

REPORTS = Path(__file__).resolve().parents[3] / "data" / "reference_reports"

# Petites tailles pour des tests rapides (la bissection synthétise N champs par étape).
class _Fast(TypedDict):
    n_elements: int
    n_u: int
    n_v: int


_FAST: _Fast = {"n_elements": 24, "n_u": 61, "n_v": 61}


def test_calibrate_returns_n_specs_and_is_reproducible() -> None:
    stats = read_report(REPORTS / "0000")
    specs_a, grid_a = calibrate(stats, seed=7, **_FAST)
    specs_b, grid_b = calibrate(stats, seed=7, **_FAST)
    # le nombre vient du paramètre (zone), pas du rapport
    assert len(specs_a) == _FAST["n_elements"]
    assert all(isinstance(s, EllipticalSpec) for s in specs_a)
    # reproductible : même seed -> mêmes specs
    assert specs_a[0].peak_gain_dbi == specs_b[0].peak_gain_dbi
    assert specs_a[-1].sigma_major == specs_b[-1].sigma_major
    assert grid_a.n_u == 61 and grid_a.n_v == 61


def test_calibrate_grid_is_the_zone_grid() -> None:
    # la grille doit couvrir la zone FRANCE (≈ ±6.5° autour du centre de zone),
    # pas un patch ±0.2° comme l'ancienne version pilotée par le rapport.
    stats = read_report(REPORTS / "0000")
    _, grid = calibrate(stats, seed=0, **_FAST)
    assert (grid.u_max - grid.u_min) > 5.0
    assert (grid.v_max - grid.v_min) > 5.0


def test_calibrate_orientations_folded() -> None:
    stats = read_report(REPORTS / "0000")
    specs, _ = calibrate(stats, seed=1, **_FAST)
    assert all(-90.0 < s.orientation_deg <= 90.0 for s in specs)


def test_calibrate_covers_the_zone() -> None:
    # σ est optimisé pour que l'enveloppe max-sur-patterns ≥ crête_moy − marge.
    stats = read_report(REPORTS / "0001")
    specs, grid = calibrate(stats, seed=0, **_FAST)
    geo = sat_frame_geometry(FRANCE, n_u=_FAST["n_u"], n_v=_FAST["n_v"])
    cov = pattern_envelope_min_in_zone(
        specs, grid, geo.zone_center, geo.zone_radius_deg, mode=stats.mode
    )
    target = float(np.mean([s.peak_gain_dbi for s in specs])) - DEFAULT_COVERAGE_MARGIN_DB
    # couverture atteinte à la tolérance de la bissection près
    assert cov >= target - 1.0


def test_write_calibrated_roundtrip(tmp_path: Path) -> None:
    stats = read_report(REPORTS / "0001")
    specs, grid = calibrate(stats, seed=0, **_FAST)
    out = write_calibrated(tmp_path / "cal.npz", grid, specs, mode=stats.mode)
    data = np.load(out, allow_pickle=False)
    n = _FAST["n_elements"]
    assert data["fields"].shape == (n, grid.n_v, grid.n_u)
    assert data["fields"].dtype == np.complex128
    assert data["sigmas_major"].shape == (n,)
    assert data["sigmas_minor"].shape == (n,)
    assert data["orientations_deg"].shape == (n,)
    assert str(data["mode"]) == "gaussian"


def test_validate_against_flags_off_specs() -> None:
    stats = read_report(REPORTS / "0001")
    specs, _ = calibrate(stats, seed=0, **_FAST)
    # specs conformes -> pas d'écart sur le caractère (crête, phase, ellipticité)
    assert validate_against(stats, specs) == []
    # crêtes très décalées -> écart signalé
    bad = [s.model_copy(update={"peak_gain_dbi": s.peak_gain_dbi + 50.0}) for s in specs]
    devs = validate_against(stats, bad)
    assert any(d["field"] == "peak_dbi" for d in devs)


def test_validate_against_flags_circular_ellipticity() -> None:
    # 0001 a des lobes elliptiques (ellipticité mesurée ~1.7) ; forcer le circulaire
    # (sigma_major == sigma_minor -> ellipticité 1.0) doit être signalé.
    stats = read_report(REPORTS / "0001")
    specs, _ = calibrate(stats, seed=0, **_FAST)
    circular = [s.model_copy(update={"sigma_minor": s.sigma_major}) for s in specs]
    devs = validate_against(stats, circular)
    assert any(d["field"] == "ellipticity" for d in devs)


def test_generate_calibrated_end_to_end(tmp_path: Path) -> None:
    out = generate_calibrated(REPORTS / "0000", tmp_path / "cal0000.npz", seed=0, **_FAST)
    assert out.exists()
    data = np.load(out, allow_pickle=False)
    assert data["fields"].shape[0] == _FAST["n_elements"]
    assert str(data["mode"]) == "airy"
