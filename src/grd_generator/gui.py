"""Application interactive (PyQt6) de calibration & visualisation.

`build_result` (pur, sans Qt) construit un `Scenario` disque depuis les paramètres
de l'interface, calibre via la zone et synthétise les champs. Le widget Qt
(`PatternStudio`, ajouté ensuite) pilote cette fonction et le rendu `plot.py`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from grd_generator.calibrate import calibrate
from grd_generator.report_ingest import read_report
from grd_generator.scenario import Scenario
from grd_generator.schemas import EllipticalSpec, UVGrid
from grd_generator.synth import elliptical_field


@dataclass(frozen=True)
class CalibrationResult:
    """Résultat d'une calibration, prêt pour le rendu."""

    grid: UVGrid
    fields: NDArray[np.complex128]
    centers_uv: NDArray[np.float64]
    peaks_dbi: NDArray[np.float64]
    mode: str
    specs: list[EllipticalSpec]


def build_result(
    *,
    antenna_lat: float,
    antenna_lon: float,
    sat_lon: float,
    zone_radius_deg: float,
    report_path: str | Path,
    n_elements: int,
    seed: int,
    coverage_margin_db: float,
    n_u: int = 161,
    n_v: int = 161,
) -> CalibrationResult:
    """Construit un `Scenario` disque depuis les paramètres et calibre l'array."""
    scenario = Scenario(
        name="custom",
        sat_lon_deg=sat_lon,
        antenna_latlon=(antenna_lat, antenna_lon),
        zone_radius_deg=zone_radius_deg,
    )
    stats = read_report(report_path)
    specs, grid = calibrate(
        stats,
        scenario=scenario,
        n_elements=n_elements,
        seed=seed,
        coverage_margin_db=coverage_margin_db,
        n_u=n_u,
        n_v=n_v,
    )
    fields = np.stack([elliptical_field(s, grid, mode=stats.mode) for s in specs])
    centers = np.array([s.center_uv for s in specs], dtype=float)
    peaks = np.array([s.peak_gain_dbi for s in specs], dtype=float)
    return CalibrationResult(
        grid=grid, fields=fields, centers_uv=centers, peaks_dbi=peaks, mode=stats.mode, specs=specs
    )


def main() -> None:  # pragma: no cover
    """Point d'entrée CLI — le widget Qt sera ajouté dans une tâche ultérieure."""
    raise NotImplementedError("GUI widget not yet implemented")
