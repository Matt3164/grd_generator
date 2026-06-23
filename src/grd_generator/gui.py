"""Application interactive (PyQt6) de calibration & visualisation.

`build_result` (pur, sans Qt) construit un `Scenario` disque depuis les paramètres
de l'interface, calibre via la zone et synthétise les champs. Le widget Qt
(`PatternStudio`, ajouté ensuite) pilote cette fonction et le rendu `plot.py`.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from grd_generator.calibrate import (
    DEFAULT_COVERAGE_MARGIN_DB,
    DEFAULT_N_ELEMENTS,
    calibrate,
    write_calibrated,
)
from grd_generator.geometry import earth_limb_angle_deg
from grd_generator.logger import configure_logging, logger
from grd_generator.plot import (
    _draw_earth_envelope,
    _draw_phase_map,
    _draw_uv_map,
    draw_earth_envelope_contours,
    draw_earth_pattern_footprints,
    draw_zone_and_antenna,
    envelope_max_dbi,
)
from grd_generator.report_ingest import read_report
from grd_generator.scenario import Scenario, sat_frame_geometry
from grd_generator.schemas import EllipticalSpec, UVGrid
from grd_generator.synth import combined_max_directivity_dbi, elliptical_field

REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "reference_reports"


@dataclass(frozen=True)
class CalibrationResult:
    """Résultat d'une calibration, prêt pour le rendu."""

    grid: UVGrid
    fields: NDArray[np.complex128]
    centers_uv: NDArray[np.float64]
    peaks_dbi: NDArray[np.float64]
    mode: str
    specs: list[EllipticalSpec]
    sat_lon: float
    zone_center: tuple[float, float]
    zone_radius_deg: float
    antenna_latlon: tuple[float, float]


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
    geo = sat_frame_geometry(scenario, n_u=n_u, n_v=n_v)
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
        grid=grid,
        fields=fields,
        centers_uv=centers,
        peaks_dbi=peaks,
        mode=stats.mode,
        specs=specs,
        sat_lon=sat_lon,
        zone_center=geo.zone_center,
        zone_radius_deg=geo.zone_radius_deg,
        antenna_latlon=(antenna_lat, antenna_lon),
    )


class PatternStudio(QMainWindow):  # type: ignore[misc]
    """Studio interactif : disque + rapport -> calibration -> 5 vues, export .npz."""

    def __init__(
        self, *, reports_dir: Path = REPORTS_DIR, n_u: int = 161, n_v: int = 161
    ) -> None:
        super().__init__()
        self.setWindowTitle("grd_generator — Pattern Studio")
        self._n_u = n_u
        self._n_v = n_v
        self._result: CalibrationResult | None = None
        self._axes: Any = None  # tableau d'axes matplotlib, peuplé par _redraw_all
        self._cb_dir: Any = None  # colorbar directivité (retirée au redraw)
        self._cb_phase: Any = None  # colorbar phase (retirée au redraw)

        self._lat = self._dspin(-90.0, 90.0, 46.6, 0.1)
        self._lon = self._dspin(-180.0, 180.0, 2.5, 0.1)
        self._sat_lon = self._dspin(-180.0, 180.0, 3.0, 0.1)
        # bornes physiques : rayon > 0 et ≤ limbe − marge (sinon hors du disque terrestre)
        zone_max = round(earth_limb_angle_deg() - 0.2, 2)
        self._zone = self._dspin(0.5, zone_max, 6.0, 0.5)
        self._report = QComboBox()
        for d in sorted(p.name for p in reports_dir.iterdir() if (p / "report.json").exists()):
            self._report.addItem(d)
        self._reports_dir = reports_dir
        self._n_elements = self._ispin(1, 400, DEFAULT_N_ELEMENTS)
        self._seed = self._ispin(0, 10_000, 0)
        self._margin = self._dspin(0.0, 30.0, DEFAULT_COVERAGE_MARGIN_DB, 0.5)
        self._element = self._ispin(0, 0, 0)
        self._element.valueChanged.connect(self.set_element)

        gen_btn = QPushButton("Générer")
        gen_btn.clicked.connect(self._on_generate)
        self._export_btn = QPushButton("Export…")
        self._export_btn.clicked.connect(self._on_export)
        self._export_btn.setEnabled(False)

        disk = QGroupBox("Disque (zone)")
        df = QFormLayout(disk)
        df.addRow("Antenne lat (°)", self._lat)
        df.addRow("Antenne lon (°)", self._lon)
        df.addRow("Sat lon (°)", self._sat_lon)
        df.addRow("Rayon zone (°)", self._zone)
        sim = QGroupBox("Simulation")
        sf = QFormLayout(sim)
        sf.addRow("Rapport", self._report)
        sf.addRow("N éléments", self._n_elements)
        sf.addRow("Seed", self._seed)
        sf.addRow("Marge couverture (dB)", self._margin)
        disp = QGroupBox("Affichage")
        pf = QFormLayout(disp)
        pf.addRow("Pattern n°", self._element)

        controls = QVBoxLayout()
        for w in (disk, sim, disp, gen_btn, self._export_btn):
            controls.addWidget(w)
        controls.addStretch(1)
        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_widget.setMaximumWidth(320)

        self._figure = Figure(figsize=(14, 8))
        self._canvas = FigureCanvasQTAgg(self._figure)  # type: ignore[no-untyped-call]

        root = QHBoxLayout()
        root.addWidget(controls_widget)
        root.addWidget(self._canvas, stretch=1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self.generate()

    def _dspin(self, lo: float, hi: float, val: float, step: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        return s

    def _ispin(self, lo: int, hi: int, val: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    def generate(self) -> None:
        """Calibre depuis les contrôles et redessine les 5 vues."""
        try:
            result = build_result(
                antenna_lat=self._lat.value(),
                antenna_lon=self._lon.value(),
                sat_lon=self._sat_lon.value(),
                zone_radius_deg=self._zone.value(),
                report_path=self._reports_dir / self._report.currentText(),
                n_elements=self._n_elements.value(),
                seed=self._seed.value(),
                coverage_margin_db=self._margin.value(),
                n_u=self._n_u,
                n_v=self._n_v,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Génération impossible", str(exc))
            return
        self._result = result
        self._element.blockSignals(True)
        self._element.setRange(0, len(result.specs) - 1)
        if self._element.value() > len(result.specs) - 1:
            self._element.setValue(0)
        self._element.blockSignals(False)
        self._export_btn.setEnabled(True)
        self._redraw_all()
        logger.info("gui_generated", n=len(result.specs), mode=result.mode)

    def set_element(self, idx: int) -> None:
        """Change le pattern affiché ; ne redessine que directivité + phase."""
        if self._result is None:
            return
        self._draw_element(idx)
        self._canvas.draw_idle()  # type: ignore[no-untyped-call]

    def export_to(self, path: str | Path) -> None:
        """Écrit le .npz du résultat courant."""
        if self._result is None:
            return
        write_calibrated(path, self._result.grid, self._result.specs, mode=self._result.mode)
        logger.info("gui_exported", file=str(path))

    def _redraw_all(self) -> None:
        assert self._result is not None
        r = self._result
        self._figure.clear()
        self._axes = self._figure.subplots(2, 3).ravel()
        self._cb_dir = None
        self._cb_phase = None
        # Vues « Terre » = repère angulaire (u, v) à la longitude sat (vue satellite).
        earth_kw: dict[str, float] = {"sat_lon": r.sat_lon}
        overlay_kw: dict[str, float] = {
            "zone_center": r.zone_center,  # type: ignore[dict-item]
            "zone_radius": r.zone_radius_deg,
            "antenna_lat": r.antenna_latlon[0],
            "antenna_lon": r.antenna_latlon[1],
        }
        self._draw_element(self._element.value(), clear=False)
        draw_earth_pattern_footprints(
            self._axes[2], r.grid, r.fields, r.centers_uv, **earth_kw  # type: ignore[arg-type]
        )
        draw_zone_and_antenna(self._axes[2], **earth_kw, **overlay_kw)  # type: ignore[arg-type]
        env_coherent = combined_max_directivity_dbi(list(r.fields))
        _draw_earth_envelope(self._axes[3], r.grid, env_coherent, **earth_kw)  # type: ignore[arg-type]
        self._axes[3].set_title("Enveloppe — vue satellite (combinaison cohérente)")
        draw_zone_and_antenna(self._axes[3], **earth_kw, **overlay_kw)  # type: ignore[arg-type]
        env_max = envelope_max_dbi(r.fields)
        draw_earth_envelope_contours(
            self._axes[4], r.grid, env_max, step_db=1.0, **earth_kw  # type: ignore[arg-type]
        )
        draw_zone_and_antenna(self._axes[4], **earth_kw, **overlay_kw)  # type: ignore[arg-type]
        self._axes[5].axis("off")
        self._figure.tight_layout()
        self._canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _draw_element(self, idx: int, *, clear: bool = True) -> None:
        assert self._result is not None
        r = self._result
        field = r.fields[idx]
        dbi = 10.0 * np.log10(np.maximum(np.abs(field) ** 2, 1e-12))
        if clear:
            # Retirer les colorbars précédentes (sinon elles s'accumulent au redraw).
            if self._cb_dir is not None:
                self._cb_dir.remove()
            if self._cb_phase is not None:
                self._cb_phase.remove()
            self._axes[0].clear()
            self._axes[1].clear()
        self._cb_dir = _draw_uv_map(self._axes[0], r.grid, dbi, f"Élément #{idx} — directivité")
        self._cb_phase = _draw_phase_map(self._axes[1], r.grid, field, f"Élément #{idx} — phase")

    def _on_generate(self) -> None:
        self.generate()

    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Exporter le .npz", "calibrated.npz", "*.npz")
        if path:
            self.export_to(path)


def main() -> None:  # pragma: no cover
    configure_logging()
    parser = argparse.ArgumentParser(description="Studio interactif de calibration de patterns.")
    parser.parse_args()
    app = QApplication.instance() or QApplication(sys.argv)
    studio = PatternStudio()
    studio.show()
    app.exec()
