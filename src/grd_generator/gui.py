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
    draw_service_zone_uv,
    draw_zone_and_antenna,
    envelope_max_dbi,
)
from grd_generator.reflector import (
    FeedSpec,
    ReflectorSpec,
    ServiceZone,
    hex_feed_positions,
    synthesize_reflector_fields,
)
from grd_generator.report_ingest import read_report
from grd_generator.scenario import Scenario, sat_frame_geometry
from grd_generator.schemas import ComplexField, EllipticalSpec, UVGrid
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


@dataclass(frozen=True)
class ReflectorResult:
    """Résultat AFR (pur, sans Qt) : champs co/cross par feed + zone + grille."""

    co_fields: list[ComplexField]
    cross_fields: list[ComplexField]
    zone: ServiceZone
    grid: UVGrid


def build_reflector_result(
    *,
    diameter_m: float,
    f_over_d: float,
    offset_clearance_m: float,
    freq_ghz: float,
    q: float,
    pitch_m: float,
    n_feeds: int,
    zone_radius_deg: float,
    grid: UVGrid,
) -> ReflectorResult:
    """Construit le résultat AFR (pur, sans Qt) depuis les paramètres de l'IHM."""
    spec = ReflectorSpec(
        diameter_m=diameter_m,
        focal_length_m=f_over_d * diameter_m,
        offset_clearance_m=offset_clearance_m,
        freq_hz=freq_ghz * 1e9,
    )
    feeds = FeedSpec(
        positions_m=hex_feed_positions(pitch_m, n_feeds, focal_radius_m=10.0 * pitch_m),
        q=q,
    )
    co, cross = synthesize_reflector_fields(spec, feeds, grid)
    return ReflectorResult(
        co_fields=co,
        cross_fields=cross,
        zone=ServiceZone(radius_deg=zone_radius_deg),
        grid=grid,
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
        # Projection sur Terre : repère NADIR (la grille est en repère nadir),
        # borésight = (0, sat_lon) — pas le pointage antenne.
        earth_kw: dict[str, float] = {"sat_lon": r.sat_lon, "b_lat": 0.0, "b_lon": r.sat_lon}
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
        self._axes[3].set_title("Enveloppe Terre — combinaison cohérente")
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


class ReflectorStudio(QMainWindow):  # type: ignore[misc]
    """Studio interactif AFR : paramètres réflecteur → champs co/cross + zone."""

    def __init__(self, *, n_u: int = 81, n_v: int = 81) -> None:
        super().__init__()
        self.setWindowTitle("grd_generator — Reflector Studio (AFR)")
        self._n_u = n_u
        self._n_v = n_v
        self._afr_result: ReflectorResult | None = None
        self._axes: Any = None
        self._cb_dir: Any = None
        self._cb_phase: Any = None
        self._cb_env: Any = None
        self._cb_env_max: Any = None

        # Paramètres réflecteur
        self._diameter = self._dspin(0.1, 10.0, 2.0, 0.1)
        self._f_over_d = self._dspin(0.5, 4.0, 1.2, 0.1)
        self._offset = self._dspin(0.0, 5.0, 0.0, 0.05)
        self._freq_ghz = self._dspin(1.0, 100.0, 20.0, 1.0)
        self._q = self._dspin(0.5, 20.0, 2.0, 0.5)
        self._pitch = self._dspin(0.005, 0.5, 0.03, 0.005)
        self._n_feeds = self._ispin(1, 127, 7)
        # bornes zone : [6, 14]° conforme à ServiceZone
        self._zone_radius = self._dspin(6.0, 14.0, 8.0, 0.5)
        self._feed_idx = self._ispin(0, 0, 0)
        self._feed_idx.valueChanged.connect(self._on_feed_changed)

        gen_btn = QPushButton("Générer")
        gen_btn.clicked.connect(self._on_generate)

        refl = QGroupBox("Réflecteur")
        rf = QFormLayout(refl)
        rf.addRow("Diamètre (m)", self._diameter)
        rf.addRow("F/D", self._f_over_d)
        rf.addRow("Offset clearance (m)", self._offset)
        rf.addRow("Fréquence (GHz)", self._freq_ghz)
        feed_grp = QGroupBox("Réseau de feeds")
        ff = QFormLayout(feed_grp)
        ff.addRow("Facteur q (cos^q)", self._q)
        ff.addRow("Pas pitch (m)", self._pitch)
        ff.addRow("N feeds", self._n_feeds)
        zone_grp = QGroupBox("Zone de service")
        zf = QFormLayout(zone_grp)
        zf.addRow("Rayon zone (°)", self._zone_radius)
        disp = QGroupBox("Affichage")
        df = QFormLayout(disp)
        df.addRow("Feed n°", self._feed_idx)

        controls = QVBoxLayout()
        for w in (refl, feed_grp, zone_grp, disp, gen_btn):
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
        """Construit le résultat AFR depuis les contrôles et redessine les vues."""
        grid = UVGrid(
            u_min=-14.0, u_max=14.0,
            v_min=-14.0, v_max=14.0,
            n_u=self._n_u, n_v=self._n_v,
        )
        try:
            result = build_reflector_result(
                diameter_m=self._diameter.value(),
                f_over_d=self._f_over_d.value(),
                offset_clearance_m=self._offset.value(),
                freq_ghz=self._freq_ghz.value(),
                q=self._q.value(),
                pitch_m=self._pitch.value(),
                n_feeds=self._n_feeds.value(),
                zone_radius_deg=self._zone_radius.value(),
                grid=grid,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Génération impossible", str(exc))
            return
        self._afr_result = result
        n = len(result.co_fields)
        self._feed_idx.blockSignals(True)
        self._feed_idx.setRange(0, n - 1)
        if self._feed_idx.value() > n - 1:
            self._feed_idx.setValue(0)
        self._feed_idx.blockSignals(False)
        self._redraw_all()
        logger.info("afr_generated", n_feeds=n)

    def _redraw_all(self) -> None:
        assert self._afr_result is not None
        r = self._afr_result
        self._figure.clear()
        self._axes = self._figure.subplots(2, 3).ravel()
        self._cb_dir = None
        self._cb_phase = None
        self._draw_feed(self._feed_idx.value(), clear=False)
        # Enveloppe co-pol (combinaison cohérente) sur les axes (u,v)
        env = combined_max_directivity_dbi(list(r.co_fields))
        self._cb_env = _draw_uv_map(self._axes[2], r.grid, env, "Enveloppe co-pol")
        draw_service_zone_uv(self._axes[2], r.zone)
        # Enveloppe simple max sur les axes (u,v)
        env_max = envelope_max_dbi(np.stack(r.co_fields))
        self._cb_env_max = _draw_uv_map(self._axes[3], r.grid, env_max, "Enveloppe max co-pol")
        draw_service_zone_uv(self._axes[3], r.zone)
        self._axes[4].axis("off")
        self._axes[5].axis("off")
        self._figure.tight_layout()
        self._canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _draw_feed(self, idx: int, *, clear: bool = True) -> None:
        assert self._afr_result is not None
        r = self._afr_result
        field = r.co_fields[idx]
        dbi = 10.0 * np.log10(np.maximum(np.abs(field) ** 2, 1e-12))
        if clear:
            if self._cb_dir is not None:
                self._cb_dir.remove()
            if self._cb_phase is not None:
                self._cb_phase.remove()
            self._axes[0].clear()
            self._axes[1].clear()
        self._cb_dir = _draw_uv_map(self._axes[0], r.grid, dbi, f"Feed #{idx} — co-pol dBi")
        draw_service_zone_uv(self._axes[0], r.zone)
        self._cb_phase = _draw_phase_map(self._axes[1], r.grid, field, f"Feed #{idx} — phase")
        draw_service_zone_uv(self._axes[1], r.zone)

    def _on_feed_changed(self, idx: int) -> None:
        if self._afr_result is None:
            return
        self._draw_feed(idx)
        self._canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _on_generate(self) -> None:
        self.generate()


def main() -> None:  # pragma: no cover
    configure_logging()
    parser = argparse.ArgumentParser(description="Studio interactif de calibration de patterns.")
    parser.parse_args()
    app = QApplication.instance() or QApplication(sys.argv)
    studio = PatternStudio()
    studio.show()
    app.exec()
