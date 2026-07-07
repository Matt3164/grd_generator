"""Application interactive (PyQt6) de calibration & visualisation.

`build_result` (pur, sans Qt) construit un `Scenario` disque depuis les paramètres
de l'interface, calibre via la zone et synthétise les champs. Le widget Qt
(`PatternStudio`, ajouté ensuite) pilote cette fonction et le rendu `plot.py`.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PyQt6.QtCore import Qt
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
    QTabWidget,
    QTextBrowser,
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
    dereference_phase,
    form_beam,
    hex_feed_positions,
    synthesize_reflector_fields,
)
from grd_generator.reflector.grd_export import (
    patterns_to_zip_bytes,
    simulation_params_dict,
)
from grd_generator.report_ingest import read_report
from grd_generator.scenario import Scenario, sat_frame_geometry
from grd_generator.schemas import ComplexField, EllipticalSpec, UVGrid
from grd_generator.synth import combined_max_directivity_dbi, elliptical_field

REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "reference_reports"

# Panneau "Beam formé — zoom" (ReflectorStudio) : fenêtre ±_ZOOM_HALF_WIDTH_DEG°
# centrée sur la cible cliquée, ré-échantillonnée à _ZOOM_N×_ZOOM_N (la crête
# du beam pleine ouverture, ~λ/D_physique, est sous-résolue sur la grille
# large ±14°/81px — voir docstring de `ReflectorStudio._draw_beam`).
_ZOOM_HALF_WIDTH_DEG = 2.0
_ZOOM_N = 81


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
    """Résultat AFR (pur, sans Qt) : champs co/cross par feed + zone + grille.

    `wavelength_m`/`aperture_center_y_m` sont dérivés du `ReflectorSpec` interne :
    exposés ici pour l'affichage (dé-référencement de phase), pas pour les champs
    eux-mêmes.
    """

    co_fields: list[ComplexField]
    cross_fields: list[ComplexField]
    zone: ServiceZone
    grid: UVGrid
    wavelength_m: float
    aperture_center_y_m: float


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
    defocus_m: float = 0.0,
    phase_error_rms_rad: float = 0.0,
    phase_corr_length_m: float = 0.05,
    phase_error_seed: int = 0,
    phase_error_shared_rms_rad: float = 0.0,
    footprint_m: float = 0.0,
    footprint_magnification: float = 0.0,
    n_aperture: int = 128,
    pad_factor: int = 4,
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
        defocus_m=defocus_m,
        phase_error_rms_rad=phase_error_rms_rad,
        phase_corr_length_m=phase_corr_length_m,
        phase_error_seed=phase_error_seed,
        phase_error_shared_rms_rad=phase_error_shared_rms_rad,
        footprint_m=footprint_m,
        footprint_magnification=footprint_magnification,
    )
    co, cross = synthesize_reflector_fields(
        spec, feeds, grid, n_aperture=n_aperture, pad_factor=pad_factor
    )
    return ReflectorResult(
        co_fields=co,
        cross_fields=cross,
        zone=ServiceZone(radius_deg=zone_radius_deg),
        grid=grid,
        wavelength_m=spec.wavelength_m,
        aperture_center_y_m=spec.aperture_center_y_m,
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

    def _dspin(
        self, lo: float, hi: float, val: float, step: float, decimals: int = 2
    ) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setDecimals(decimals)
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
        self._cb_beam_dir: Any = None
        self._cb_beam_zoom: Any = None
        self._target_uv: tuple[float, float] = (0.0, 0.0)  # cible du beam formé (deg)
        # Cache du résultat AFR zoomé (fenêtre ±2° sur la cible) : ré-synthétiser
        # coûte plusieurs secondes (selon la résolution ouverture n choisie), donc
        # uniquement si (cible, paramètres réflecteur/feeds, n) ont changé depuis
        # le dernier calcul.
        self._zoom_cache_key: tuple[Any, ...] | None = None
        self._zoom_cache_result: ReflectorResult | None = None
        # La re-synthèse du zoom (panneau 5) est paresseuse : elle double le coût
        # d'un generate() (synthèse à n_aperture sur une 2e grille), donc on la
        # saute tant que l'utilisateur n'a pas cliqué de cible au moins une fois
        # (le generate() initial de fin de __init__ affiche un placeholder à la
        # place). Une fois activée (premier clic), elle reste active — un
        # generate() explicite (bouton) rafraîchit alors le zoom normalement.
        self._zoom_active = False

        # Paramètres réflecteur — diamètre physique réaliste par défaut (2.2 m) ;
        # avec le modèle deux échelles, l'ouverture effective par élément est
        # gouvernée par l'empreinte (voir groupe "Réseau de feeds"), pas par ce
        # diamètre.
        self._diameter = self._dspin(0.1, 25.0, 2.2, 0.1)
        self._f_over_d = self._dspin(0.5, 4.0, 1.2, 0.1)
        self._offset = self._dspin(0.0, 5.0, 0.15, 0.05)
        self._freq_ghz = self._dspin(1.0, 100.0, 20.0, 1.0)
        # Résolution de l'échantillonnage d'ouverture (n x n) : voir onglet Infos
        # pour le compromis dx = D/n vs corrélation/empreinte.
        self._n_aperture = self._ispin(64, 1024, 256)
        self._q = self._dspin(0.5, 20.0, 2.0, 0.5)
        self._pitch = self._dspin(0.002, 0.5, 0.025, 0.001, decimals=3)
        self._n_feeds = self._ispin(1, 127, 91)
        self._defocus = self._dspin(-5.0, 5.0, 0.0, 0.1)
        self._phase_rms = self._dspin(0.0, 3.0, 0.30, 0.05, decimals=2)
        self._phase_shared_rms = self._dspin(0.0, 3.0, 1.25, 0.05, decimals=2)
        self._phase_corr = self._dspin(0.005, 0.5, 0.015, 0.005, decimals=3)
        self._footprint = self._dspin(0.0, 5.0, 0.16, 0.01, decimals=3)
        self._footprint_mag = self._dspin(-200.0, 200.0, 6.0, 1.0, decimals=2)
        # bornes zone : [6, 14]° conforme à ServiceZone
        self._zone_radius = self._dspin(6.0, 14.0, 8.0, 0.5)
        self._feed_idx = self._ispin(0, 0, 0)
        self._feed_idx.valueChanged.connect(self._on_feed_changed)

        gen_btn = QPushButton("Générer")
        gen_btn.clicked.connect(self._on_generate)
        grd_btn = QPushButton("Export GRD (ZIP)…")
        grd_btn.clicked.connect(self._on_export_grd)
        params_btn = QPushButton("Export params (JSON)…")
        params_btn.clicked.connect(self._on_export_params)

        refl = QGroupBox("Réflecteur")
        rf = QFormLayout(refl)
        rf.addRow("Diamètre (m)", self._diameter)
        rf.addRow("F/D", self._f_over_d)
        rf.addRow("Offset clearance (m)", self._offset)
        rf.addRow("Fréquence (GHz)", self._freq_ghz)
        rf.addRow("Résolution ouverture (n)", self._n_aperture)
        feed_grp = QGroupBox("Réseau de feeds")
        ff = QFormLayout(feed_grp)
        ff.addRow("Facteur q (cos^q)", self._q)
        ff.addRow("Pas pitch (m)", self._pitch)
        ff.addRow("N feeds", self._n_feeds)
        ff.addRow("Defocus (m)", self._defocus)
        ff.addRow("σ par feed (rad)", self._phase_rms)
        ff.addRow("σ commun (rad)", self._phase_shared_rms)
        ff.addRow("Corrélation (m)", self._phase_corr)
        ff.addRow("Empreinte (m)", self._footprint)
        ff.addRow("Magnification", self._footprint_mag)
        zone_grp = QGroupBox("Zone de service")
        zf = QFormLayout(zone_grp)
        zf.addRow("Rayon zone (°)", self._zone_radius)
        disp = QGroupBox("Affichage")
        df = QFormLayout(disp)
        df.addRow("Feed n°", self._feed_idx)

        controls = QVBoxLayout()
        for w in (refl, feed_grp, zone_grp, disp, gen_btn, grd_btn, params_btn):
            controls.addWidget(w)
        controls.addStretch(1)
        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_widget.setMaximumWidth(320)

        self._figure = Figure(figsize=(14, 8))
        self._canvas = FigureCanvasQTAgg(self._figure)  # type: ignore[no-untyped-call]
        # Clic sur une carte (u,v) → recale la cible du beam formé.
        self._canvas.mpl_connect("button_press_event", self._on_click)

        studio_root = QHBoxLayout()
        studio_root.addWidget(controls_widget)
        studio_root.addWidget(self._canvas, stretch=1)
        studio_tab = QWidget()
        studio_tab.setLayout(studio_root)

        info_tab = QTextBrowser()
        info_tab.setHtml(_reflector_info_html())

        tabs = QTabWidget()
        tabs.addTab(studio_tab, "Studio")
        tabs.addTab(info_tab, "Infos")
        self.setCentralWidget(tabs)

        self.generate()

    def _dspin(
        self, lo: float, hi: float, val: float, step: float, decimals: int = 2
    ) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setDecimals(decimals)
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
        """Construit le résultat AFR depuis les contrôles et redessine les vues.

        Curseur d'attente pendant toute la méthode. Tant qu'aucune cible n'a été
        cliquée (`_zoom_active` False), la re-synthèse du zoom (`_redraw_all` →
        `_draw_beam`) est sautée : seule la synthèse principale coûte (~20s avec
        91 feeds à n_aperture=256 défaut). Après un premier clic, elle s'ajoute
        au coût de la génération et peut prendre plusieurs secondes de plus
        (selon la résolution ouverture n).
        """
        grid = UVGrid(
            u_min=-14.0, u_max=14.0,
            v_min=-14.0, v_max=14.0,
            n_u=self._n_u, n_v=self._n_v,
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
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
                    defocus_m=self._defocus.value(),
                    phase_error_rms_rad=self._phase_rms.value(),
                    phase_corr_length_m=self._phase_corr.value(),
                    phase_error_shared_rms_rad=self._phase_shared_rms.value(),
                    footprint_m=self._footprint.value(),
                    footprint_magnification=self._footprint_mag.value(),
                    n_aperture=self._n_aperture.value(),
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
        finally:
            QApplication.restoreOverrideCursor()

    def _redraw_all(self) -> None:
        assert self._afr_result is not None
        r = self._afr_result
        self._figure.clear()
        self._axes = self._figure.subplots(2, 3).ravel()
        self._cb_dir = None
        self._cb_phase = None
        self._cb_beam_dir = None
        self._cb_beam_zoom = None
        self._draw_feed(self._feed_idx.value(), clear=False)
        # Enveloppe MRC (combinaison cohérente, beamforming idéal) sur les axes (u,v)
        env = combined_max_directivity_dbi(list(r.co_fields))
        self._cb_env = _draw_uv_map(self._axes[2], r.grid, env, "Enveloppe MRC (beamforming idéal)")
        draw_service_zone_uv(self._axes[2], r.zone)
        # Enveloppe max par élément (festonnage réel, pas de recombinaison) sur les axes (u,v)
        env_max = envelope_max_dbi(np.stack(r.co_fields))
        self._cb_env_max = _draw_uv_map(self._axes[3], r.grid, env_max, "Enveloppe max par élément")
        draw_service_zone_uv(self._axes[3], r.zone)
        # Beam formé vers la cible (filtre adapté) : directivité + phase
        self._draw_beam(clear=False)
        self._figure.tight_layout()
        self._canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _current_reflector_params(self) -> tuple[float, ...]:
        """Paramètres réflecteur/feeds courants (hors cible) : clé de cache du zoom."""
        return (
            self._diameter.value(),
            self._f_over_d.value(),
            self._offset.value(),
            self._freq_ghz.value(),
            self._q.value(),
            self._pitch.value(),
            float(self._n_feeds.value()),
            self._defocus.value(),
            self._phase_rms.value(),
            self._phase_shared_rms.value(),
            self._phase_corr.value(),
            self._footprint.value(),
            self._footprint_mag.value(),
            self._zone_radius.value(),
            float(self._n_aperture.value()),
        )

    def _zoomed_beam_result(self) -> ReflectorResult:
        """Ré-synthétise les champs sur une fenêtre ±2° centrée sur la cible.

        Mémorise le dernier résultat par (cible, paramètres, n) : ne recalcule que si
        l'un des trois a changé depuis le dernier appel (coût dépendant de n).
        """
        key = (self._target_uv, self._current_reflector_params())
        if key == self._zoom_cache_key and self._zoom_cache_result is not None:
            return self._zoom_cache_result
        tu, tv = self._target_uv
        zoom_grid = UVGrid(
            u_min=tu - _ZOOM_HALF_WIDTH_DEG, u_max=tu + _ZOOM_HALF_WIDTH_DEG,
            v_min=tv - _ZOOM_HALF_WIDTH_DEG, v_max=tv + _ZOOM_HALF_WIDTH_DEG,
            n_u=_ZOOM_N, n_v=_ZOOM_N,
        )
        result = build_reflector_result(
            diameter_m=self._diameter.value(),
            f_over_d=self._f_over_d.value(),
            offset_clearance_m=self._offset.value(),
            freq_ghz=self._freq_ghz.value(),
            q=self._q.value(),
            pitch_m=self._pitch.value(),
            n_feeds=self._n_feeds.value(),
            zone_radius_deg=self._zone_radius.value(),
            grid=zoom_grid,
            defocus_m=self._defocus.value(),
            phase_error_rms_rad=self._phase_rms.value(),
            phase_corr_length_m=self._phase_corr.value(),
            phase_error_shared_rms_rad=self._phase_shared_rms.value(),
            footprint_m=self._footprint.value(),
            footprint_magnification=self._footprint_mag.value(),
            n_aperture=self._n_aperture.value(),
        )
        self._zoom_cache_key = key
        self._zoom_cache_result = result
        return result

    def _draw_beam(self, *, clear: bool = True) -> None:
        """Beam formé par filtre adapté vers `self._target_uv` (axe 4) + zoom (axe 5).

        Le beam pleine ouverture (axe 4) vient des champs déjà synthétisés : coût
        négligeable, toujours calculé. Le beam pleine ouverture a une crête
        ~λ/D_physique de large : sur la grille large ±14°/81px elle ne fait qu'un
        pixel et n'est pas visible (seul le plancher de speckle l'est). Le
        panneau 5 ré-échantillonne donc les champs sur une fenêtre ±2° centrée
        sur la cible pour la résoudre — mais cette re-synthèse (coût ~ celui du
        generate() principal) est paresseuse : tant qu'aucun clic n'a fixé de
        cible (`self._zoom_active` toujours False), le panneau affiche un
        placeholder au lieu de payer le calcul.
        """
        assert self._afr_result is not None
        r = self._afr_result
        beam = form_beam(list(r.co_fields), r.grid, self._target_uv)
        dbi = 10.0 * np.log10(np.maximum(np.abs(beam) ** 2, 1e-12))
        if clear:
            if self._cb_beam_dir is not None:
                self._cb_beam_dir.remove()
            if self._cb_beam_zoom is not None:
                self._cb_beam_zoom.remove()
            self._axes[4].clear()
            self._axes[5].clear()
        tu, tv = self._target_uv
        peak_dbi = float(dbi.max())
        title = f"Beam formé ({tu:.1f}, {tv:.1f})° — crête {peak_dbi:.1f} dBi"
        self._cb_beam_dir = _draw_uv_map(self._axes[4], r.grid, dbi, title)
        draw_service_zone_uv(self._axes[4], r.zone)
        self._axes[4].plot(tu, tv, "rx", markersize=10, markeredgewidth=2)

        if not self._zoom_active:
            self._axes[5].set_axis_off()
            self._axes[5].text(
                0.5, 0.5, "Cliquez une cible sur une carte\npour calculer le zoom",
                ha="center", va="center", transform=self._axes[5].transAxes,
            )
            return

        zoom = self._zoomed_beam_result()
        zoom_beam = form_beam(list(zoom.co_fields), zoom.grid, self._target_uv)
        zoom_dbi = 10.0 * np.log10(np.maximum(np.abs(zoom_beam) ** 2, 1e-12))
        zoom_title = (
            f"Beam formé — zoom ±{_ZOOM_HALF_WIDTH_DEG:.0f}°"
            f" (crête {float(zoom_dbi.max()):.1f} dBi)"
        )
        self._cb_beam_zoom = _draw_uv_map(self._axes[5], zoom.grid, zoom_dbi, zoom_title)
        draw_service_zone_uv(self._axes[5], zoom.zone)
        self._axes[5].plot(tu, tv, "rx", markersize=10, markeredgewidth=2)
        # Le cercle de zone peut être bien plus large que la fenêtre de zoom :
        # borner explicitement la vue pour ne pas laisser matplotlib ré-élargir
        # les axes afin de l'inclure en entier.
        self._axes[5].set_xlim(zoom.grid.u_min, zoom.grid.u_max)
        self._axes[5].set_ylim(zoom.grid.v_min, zoom.grid.v_max)

    def _on_click(self, event: Any) -> None:
        """Clic sur une carte (u,v) → recale la cible et redessine le beam formé.

        Active la re-synthèse du zoom (panneau 5) si ce n'est pas déjà fait —
        voir `_zoom_active` — puis curseur d'attente pendant la ré-synthèse
        (voir `_zoomed_beam_result`).
        """
        if self._afr_result is None or event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._target_uv = (float(event.xdata), float(event.ydata))
        self._zoom_active = True
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._draw_beam()
        finally:
            QApplication.restoreOverrideCursor()
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
        # Dé-référencement d'affichage uniquement (voir docstring `dereference_phase`) :
        # retire la porteuse linéaire due au centre d'ouverture décalé, qui sinon se
        # replie en rayures sur cette grille grossière. `field` (utilisé pour le
        # beamforming/exports) n'est pas modifié.
        dephased_angle = dereference_phase(field, r.grid, r.wavelength_m, r.aperture_center_y_m)
        dephased_field = np.exp(1j * dephased_angle)
        self._cb_phase = _draw_phase_map(
            self._axes[1], r.grid, dephased_field, f"Feed #{idx} — phase (dé-référencée)"
        )
        draw_service_zone_uv(self._axes[1], r.zone)

    def _on_feed_changed(self, idx: int) -> None:
        if self._afr_result is None:
            return
        self._draw_feed(idx)
        self._canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _on_generate(self) -> None:
        self.generate()

    def _sim_params(self) -> dict[str, Any]:
        """Paramètres de simulation courants (valeurs des contrôles) pour l'export."""
        return simulation_params_dict(
            diameter_m=self._diameter.value(),
            f_over_d=self._f_over_d.value(),
            offset_clearance_m=self._offset.value(),
            freq_ghz=self._freq_ghz.value(),
            q=self._q.value(),
            pitch_m=self._pitch.value(),
            n_feeds=self._n_feeds.value(),
            zone_radius_deg=self._zone_radius.value(),
            defocus_m=self._defocus.value(),
            phase_error_rms_rad=self._phase_rms.value(),
            phase_corr_length_m=self._phase_corr.value(),
            phase_error_shared_rms_rad=self._phase_shared_rms.value(),
            footprint_m=self._footprint.value(),
            footprint_magnification=self._footprint_mag.value(),
        )

    def _on_export_grd(self) -> None:
        """Exporte tous les patterns (co + cross) en `.grd` TICRA, un par feed, zippés."""
        if self._afr_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter les patterns GRD (ZIP)", "patterns_grd.zip", "*.zip"
        )
        if not path:
            return
        r = self._afr_result
        blob = patterns_to_zip_bytes(
            list(r.co_fields),
            list(r.cross_fields),
            r.grid,
            freq_hz=self._freq_ghz.value() * 1e9,
            prefix="feed",
        )
        Path(path).write_bytes(blob)
        logger.info("afr_grd_exported", file=path, n_feeds=len(r.co_fields))

    def _on_export_params(self) -> None:
        """Exporte uniquement les paramètres de simulation au format JSON."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter les paramètres (JSON)", "sim_params.json", "*.json"
        )
        if not path:
            return
        Path(path).write_text(json.dumps(self._sim_params(), indent=2), encoding="utf-8")
        logger.info("afr_params_exported", file=path)


def _reflector_info_html() -> str:  # pragma: no cover
    """Contenu HTML de l'onglet Infos : définitions des paramètres + modèle de simulation."""
    return """
<h2>Reflector Studio (AFR) — aide</h2>
<p>Modèle <b>array-fed reflector</b> : un réflecteur parabolique offset illuminé
par un réseau de feeds dans le plan focal. Chaque feed produit un faisceau
secondaire ; les cartes montrent chaque faisceau, leurs enveloppes, et un
faisceau <i>formé</i> vers une cible.</p>

<h3>Paramètres modifiables</h3>
<table cellpadding="4">
<tr><th align="left">Paramètre</th><th align="left">Définition exacte</th></tr>
<tr><td><b>Diamètre (m)</b></td>
    <td>Diamètre <i>D</i> physique du réflecteur (disque réel, ex. 2,2 m). Avec
    une Empreinte &gt; 0, ce diamètre fixe l'étendue du grand réflecteur, mais
    <i>pas</i> la largeur de faisceau élémentaire par feed : voir Empreinte
    ci-dessous (modèle deux échelles). Sans empreinte (=0, pleine ouverture),
    <i>D</i> gouverne directement θ₃dB ≈ 1,15·λ/<i>D</i>.</td></tr>
<tr><td><b>F/D</b></td>
    <td>Rapport focale/diamètre. La focale vaut <i>F</i> = (F/D)·<i>D</i>. Gouverne
    le dépointage par feed (θ ≈ BDF·offset/<i>F</i>, donc ∝ 1/<i>F</i>) et
    l'illumination du bord.</td></tr>
<tr><td><b>Offset clearance (m)</b></td>
    <td>Dégagement vertical de l'ouverture sous l'axe du paraboloïde parent.
    Centre de l'ouverture projetée : <i>clearance</i> + <i>D</i>/2.</td></tr>
<tr><td><b>Fréquence (GHz)</b></td>
    <td>Fréquence d'exploitation. Longueur d'onde λ = c/f. Intervient dans la
    largeur de faisceau (λ/<i>D</i>) et le gradient de phase.</td></tr>
<tr><td><b>Résolution ouverture (n)</b></td>
    <td>Nombre d'échantillons par côté de la grille d'ouverture (n×n) utilisée
    pour la FFT du champ. Fixe le pas d'échantillonnage dx = Diamètre/n : pour ne
    pas sous-résoudre l'écran de phase (Corrélation) ni l'Empreinte, il faut
    dx ≲ Corrélation/3 et dx ≲ Empreinte/20 — sinon le pattern paraît faussement
    bruité (aliasing). 256 (défaut) est un compromis raisonnable (~20 s à 91
    feeds) ; 448 donne la pleine fidélité (~80 s) ; 128 reste un aperçu rapide
    mais sous-résout déjà une corrélation de 15 mm.</td></tr>
<tr><td><b>Facteur q (cos^q)</b></td>
    <td>Taper d'amplitude du feed : diagramme cos<sup>q</sup>(ψ). <i>q</i> élevé ⇒
    bord du réflecteur sous-illuminé ⇒ faisceau secondaire un peu plus large et
    moins de spillover, au prix de l'efficacité.</td></tr>
<tr><td><b>Pas pitch (m)</b></td>
    <td>Distance centre-à-centre entre deux feeds voisins dans le plan focal
    (maille hexagonale). Espacement angulaire des faisceaux : Δθ = BDF·pitch/<i>F</i>.
    Pour 80 feeds, le feed le plus externe est à ≈ 4,7·pitch.</td></tr>
<tr><td><b>N feeds</b></td>
    <td>Nombre de feeds (points hexagonaux les plus proches du centre retenus).
    Remplit un disque de couverture d'autant plus grand.</td></tr>
<tr><td><b>Defocus (m)</b></td>
    <td>Déplacement axial du plan des feeds le long de l'axe focal (&gt;0 = feed
    reculé). Ajoute une phase d'erreur quadratique en ouverture qui étale le
    pattern par feed (crête plus basse, enveloppe plus large) ; le beamforming
    (filtre adapté déjà en place) refocalise et recombine partiellement
    l'énergie perdue, sans l'annuler entièrement.</td></tr>
<tr><td><b>σ par feed (rad)</b></td>
    <td>Écart-type de l'écran de phase aléatoire corrélé propre à chaque feed
    (erreurs type Ruze : imperfections de surface/alignement statistiques
    individuelles, RNG dédié par feed). Ce terme <i>disperse</i> : il dégrade
    chaque feed de façon indépendante, donc augmente la dispersion de crête
    d'un feed à l'autre en plus de la dégradation moyenne (crête ↓, lobes
    secondaires ↑) — indépendamment du pitch. σ=0 (défaut) : aucun écran,
    comportement inchangé.</td></tr>
<tr><td><b>σ commun (rad)</b></td>
    <td>Écart-type de l'écran de phase aléatoire corrélé COMMUN à tous les
    feeds (erreurs de surface du réflecteur lui-même, un seul écran partagé).
    Ce terme dégrade chaque feed de la même façon (crête ↓, lobes secondaires
    ↑) <i>sans</i> disperser les crêtes entre feeds, contrairement au σ par
    feed. σ=0 (défaut) : aucun écran commun.</td></tr>
<tr><td><b>Corrélation (m)</b></td>
    <td>Longueur de corrélation spatiale du lissage gaussien appliqué aux deux
    bruits de phase (par feed et commun) : contrôle l'échelle du « speckle »
    (grain fin ⇒ corrélation courte, texture plus large ⇒ corrélation
    longue), à σ fixé.
    Sans effet si σ=0.</td></tr>
<tr><td><b>Empreinte (m)</b></td>
    <td>Modèle deux échelles : le Diamètre ci-dessus est le vrai réflecteur
    physique, mais chaque feed n'en illumine qu'une empreinte gaussienne
    limitée, de ce diamètre (FWHM en amplitude). C'est elle, et non plus le
    diamètre du réflecteur, qui fixe la largeur de faisceau et le gain
    élémentaires par feed. 0 (défaut) : désactivée, pleine ouverture,
    comportement inchangé.</td></tr>
<tr><td><b>Magnification</b></td>
    <td>Déplacement du centre de l'empreinte par mètre de décalage du feed
    dans le plan focal (m/m) : en balayant le cluster de feeds, les
    empreintes se déplacent sur le grand réflecteur et pavent une zone plus
    large que l'empreinte seule. Le beam formé (rangée du bas) recombine
    alors ces empreintes en une pleine ouverture effective — c'est l'intérêt
    de la magnification. 0 (défaut) : empreinte centrée pour tous les feeds
    (pas de balayage). Sans effet si Empreinte=0.</td></tr>
<tr><td><b>Rayon zone (°)</b></td>
    <td>Rayon du disque de service, tracé en cercle blanc pointillé sur toutes les
    cartes (u,v). N'agit pas sur les champs, seulement sur l'affichage.</td></tr>
<tr><td><b>Feed n°</b></td>
    <td>Indice du feed dont le pattern individuel (co-pol dBi + phase) est affiché
    en haut à gauche.</td></tr>
<tr><td><b>Cible du beam (clic)</b></td>
    <td>Cliquer sur n'importe quelle carte large (u,v) fixe la coordonnée cible
    (u,v)° du beam formé par filtre adapté (rangée du bas) — directivité (axe 4)
    et zoom (axe 5), qui suivent tous les deux le clic.</td></tr>
</table>

<h3>Détail de la simulation</h3>
<ul>
<li><b>Champ d'ouverture (optique géométrique)</b> : un point d'ouverture est vu
du feed sous l'angle ψ = 2·arctan(ρ/2<i>F</i>) ; amplitude cos<sup>q</sup>(ψ).
Un feed décalé impose un gradient de phase linéaire (beam deviation factor) qui
dépointe le faisceau secondaire.</li>
<li><b>Far-field</b> : transformée de Fourier 2D du champ d'ouverture, échantillonnée
en cosinus directeurs (l,m) = (sinθcosφ, sinθsinφ), puis normalisée en directivité
(référence isotrope) et rééchantillonnée sur la grille (u,v) en degrés.</li>
<li><b>Co / cross-pol</b> : décomposition de Ludwig-3 ; la cross-pol vient de la
géométrie offset.</li>
<li><b>Beam formé</b> : filtre adapté (conjugate beamforming), poids
wᵢ = conj(Eᵢ(cible)) normalisés en norme L2 ⇒ la crête touche l'enveloppe max
par élément au point visé.</li>
<li><b>Beam formé — zoom</b> : la crête du beam formé pleine ouverture est large de
~λ/D<sub>physique</sub> (ex. ~0,35° à 20 GHz/2,2 m) — souvent moins large qu'un
pixel de la grille d'affichage ±14°/81px, donc invisible dans le panneau
directivité (axe 4), qui ne montre alors que le plancher de speckle. Le panneau
zoom (axe 5) ré-échantillonne les champs sur une fenêtre ±2° centrée sur la
cible pour résoudre cette crête ; mémorisé par (cible, paramètres, résolution
ouverture), il ne recalcule que si l'un des trois change. Cette re-synthèse est
<i>paresseuse</i> : au démarrage (avant tout clic), le panneau affiche un
placeholder plutôt que de payer ce calcul (coût comparable à la génération
principale) ; le premier clic sur une carte l'active définitivement.</li>
<li><b>Enveloppes</b> : « Enveloppe MRC (beamforming idéal) » = RSS
10·log₁₀ Σ|Eᵢ|² (directivité max <i>atteignable</i> par recombinaison
cohérente, lisse) ; « Enveloppe max par élément » = max des faisceaux
individuels sans recombinaison (montre le festonnage réel). Comparer une cible
de gain à la seconde, pas à la première.</li>
<li><b>Phase dé-référencée</b> : le panneau phase du feed (axe 1) retire, pour
l'affichage seulement, la porteuse linéaire exp(i·k·Y₀·sin v) due au centre
d'ouverture décalé Y₀ (offset clearance + D/2) — sans elle, cette porteuse se
replie en rayures (moiré) sur la grille d'affichage grossière. Les champs
utilisés pour le beamforming et les exports GRD restent inchangés (vérité
physique).</li>
</ul>

<h3>Exports</h3>
<ul>
<li><b>Export GRD (ZIP)</b> : un fichier <code>.grd</code> TICRA GRASP ASCII par feed
(co + cross, Ludwig-3, grille en cosinus directeurs IGRID=1), rassemblés dans un ZIP.</li>
<li><b>Export params (JSON)</b> : uniquement les paramètres de simulation courants
(dont la focale dérivée).</li>
</ul>
"""


def main() -> None:  # pragma: no cover
    configure_logging()
    parser = argparse.ArgumentParser(description="Studio interactif de calibration de patterns.")
    parser.add_argument(
        "--mode",
        choices=["pattern", "reflector"],
        default="pattern",
        help=(
            "Mode de lancement : 'pattern' (défaut) → PatternStudio ;"
            " 'reflector' → ReflectorStudio (AFR)."
        ),
    )
    args = parser.parse_args()
    app = QApplication.instance() or QApplication(sys.argv)
    if args.mode == "reflector":
        studio: QMainWindow = ReflectorStudio()
    else:
        studio = PatternStudio()
    studio.show()
    app.exec()


if __name__ == "__main__":
    main()
