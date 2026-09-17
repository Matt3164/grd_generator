"""
Exemple : calibrage du jeu Ku multi-fréquence (`grd-batch`)
============================================================
Balaie une petite grille de paramètres (`defocus_m`, `pitch_m`, `q`, `n_feeds`)
à une seule fréquence (11 400 MHz par défaut), calcule les métriques
(`reflector.metrics`) sur les champs AFR obtenus — **sans écrire de fichier
GRD** — et imprime/logue le tableau des résultats, avec une colonne
« critères tenus » (voir la spec, § Références chiffrées). Sert de boucle de
calibrage manuel pour choisir les paramètres avant de figer `batch-ku.json`
(spec, § Calibrage).

Avec `--figure PATH`, rend en plus une figure à trois panneaux (10 700 MHz,
`--freq-mhz`, 12 750 MHz) de l'enveloppe (`Σ|co|²`) pour la **première**
combinaison du balayage (passer des listes à un seul élément par axe pour
choisir précisément la combinaison figurée), avec, si `--mission-plan` est
fourni, les centres de faisceaux du cas en surimpression (conversion
`(u, v) = (sin θx, sin θy)`, voir la spec, § Après B3 / point ouvert n°1).

Usage :
    uv run python src/grd_generator/examples/example_batch_ku.py
    uv run python src/grd_generator/examples/example_batch_ku.py \\
        --defocus-m 0.53 0.75 0.93 1.1 --pitch-m 0.032 0.036 0.040 \\
        --q 2.0 3.0 --n-feeds 91 127
    uv run python src/grd_generator/examples/example_batch_ku.py \\
        --defocus-m 0.53 --pitch-m 0.032 --q 2.0 --n-feeds 91 \\
        --figure calibrage.png \\
        --mission-plan chemin/vers/mission_plan.json
"""

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from grd_generator.logger import configure_logging, logger
from grd_generator.reflector.metrics import FrequencyMetrics, compute_frequency_metrics
from grd_generator.reflector.spec import FeedSpec, ReflectorSpec
from grd_generator.reflector.synth_afr import hex_feed_positions, synthesize_reflector_fields
from grd_generator.schemas import ComplexField, UVGrid
from grd_generator.synth import combined_max_directivity_dbi

# Point de départ du calibrage (jeu d'août, spec § Décisions validées) :
# valeurs par défaut de ce script, PAS des valeurs validées.
_DEFAULT_DIAMETER_M = 1.3
_DEFAULT_F_OVER_D = 1.5
_DEFAULT_OFFSET_CLEARANCE_M = 0.2
_DEFAULT_UV_HALF_WIDTH = 0.18
_DEFAULT_N_GRID = 161
_DEFAULT_N_APERTURE = 128
_DEFAULT_PAD_FACTOR = 8
# Même convention que `gui.build_reflector_result`/`batch._feed_positions`.
_FOCAL_RADIUS_PITCH_FACTOR = 10.0
_FIGURE_LOW_HIGH_MHZ = (10700, 12750)  # bornes basse/haute ; la 3e freq = --freq-mhz

_CRITERIA_HEADER = (
    "defocus_m", "pitch_m", "q", "n_feeds",
    "peak_dbi_med", "width_3db_med", "peak_r_max",
    "env_max_dbi", "env_r1db_deg", "env_r3db_deg", "env_r39dbi_deg",
    "criteres",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq-mhz", type=int, default=11400)
    parser.add_argument("--defocus-m", type=float, nargs="+", default=[0.53])
    parser.add_argument("--pitch-m", type=float, nargs="+", default=[0.032])
    parser.add_argument("--q", type=float, nargs="+", default=[2.0])
    parser.add_argument("--n-feeds", type=int, nargs="+", default=[91])
    parser.add_argument("--diameter-m", type=float, default=_DEFAULT_DIAMETER_M)
    parser.add_argument("--f-over-d", type=float, default=_DEFAULT_F_OVER_D)
    parser.add_argument("--offset-clearance-m", type=float, default=_DEFAULT_OFFSET_CLEARANCE_M)
    parser.add_argument(
        "--centered", action=argparse.BooleanOptionalAction, default=True,
        help="ouverture centrée sur l'axe (défaut, comme la config de départ de la spec)",
    )
    parser.add_argument("--uv-half-width", type=float, default=_DEFAULT_UV_HALF_WIDTH)
    parser.add_argument("--n-grid", type=int, default=_DEFAULT_N_GRID)
    parser.add_argument("--n-aperture", type=int, default=_DEFAULT_N_APERTURE)
    parser.add_argument("--pad-factor", type=int, default=_DEFAULT_PAD_FACTOR)
    parser.add_argument("--figure", type=Path, default=None, help="chemin du PNG de calibrage")
    parser.add_argument(
        "--mission-plan", type=Path, default=None,
        help="mission_plan.json à surimprimer sur la figure (--figure requis)",
    )
    return parser.parse_args(argv)


def build_grid(args: argparse.Namespace) -> UVGrid:
    return UVGrid(
        u_min=-args.uv_half_width, u_max=args.uv_half_width,
        v_min=-args.uv_half_width, v_max=args.uv_half_width,
        n_u=args.n_grid, n_v=args.n_grid,
    )


def synthesize_combo(
    args: argparse.Namespace,
    grid: UVGrid,
    *,
    freq_mhz: float,
    defocus_m: float,
    pitch_m: float,
    q: float,
    n_feeds: int,
) -> tuple[list[ComplexField], list[ComplexField]]:
    """Synthétise les champs co/cross AFR pour une combinaison de calibrage."""
    spec = ReflectorSpec(
        diameter_m=args.diameter_m,
        focal_length_m=args.f_over_d * args.diameter_m,
        offset_clearance_m=args.offset_clearance_m,
        freq_hz=freq_mhz * 1.0e6,
        centered_aperture=args.centered,
    )
    positions = hex_feed_positions(
        pitch_m, n_feeds, focal_radius_m=_FOCAL_RADIUS_PITCH_FACTOR * pitch_m
    )
    feeds = FeedSpec(positions_m=positions, q=q, defocus_m=defocus_m)
    return synthesize_reflector_fields(
        spec, feeds, grid, n_aperture=args.n_aperture, pad_factor=args.pad_factor
    )


def feed_peaks_uv(co_fields: list[ComplexField], grid: UVGrid) -> NDArray[np.float64]:
    """Position (u, v) de la crête de chaque feed."""
    gu, gv = grid.meshgrid()
    peaks = np.zeros((len(co_fields), 2), dtype=np.float64)
    for i, co in enumerate(co_fields):
        idx = np.unravel_index(np.argmax(np.abs(co) ** 2), co.shape)
        peaks[i] = (gu[idx], gv[idx])
    return peaks


def load_mission_plan_uv(path: Path) -> NDArray[np.float64]:
    """Centres de faisceaux du cas, `(thetaX, thetaY)` en degrés → `(u, v)`.

    `u = sin(θx)`, `v = sin(θy)` (chaque axe converti indépendamment — la
    grille est en cosinus directeurs, le plan de mission en degrés « répartis
    sur les deux axes », voir la spec, section « Après B3 »)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    centers: list[tuple[float, float]] = []
    for beam in data.get("beams", []):
        for coverage in beam.get("coverages", []):
            theta_x_deg, theta_y_deg = coverage["geometry"]["center"]
            centers.append(
                (math.sin(math.radians(theta_x_deg)), math.sin(math.radians(theta_y_deg)))
            )
    return np.array(centers, dtype=np.float64)


def criteria_status(fm: FrequencyMetrics) -> str:
    """Résume les critères d'acceptation de la spec (tableau « Références
    chiffrées ») pour une combinaison : `"OK"` ou la liste des critères ratés."""
    checks = {
        "peak_med_27-28.5": 27.0 <= fm.peak_dbi_median <= 28.5,
        "width_med_6-7.5": 6.0 <= fm.width_3db_deg_median <= 7.5,
        "peak_r_max<=6.2": fm.peak_radius_deg_max <= 6.2,
        "env_max_42-44.5": 42.0 <= fm.envelope.max_dbi <= 44.5,
        "env_r3db>=4.0(bloquant)": fm.envelope.radius_3db_deg >= 4.0,
        "env_r39dbi>=4.0": fm.envelope.radius_39dbi_deg >= 4.0,
    }
    failed = [name for name, ok in checks.items() if not ok]
    return "OK" if not failed else "; ".join(failed)


def _print_table(rows: list[dict[str, Any]]) -> None:
    # print() délibéré : c'est le RÉSULTAT de ce script (tableau lu par un
    # humain en calibrage manuel, cf. spec « imprime/logue »), pas un message
    # de statut — ceux-ci passent par loguru (`logger.info("sweep_combo", …)`
    # à l'appel de `run_sweep`, juste avant).
    widths = [
        max(len(col), max(len(str(row[col])) for row in rows)) for col in _CRITERIA_HEADER
    ]

    def _fmt(values: list[Any]) -> str:
        return "  ".join(str(v).rjust(w) for v, w in zip(values, widths, strict=True))

    print(_fmt(list(_CRITERIA_HEADER)))
    print(_fmt(["-" * w for w in widths]))
    for row in rows:
        print(_fmt([row[col] for col in _CRITERIA_HEADER]))


def run_sweep(args: argparse.Namespace, grid: UVGrid) -> list[dict[str, Any]]:
    """Balaie le produit cartésien des paramètres, logue et renvoie les lignes."""
    rows: list[dict[str, Any]] = []
    combos = itertools.product(args.defocus_m, args.pitch_m, args.q, args.n_feeds)
    for defocus_m, pitch_m, q, n_feeds in combos:
        co_fields, cross_fields = synthesize_combo(
            args, grid, freq_mhz=args.freq_mhz,
            defocus_m=defocus_m, pitch_m=pitch_m, q=q, n_feeds=n_feeds,
        )
        fm = compute_frequency_metrics(args.freq_mhz, co_fields, cross_fields, grid)
        row = {
            "defocus_m": defocus_m, "pitch_m": pitch_m, "q": q, "n_feeds": n_feeds,
            "peak_dbi_med": round(fm.peak_dbi_median, 2),
            "width_3db_med": round(fm.width_3db_deg_median, 2),
            "peak_r_max": round(fm.peak_radius_deg_max, 2),
            "env_max_dbi": round(fm.envelope.max_dbi, 2),
            "env_r1db_deg": round(fm.envelope.radius_1db_deg, 2),
            "env_r3db_deg": round(fm.envelope.radius_3db_deg, 2),
            "env_r39dbi_deg": round(fm.envelope.radius_39dbi_deg, 2),
            "criteres": criteria_status(fm),
        }
        logger.info("sweep_combo", freq_mhz=args.freq_mhz, **row)
        rows.append(row)
    return rows


def render_figure(args: argparse.Namespace, grid: UVGrid) -> None:
    """Figure à 3 panneaux (bas/`--freq-mhz`/haut) pour la 1ère combinaison du
    balayage — voir la docstring du module."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from grd_generator.plot import draw_calibration_envelope_map

    defocus_m, pitch_m, q, n_feeds = (
        args.defocus_m[0], args.pitch_m[0], args.q[0], args.n_feeds[0]
    )
    mission_plan_uv = load_mission_plan_uv(args.mission_plan) if args.mission_plan else None
    freqs_mhz = (_FIGURE_LOW_HIGH_MHZ[0], args.freq_mhz, _FIGURE_LOW_HIGH_MHZ[1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, freq_mhz in zip(axes, freqs_mhz, strict=True):
        co_fields, _ = synthesize_combo(
            args, grid, freq_mhz=freq_mhz,
            defocus_m=defocus_m, pitch_m=pitch_m, q=q, n_feeds=n_feeds,
        )
        envelope_dbi = combined_max_directivity_dbi(co_fields)
        draw_calibration_envelope_map(
            ax, grid, envelope_dbi,
            feed_peaks_uv=feed_peaks_uv(co_fields, grid),
            mission_plan_uv=mission_plan_uv,
            title=f"{freq_mhz} MHz",
        )
    fig.suptitle(
        f"Calibrage : defocus_m={defocus_m}, pitch_m={pitch_m}, q={q}, n_feeds={n_feeds}"
    )
    fig.tight_layout()
    fig.savefig(args.figure, dpi=150)
    plt.close(fig)
    logger.info("figure_written", path=str(args.figure))


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    logger.info(
        "example_start", feature="batch_ku_calibration", freq_mhz=args.freq_mhz,
        n_combos=len(args.defocus_m) * len(args.pitch_m) * len(args.q) * len(args.n_feeds),
    )

    grid = build_grid(args)
    rows = run_sweep(args, grid)
    _print_table(rows)

    if args.figure is not None:
        render_figure(args, grid)

    logger.info("example_done", feature="batch_ku_calibration")


if __name__ == "__main__":
    main()
