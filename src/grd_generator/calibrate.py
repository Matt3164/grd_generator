"""Génération calibrée d'un array : maillage dicté par la zone, élément par le rapport.

Offline. La **géométrie** (maille hexagonale, espacement, échelle σ du lobe) vient
de la **zone de couverture** (scénario, ex. FRANCE) : centre/rayon de zone via
`sat_frame_geometry`, espacement via `auto_spacing`, et σ dimensionné par bissection
pour que l'enveloppe (max sur patterns) ≥ crête−marge **sur la zone** (esprit
`optimize_sigma`). Le **rapport** ne fournit que le *caractère* d'élément — mode,
ratio d'ellipticité, orientation, crête, pente de phase — échantillonné par élément
(seed fixe). `validate_against` reste un diagnostic (warnings, jamais bloquant).

Usage :
    uv run grd-calibrate --report data/reference_reports/0000 --out data/processed/cal0000.npz
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from grd_generator.generate import auto_spacing
from grd_generator.logger import configure_logging, logger
from grd_generator.report_ingest import FieldDist, MeasuredStats, read_report
from grd_generator.scenario import FRANCE, Scenario, sat_frame_geometry
from grd_generator.schemas import EllipticalSpec, UVGrid
from grd_generator.synth import elliptical_field, hex_centers_uv, pattern_envelope_min_in_zone

_SIGMA_FLOOR = 1e-6  # plancher strict pour σ (deg)
_SIGMA_HI = 5.0  # échelle σ (moyenne géométrique) max explorée (deg)
_SIGMA_TOL = 1e-3  # tolérance de la bissection sur σ
DEFAULT_N_ELEMENTS = 80
DEFAULT_COVERAGE_MARGIN_DB = 3.0  # seuil de couverture = crête moyenne − marge
DEFAULT_OUTPUT = Path("data/processed/calibrated.npz")


@dataclass(frozen=True)
class _Character:
    """Caractère d'élément du rapport (indépendant de l'échelle σ, fixée par la zone)."""

    ellipticity: float  # ratio d'ellipticité (constant : rapport des largeurs moyennes)
    rel_width: NDArray[np.float64]  # dispersion relative de taille autour de 1
    orientation_deg: NDArray[np.float64]
    peak_dbi: NDArray[np.float64]
    phase_slope: NDArray[np.float64]


def _fold_orientation(deg: float) -> float:
    """Replie un angle dans l'intervalle (−90, 90] (axe d'ellipse, période 180°)."""
    return 90.0 - ((90.0 - deg) % 180.0)


def _sample_character(stats: MeasuredStats, n: int, seed: int) -> _Character:
    """Tire le caractère d'élément (ellipticité, orientation, crête, phase) du rapport."""
    rng = np.random.default_rng(seed)

    def draw(dist: FieldDist) -> NDArray[np.float64]:
        return np.asarray(rng.normal(dist.mean, dist.std, size=n), dtype=np.float64)

    # Ratio d'ellipticité = rapport des largeurs MOYENNES (constant) — même grandeur
    # que la cible diagnostiquée, sans biais de Jensen d'un tirage max/min par élément.
    ellipticity = stats.lobe_width_major_deg.mean / stats.lobe_width_minor_deg.mean
    # Dispersion de TAILLE relative (pas d'ellipticité) : moyenne géométrique des
    # largeurs tirées, normalisée à 1.
    wmaj = np.maximum(_SIGMA_FLOOR, draw(stats.lobe_width_major_deg))
    wmin = np.maximum(_SIGMA_FLOOR, draw(stats.lobe_width_minor_deg))
    gmean = np.sqrt(wmaj * wmin)
    rel_width = gmean / float(gmean.mean())
    orientation = np.array([_fold_orientation(float(x)) for x in draw(stats.orientation_deg)])
    return _Character(
        ellipticity=float(ellipticity),
        rel_width=rel_width,
        orientation_deg=orientation,
        peak_dbi=draw(stats.peak_dbi),
        phase_slope=draw(stats.phase_slope_rad_per_deg),
    )


def _zone_centers(
    zone_center: tuple[float, float], zone_radius: float, n: int
) -> tuple[list[tuple[float, float]], float]:
    """`n` centres hex couvrant la zone ; resserre l'espacement jusqu'à loger n.

    `auto_spacing` donne l'espacement théorique par aire ; le pavage d'un disque
    en loge un peu moins (effets de bord), surtout pour les petits n — on resserre.
    """
    spacing = auto_spacing(zone_radius, n)
    while True:
        try:
            return hex_centers_uv(zone_center, zone_radius, n, spacing), spacing
        except ValueError:
            spacing *= 0.9


def _specs_at_scale(
    centers: list[tuple[float, float]], char: _Character, scale: float
) -> list[EllipticalSpec]:
    """Construit les specs elliptiques à l'échelle σ donnée (caractère fixé)."""
    root = float(np.sqrt(char.ellipticity))
    specs: list[EllipticalSpec] = []
    for k, (cu, cv) in enumerate(centers):
        sigma = max(_SIGMA_FLOOR, scale * float(char.rel_width[k]))
        specs.append(
            EllipticalSpec(
                center_uv=(cu, cv),
                sigma_major=max(_SIGMA_FLOOR, sigma * root),
                sigma_minor=max(_SIGMA_FLOOR, sigma / root),
                orientation_deg=float(char.orientation_deg[k]),
                peak_gain_dbi=float(char.peak_dbi[k]),
                phase_slope_radial=float(char.phase_slope[k]),
            )
        )
    return specs


def calibrate(
    stats: MeasuredStats,
    *,
    scenario: Scenario = FRANCE,
    n_elements: int = DEFAULT_N_ELEMENTS,
    seed: int = 0,
    coverage_margin_db: float = DEFAULT_COVERAGE_MARGIN_DB,
    n_u: int = 161,
    n_v: int = 161,
) -> tuple[list[EllipticalSpec], UVGrid]:
    """Calibre un array : maille hex couvrant la zone, σ optimisé pour la couverture.

    La géométrie (centre/rayon de zone, grille) vient de `scenario` ; l'espacement
    de `auto_spacing(rayon, n_elements)` ; l'échelle σ du lobe est trouvée par
    bissection pour que l'enveloppe max-sur-patterns ≥ `crête_moyenne −
    coverage_margin_db` à l'intérieur de la zone. Le `stats` fournit le caractère
    d'élément (mode, ellipticité, orientation, crête, phase).
    """
    geo = sat_frame_geometry(scenario, n_u=n_u, n_v=n_v)
    centers, spacing = _zone_centers(geo.zone_center, geo.zone_radius_deg, n_elements)
    char = _sample_character(stats, n_elements, seed)
    target_dbi = float(char.peak_dbi.mean()) - coverage_margin_db

    def covers(scale: float) -> bool:
        specs = _specs_at_scale(centers, char, scale)
        cov = pattern_envelope_min_in_zone(
            specs, geo.grid, geo.zone_center, geo.zone_radius_deg, mode=stats.mode
        )
        return cov >= target_dbi

    if covers(_SIGMA_HI):
        lo, hi = _SIGMA_FLOOR, _SIGMA_HI
        while hi - lo > _SIGMA_TOL:
            mid = 0.5 * (lo + hi)
            if covers(mid):
                hi = mid
            else:
                lo = mid
        scale = hi
    else:
        logger.warning(
            "calibration_coverage_infeasible",
            target_dbi=round(target_dbi, 2),
            sigma_hi=_SIGMA_HI,
        )
        scale = _SIGMA_HI

    specs = _specs_at_scale(centers, char, scale)
    logger.info(
        "calibrated_specs",
        n=len(specs),
        mode=stats.mode,
        sigma_scale_deg=round(scale, 4),
        spacing_deg=round(spacing, 4),
        target_dbi=round(target_dbi, 2),
    )
    return specs, geo.grid


def write_calibrated(
    output_path: str | Path,
    grid: UVGrid,
    specs: list[EllipticalSpec],
    *,
    mode: str,
) -> Path:
    """Synthétise les champs elliptiques et écrit le .npz (clés étendues)."""
    fields = np.stack([elliptical_field(s, grid, mode=mode) for s in specs])
    ids = np.array([f"ant_{k:03d}" for k in range(len(specs))])
    centers_uv = np.array([s.center_uv for s in specs], dtype=float)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        u_min=grid.u_min,
        u_max=grid.u_max,
        v_min=grid.v_min,
        v_max=grid.v_max,
        n_u=grid.n_u,
        n_v=grid.n_v,
        fields=fields,
        antenna_ids=ids,
        centers_uv=centers_uv,
        mode=np.array(mode),
        peaks_dbi=np.array([s.peak_gain_dbi for s in specs], dtype=float),
        phase_slopes=np.array([s.phase_slope_radial for s in specs], dtype=float),
        sigmas_major=np.array([s.sigma_major for s in specs], dtype=float),
        sigmas_minor=np.array([s.sigma_minor for s in specs], dtype=float),
        orientations_deg=np.array([s.orientation_deg for s in specs], dtype=float),
    )
    logger.info("calibrated_written", file=str(out), n_antennas=len(specs))
    return out


def validate_against(
    stats: MeasuredStats,
    specs: list[EllipticalSpec],
    *,
    tol: float = 0.15,
) -> list[dict[str, object]]:
    """Diagnostic : caractère généré vs mesuré (crête, phase, ellipticité) ; warnings.

    L'échelle σ et l'espacement ne sont PAS diagnostiqués : ils viennent de la zone,
    pas du rapport. Renvoie la liste des écarts (vide si tout est dans la tolérance) ;
    ne lève jamais.
    """
    gen_peak = float(np.mean([s.peak_gain_dbi for s in specs]))
    gen_phase = float(np.mean([s.phase_slope_radial for s in specs]))
    gen_sigma_major = float(np.mean([s.sigma_major for s in specs]))
    gen_sigma_minor = float(np.mean([s.sigma_minor for s in specs]))
    # Ellipticité = rapport des moyennes (même définition que la cible, sans biais de Jensen).
    gen_ellipticity = gen_sigma_major / gen_sigma_minor
    measured_ellipticity = stats.lobe_width_major_deg.mean / stats.lobe_width_minor_deg.mean
    checks: list[tuple[str, float, float]] = [
        ("peak_dbi", stats.peak_dbi.mean, gen_peak),
        ("phase_slope_rad_per_deg", stats.phase_slope_rad_per_deg.mean, gen_phase),
        ("ellipticity", measured_ellipticity, gen_ellipticity),
    ]
    deviations: list[dict[str, object]] = []
    for field, measured, generated in checks:
        if measured != 0.0 and abs(generated - measured) / abs(measured) > tol:
            deviations.append({"field": field, "measured": measured, "generated": generated})
            logger.warning(
                "calibration_stat_deviation",
                field=field,
                measured=round(measured, 4),
                generated=round(generated, 4),
                tol=tol,
            )
    return deviations


def generate_calibrated(
    report_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    scenario: Scenario = FRANCE,
    n_elements: int = DEFAULT_N_ELEMENTS,
    seed: int = 0,
    coverage_margin_db: float = DEFAULT_COVERAGE_MARGIN_DB,
    n_u: int = 161,
    n_v: int = 161,
) -> Path:
    """Bout-en-bout : lit le rapport, calibre sur la zone, écrit le .npz, logue le diagnostic."""
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
    out = write_calibrated(output_path, grid, specs, mode=stats.mode)
    validate_against(stats, specs)
    return out


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Génère un array calibré (maillage = zone FRANCE, élément = rapport)."
    )
    parser.add_argument("--report", required=True, help="report.json (fichier ou dossier)")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="chemin du .npz de sortie")
    parser.add_argument(
        "--n-elements", type=int, default=DEFAULT_N_ELEMENTS, help="nombre d'éléments"
    )
    parser.add_argument(
        "--coverage-margin-db",
        type=float,
        default=DEFAULT_COVERAGE_MARGIN_DB,
        help="marge sous la crête moyenne pour le seuil de couverture (dB)",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed du tirage (reproductibilité)")
    args = parser.parse_args()
    generate_calibrated(
        args.report,
        args.out,
        n_elements=args.n_elements,
        seed=args.seed,
        coverage_margin_db=args.coverage_margin_db,
    )


if __name__ == "__main__":
    main()
