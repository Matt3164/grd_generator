"""Génération calibrée d'un array à partir d'un rapport grd_analyzer.

Offline : tire des éléments à lobe elliptique dans les distributions mesurées
(seed fixe), les place sur une maille hexagonale à l'espacement mesuré, synthétise
et écrit le .npz. `validate_against` re-mesure les stats générées et logue un
warning par champ hors tolérance (diagnostic, jamais bloquant).

Usage :
    uv run grd-calibrate --report data/reference_reports/0000 --out data/processed/cal0000.npz
"""

import argparse
from pathlib import Path

import numpy as np

from grd_generator.logger import configure_logging, logger
from grd_generator.report_ingest import FieldDist, MeasuredStats, read_report
from grd_generator.schemas import EllipticalSpec, UVGrid
from grd_generator.synth import elliptical_field, hex_centers_uv, widths_to_sigmas

_SIGMA_FLOOR = 1e-6  # plancher pour σ/largeurs tirées (strictement > 0)
DEFAULT_OUTPUT = Path("data/processed/calibrated.npz")


def _hex_centers(n: int, spacing: float) -> list[tuple[float, float]]:
    """`n` centres sur maille hex à `spacing` ; élargit le disque jusqu'à loger n."""
    # rayon ~ inverse de auto_spacing ; on élargit si la maille loge trop peu.
    radius = float(spacing * np.sqrt(np.sqrt(3.0) * n / (2.0 * np.pi)))
    while True:
        try:
            return hex_centers_uv((0.0, 0.0), radius, n, spacing)
        except ValueError:
            radius *= 1.3


def _fold_orientation(deg: float) -> float:
    """Replie un angle dans l'intervalle (−90, 90] (axe d'ellipse, période 180°)."""
    return 90.0 - ((90.0 - deg) % 180.0)


def _grid_for(
    centers: list[tuple[float, float]],
    specs: list[EllipticalSpec],
    n_u: int,
    n_v: int,
    margin_sigma: float,
) -> UVGrid:
    """Grille couvrant la bbox des centres élargie de `margin_sigma·max(σ_major)`."""
    cu = np.array([c[0] for c in centers])
    cv = np.array([c[1] for c in centers])
    margin = margin_sigma * max(s.sigma_major for s in specs)
    return UVGrid(
        u_min=float(cu.min()) - margin,
        u_max=float(cu.max()) + margin,
        v_min=float(cv.min()) - margin,
        v_max=float(cv.max()) + margin,
        n_u=n_u,
        n_v=n_v,
    )


def calibrate(
    stats: MeasuredStats,
    *,
    seed: int = 0,
    n_u: int = 161,
    n_v: int = 161,
    grid_margin_sigma: float = 6.0,
) -> tuple[list[EllipticalSpec], UVGrid]:
    """Tire `stats.n_elements` specs elliptiques dans les distributions mesurées."""
    rng = np.random.default_rng(seed)
    centers = _hex_centers(stats.n_elements, stats.spacing_deg)

    def draw(dist: FieldDist) -> float:
        return float(rng.normal(dist.mean, dist.std))

    specs: list[EllipticalSpec] = []
    for cu, cv in centers:
        wmaj = max(_SIGMA_FLOOR, draw(stats.lobe_width_major_deg))
        wmin = max(_SIGMA_FLOOR, draw(stats.lobe_width_minor_deg))
        first_null = (
            None
            if stats.first_null_radius_deg is None
            else max(_SIGMA_FLOOR, draw(stats.first_null_radius_deg))
        )
        smaj, smin = widths_to_sigmas(stats.mode, wmaj, wmin, first_null)
        specs.append(
            EllipticalSpec(
                center_uv=(cu, cv),
                sigma_major=max(_SIGMA_FLOOR, smaj),
                sigma_minor=max(_SIGMA_FLOOR, smin),
                orientation_deg=_fold_orientation(draw(stats.orientation_deg)),
                peak_gain_dbi=draw(stats.peak_dbi),
                phase_slope_radial=draw(stats.phase_slope_rad_per_deg),
            )
        )
    grid = _grid_for(centers, specs, n_u, n_v, grid_margin_sigma)
    logger.info("calibrated_specs", n=len(specs), mode=stats.mode)
    return specs, grid


def write_calibrated(
    output_path: str | Path,
    grid: UVGrid,
    specs: list[EllipticalSpec],
    *,
    mode: str,
) -> Path:
    """Synthétise les champs elliptiques et écrit le .npz (clés étendues)."""
    fields = np.stack([elliptical_field(s, grid, mode=mode) for s in specs])
    ids = np.array([f"ant_{k:02d}" for k in range(len(specs))])
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
    """Diagnostic : compare stats générées vs mesurées ; warning par champ hors tol.

    Renvoie la liste des écarts détectés (vide si tout est dans la tolérance).
    Ne lève jamais : c'est un diagnostic, pas un gate.
    """
    gen_peak = float(np.mean([s.peak_gain_dbi for s in specs]))
    gen_phase = float(np.mean([s.phase_slope_radial for s in specs]))
    gen_sigma_major = float(np.mean([s.sigma_major for s in specs]))
    gen_sigma_minor = float(np.mean([s.sigma_minor for s in specs]))
    # Ellipticité = rapport des moyennes (et non moyenne des rapports) : même
    # définition que la cible mesurée, sans le biais de Jensen sur les ratios.
    gen_ellipticity = gen_sigma_major / gen_sigma_minor
    # Cibles mesurées pour la valeur ajoutée du feature (lobes elliptiques) :
    # l'ellipticité = rapport des largeurs ; l'échelle σ_major via la même
    # conversion largeur→σ que la génération. L'orientation n'est PAS diagnostiquée
    # par sa moyenne (≈ 0, distribution quasi-uniforme d'axes : moyenne non informative).
    measured_ellipticity = stats.lobe_width_major_deg.mean / stats.lobe_width_minor_deg.mean
    first_null_mean = (
        None if stats.first_null_radius_deg is None else stats.first_null_radius_deg.mean
    )
    measured_sigma_major, _ = widths_to_sigmas(
        stats.mode,
        stats.lobe_width_major_deg.mean,
        stats.lobe_width_minor_deg.mean,
        first_null_mean,
    )
    checks: list[tuple[str, float, float]] = [
        ("peak_dbi", stats.peak_dbi.mean, gen_peak),
        ("phase_slope_rad_per_deg", stats.phase_slope_rad_per_deg.mean, gen_phase),
        ("ellipticity", measured_ellipticity, gen_ellipticity),
        ("sigma_major_deg", measured_sigma_major, gen_sigma_major),
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
    seed: int = 0,
) -> Path:
    """Bout-en-bout : lit le rapport, calibre, écrit le .npz, logue le diagnostic."""
    stats = read_report(report_path)
    specs, grid = calibrate(stats, seed=seed)
    out = write_calibrated(output_path, grid, specs, mode=stats.mode)
    validate_against(stats, specs)
    return out


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Génère un array calibré depuis un report.json.")
    parser.add_argument("--report", required=True, help="report.json (fichier ou dossier)")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="chemin du .npz de sortie")
    parser.add_argument("--seed", type=int, default=0, help="seed du tirage (reproductibilité)")
    args = parser.parse_args()
    generate_calibrated(args.report, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
