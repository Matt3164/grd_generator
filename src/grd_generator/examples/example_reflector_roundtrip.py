"""
Exemple : boucle round-trip reflector (AFR) -> .grd -> grd_analyzer -> comparaison
====================================================================================
Génère les champs co/cross d'un réflecteur alimenté par réseau (AFR) avec les
paramètres donnés, les exporte en fichiers `.grd` TICRA GRASP (un par feed),
lance le projet frère `grd_analyzer` dessus, puis compare le `report.json`
obtenu à un rapport de référence (mesuré sur des données réelles) stat par
stat.

Sert de boucle de calibrage manuel pour faire coller le mode reflector aux
statistiques de données réelles : on ajuste les paramètres AFR (diamètre,
F/D, pitch, q, ...) jusqu'à ce que les deltas `sim` / `ref` se resserrent.

Important — confidentialité : les crêtes absolues (`peak_dbi_mean`) du
rapport de référence sont relativisées (offset arbitraire) pour ne pas
exposer le gain réel de l'antenne. Ne pas comparer `peak_dbi_mean` entre sim
et ref : comparer plutôt la dispersion des crêtes (`peak_dbi_std`), qui elle
est invariante à un décalage constant.

Usage :
    uv run python src/grd_generator/examples/example_reflector_roundtrip.py
    uv run python src/grd_generator/examples/example_reflector_roundtrip.py \\
        --diameter-m 2.0 --f-over-d 1.2 --pitch-m 0.03 --n-feeds 7

Prérequis : le dossier `grd_analyzer` (projet frère) doit être présent, par
défaut en sibling du repo courant (`../grd_analyzer`) ou pointé par la
variable d'environnement `GRD_ANALYZER_DIR`.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from grd_generator.gui import build_reflector_result
from grd_generator.logger import configure_logging, logger
from grd_generator.reflector.grd_export import pattern_to_grd
from grd_generator.schemas import UVGrid

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_REPORT = REPO_ROOT / "data" / "reference_reports" / "0000" / "report.json"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "export" / "roundtrip" / "default"

# Grille reflector en cosinus directeurs (u,v) = (sinθcosφ, sinθsinφ), conforme
# GRASP IGRID=1. ±0.25 ≈ ±14° (sin(14°)=0.242) : même champ de vue que l'ancienne
# grille en degrés.
_UV_HALF_WIDTH = 0.25

# Stats comparées entre le rapport simulé et le rapport de référence. Voir la
# note de confidentialité en tête de module : `peak_dbi_mean` est exclu.
COMPARED_STATS = (
    "lobe_width_deg_mean",
    "lobe_ellipticity_mean",
    "first_null_radius_deg_mean",
    "first_sidelobe_db_mean",
    "spacing_deg_mean",
    "peak_dbi_std",
    "cross_pol_max_db",
    "phase_slope_est_mean_rad_per_deg",
)


def _default_grd_analyzer_dir() -> Path:
    """Résout le dossier grd_analyzer : sibling du repo courant par défaut."""
    env_dir = os.environ.get("GRD_ANALYZER_DIR")
    if env_dir:
        return Path(env_dir)
    return REPO_ROOT.parent / "grd_analyzer"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diameter-m", type=float, default=2.0)
    parser.add_argument("--f-over-d", type=float, default=1.2)
    parser.add_argument("--offset-m", type=float, default=0.0)
    parser.add_argument(
        "--centered",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ouverture centrée sur l'axe (axisymétrique) au lieu de l'offset réel (défaut).",
    )
    parser.add_argument("--freq-ghz", type=float, default=20.0)
    parser.add_argument("--q", type=float, default=2.0)
    parser.add_argument("--pitch-m", type=float, default=0.03)
    parser.add_argument("--n-feeds", type=int, default=7)
    parser.add_argument("--defocus-m", type=float, default=0.0)
    parser.add_argument("--zone-radius-deg", type=float, default=8.0)
    parser.add_argument("--n-grid", type=int, default=81)
    parser.add_argument(
        "--reference-report",
        type=Path,
        default=DEFAULT_REFERENCE_REPORT,
        help="report.json de référence (mesuré sur données réelles)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="dossier de sortie (grd/, analysis/, comparison.json)",
    )
    parser.add_argument(
        "--grd-analyzer-dir",
        type=Path,
        default=None,
        help="dossier du projet grd_analyzer (défaut : sibling du repo, ou $GRD_ANALYZER_DIR)",
    )
    return parser.parse_args(argv)


def generate_and_export_grd(args: argparse.Namespace, grd_dir: Path) -> int:
    """Génère les champs AFR et écrit un `.grd` (co+cross) par feed dans `grd_dir`."""
    grid = UVGrid(
        u_min=-_UV_HALF_WIDTH, u_max=_UV_HALF_WIDTH,
        v_min=-_UV_HALF_WIDTH, v_max=_UV_HALF_WIDTH,
        n_u=args.n_grid, n_v=args.n_grid,
    )
    result = build_reflector_result(
        diameter_m=args.diameter_m,
        f_over_d=args.f_over_d,
        offset_clearance_m=args.offset_m,
        freq_ghz=args.freq_ghz,
        q=args.q,
        pitch_m=args.pitch_m,
        n_feeds=args.n_feeds,
        zone_radius_deg=args.zone_radius_deg,
        grid=grid,
        defocus_m=args.defocus_m,
        centered_aperture=args.centered,
    )
    grd_dir.mkdir(parents=True, exist_ok=True)
    freq_hz = args.freq_ghz * 1e9
    n = len(result.co_fields)
    for i, (co, cross) in enumerate(zip(result.co_fields, result.cross_fields, strict=True)):
        text = pattern_to_grd(co, cross, result.grid, freq_hz=freq_hz, title=f"feed #{i}")
        (grd_dir / f"feed_{i:03d}.grd").write_text(text, encoding="utf-8")
    logger.info("roundtrip_grd_written", directory=str(grd_dir), n_feeds=n)
    return n


def run_grd_analyzer(
    *, grd_analyzer_dir: Path, grd_dir: Path, analysis_dir: Path, freq_ghz: float
) -> None:
    """Lance `grd-analyzer` en subprocess via `uv run --project`."""
    if not grd_analyzer_dir.is_dir():
        logger.error("grd_analyzer_dir_missing", directory=str(grd_analyzer_dir))
        raise SystemExit(f"dossier grd_analyzer introuvable : {grd_analyzer_dir}")
    cmd = [
        "uv",
        "run",
        "--project",
        str(grd_analyzer_dir),
        "grd-analyzer",
        "--grd-dir",
        str(grd_dir),
        "--out-dir",
        str(analysis_dir),
        "--frequency-ghz",
        str(freq_ghz),
    ]
    logger.info("grd_analyzer_start", command=" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(
            "grd_analyzer_failed",
            returncode=result.returncode,
            stdout=result.stdout[-4000:],
            stderr=result.stderr[-4000:],
        )
        raise SystemExit(f"grd-analyzer a échoué (code {result.returncode}) — voir logs.")
    logger.info("grd_analyzer_done", out_dir=str(analysis_dir))


def compare_reports(sim_report: dict[str, Any], ref_report: dict[str, Any]) -> dict[str, Any]:
    """Compare les stats de `COMPARED_STATS` entre rapport simulé et de référence.

    Les valeurs `None` (stat absente/non calculable dans un des deux rapports)
    sont conservées telles quelles ; le delta n'est calculé que si les deux
    valeurs sont numériques.
    """
    comparison: dict[str, Any] = {}
    for stat in COMPARED_STATS:
        sim_val = sim_report.get(stat)
        ref_val = ref_report.get(stat)
        delta: float | None = None
        if isinstance(sim_val, (int, float)) and isinstance(ref_val, (int, float)):
            delta = sim_val - ref_val
        comparison[stat] = {"sim": sim_val, "ref": ref_val, "delta": delta}
        logger.info("stat_compare", stat=stat, sim=sim_val, ref=ref_val, delta=delta)
    return comparison


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    logger.info(
        "example_start",
        feature="reflector_roundtrip",
        diameter_m=args.diameter_m,
        f_over_d=args.f_over_d,
        pitch_m=args.pitch_m,
        n_feeds=args.n_feeds,
    )

    grd_analyzer_dir = args.grd_analyzer_dir or _default_grd_analyzer_dir()
    out_dir = args.out_dir
    grd_dir = out_dir / "grd"
    analysis_dir = out_dir / "analysis"

    generate_and_export_grd(args, grd_dir)
    run_grd_analyzer(
        grd_analyzer_dir=grd_analyzer_dir,
        grd_dir=grd_dir,
        analysis_dir=analysis_dir,
        freq_ghz=args.freq_ghz,
    )

    sim_report = json.loads((analysis_dir / "report.json").read_text(encoding="utf-8"))
    ref_report = json.loads(Path(args.reference_report).read_text(encoding="utf-8"))
    comparison = compare_reports(sim_report, ref_report)

    comparison_path = out_dir / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    logger.info("comparison_written", file=str(comparison_path))

    logger.info("example_done", feature="reflector_roundtrip")


if __name__ == "__main__":
    main()
