"""Générateur offline du jeu de référence pattern (80 éléments) → .npz.

Autonome et séparé de toute analyse : porte l'optimisation σ (bissection sous
contrainte enveloppe ≥ 44 dBi) puis écrit grille + 80 champs complexes +
métadonnées dans `data/processed/reference_array.npz`. Lancé à la main.

Usage :
    uv run grd-generate
    python src/grd_generator/generate.py
"""

from pathlib import Path

import numpy as np

from grd_generator.logger import configure_logging, logger
from grd_generator.scenario import FRANCE, Scenario, sat_frame_geometry
from grd_generator.schemas import GaussianSpec, UVGrid
from grd_generator.synth import (
    check_power_conservation,
    field_generator,
    hex_centers_uv,
    optimize_sigma,
)

# Paramètres d'array de référence.
N_ANTENNAS = 80
PEAK_GAIN_DBI = 47.0  # gain crête par élément (dBi)
# Plancher d'enveloppe imposé dans la zone = seuil de couverture arbitré (44 dBi).
TARGET_DBI = 44.0
# Pente de phase **radiale** (rad/°) commune à tous les éléments :
# φ = pente·‖uv−centre‖. N'affecte pas |E| ni la couverture, mais rend les poids
# de combinaison complexes et fait apparaître des ripples d'interférence.
PHASE_SLOPE_RADIAL = 3.0
GRID_N_U = 161
GRID_N_V = 161
SIGMA_LO = 1e-3
SIGMA_HI = 5.0
DEFAULT_OUTPUT = Path("data/processed/reference_array.npz")


def auto_spacing(zone_radius: float, n: int) -> float:
    """Pas hexagonal pour loger ~`n` points dans le disque (aire = π r²)."""
    return float(np.sqrt(2.0 * np.pi * zone_radius**2 / (np.sqrt(3.0) * n)))


def write_reference(
    output_path: str | Path,
    grid: UVGrid,
    specs: list[GaussianSpec],
    *,
    mode: str = "gaussian",
) -> Path:
    """Synthétise les champs des `specs` (selon le `mode`) et écrit le .npz."""
    generate = field_generator(mode)
    fields = np.stack([generate(s, grid) for s in specs])
    ids = np.array([f"ant_{k:02d}" for k in range(len(specs))])
    centers_uv = np.array([s.center_uv for s in specs], dtype=float)
    sigmas = np.array([s.sigma for s in specs], dtype=float)
    peaks = np.array([s.peak_gain_dbi for s in specs], dtype=float)
    phase_slopes = np.array([s.phase_slope_radial for s in specs], dtype=float)

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
        sigmas=sigmas,
        peaks_dbi=peaks,
        phase_slopes=phase_slopes,
    )
    logger.info("pattern_gen_done", file=str(out), n_antennas=len(specs))
    return out


def generate_reference(
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    mode: str = "gaussian",
    scenario: Scenario = FRANCE,
) -> Path:
    """Optimise la largeur, synthétise les 80 champs et écrit le .npz de référence.

    `mode` sélectionne le générateur d'éléments (`gaussian` par défaut, `airy`
    pour des éléments à lobes secondaires). `scenario` fixe la géométrie
    (défaut FRANCE). Géométrie en repère SAT (nadir) dérivée du scénario.
    """
    geo = sat_frame_geometry(scenario, n_u=GRID_N_U, n_v=GRID_N_V)
    spacing = auto_spacing(geo.zone_radius_deg, N_ANTENNAS)
    centers = hex_centers_uv(geo.zone_center, geo.zone_radius_deg, N_ANTENNAS, spacing)

    logger.info("pattern_gen_start", n_antennas=N_ANTENNAS, spacing_deg=round(spacing, 4))
    sigma, specs = optimize_sigma(
        geo.grid,
        centers,
        peak_gain_dbi=PEAK_GAIN_DBI,
        zone_center=geo.zone_center,
        zone_radius=geo.zone_radius_deg,
        target_dbi=TARGET_DBI,
        sigma_lo=SIGMA_LO,
        sigma_hi=SIGMA_HI,
        phase_slope_radial=PHASE_SLOPE_RADIAL,
        mode=mode,
    )
    logger.info("pattern_gen_sigma", sigma_deg=round(sigma, 4))

    # Conservation de la puissance : crête et largeur sont liées (D·Ω ≈ 4π).
    # Warning seulement.
    check_power_conservation(mode, sigma, PEAK_GAIN_DBI)

    return write_reference(output_path, geo.grid, specs, mode=mode)


def main() -> None:
    configure_logging()
    generate_reference()


if __name__ == "__main__":
    main()
