"""Synthèse des centres de feed et combinaison de champs (chemin réflecteur AFR).

Sous-ensemble du module de synthèse pure amont, réduit aux deux fonctions
encore requises par le chemin réflecteur : `hex_centers_uv` place les centres
de feed sur une maille hexagonale dans le disque de service
(`reflector/synth_afr.py`) ; `combined_max_directivity_dbi` calcule la
directivité max atteignable après combinaison cohérente de champs (`gui.py`).
Ce module est entièrement testé (100 %).
"""

import numpy as np

from grd_generator.schemas import ComplexField, DirectivityMap

_FLOOR = 1e-12  # plancher de |E|² pour éviter log10(0) hors lobe


def combined_max_directivity_dbi(fields: list[ComplexField]) -> DirectivityMap:
    """Directivité **max atteignable après combinaison cohérente**, point par point.

    En pointant un faisceau formé optimal (filtre adapté `wᵢ = conj(Eᵢ)`, poids à
    norme L2 unité) sur un point, la directivité atteignable y vaut
    `10·log10(Σ|Eᵢ|²)` — borne de Cauchy-Schwarz `|Σ wᵢ Eᵢ| ≤ ‖w‖·‖E‖`. C'est
    le RSS des champs : ≥ le meilleur élément seul, et exactement la crête du
    faisceau cohérent (combinaison en filtre adapté, lobe pointé) lorsqu'on y pointe.
    """
    power = np.sum([np.abs(f) ** 2 for f in fields], axis=0)
    dbi: DirectivityMap = (10.0 * np.log10(np.maximum(power, _FLOOR))).astype(np.float64)
    return dbi


def hex_centers_uv(
    zone_center: tuple[float, float],
    zone_radius: float,
    n: int,
    spacing: float,
) -> list[tuple[float, float]]:
    """`n` centres sur une maille hexagonale, pris dans le disque de service.

    Pavage hexagonal (lignes décalées d'un demi-pas) de pas `spacing` ; on garde
    les points inclus dans le disque et on retient les `n` plus proches du centre.
    Lève ValueError si moins de `n` points tiennent dans le disque.
    """
    cu, cv = zone_center
    dy = spacing * np.sqrt(3.0) / 2.0
    rows = int(np.ceil(2.0 * zone_radius / dy)) + 2
    cols = int(np.ceil(2.0 * zone_radius / spacing)) + 2

    pts: list[tuple[float, float]] = []
    for i in range(-rows, rows + 1):
        y = cv + i * dy
        offset = (spacing / 2.0) if (i % 2) else 0.0
        for j in range(-cols, cols + 1):
            x = cu + offset + j * spacing
            if (x - cu) ** 2 + (y - cv) ** 2 <= zone_radius**2:
                pts.append((x, y))

    if len(pts) < n:
        raise ValueError(f"maille trop lâche : {len(pts)} < {n} centres dans la zone")
    pts.sort(key=lambda p: (p[0] - cu) ** 2 + (p[1] - cv) ** 2)
    return pts[:n]


__all__ = [
    "combined_max_directivity_dbi",
    "hex_centers_uv",
]
