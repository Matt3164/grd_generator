"""Synthèse pure des diagrammes d'éléments et optimisation de largeur (générateur).

Tout est calculé en espace angulaire (u, v) abstrait, sans géométrie terrestre.
Deux **modes de génération** enfichables (registre `_MODES`, défaut `gaussian`) :
`gaussian` (lisse, sans lobes) et `airy` (ouverture circulaire uniforme, lobes à
−17,6 dB et nulls francs — cf. `docs/biblio-patterns.md`). Ce module est
entièrement testé (100 %) ; le script d'écriture du .npz vit dans
`generate.py` (exclu de couverture).
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from grd_generator.logger import logger
from grd_generator.schemas import (
    ComplexField,
    DirectivityMap,
    GaussianSpec,
    UVGrid,
)

_FLOOR = 1e-12  # plancher de |E|² pour éviter log10(0) hors lobe

FloatArray = NDArray[np.float64]
# Un mode de génération : (spec, grille) → champ complexe E(u, v).
FieldGenerator = Callable[["GaussianSpec", UVGrid], ComplexField]


def gaussian_field(spec: GaussianSpec, grid: UVGrid) -> ComplexField:
    """Champ complexe E(u, v) d'un élément gaussien sur la grille.

    A = 10^(peak/20)·exp(−‖uv−c‖²/(2σ²)) ; φ = pente_radiale·‖uv−c‖ ; E = A·e^{jφ}.
    """
    gu, gv = grid.meshgrid()
    cu, cv = spec.center_uv
    du = gu - cu
    dv = gv - cv
    r2 = du**2 + dv**2
    amplitude = 10.0 ** (spec.peak_gain_dbi / 20.0) * np.exp(-r2 / (2.0 * spec.sigma**2))
    phase = spec.phase_slope_radial * np.sqrt(r2)
    field: ComplexField = (amplitude * np.exp(1j * phase)).astype(np.complex128)
    return field


def _bessel_j1(x: FloatArray) -> FloatArray:
    """J₁(x) par l'approximation rationnelle classique (A&S §9.4 / Num. Recipes).

    Vectorisée numpy, précision ~1e-7 — suffisant pour la synthèse, sans ajouter
    scipy au projet (décision de `docs/biblio-patterns.md`). Fonction impaire.
    """
    ax = np.abs(np.asarray(x, dtype=np.float64))
    small = ax < 8.0

    # |x| < 8 : fraction rationnelle en x².
    y = ax * ax
    p_near = ax * (
        72362614232.0
        + y
        * (
            -7895059235.0
            + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))
        )
    )
    q_near = 144725228442.0 + y * (
        2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y)))
    )
    near = p_near / q_near

    # |x| ≥ 8 : forme asymptotique amortie en 1/√x.
    safe_ax = np.maximum(ax, 8.0)  # n'est utilisé que là où small est faux
    z = 8.0 / safe_ax
    y2 = z * z
    xx = safe_ax - 2.356194491  # 3π/4
    p_far = 1.0 + y2 * (
        0.183105e-2 + y2 * (-0.3516396496e-4 + y2 * (0.2457520174e-5 + y2 * (-0.240337019e-6)))
    )
    q_far = 0.04687499995 + y2 * (
        -0.2002690873e-3 + y2 * (0.8449199096e-5 + y2 * (-0.88228987e-6 + y2 * 0.105787412e-6))
    )
    far = np.sqrt(0.636619772 / safe_ax) * (np.cos(xx) * p_far - z * np.sin(xx) * q_far)

    magnitude = np.where(small, near, far)
    result: FloatArray = (magnitude * np.sign(x)).astype(np.float64)
    return result


_J1_FIRST_NULL = 3.8317059702075125  # premier zéro de J₁ (rayon du 1er null d'Airy)


def airy_field(spec: GaussianSpec, grid: UVGrid) -> ComplexField:
    """Champ d'ouverture circulaire uniforme : profil jinc `2·J₁(x)/x`.

    `spec.sigma` est réinterprété comme le **rayon du premier null** (°) ;
    crête `peak_gain_dbi` au centre ; même rampe de phase radiale que la
    gaussienne. Contrairement à elle : **lobes secondaires à −17,6 dB et nulls
    francs** (l'amplitude change de signe) — la combinaison devient un vrai
    problème d'interférences (cf. `docs/biblio-patterns.md`).
    """
    gu, gv = grid.meshgrid()
    cu, cv = spec.center_uv
    r = np.hypot(gu - cu, gv - cv)
    x = _J1_FIRST_NULL * r / spec.sigma
    jinc = np.where(x == 0.0, 1.0, 2.0 * _bessel_j1(x) / np.where(x == 0.0, 1.0, x))
    amplitude = 10.0 ** (spec.peak_gain_dbi / 20.0) * jinc
    phase = spec.phase_slope_radial * r
    field: ComplexField = (amplitude * np.exp(1j * phase)).astype(np.complex128)
    return field


_MODES: dict[str, FieldGenerator] = {
    "gaussian": gaussian_field,
    "airy": airy_field,
}

_DEG_TO_RAD = np.pi / 180.0


def power_fraction(field: ComplexField, grid: UVGrid) -> float:
    """Fraction de la puissance totale rayonnée capturée par la grille.

    Un champ `.grd`/synthétique est E rapporté à la référence isotrope : |E|²
    est la **directivité linéaire** D(u, v) et la conservation de la puissance
    impose ∮_sphère D dΩ = 4π. La fraction (1/4π)·Σ |E|²·dΩ — avec
    dΩ ≈ (π/180)²·du·dv en **approximation petit angle** (champ de vue GEO
    ±9° : le jacobien complet 1/√(1−sin²θ) en dévie de < 1,3 %) — doit donc
    rester ≤ 1. Au-delà : l'élément « rayonne plus que sa puissance totale »
    (non physique) ; très en deçà : le lobe est largement hors grille.
    """
    u, v = grid.axes()
    cell_sr = (float(u[1] - u[0]) * _DEG_TO_RAD) * (float(v[1] - v[0]) * _DEG_TO_RAD)
    return float((np.abs(field) ** 2).sum()) * cell_sr / (4.0 * np.pi)


def _gaussian_max_peak_dbi(sigma_deg: float) -> float:
    """Borne gaussienne : |E|² = exp(−r²/σ²) ⇒ Ω = π·σ_rad² ⇒ crête ≤ 4/σ_rad²."""
    return float(10.0 * np.log10(4.0 / (sigma_deg * _DEG_TO_RAD) ** 2))


def _airy_max_peak_dbi(sigma_deg: float) -> float:
    """Borne airy : ouverture circulaire uniforme, 1ᵉʳ null θ₀ = 3,83/(πD/λ)
    ⇒ crête = (πD/λ)² = (3,8317/θ₀_rad)²."""
    return float(20.0 * np.log10(_J1_FIRST_NULL / (sigma_deg * _DEG_TO_RAD)))


# Chaque mode du registre déclare sa crête maximale physique (conservation de
# la puissance) — un nouveau mode DOIT s'enregistrer ici aussi (test verrou).
_MODE_PEAK_BOUNDS: dict[str, Callable[[float], float]] = {
    "gaussian": _gaussian_max_peak_dbi,
    "airy": _airy_max_peak_dbi,
}


def max_peak_dbi(mode: str, sigma: float) -> float:
    """Crête maximale physique (dBi) du mode pour la largeur `sigma` (en degrés).

    Découle de D_crête·Ω_lobe ≈ 4π (conservation de la puissance) — au-delà,
    le champ généré rayonnerait plus que sa puissance totale.
    """
    if mode not in _MODE_PEAK_BOUNDS:
        raise ValueError(f"mode de génération inconnu : {mode!r} (connus : {available_modes()})")
    return _MODE_PEAK_BOUNDS[mode](sigma)


def check_power_conservation(mode: str, sigma: float, peak_gain_dbi: float) -> bool:
    """Garde de génération : la crête demandée respecte-t-elle la borne du mode ?

    Comportement par défaut **en attente d'arbitrage** (arb-power-policy au
    BACKLOG : normaliser, plafonner ou avertir ?) : on signale par un warning
    structuré et on laisse générer — aucun changement de comportement tant que
    Matthieu n'a pas tranché.
    """
    bound = max_peak_dbi(mode, sigma)
    compliant = peak_gain_dbi <= bound
    if not compliant:
        logger.warning(
            "generation_peak_exceeds_power_bound",
            mode=mode,
            sigma_deg=round(sigma, 4),
            peak_gain_dbi=peak_gain_dbi,
            max_peak_dbi=round(bound, 2),
        )
    return compliant


def available_modes() -> list[str]:
    """Modes de génération disponibles (registre, défaut `gaussian`)."""
    return sorted(_MODES)


def field_generator(mode: str) -> FieldGenerator:
    """Résout un mode de génération, ou `ValueError` explicite."""
    if mode not in _MODES:
        raise ValueError(f"mode de génération inconnu : {mode!r} (connus : {available_modes()})")
    return _MODES[mode]


def directivity_dbi_from_field(field: ComplexField) -> DirectivityMap:
    """Directivité |E|² d'un champ, en dBi, avec plancher anti-log(0)."""
    power = np.abs(field) ** 2
    dbi: DirectivityMap = (10.0 * np.log10(np.maximum(power, _FLOOR))).astype(np.float64)
    return dbi


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


def max_envelope(
    specs: list[GaussianSpec], grid: UVGrid, *, mode: str = "gaussian"
) -> DirectivityMap:
    """Enveloppe de directivité **max atteignable après combinaison**, point par point.

    Cf. `combined_max_directivity_dbi` : `10·log10(Σ|Eᵢ|²)`, le plafond qu'un
    faisceau formé peut atteindre en chaque point (et non le max élément seul).
    `mode` sélectionne le générateur de champ (registre, défaut `gaussian`).
    """
    generate = field_generator(mode)
    return combined_max_directivity_dbi([generate(s, grid) for s in specs])


def envelope_min_dbi(
    specs: list[GaussianSpec],
    grid: UVGrid,
    zone_center: tuple[float, float],
    zone_radius: float,
    *,
    mode: str = "gaussian",
) -> float:
    """Directivité d'enveloppe la plus faible à l'intérieur du disque de service."""
    gu, gv = grid.meshgrid()
    inside = (gu - zone_center[0]) ** 2 + (gv - zone_center[1]) ** 2 <= zone_radius**2
    envelope = max_envelope(specs, grid, mode=mode)
    return float(envelope[inside].min())


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


def optimize_sigma(
    grid: UVGrid,
    centers: list[tuple[float, float]],
    peak_gain_dbi: float,
    zone_center: tuple[float, float],
    zone_radius: float,
    target_dbi: float = 45.0,
    sigma_lo: float = 1e-3,
    sigma_hi: float = 5.0,
    tol: float = 1e-3,
    phase_slope_radial: float = 0.0,
    *,
    mode: str = "gaussian",
) -> tuple[float, list[GaussianSpec]]:
    """Plus petite largeur globale telle que l'enveloppe ≥ `target_dbi` en zone.

    Pour `gaussian`, σ est l'écart-type ; pour `airy`, le rayon du premier null.
    L'enveloppe minimale sur la zone croît avec σ ; on cherche le σ frontière par
    bissection entre `sigma_lo` (lobes serrés) et `sigma_hi` (lobes larges).
    ⚠️ Avec un mode à nulls (`airy`), la monotonie n'est plus garantie : le σ
    retourné reste **toujours faisable** (invariant de la bissection — `hi` ne
    prend que des valeurs faisables), mais peut ne pas être minimal
    (cf. caveat de `docs/biblio-patterns.md`).
    `phase_slope_radial` (rad/°) est la pente de phase **radiale** commune à tous les
    éléments (φ = pente·‖uv−centre‖, courbure par élément) : elle n'affecte ni `|E|`
    ni donc l'optimisation, mais elle est portée par les specs retournées — d'où des
    poids complexes et des **ripples** d'interférence dans le faisceau combiné.
    Renvoie (σ, specs) au σ retenu. Lève ValueError si même `sigma_hi` ne satisfait
    pas la contrainte.
    """

    def specs_for(sigma: float) -> list[GaussianSpec]:
        return [
            GaussianSpec(
                center_uv=c,
                sigma=sigma,
                peak_gain_dbi=peak_gain_dbi,
                phase_slope_radial=phase_slope_radial,
            )
            for c in centers
        ]

    def feasible(sigma: float) -> bool:
        margin = envelope_min_dbi(specs_for(sigma), grid, zone_center, zone_radius, mode=mode)
        return margin >= target_dbi

    if not feasible(sigma_hi):
        raise ValueError(
            f"contrainte infaisable : σ_hi={sigma_hi} donne une enveloppe "
            f"< {target_dbi} dBi ; élargir sigma_hi ou augmenter peak_gain_dbi"
        )

    lo, hi = sigma_lo, sigma_hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return hi, specs_for(hi)


__all__ = [
    "gaussian_field",
    "airy_field",
    "available_modes",
    "field_generator",
    "FieldGenerator",
    "power_fraction",
    "max_peak_dbi",
    "check_power_conservation",
    "directivity_dbi_from_field",
    "combined_max_directivity_dbi",
    "max_envelope",
    "envelope_min_dbi",
    "hex_centers_uv",
    "optimize_sigma",
]
