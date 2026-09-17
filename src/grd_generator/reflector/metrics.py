"""Métriques de calibrage du jeu Ku (lot B3) — module pur, sans E/S.

Appliqué par `batch.run_batch` aux champs qu'il écrit (→ `meta/metrics.json`) et
par `examples/example_batch_ku.py` au balayage de calibrage. Voir
`docs/superpowers/specs/2026-09-17-batch-ku-hdf5-design.md`, sections
« Références chiffrées… » (tableau des critères) et « Métriques et calibrage ».

Convention d'angle : la grille `UVGrid` est en cosinus directeurs
(u, v) = (sinθcosφ, sinθsinφ), donc `u² + v² = sin²θ` **quel que soit
l'azimut φ** — le rayon angulaire depuis le nadir d'un nœud de grille est
exactement `θ = asin(√(u²+v²))`, sans approximation petits angles.
"""

import math

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from grd_generator.schemas import ComplexField, DirectivityMap, UVGrid
from grd_generator.synth import combined_max_directivity_dbi

_FLOOR = 1e-12


def dbi_map(field: ComplexField) -> DirectivityMap:
    """Carte de directivité (dBi) d'un champ complexe unique (`10·log10(|E|²)`)."""
    dbi: DirectivityMap = (10.0 * np.log10(np.maximum(np.abs(field) ** 2, _FLOOR))).astype(
        np.float64
    )
    return dbi


def radius_deg(grid: UVGrid) -> NDArray[np.float64]:
    """Rayon angulaire (deg) depuis le nadir de chaque nœud de `grid`.

    `θ = asin(√(u²+v²))` — exact (voir docstring du module), pas une
    approximation petits angles.
    """
    gu, gv = grid.meshgrid()
    r_uv = np.clip(np.hypot(gu, gv), 0.0, 1.0)
    out: NDArray[np.float64] = np.degrees(np.arcsin(r_uv))
    return out


def peak_dbi(field: ComplexField) -> float:
    """Crête (dBi) d'un champ complexe."""
    return float(dbi_map(field).max())


def peak_radius_deg(field: ComplexField, grid: UVGrid) -> float:
    """Rayon angulaire (deg) depuis le nadir de la crête d'un champ complexe."""
    dbi = dbi_map(field)
    r = radius_deg(grid)
    idx = np.unravel_index(np.argmax(dbi), dbi.shape)
    return float(r[idx])


def solid_angle_sr(mask: NDArray[np.bool_], grid: UVGrid) -> float:
    """Angle solide (sr) couvert par les nœuds `mask` de `grid`.

    Élément d'angle solide exact `dΩ = du·dv / cosθ` (même formule que
    `farfield.normalize_to_directivity`), sommé sur les nœuds retenus.
    """
    gu, gv = grid.meshgrid()
    cos_theta = np.sqrt(np.clip(1.0 - gu**2 - gv**2, _FLOOR, None))
    u_axis, v_axis = grid.axes()
    du = float(u_axis[1] - u_axis[0])
    dv = float(v_axis[1] - v_axis[0])
    return float(np.sum(du * dv / cos_theta[mask]))


def equivalent_diameter_deg(omega_sr: float) -> float:
    """Diamètre (deg) de la calotte sphérique de même angle solide `omega_sr`.

    `Ω = 2π(1 − cosα)` pour une calotte de demi-angle `α`, **quel que soit
    son centrage sur la sphère** (symétrie de rotation) : la conversion est
    donc exacte comme définition de « diamètre équivalent » d'une région
    d'angle solide `omega_sr` — que cette région soit elle-même une calotte
    exacte (un lobe centré au nadir, où le cercle en (u,v) est bien une
    calotte) ou non (un lobe dépointé, où la conversion reste un diamètre
    équivalent au sens de l'aire, pas une calotte géodésique réelle).
    """
    cos_alpha = np.clip(1.0 - omega_sr / (2.0 * math.pi), -1.0, 1.0)
    return float(2.0 * math.degrees(math.acos(cos_alpha)))


def width_3db_deg(field: ComplexField, grid: UVGrid) -> float:
    """Largeur à −3 dB (diamètre équivalent, deg) d'un champ complexe."""
    dbi = dbi_map(field)
    peak = float(dbi.max())
    mask = dbi >= (peak - 3.0)
    return equivalent_diameter_deg(solid_angle_sr(mask, grid))


def cross_pol_isolation_db(co: ComplexField, cross: ComplexField) -> float:
    """Écart co/cross (dB) **à la crête co-polaire** (pas à la crête cross)."""
    dbi_co = dbi_map(co)
    idx = np.unravel_index(np.argmax(dbi_co), dbi_co.shape)
    dbi_cross = dbi_map(cross)
    return float(dbi_co[idx] - dbi_cross[idx])


def guaranteed_radius_deg(dbi: DirectivityMap, grid: UVGrid, threshold_dbi: float) -> float:
    """Plus grand rayon (deg) autour du nadir tel que **tout** nœud à ce rayon
    ou moins reste `>= threshold_dbi`.

    Calculé comme le rayon du nœud en échec le plus proche du nadir (aucun
    nœud plus proche ne peut alors être garanti à l'intérieur d'un disque plus
    grand) ; si aucun nœud n'échoue, renvoie le rayon maximal couvert par la
    grille (toute la grille tient le seuil).
    """
    r = radius_deg(grid)
    failing = dbi < threshold_dbi
    if not np.any(failing):
        return float(r.max())
    return float(r[failing].min())


class FeedMetrics(BaseModel):
    """Métriques d'un feed unique, à une fréquence donnée."""

    feed: int
    peak_dbi: float
    peak_radius_deg: float
    width_3db_deg: float
    cross_pol_isolation_db: float


class EnvelopeMetrics(BaseModel):
    """Métriques de l'enveloppe (`Σ|co|²` sur les feeds), à une fréquence donnée."""

    max_dbi: float
    radius_1db_deg: float
    radius_3db_deg: float
    radius_39dbi_deg: float


class FrequencyMetrics(BaseModel):
    """Métriques complètes (feeds + enveloppe) à une fréquence donnée."""

    freq_mhz: int
    feeds: list[FeedMetrics]
    envelope: EnvelopeMetrics
    peak_dbi_median: float
    width_3db_deg_median: float
    peak_radius_deg_max: float

    def passes_blocking_criterion(self, *, min_radius_3db_deg: float = 4.0) -> bool:
        """Critère bloquant de la spec : rayon d'enveloppe à −3 dB ≥ 4,0°."""
        return self.envelope.radius_3db_deg >= min_radius_3db_deg


def compute_frequency_metrics(
    freq_mhz: int,
    co_fields: list[ComplexField],
    cross_fields: list[ComplexField],
    grid: UVGrid,
) -> FrequencyMetrics:
    """Calcule `FrequencyMetrics` pour un jeu de champs co/cross (une fréquence)."""
    feeds = [
        FeedMetrics(
            feed=i + 1,
            peak_dbi=peak_dbi(co),
            peak_radius_deg=peak_radius_deg(co, grid),
            width_3db_deg=width_3db_deg(co, grid),
            cross_pol_isolation_db=cross_pol_isolation_db(co, cross),
        )
        for i, (co, cross) in enumerate(zip(co_fields, cross_fields, strict=True))
    ]
    envelope_dbi = combined_max_directivity_dbi(co_fields)
    max_dbi = float(envelope_dbi.max())
    envelope = EnvelopeMetrics(
        max_dbi=max_dbi,
        radius_1db_deg=guaranteed_radius_deg(envelope_dbi, grid, max_dbi - 1.0),
        radius_3db_deg=guaranteed_radius_deg(envelope_dbi, grid, max_dbi - 3.0),
        radius_39dbi_deg=guaranteed_radius_deg(envelope_dbi, grid, 39.0),
    )
    return FrequencyMetrics(
        freq_mhz=freq_mhz,
        feeds=feeds,
        envelope=envelope,
        peak_dbi_median=float(np.median([f.peak_dbi for f in feeds])),
        width_3db_deg_median=float(np.median([f.width_3db_deg for f in feeds])),
        peak_radius_deg_max=float(max(f.peak_radius_deg for f in feeds)),
    )


__all__ = [
    "EnvelopeMetrics",
    "FeedMetrics",
    "FrequencyMetrics",
    "compute_frequency_metrics",
    "cross_pol_isolation_db",
    "dbi_map",
    "equivalent_diameter_deg",
    "guaranteed_radius_deg",
    "peak_dbi",
    "peak_radius_deg",
    "radius_deg",
    "solid_angle_sr",
    "width_3db_deg",
]
