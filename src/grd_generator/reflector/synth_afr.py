"""Assemblage AFR : feeds → champs secondaires complexes (co + cross) sur (u,v)."""

import numpy as np
from numpy.typing import NDArray

from grd_generator.reflector import farfield, optics
from grd_generator.reflector.spec import FeedSpec, ReflectorSpec
from grd_generator.schemas import ComplexField, UVGrid
from grd_generator.synth import hex_centers_uv

FloatArray = NDArray[np.float64]


def hex_feed_positions(
    pitch_m: float, n_feeds: int, focal_radius_m: float
) -> list[tuple[float, float]]:
    """Lattice hexagonal de positions de feeds (m) dans le plan focal, centré."""
    return hex_centers_uv((0.0, 0.0), focal_radius_m, n_feeds, pitch_m)


def form_beam(
    co_fields: list[ComplexField], grid: UVGrid, target_uv: tuple[float, float]
) -> ComplexField:
    """Beam formé par filtre adapté (conjugate beamforming) vers `target_uv` (deg).

    Poids `wᵢ = conj(Eᵢ(cible))` normalisés en norme L2 unité ; champ combiné
    `B(u,v) = Σᵢ wᵢ·Eᵢ(u,v)`. Par Cauchy-Schwarz, `|B(cible)|² = Σᵢ|Eᵢ(cible)|²`,
    donc la crête du beam touche l'enveloppe max co-pol au point visé. La cible
    est projetée sur la cellule de grille la plus proche.
    """
    stack = np.stack(co_fields)  # (n_feeds, n_v, n_u)
    u_axis, v_axis = grid.axes()
    u0, v0 = target_uv
    iu = int(np.argmin(np.abs(u_axis - u0)))
    iv = int(np.argmin(np.abs(v_axis - v0)))
    samples = stack[:, iv, iu]  # Eᵢ(cible), forme (n_feeds,)
    norm = float(np.sqrt(np.sum(np.abs(samples) ** 2)))
    if norm == 0.0:
        return np.zeros_like(stack[0])
    weights = np.conj(samples) / norm
    beam: ComplexField = np.tensordot(weights, stack, axes=([0], [0])).astype(np.complex128)
    return beam


def dereference_phase(
    field: ComplexField, grid: UVGrid, wavelength_m: float, aperture_center_y_m: float
) -> FloatArray:
    """Phase de `field` dé-référencée de la porteuse due au centre d'ouverture décalé.

    Un centre d'ouverture excentré en Y₀ = `aperture_center_y_m` (offset) impose au
    champ lointain une porteuse de phase quasi-linéaire en v :
    `exp(-i·k·Y₀·sin v)` (k = 2π/λ) — signe hérité de la convention de
    `numpy.fft.fft2` utilisée par `farfield.far_field_fft` (TF directe en
    exp(-i·2π·f·x) : un décalage +Y₀ de l'ouverture multiplie son spectre par
    exp(-i·2π·f·Y₀) = exp(-i·k·Y₀·sin v)). Sur une grille d'affichage grossière,
    cette porteuse se replie visuellement en rayures horizontales (moiré) qui
    masquent la structure de phase utile. Cette fonction ne change PAS la
    physique : elle retire cette porteuse *pour l'affichage seul*, en renvoyant
    `angle(field · exp(i·k·Y₀·sin v))` (conjugué de la porteuse, pour
    l'annuler). Les champs utilisés pour le beamforming (`form_beam`) et les
    exports GRD doivent rester les champs bruts, pas ce résultat.
    """
    _, gv = grid.meshgrid()
    k = 2.0 * np.pi / wavelength_m
    m = np.sin(np.deg2rad(gv))
    carrier = np.exp(1j * k * aperture_center_y_m * m)
    dereferenced: FloatArray = np.angle(field * carrier)
    return dereferenced


def synthesize_reflector_fields(
    spec: ReflectorSpec,
    feeds: FeedSpec,
    grid: UVGrid,
    *,
    n_aperture: int = 128,
    pad_factor: int = 4,
) -> tuple[list[ComplexField], list[ComplexField]]:
    """Pour chaque feed : champ co-pol et cross-pol (u,v), normalisés en directivité.

    `n_aperture` fixe le pas physique de la grille `dx = diameter_m /
    n_aperture` (voir `optics.aperture_grid`) ; `pad_factor` contrôle le
    zéro-padding, donc la résolution angulaire du far-field.
    """
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture, pad_factor)
    ex, ey = optics.aperture_pol_vectors(spec, X, Y)
    co_fields: list[ComplexField] = []
    cross_fields: list[ComplexField] = []

    for feed_xy in feeds.positions_m:
        scalar = optics.aperture_field(
            spec,
            feed_xy,
            X,
            Y,
            inside,
            feeds.q,
            defocus_m=feeds.defocus_m,
        )
        Fx, L, M = farfield.far_field_fft(scalar * ex, dx, spec.wavelength_m)
        Fy, _, _ = farfield.far_field_fft(scalar * ey, dx, spec.wavelength_m)
        e_co, e_cross = farfield.ludwig3_co_cross(Fx, Fy, L, M)
        dL = float(L[0, 1] - L[0, 0])
        dM = float(M[1, 0] - M[0, 0])
        norm = farfield.normalize_to_directivity(
            np.abs(e_co) ** 2 + np.abs(e_cross) ** 2, L, M, dL, dM
        )
        lin = L[0, :]
        co_fields.append(farfield.resample_to_uvgrid(e_co * norm, lin, grid))
        cross_fields.append(farfield.resample_to_uvgrid(e_cross * norm, lin, grid))
    return co_fields, cross_fields
