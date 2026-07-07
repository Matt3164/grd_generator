"""Assemblage AFR : feeds → champs secondaires complexes (co + cross) sur (u,v)."""

import numpy as np

from grd_generator.reflector import farfield, optics
from grd_generator.reflector.spec import FeedSpec, ReflectorSpec
from grd_generator.schemas import ComplexField, UVGrid
from grd_generator.synth import hex_centers_uv

# Décalage de seed pour l'écran de phase COMMUN (`phase_error_shared_rms_rad`),
# ajouté à `phase_error_seed` plutôt que soustrait : `np.random.default_rng`
# rejette les seeds négatifs, et `phase_error_seed` vaut 0 par défaut (donc
# `seed - 1` casserait le cas par défaut). Un grand décalage fixe reste
# distinct de tous les seeds par-feed `phase_error_seed + i` pour tout nombre
# de feeds réaliste (i < _SHARED_SEED_OFFSET).
_SHARED_SEED_OFFSET = 1_000_003


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


def synthesize_reflector_fields(
    spec: ReflectorSpec,
    feeds: FeedSpec,
    grid: UVGrid,
    *,
    n_aperture: int = 128,
    pad_factor: int = 4,
) -> tuple[list[ComplexField], list[ComplexField]]:
    """Pour chaque feed : champ co-pol et cross-pol (u,v), normalisés en directivité.

    Si `feeds.phase_error_rms_rad > 0`, un écran de phase aléatoire corrélé
    (`optics.random_phase_screen`) est construit par feed avec un RNG dédié
    `np.random.default_rng(feeds.phase_error_seed + i)` (i = indice du feed
    dans `positions_m`) et ajouté à la phase d'ouverture — déterministe :
    mêmes spec/feeds → mêmes champs. Si `phase_error_rms_rad == 0`, aucun RNG
    n'est instancié et le comportement est strictement identique à l'absence
    d'écran.

    Si `feeds.phase_error_shared_rms_rad > 0`, un second écran, COMMUN à tous
    les feeds (erreurs de surface du réflecteur, par opposition aux erreurs
    propres au feed ci-dessus), est construit une seule fois avec
    `np.random.default_rng(feeds.phase_error_seed + _SHARED_SEED_OFFSET)`
    (seed distinct de tous les seeds par-feed `phase_error_seed + i`, i >= 0
    — voir commentaire module) et ajouté à l'écran de chaque feed (les deux
    écrans se cumulent par simple somme). Si les deux rms sont nuls, aucun
    RNG n'est instancié et le comportement est strictement identique à
    l'absence d'écran.
    """
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture, pad_factor)
    ex, ey = optics.aperture_pol_vectors(spec, X, Y)
    co_fields: list[ComplexField] = []
    cross_fields: list[ComplexField] = []

    shared_screen = None
    if feeds.phase_error_shared_rms_rad > 0.0:
        shared_rng = np.random.default_rng(feeds.phase_error_seed + _SHARED_SEED_OFFSET)
        shared_screen = optics.random_phase_screen(
            X,
            Y,
            inside,
            dx,
            rms_rad=feeds.phase_error_shared_rms_rad,
            corr_length_m=feeds.phase_corr_length_m,
            rng=shared_rng,
        )

    for i, feed_xy in enumerate(feeds.positions_m):
        extra_phase = None
        if feeds.phase_error_rms_rad > 0.0:
            rng = np.random.default_rng(feeds.phase_error_seed + i)
            extra_phase = optics.random_phase_screen(
                X,
                Y,
                inside,
                dx,
                rms_rad=feeds.phase_error_rms_rad,
                corr_length_m=feeds.phase_corr_length_m,
                rng=rng,
            )
        if shared_screen is not None:
            extra_phase = shared_screen if extra_phase is None else extra_phase + shared_screen
        scalar = optics.aperture_field(
            spec,
            feed_xy,
            X,
            Y,
            inside,
            feeds.q,
            defocus_m=feeds.defocus_m,
            extra_phase=extra_phase,
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
