import numpy as np
import pytest

from grd_generator.reflector import (
    FeedSpec,
    ReflectorSpec,
    hex_feed_positions,
    synthesize_reflector_fields,
)
from grd_generator.schemas import UVGrid


def _grid() -> UVGrid:
    return UVGrid(u_min=-14, u_max=14, v_min=-14, v_max=14, n_u=81, n_v=81)


def test_centered_feed_beam_points_at_boresight() -> None:
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.0, freq_hz=20e9)
    feeds = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0)
    co, cross = synthesize_reflector_fields(spec, feeds, _grid(), n_aperture=64, pad_factor=4)
    assert len(co) == 1 and len(cross) == 1
    dbi = 10.0 * np.log10(np.abs(co[0]) ** 2)
    iv, iu = np.unravel_index(np.argmax(dbi), dbi.shape)
    gu, gv = _grid().meshgrid()
    assert abs(gu[iv, iu]) < 1.0 and abs(gv[iv, iu]) < 1.0  # peak near (0,0)


def test_displaced_feed_beam_scans() -> None:
    # A feed displaced +x in the focal plane scans the beam off boresight.
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.0, freq_hz=20e9)
    feeds = FeedSpec(positions_m=[(0.05, 0.0)], q=2.0)
    co, _ = synthesize_reflector_fields(spec, feeds, _grid(), n_aperture=64, pad_factor=4)
    dbi = 10.0 * np.log10(np.abs(co[0]) ** 2)
    iv, iu = np.unravel_index(np.argmax(dbi), dbi.shape)
    gu, _ = _grid().meshgrid()
    assert abs(gu[iv, iu]) > 1.0  # beam moved away from boresight in u


def test_hex_feed_positions_count_and_centering() -> None:
    pos = hex_feed_positions(pitch_m=0.03, n_feeds=7, focal_radius_m=0.2)
    assert len(pos) == 7
    assert pos[0] == pytest.approx((0.0, 0.0))  # closest to center first


def test_axial_defocus_lowers_peak_and_widens_main_lobe() -> None:
    # D=2m, 20 GHz, feed central seul : le defocus_m ci-dessous introduit
    # plusieurs longueurs d'onde de différence de marche au bord de
    # l'ouverture (voir exploration numérique en tâche), donc une chute nette
    # de crête et un étalement net du lobe principal.
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.0, freq_hz=20e9)
    grid = _grid()

    feeds_focused = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0)
    co_focused, _ = synthesize_reflector_fields(
        spec, feeds_focused, grid, n_aperture=64, pad_factor=4
    )
    dbi_focused = 10.0 * np.log10(np.maximum(np.abs(co_focused[0]) ** 2, 1e-12))
    peak_focused = float(dbi_focused.max())

    feeds_defocused = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0, defocus_m=0.15)
    co_defocused, _ = synthesize_reflector_fields(
        spec, feeds_defocused, grid, n_aperture=64, pad_factor=4
    )
    dbi_defocused = 10.0 * np.log10(np.maximum(np.abs(co_defocused[0]) ** 2, 1e-12))
    peak_defocused = float(dbi_defocused.max())

    # Le champ défocalisé n'est pas dégénéré/nul (évite le faux-positif d'un
    # bug qui mettrait tout à zéro et ferait "baisser la crête" trivialement).
    assert np.any(np.abs(co_defocused[0]) > 1e-6)

    assert peak_defocused < peak_focused - 3.0

    # Étalement : nombre de points au-dessus de crête-3dB, plus grand quand
    # défocalisé (le lobe principal s'élargit).
    n_above_focused = int(np.sum(dbi_focused > peak_focused - 3.0))
    n_above_defocused = int(np.sum(dbi_defocused > peak_defocused - 3.0))
    assert n_above_defocused > n_above_focused


def test_axial_defocus_offset_aperture_does_not_squint() -> None:
    # Régression : sur une ouverture fortement offset (petit réflecteur,
    # large clearance), le terme de defocus dépointait numériquement le
    # faisceau (jusqu'à +11.5° mesurés à δz=0.12 avant correctif) au lieu de
    # seulement l'étaler — voir docstring du module optics.aperture_field.
    # Le pic doit rester au pointage nominal quel que soit δz, avec une
    # crête qui baisse (l'étalement, lui, demeure).
    spec = ReflectorSpec(
        diameter_m=0.28, focal_length_m=0.336, offset_clearance_m=0.15, freq_hz=20e9
    )
    grid = UVGrid(u_min=-30, u_max=30, v_min=-30, v_max=30, n_u=121, n_v=121)
    du = (grid.u_max - grid.u_min) / (grid.n_u - 1)
    dv = (grid.v_max - grid.v_min) / (grid.n_v - 1)
    gu, gv = grid.meshgrid()

    def _peak(defocus_m: float) -> tuple[float, float, float]:
        feeds = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0, defocus_m=defocus_m)
        co, _ = synthesize_reflector_fields(spec, feeds, grid, n_aperture=64, pad_factor=4)
        dbi = 10.0 * np.log10(np.maximum(np.abs(co[0]) ** 2, 1e-12))
        iv, iu = np.unravel_index(np.argmax(dbi), dbi.shape)
        return float(dbi[iv, iu]), float(gu[iv, iu]), float(gv[iv, iu])

    peak_focused, u0, v0 = _peak(0.0)
    peak_defocused, u1, v1 = _peak(0.12)

    # Le pic reste à moins d'un pas de grille du pic sans defocus (pas de
    # dépointage introduit par le terme de defocus).
    assert abs(u1 - u0) <= du + 1e-9
    assert abs(v1 - v0) <= dv + 1e-9

    # L'étalement demeure : la crête baisse toujours avec le defocus.
    assert peak_defocused < peak_focused


def test_phase_error_rms_zero_leaves_fields_unchanged() -> None:
    # rms=0 (défaut) : le champ doit être strictement identique à l'existant,
    # que phase_error_rms_rad soit omis ou explicitement à 0.
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.0, freq_hz=20e9)
    grid = _grid()
    feeds_default = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0)
    feeds_explicit_zero = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0, phase_error_rms_rad=0.0)
    co_default, cross_default = synthesize_reflector_fields(
        spec, feeds_default, grid, n_aperture=64, pad_factor=4
    )
    co_zero, cross_zero = synthesize_reflector_fields(
        spec, feeds_explicit_zero, grid, n_aperture=64, pad_factor=4
    )
    np.testing.assert_array_equal(co_default[0], co_zero[0])
    np.testing.assert_array_equal(cross_default[0], cross_zero[0])


def test_phase_error_is_deterministic_given_same_spec_and_seed() -> None:
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.0, freq_hz=20e9)
    grid = _grid()
    feeds = FeedSpec(
        positions_m=[(0.0, 0.0)],
        q=2.0,
        phase_error_rms_rad=0.5,
        phase_corr_length_m=0.05,
        phase_error_seed=3,
    )
    co_a, _ = synthesize_reflector_fields(spec, feeds, grid, n_aperture=64, pad_factor=4)
    co_b, _ = synthesize_reflector_fields(spec, feeds, grid, n_aperture=64, pad_factor=4)
    np.testing.assert_array_equal(co_a[0], co_b[0])


def test_phase_error_differs_between_feeds() -> None:
    # Deux feeds différents doivent recevoir des écrans de phase différents
    # (RNG seedé par index de feed) : les champs par feed ne peuvent donc pas
    # être identiques au-delà du tilt commun (positions de feed différentes,
    # donc le tilt diffère déjà, mais on vérifie surtout que l'ajout d'un
    # bruit indépendant n'est pas trivialement partagé).
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.0, freq_hz=20e9)
    grid = _grid()
    feeds = FeedSpec(
        positions_m=[(0.0, 0.0), (0.0, 0.0)],  # même position -> seul le bruit diffère
        q=2.0,
        phase_error_rms_rad=0.5,
        phase_corr_length_m=0.05,
        phase_error_seed=0,
    )
    co, _ = synthesize_reflector_fields(spec, feeds, grid, n_aperture=64, pad_factor=4)
    assert not np.allclose(co[0], co[1])


def test_phase_error_lowers_peak_and_raises_sidelobe_floor() -> None:
    # D=0.28m, F/D=1.2, clearance=0.15m, 20 GHz, feed central : avec un bruit
    # de phase de rms ~1 rad et une corrélation courte (D/10), la crête doit
    # baisser nettement et le niveau max hors lobe principal (au-delà de deux
    # largeurs de lobe environ) doit remonter nettement (voir tâche).
    spec = ReflectorSpec(
        diameter_m=0.28, focal_length_m=0.336, offset_clearance_m=0.15, freq_hz=20e9
    )
    grid = UVGrid(u_min=-30, u_max=30, v_min=-30, v_max=30, n_u=121, n_v=121)
    gu, gv = grid.meshgrid()

    feeds_clean = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0)
    co_clean, _ = synthesize_reflector_fields(spec, feeds_clean, grid, n_aperture=64, pad_factor=4)
    dbi_clean = 10.0 * np.log10(np.maximum(np.abs(co_clean[0]) ** 2, 1e-12))
    peak_clean = float(dbi_clean.max())

    feeds_noisy = FeedSpec(
        positions_m=[(0.0, 0.0)],
        q=2.0,
        phase_error_rms_rad=1.0,
        phase_corr_length_m=spec.diameter_m / 10.0,
        phase_error_seed=0,
    )
    co_noisy, _ = synthesize_reflector_fields(spec, feeds_noisy, grid, n_aperture=64, pad_factor=4)
    dbi_noisy = 10.0 * np.log10(np.maximum(np.abs(co_noisy[0]) ** 2, 1e-12))
    peak_noisy = float(dbi_noisy.max())

    assert peak_noisy < peak_clean - 2.0

    # Masque "hors lobe principal" : au-delà d'environ 2 largeurs de lobe
    # (approx. λ/D en degrés) du pointage nominal (0,0).
    lobe_width_deg = np.degrees(spec.wavelength_m / spec.diameter_m)
    outside_mask = np.hypot(gu, gv) > 2.0 * lobe_width_deg
    floor_clean = float(dbi_clean[outside_mask].max())
    floor_noisy = float(dbi_noisy[outside_mask].max())
    assert floor_noisy > floor_clean + 5.0


def test_shared_phase_error_zero_leaves_fields_unchanged() -> None:
    # phase_error_shared_rms_rad=0 (défaut) : le champ doit être strictement
    # identique à l'existant, écran par-feed actif ou non.
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.0, freq_hz=20e9)
    grid = _grid()
    feeds_no_shared = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0, phase_error_rms_rad=0.4)
    feeds_explicit_zero_shared = FeedSpec(
        positions_m=[(0.0, 0.0)],
        q=2.0,
        phase_error_rms_rad=0.4,
        phase_error_shared_rms_rad=0.0,
    )
    co_a, cross_a = synthesize_reflector_fields(
        spec, feeds_no_shared, grid, n_aperture=64, pad_factor=4
    )
    co_b, cross_b = synthesize_reflector_fields(
        spec, feeds_explicit_zero_shared, grid, n_aperture=64, pad_factor=4
    )
    np.testing.assert_array_equal(co_a[0], co_b[0])
    np.testing.assert_array_equal(cross_a[0], cross_b[0])


def test_shared_and_individual_phase_error_combine_and_differ_from_each_alone() -> None:
    # Le cumul (shared + individuel) doit différer de chacun pris seul : le
    # champ résultant n'est ni celui de l'écran individuel seul, ni celui de
    # l'écran commun seul.
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.0, freq_hz=20e9)
    grid = _grid()

    feeds_individual_only = FeedSpec(
        positions_m=[(0.0, 0.0)], q=2.0, phase_error_rms_rad=0.4, phase_error_seed=0
    )
    feeds_shared_only = FeedSpec(
        positions_m=[(0.0, 0.0)], q=2.0, phase_error_shared_rms_rad=0.4, phase_error_seed=0
    )
    feeds_both = FeedSpec(
        positions_m=[(0.0, 0.0)],
        q=2.0,
        phase_error_rms_rad=0.4,
        phase_error_shared_rms_rad=0.4,
        phase_error_seed=0,
    )

    co_individual, _ = synthesize_reflector_fields(
        spec, feeds_individual_only, grid, n_aperture=64, pad_factor=4
    )
    co_shared, _ = synthesize_reflector_fields(
        spec, feeds_shared_only, grid, n_aperture=64, pad_factor=4
    )
    co_both, _ = synthesize_reflector_fields(spec, feeds_both, grid, n_aperture=64, pad_factor=4)

    assert not np.allclose(co_both[0], co_individual[0])
    assert not np.allclose(co_both[0], co_shared[0])


def test_shared_phase_error_is_deterministic_given_same_spec_and_seed() -> None:
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.0, freq_hz=20e9)
    grid = _grid()
    feeds = FeedSpec(
        positions_m=[(0.0, 0.0), (0.01, 0.0)],
        q=2.0,
        phase_error_rms_rad=0.3,
        phase_error_shared_rms_rad=0.6,
        phase_corr_length_m=0.04,
        phase_error_seed=5,
    )
    co_a, cross_a = synthesize_reflector_fields(spec, feeds, grid, n_aperture=64, pad_factor=4)
    co_b, cross_b = synthesize_reflector_fields(spec, feeds, grid, n_aperture=64, pad_factor=4)
    for i in range(2):
        np.testing.assert_array_equal(co_a[i], co_b[i])
        np.testing.assert_array_equal(cross_a[i], cross_b[i])
