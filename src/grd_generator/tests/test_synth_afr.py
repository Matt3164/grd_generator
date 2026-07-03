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
