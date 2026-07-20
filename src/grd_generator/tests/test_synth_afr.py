import numpy as np
import pytest

from grd_generator.reflector import (
    FeedSpec,
    ReflectorSpec,
    dereference_phase,
    hex_feed_positions,
    synthesize_reflector_fields,
)
from grd_generator.schemas import UVGrid

# Grille reflector en cosinus directeurs : ±0.25 ≈ ±14° (sin(14°)=0.242).
_UV_HALF_WIDTH = 0.25
# Seuil "≈1°" (proximité du pic au boresight, écart de scan), converti en
# cosinus directeur : sin(radians(1.0)).
_ONE_DEG_UV = float(np.sin(np.radians(1.0)))


def _grid() -> UVGrid:
    return UVGrid(
        u_min=-_UV_HALF_WIDTH, u_max=_UV_HALF_WIDTH,
        v_min=-_UV_HALF_WIDTH, v_max=_UV_HALF_WIDTH,
        n_u=81, n_v=81,
    )


def _grid_161() -> UVGrid:
    return UVGrid(
        u_min=-_UV_HALF_WIDTH, u_max=_UV_HALF_WIDTH,
        v_min=-_UV_HALF_WIDTH, v_max=_UV_HALF_WIDTH,
        n_u=161, n_v=161,
    )


def _minus3db_width_deg(dbi: np.ndarray, grid: UVGrid) -> float:
    """Largeur -3dB (deg) via un profil 1D interpolé le long de u, à v=v_peak.

    Interpolation linéaire aux croisements pour une précision sous-pixel
    (plus robuste qu'un simple comptage de pixels au-dessus du seuil pour
    comparer des largeurs de lobe entre configurations différentes).
    """
    u_axis, _ = grid.axes()
    iv, iu_raw = np.unravel_index(np.argmax(dbi), dbi.shape)
    iu: int = int(iu_raw)
    peak = dbi[iv, iu]
    profile = dbi[iv, :]
    half = peak - 3.0
    above = profile >= half
    left = iu
    while left - 1 >= 0 and above[left - 1]:
        left -= 1
    right = iu
    while right + 1 < len(profile) and above[right + 1]:
        right += 1

    def _cross(i0: int, i1: int) -> float:
        y0, y1 = profile[i0], profile[i1]
        u0, u1 = u_axis[i0], u_axis[i1]
        if y1 == y0:
            return float(u_axis[i0])
        t = (half - y0) / (y1 - y0)
        return float(u0 + t * (u1 - u0))

    u_left = _cross(left - 1, left) if left > 0 else float(u_axis[left])
    u_right = _cross(right, right + 1) if right < len(profile) - 1 else float(u_axis[right])
    return u_right - u_left


def test_centered_feed_beam_points_at_boresight() -> None:
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.0, freq_hz=20e9)
    feeds = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0)
    co, cross = synthesize_reflector_fields(spec, feeds, _grid(), n_aperture=64, pad_factor=4)
    assert len(co) == 1 and len(cross) == 1
    dbi = 10.0 * np.log10(np.abs(co[0]) ** 2)
    iv, iu = np.unravel_index(np.argmax(dbi), dbi.shape)
    gu, gv = _grid().meshgrid()
    assert abs(gu[iv, iu]) < _ONE_DEG_UV and abs(gv[iv, iu]) < _ONE_DEG_UV  # peak near (0,0)


def test_displaced_feed_beam_scans() -> None:
    # A feed displaced +x in the focal plane scans the beam off boresight.
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.0, freq_hz=20e9)
    feeds = FeedSpec(positions_m=[(0.05, 0.0)], q=2.0)
    co, _ = synthesize_reflector_fields(spec, feeds, _grid(), n_aperture=64, pad_factor=4)
    dbi = 10.0 * np.log10(np.abs(co[0]) ** 2)
    iv, iu = np.unravel_index(np.argmax(dbi), dbi.shape)
    gu, _ = _grid().meshgrid()
    assert abs(gu[iv, iu]) > _ONE_DEG_UV  # beam moved away from boresight in u


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
    # sin(30°) = 0.5 exactement : bornes cosinus directeurs équivalentes à ±30°.
    grid = UVGrid(u_min=-0.5, u_max=0.5, v_min=-0.5, v_max=0.5, n_u=121, n_v=121)
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


def _v_asymmetry_db(dbi: np.ndarray, grid: UVGrid, *, u0: float = 0.0) -> float:
    """Asymétrie haut/bas (dB) de la coupe u=u0 : max |dbi(v) - dbi(-v)| près du pic.

    Restreint aux points à moins de 10 dB de la crête pour ignorer le bruit
    numérique des lobes très bas (planchers -12 dB imposés par `maximum`).
    Suppose une grille symétrique en v (v_min = -v_max, n_v impair) pour que
    v=0 tombe exactement sur un point de grille.
    """
    u_axis, v_axis = grid.axes()
    iu0 = int(np.argmin(np.abs(u_axis - u0)))
    profile = dbi[:, iu0]
    mid = len(profile) // 2
    assert v_axis[mid] == pytest.approx(0.0, abs=1e-9)
    peak = float(profile.max())
    near_peak = profile > peak - 10.0
    diffs = [
        abs(profile[mid + k] - profile[mid - k])
        for k in range(1, mid)
        if near_peak[mid + k] or near_peak[mid - k]
    ]
    return max(diffs) if diffs else 0.0


def test_centered_aperture_defocus_gives_symmetric_farfield_cut() -> None:
    # Contexte (diagnostic) : sur ce modèle AFR, l'ouverture offset produit de
    # la coma sous defocus (asymétrie haut/bas marquée, "oreilles"). Une
    # ouverture centrée (centered_aperture=True, Y0=0) est axisymétrique : le
    # champ d'ouverture scalaire ne dépend que de rho=||(X,Y)|| pour un feed
    # central, et la décomposition Ludwig-3 préserve la parité en v (preuve :
    # e_co(L,-M)=e_co(L,M) car Fx est pair en Y, Fy impair, phi impair en M).
    # La coupe u=0 doit donc être symétrique en v à la précision numérique
    # près, bien au-dessous du seuil de garde-fou de 0.5 dB.
    spec = ReflectorSpec(
        diameter_m=0.28, focal_length_m=0.336, freq_hz=20e9, centered_aperture=True
    )
    grid = _grid_161()
    feeds = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0, defocus_m=0.2)
    co, _ = synthesize_reflector_fields(spec, feeds, grid, n_aperture=64, pad_factor=4)
    dbi = 10.0 * np.log10(np.maximum(np.abs(co[0]) ** 2, 1e-12))
    assert _v_asymmetry_db(dbi, grid) < 0.5


def test_offset_aperture_defocus_gives_asymmetric_farfield_cut() -> None:
    # Garde-fou : la même config, mais avec l'ouverture offset réelle (défaut
    # du modèle, `centered_aperture=False`), doit rester nettement asymétrique
    # (coma) — sinon `test_centered_aperture_...` ci-dessus ne testerait rien
    # (un bug qui ignorerait `centered_aperture` passerait quand même).
    spec = ReflectorSpec(
        diameter_m=0.28, focal_length_m=0.336, offset_clearance_m=0.15, freq_hz=20e9
    )
    grid = _grid_161()
    feeds = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0, defocus_m=0.2)
    co, _ = synthesize_reflector_fields(spec, feeds, grid, n_aperture=64, pad_factor=4)
    dbi = 10.0 * np.log10(np.maximum(np.abs(co[0]) ** 2, 1e-12))
    assert _v_asymmetry_db(dbi, grid) > 5.0


# Demi-largeur ±2° convertie en cosinus directeur (grille reflector), utilisée
# par `_unaliased_dereference_grid` pour reproduire le même gradient de phase
# par pixel qu'avant la conversion (voir sa docstring).
_UNALIASED_HALF_WIDTH_UV = float(np.sin(np.radians(2.0)))


def _unaliased_dereference_grid() -> UVGrid:
    """Grille (u,v) NON aliasée pour la porteuse k·Y0·v des tests de phase.

    `v` est ici directement le cosinus directeur (grille reflector, pas de
    `sin` supplémentaire dans `dereference_phase`). Sur la grille d'affichage
    standard (`_grid()`, ±0.25/81px), le gradient de phase par pixel de la
    porteuse dépasse π (repliement) : un bug de signe y est masqué par
    l'aliasing lui-même (voir tâche). Ici (±sin(2°)/51px, la même couverture
    angulaire physique qu'avant la conversion aux cosinus directeurs), le
    gradient de phase de la porteuse par pixel ≈ 0.73 rad < π/4 : la grille
    résout correctement la porteuse, donc un signe inversé produit un résidu
    net (non masqué).
    """
    return UVGrid(
        u_min=-_UNALIASED_HALF_WIDTH_UV, u_max=_UNALIASED_HALF_WIDTH_UV,
        v_min=-_UNALIASED_HALF_WIDTH_UV, v_max=_UNALIASED_HALF_WIDTH_UV,
        n_u=21, n_v=51,
    )


def test_dereference_phase_removes_pure_carrier() -> None:
    # Un champ qui N'EST QUE la porteuse physique exp(-i·k·Y0·v) (signe de
    # la convention FFT de `farfield.far_field_fft`, voir docstring de
    # `dereference_phase`) doit être dé-référencé à une phase ~0 partout :
    # c'est exactement le terme retiré.
    grid = _unaliased_dereference_grid()
    wavelength_m = 0.015  # 20 GHz
    y0 = 1.25  # aperture_center_y_m typique (D=2.2, clearance=0.15)
    _, gv = grid.meshgrid()
    k = 2.0 * np.pi / wavelength_m
    m = gv
    field = np.exp(-1j * k * y0 * m)

    dephased = dereference_phase(field, grid, wavelength_m, y0)

    np.testing.assert_allclose(dephased, 0.0, atol=1e-9)


def test_dereference_phase_constant_field_yields_carrier() -> None:
    # Un champ constant (phase nulle partout) n'a pas de porteuse à retirer : le
    # résultat est donc exactement la porteuse conjuguée k·Y0·v appliquée
    # par la fonction (voir docstring).
    grid = _unaliased_dereference_grid()
    wavelength_m = 0.015
    y0 = 1.25
    field = np.ones((grid.n_v, grid.n_u), dtype=np.complex128)

    dephased = dereference_phase(field, grid, wavelength_m, y0)

    _, gv = grid.meshgrid()
    k = 2.0 * np.pi / wavelength_m
    expected = np.angle(np.exp(1j * k * y0 * gv))
    np.testing.assert_allclose(dephased, expected, atol=1e-9)


def test_dereference_phase_reduces_v_gradient_for_carrier_plus_slow_structure() -> None:
    # Champ réaliste = porteuse physique (grande, due à Y0) + structure lente
    # (utile, petite devant la porteuse) : après dé-référencement, le gradient
    # de phase selon v doit être nettement plus petit qu'avant (la porteuse
    # dominait le gradient brut et masquait la structure lente).
    grid = _unaliased_dereference_grid()
    wavelength_m = 0.015
    y0 = 1.25
    _, gv = grid.meshgrid()
    k = 2.0 * np.pi / wavelength_m
    m = gv
    # Coefficient choisi pour que |slow_phase| reste du même ordre de grandeur
    # (~0.1 rad) qu'avant la conversion aux cosinus directeurs, où `gv`
    # parcourait ±2° au lieu de ±sin(2°) ici : facteur 2.0/sin(2°) ≈ 57.3.
    slow_phase = 0.05 * (2.0 / _UNALIASED_HALF_WIDTH_UV) * gv
    field = np.exp(1j * (-k * y0 * m + slow_phase))

    before = np.unwrap(np.angle(field), axis=0)
    after = np.unwrap(dereference_phase(field, grid, wavelength_m, y0), axis=0)
    grad_before = float(np.mean(np.abs(np.diff(before, axis=0))))
    grad_after = float(np.mean(np.abs(np.diff(after, axis=0))))

    assert grad_after < grad_before / 10.0


def test_dereference_phase_sign_matters_on_unaliased_grid() -> None:
    # Garde-fou anti-régression du bug de signe : sur une grille NON aliasée
    # (voir `_unaliased_dereference_grid`), l'implémentation correcte laisse un
    # résidu de gradient quasi nul, tandis que le signe inversé (bug historique
    # : `exp(-i·k·Y0·v)` au lieu de `exp(+i·k·Y0·v)`) laisse un résidu
    # de phase largement supérieur à 1 rad — preuve que ce test distinguerait
    # bien une régression de signe (contrairement à une grille aliasée où le
    # repliement masquerait l'erreur, cf. tâche : 3.09 -> 0.02 rad/pixel).
    grid = _unaliased_dereference_grid()
    wavelength_m = 0.015
    y0 = 1.25
    _, gv = grid.meshgrid()
    k = 2.0 * np.pi / wavelength_m
    m = gv
    field = np.exp(-1j * k * y0 * m)  # porteuse physique réelle

    correct = dereference_phase(field, grid, wavelength_m, y0)
    buggy = np.angle(field * np.exp(-1j * k * y0 * m))  # signe historique (bug)

    def _grad(arr: np.ndarray) -> float:
        unwrapped = np.unwrap(arr, axis=0)
        return float(np.mean(np.abs(np.diff(unwrapped, axis=0))))

    assert _grad(correct) < 0.1
    assert _grad(buggy) > 1.0
