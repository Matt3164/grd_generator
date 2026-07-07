import numpy as np
import pytest

from grd_generator.reflector import optics
from grd_generator.reflector.spec import ReflectorSpec


def _spec(offset: float = 0.0) -> ReflectorSpec:
    return ReflectorSpec(diameter_m=2.0, focal_length_m=2.0,
                         offset_clearance_m=offset, freq_hz=20e9)


def test_aperture_grid_disk_and_spacing() -> None:
    X, Y, inside, dx = optics.aperture_grid(_spec(), n_aperture=64, pad_factor=2)
    assert X.shape == (128, 128)
    assert dx == pytest.approx(2.0 / 64)
    # disk centered on (0, D/2)=(0,1): a point at (0,1) is inside, (0,3) outside.
    assert inside.sum() > 0
    r2 = X**2 + (Y - 1.0) ** 2
    np.testing.assert_array_equal(inside, r2 <= 1.0**2)


def test_illumination_angle_zero_on_axis() -> None:
    X = np.array([[0.0]])
    Y = np.array([[0.0]])
    assert optics.illumination_angle(X, Y, 2.0)[0, 0] == pytest.approx(0.0)


def test_aperture_field_uniform_when_q0_and_centered_feed() -> None:
    spec = _spec()
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture=32, pad_factor=2)
    field = optics.aperture_field(spec, (0.0, 0.0), X, Y, inside, q=0.0)
    # q=0 → cos^0 = 1 inside; centered feed → zero phase tilt → real & flat.
    assert np.allclose(field[inside], 1.0)
    assert np.allclose(field[~inside], 0.0)


def test_pol_vectors_pure_copol_on_axis() -> None:
    spec = _spec()
    X = np.array([[0.0]])
    Y = np.array([[0.0]])
    ex, ey = optics.aperture_pol_vectors(spec, X, Y)
    assert abs(ey[0, 0]) < 1e-12          # no cross-pol on axis
    assert abs(abs(ex[0, 0]) - 1.0) < 1e-12


def test_pol_vectors_cross_in_diagonal() -> None:
    spec = _spec()
    X = np.array([[0.4]])
    Y = np.array([[0.4]])
    ex, ey = optics.aperture_pol_vectors(spec, X, Y)
    assert abs(ey[0, 0]) > 1e-6           # diagonal region → cross-pol present


def test_aperture_field_defocus_zero_is_identical_to_default() -> None:
    spec = _spec()
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture=32, pad_factor=2)
    field_default = optics.aperture_field(spec, (0.01, 0.0), X, Y, inside, q=2.0)
    field_explicit_zero = optics.aperture_field(
        spec, (0.01, 0.0), X, Y, inside, q=2.0, defocus_m=0.0
    )
    np.testing.assert_array_equal(field_default, field_explicit_zero)


def test_aperture_field_defocus_adds_expected_quadratic_phase_up_to_plane() -> None:
    # L'ouverture est un disque offset (centré en Y0 = D/2 ≠ 0, cf. spec) :
    # k·δz·(1−cosψ) contient, en plus du terme quadratique attendu, une
    # composante plane a+b·X+c·Y qui agirait comme un dépointage supplémentaire
    # si elle n'était pas retirée (voir docstring du module/aperture_field).
    # On vérifie donc l'égalité à ce plan près : on l'ajuste ici même sur la
    # phase brute, exactement comme aperture_field est censé le faire en
    # interne, et on compare le résidu à la phase effectivement appliquée.
    spec = _spec()
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture=32, pad_factor=2)
    defocus_m = 0.2
    for feed_xy in [(0.0, 0.0), (0.02, -0.01)]:
        field_nodefocus = optics.aperture_field(spec, feed_xy, X, Y, inside, q=2.0)
        field_defocus = optics.aperture_field(
            spec, feed_xy, X, Y, inside, q=2.0, defocus_m=defocus_m
        )
        psi = optics.illumination_angle(X, Y, spec.focal_length_m)
        k = 2.0 * np.pi / spec.wavelength_m
        raw_phase = k * defocus_m * (1.0 - np.cos(psi))

        design = np.column_stack((np.ones(int(inside.sum())), X[inside], Y[inside]))
        coeffs, *_ = np.linalg.lstsq(design, raw_phase[inside], rcond=None)
        plane = coeffs[0] + coeffs[1] * X + coeffs[2] * Y
        expected_extra_phase = raw_phase - plane

        # Le plan retiré est non trivial (ouverture offset → gradient en Y) :
        # ce test distinguerait bien un correctif qui ne retirerait rien.
        assert abs(coeffs[2]) > 1e-6

        # Ratio plutôt que différence brute pour éviter le wrap de phase ; on
        # ramène aussi l'attendu dans (-pi, pi] par la même transformation,
        # car la phase attendue peut dépasser un tour complet sur l'ouverture.
        extra_phase = np.angle(field_defocus[inside] / field_nodefocus[inside])
        expected_wrapped = np.angle(np.exp(1j * expected_extra_phase[inside]))
        np.testing.assert_allclose(extra_phase, expected_wrapped, atol=1e-9)


def test_aperture_field_defocus_sign_flips_extra_phase() -> None:
    spec = _spec()
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture=32, pad_factor=2)
    feed_xy = (0.0, 0.0)
    field_nodefocus = optics.aperture_field(spec, feed_xy, X, Y, inside, q=2.0)
    field_pos = optics.aperture_field(spec, feed_xy, X, Y, inside, q=2.0, defocus_m=0.2)
    field_neg = optics.aperture_field(spec, feed_xy, X, Y, inside, q=2.0, defocus_m=-0.2)
    extra_phase_pos = np.angle(field_pos[inside] / field_nodefocus[inside])
    extra_phase_neg = np.angle(field_neg[inside] / field_nodefocus[inside])
    np.testing.assert_allclose(extra_phase_pos, -extra_phase_neg, atol=1e-9)


def test_aperture_field_extra_phase_none_is_identical_to_default() -> None:
    spec = _spec()
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture=32, pad_factor=2)
    field_default = optics.aperture_field(spec, (0.01, 0.0), X, Y, inside, q=2.0)
    field_explicit_none = optics.aperture_field(
        spec, (0.01, 0.0), X, Y, inside, q=2.0, extra_phase=None
    )
    np.testing.assert_array_equal(field_default, field_explicit_none)


def test_aperture_field_extra_phase_is_added_to_total_phase() -> None:
    spec = _spec()
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture=32, pad_factor=2)
    feed_xy = (0.01, -0.005)
    field_base = optics.aperture_field(spec, feed_xy, X, Y, inside, q=2.0, defocus_m=0.1)
    extra_phase = 0.3 * np.sin(X) + 0.2 * np.cos(Y)
    field_extra = optics.aperture_field(
        spec, feed_xy, X, Y, inside, q=2.0, defocus_m=0.1, extra_phase=extra_phase
    )
    observed_extra = np.angle(field_extra[inside] / field_base[inside])
    expected_extra = np.angle(np.exp(1j * extra_phase[inside]))
    np.testing.assert_allclose(observed_extra, expected_extra, atol=1e-9)


def test_random_phase_screen_matches_requested_rms_and_zero_mean() -> None:
    spec = _spec()
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture=64, pad_factor=2)
    rng = np.random.default_rng(0)
    rms_rad = 0.7
    screen = optics.random_phase_screen(
        X, Y, inside, dx, rms_rad=rms_rad, corr_length_m=0.05, rng=rng
    )
    assert screen[inside].mean() == pytest.approx(0.0, abs=1e-9)
    assert screen[inside].std() == pytest.approx(rms_rad, rel=0.01)


def test_random_phase_screen_smoothing_reduces_gradient_energy() -> None:
    # Un noyau de lissage 4x plus large doit produire un champ nettement plus
    # lisse : l'énergie de gradient (somme des diffs au carré) sur `inside`
    # doit être nettement plus faible.
    spec = _spec()
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture=64, pad_factor=2)

    def _gradient_energy(screen: np.ndarray) -> float:
        dgx = np.diff(screen, axis=1)
        dgy = np.diff(screen, axis=0)
        mask_x = inside[:, :-1] & inside[:, 1:]
        mask_y = inside[:-1, :] & inside[1:, :]
        return float(np.sum(dgx[mask_x] ** 2) + np.sum(dgy[mask_y] ** 2))

    screen_short = optics.random_phase_screen(
        X, Y, inside, dx, rms_rad=1.0, corr_length_m=0.01, rng=np.random.default_rng(1)
    )
    screen_long = optics.random_phase_screen(
        X, Y, inside, dx, rms_rad=1.0, corr_length_m=0.04, rng=np.random.default_rng(1)
    )
    energy_short = _gradient_energy(screen_short)
    energy_long = _gradient_energy(screen_long)
    assert energy_long < energy_short * 0.5


def test_random_phase_screen_reproducible_with_same_rng_seed() -> None:
    spec = _spec()
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture=32, pad_factor=2)
    screen_a = optics.random_phase_screen(
        X, Y, inside, dx, rms_rad=0.5, corr_length_m=0.03, rng=np.random.default_rng(7)
    )
    screen_b = optics.random_phase_screen(
        X, Y, inside, dx, rms_rad=0.5, corr_length_m=0.03, rng=np.random.default_rng(7)
    )
    np.testing.assert_array_equal(screen_a, screen_b)


def test_shared_extra_phase_ratio_identical_across_feeds() -> None:
    # Un écran de phase COMMUN (même array `extra_phase`) appliqué à deux
    # feeds différents doit produire, dans le domaine d'ouverture, le même
    # facteur multiplicatif complexe exp(i*screen) : le ratio
    # champ_avec_écran/champ_sans_écran ne dépend que de l'écran, pas du feed
    # (tilt et taper d'amplitude s'annulent dans le ratio).
    spec = _spec()
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture=32, pad_factor=2)
    shared_screen = optics.random_phase_screen(
        X, Y, inside, dx, rms_rad=0.8, corr_length_m=0.04, rng=np.random.default_rng(0)
    )
    feed_a = (0.0, 0.0)
    feed_b = (0.02, -0.01)
    field_a_plain = optics.aperture_field(spec, feed_a, X, Y, inside, q=2.0)
    field_a_shared = optics.aperture_field(
        spec, feed_a, X, Y, inside, q=2.0, extra_phase=shared_screen
    )
    field_b_plain = optics.aperture_field(spec, feed_b, X, Y, inside, q=2.0)
    field_b_shared = optics.aperture_field(
        spec, feed_b, X, Y, inside, q=2.0, extra_phase=shared_screen
    )
    ratio_a = field_a_shared[inside] / field_a_plain[inside]
    ratio_b = field_b_shared[inside] / field_b_plain[inside]
    np.testing.assert_allclose(ratio_a, ratio_b, atol=1e-9)
