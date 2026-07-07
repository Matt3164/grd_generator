import numpy as np

from grd_generator.reflector import (
    FeedSpec,
    ReflectorSpec,
    beamform_weights,
    directivity_barycenters,
    form_beam,
    hex_feed_positions,
    synthesize_reflector_fields,
)
from grd_generator.schemas import UVGrid


def _grid() -> UVGrid:
    # Grille reflector en cosinus directeurs : ±0.25 ≈ ±14° (sin(14°)=0.242).
    return UVGrid(u_min=-0.25, u_max=0.25, v_min=-0.25, v_max=0.25, n_u=81, n_v=81)


def _fields() -> list[np.ndarray]:
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.4, freq_hz=20e9)
    feeds = FeedSpec(positions_m=hex_feed_positions(0.03, 7, focal_radius_m=0.3), q=2.0)
    co, _ = synthesize_reflector_fields(spec, feeds, _grid(), n_aperture=64, pad_factor=4)
    return co


def test_form_beam_shape_matches_grid() -> None:
    co = _fields()
    beam = form_beam(co, _grid(), (0.0, 0.0))
    assert beam.shape == co[0].shape
    assert beam.dtype == np.complex128


def test_beam_peak_reaches_envelope_at_target() -> None:
    # Matched filter (wᵢ = conj Eᵢ(cible), ‖w‖=1) → |B(cible)|² = Σᵢ|Eᵢ(cible)|²,
    # la borne de Cauchy-Schwarz : la crête du beam touche l'enveloppe max co-pol.
    co = _fields()
    grid = _grid()
    u, v = grid.axes()
    target = (float(u[55]), float(v[40]))  # un point off-boresight arbitraire
    beam = form_beam(co, grid, target)
    stack = np.stack(co)
    iu = int(np.argmin(np.abs(u - target[0])))
    iv = int(np.argmin(np.abs(v - target[1])))
    envelope_power = float(np.sum(np.abs(stack[:, iv, iu]) ** 2))
    assert np.isclose(np.abs(beam[iv, iu]) ** 2, envelope_power)


def test_beam_peak_located_at_target() -> None:
    # Cible = pic d'un feed décalé → forcément dans la couverture ; le beam formé
    # vers ce point doit y crêter (à une cellule près).
    co = _fields()
    grid = _grid()
    u, v = grid.axes()
    fiv, fiu = np.unravel_index(int(np.argmax(np.abs(co[1]))), co[1].shape)
    target = (float(u[fiu]), float(v[fiv]))
    beam = form_beam(co, grid, target)
    biv, biu = np.unravel_index(int(np.argmax(np.abs(beam))), beam.shape)
    du = float(u[1] - u[0])
    dv = float(v[1] - v[0])
    assert abs(u[biu] - target[0]) <= du
    assert abs(v[biv] - target[1]) <= dv


def test_single_feed_beam_matches_that_feed_magnitude() -> None:
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.4, freq_hz=20e9)
    feeds = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0)
    co, _ = synthesize_reflector_fields(spec, feeds, _grid(), n_aperture=64, pad_factor=4)
    beam = form_beam(co, _grid(), (0.0, 0.0))
    # Un seul feed : le beam est ce feed à un déphasage global près → même |·|.
    assert np.allclose(np.abs(beam), np.abs(co[0]))


def test_zero_field_returns_zeros() -> None:
    grid = _grid()
    zeros = [np.zeros((grid.n_v, grid.n_u), dtype=np.complex128)]
    beam = form_beam(zeros, grid, (0.0, 0.0))
    assert np.all(beam == 0.0)


def test_beamform_weights_l2_norm_is_unity() -> None:
    co = _fields()
    grid = _grid()
    w = beamform_weights(co, grid, (0.01, -0.02))
    assert np.isclose(float(np.sum(np.abs(w) ** 2)), 1.0)


def test_beamform_weights_matches_conjugate_field_up_to_global_phase() -> None:
    # wᵢ = conj(Eᵢ(cible)) normalisés ; le référencement (`reference_strongest`)
    # ne fait qu'ajouter un déphasage global commun à tous les i.
    co = _fields()
    grid = _grid()
    target = (0.01, -0.02)
    u, v = grid.axes()
    iu = int(np.argmin(np.abs(u - target[0])))
    iv = int(np.argmin(np.abs(v - target[1])))
    samples = np.stack(co)[:, iv, iu]
    raw = np.conj(samples) / np.sqrt(np.sum(np.abs(samples) ** 2))
    w = beamform_weights(co, grid, target, reference_strongest=True)
    assert np.allclose(np.abs(w), np.abs(raw))
    ratio = w / raw
    assert np.allclose(ratio, ratio[0])


def test_beamform_weights_reference_strongest_max_feed_is_real_positive() -> None:
    co = _fields()
    grid = _grid()
    w = beamform_weights(co, grid, (0.01, -0.02), reference_strongest=True)
    j = int(np.argmax(np.abs(w)))
    assert abs(w[j].imag) < 1e-9
    assert w[j].real > 0.0


def test_beamform_weights_zero_field_returns_zeros() -> None:
    grid = _grid()
    zeros = [np.zeros((grid.n_v, grid.n_u), dtype=np.complex128)]
    w = beamform_weights(zeros, grid, (0.0, 0.0))
    assert np.all(w == 0.0)
    assert w.shape == (1,)


def test_form_beam_matches_pre_refactor_reference_implementation() -> None:
    # Vérifie que le refactor via `beamform_weights(reference_strongest=False)`
    # ne change pas le champ combiné, comparé à l'ancienne implémentation inline.
    co = _fields()
    grid = _grid()
    target = (0.01, -0.02)
    stack = np.stack(co)
    u, v = grid.axes()
    iu = int(np.argmin(np.abs(u - target[0])))
    iv = int(np.argmin(np.abs(v - target[1])))
    samples = stack[:, iv, iu]
    norm = float(np.sqrt(np.sum(np.abs(samples) ** 2)))
    reference_weights = np.conj(samples) / norm
    reference_beam = np.tensordot(reference_weights, stack, axes=([0], [0]))
    beam = form_beam(co, grid, target)
    assert np.allclose(np.abs(beam), np.abs(reference_beam))


def test_directivity_barycenters_shape() -> None:
    co = _fields()
    grid = _grid()
    centers = directivity_barycenters(co, grid)
    assert centers.shape == (len(co), 2)


def test_directivity_barycenters_peak_matches_known_point() -> None:
    grid = _grid()
    u, v = grid.axes()
    iu0, iv0 = 20, 60
    field = np.zeros((grid.n_v, grid.n_u), dtype=np.complex128)
    field[iv0, iu0] = 1.0 + 0j
    centers = directivity_barycenters([field], grid)
    assert np.isclose(centers[0, 0], u[iu0])
    assert np.isclose(centers[0, 1], v[iv0])


def test_directivity_barycenters_zero_power_returns_origin() -> None:
    grid = _grid()
    zeros = np.zeros((grid.n_v, grid.n_u), dtype=np.complex128)
    centers = directivity_barycenters([zeros], grid)
    assert np.allclose(centers[0], (0.0, 0.0))
