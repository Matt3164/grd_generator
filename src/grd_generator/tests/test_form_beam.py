import numpy as np

from grd_generator.reflector import (
    FeedSpec,
    ReflectorSpec,
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
