import numpy as np

from grd_generator.reflector import FeedSpec, ReflectorSpec, ServiceZone, hex_feed_positions
from grd_generator.reflector.generate_afr import build_reflector_reference
from grd_generator.schemas import UVGrid


def test_build_reflector_reference_shapes() -> None:
    # Grille reflector en cosinus directeurs : ±0.25 ≈ ±14° (sin(14°)=0.242).
    grid = UVGrid(u_min=-0.25, u_max=0.25, v_min=-0.25, v_max=0.25, n_u=41, n_v=41)
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.0, freq_hz=20e9)
    feeds = FeedSpec(positions_m=hex_feed_positions(0.03, 7, 0.2), q=2.0)
    zone = ServiceZone(radius_deg=8.0)
    ref = build_reflector_reference(spec, feeds, zone, grid)
    assert ref["fields"].shape == (7, 41, 41)
    assert ref["cross"].shape == (7, 41, 41)
    assert ref["zone_radius_deg"] == 8.0
    assert ref["freq_hz"] == 20e9
    assert np.iscomplexobj(ref["fields"])
