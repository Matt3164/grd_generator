import pytest
from pydantic import ValidationError

from grd_generator.reflector.spec import C_LIGHT, FeedSpec, ReflectorSpec


def test_reflector_spec_derived_quantities() -> None:
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.4, freq_hz=20e9)
    assert spec.wavelength_m == pytest.approx(C_LIGHT / 20e9)
    assert spec.f_over_d == pytest.approx(1.2)
    assert spec.aperture_center_y_m == pytest.approx(1.0)  # 0 + D/2


def test_reflector_spec_rejects_nonpositive() -> None:
    with pytest.raises(ValidationError):
        ReflectorSpec(diameter_m=0.0, focal_length_m=2.4, freq_hz=20e9)


def test_feed_spec_counts() -> None:
    feeds = FeedSpec(positions_m=[(0.0, 0.0), (0.01, 0.0)], q=2.0)
    assert feeds.n_feeds == 2
