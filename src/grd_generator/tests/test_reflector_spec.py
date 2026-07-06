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


def test_feed_spec_defocus_defaults_to_zero_and_accepts_negative() -> None:
    feeds = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0)
    assert feeds.defocus_m == 0.0
    feeds_negative = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0, defocus_m=-0.3)
    assert feeds_negative.defocus_m == pytest.approx(-0.3)


def test_feed_spec_phase_error_defaults() -> None:
    feeds = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0)
    assert feeds.phase_error_rms_rad == 0.0
    assert feeds.phase_corr_length_m == pytest.approx(0.05)
    assert feeds.phase_error_seed == 0


def test_feed_spec_phase_error_rejects_negative_rms() -> None:
    with pytest.raises(ValidationError):
        FeedSpec(positions_m=[(0.0, 0.0)], q=2.0, phase_error_rms_rad=-0.1)


def test_feed_spec_phase_error_rejects_nonpositive_corr_length() -> None:
    with pytest.raises(ValidationError):
        FeedSpec(positions_m=[(0.0, 0.0)], q=2.0, phase_corr_length_m=0.0)


def test_feed_spec_phase_error_accepts_custom_values() -> None:
    feeds = FeedSpec(
        positions_m=[(0.0, 0.0)],
        q=2.0,
        phase_error_rms_rad=1.0,
        phase_corr_length_m=0.03,
        phase_error_seed=42,
    )
    assert feeds.phase_error_rms_rad == pytest.approx(1.0)
    assert feeds.phase_corr_length_m == pytest.approx(0.03)
    assert feeds.phase_error_seed == 42


def test_feed_spec_phase_error_shared_defaults_to_zero() -> None:
    feeds = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0)
    assert feeds.phase_error_shared_rms_rad == 0.0


def test_feed_spec_phase_error_shared_rejects_negative_rms() -> None:
    with pytest.raises(ValidationError):
        FeedSpec(positions_m=[(0.0, 0.0)], q=2.0, phase_error_shared_rms_rad=-0.2)


def test_feed_spec_phase_error_shared_accepts_custom_value() -> None:
    feeds = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0, phase_error_shared_rms_rad=1.5)
    assert feeds.phase_error_shared_rms_rad == pytest.approx(1.5)
