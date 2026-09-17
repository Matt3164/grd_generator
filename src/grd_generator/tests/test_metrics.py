import math

import numpy as np
import pytest

from grd_generator.reflector.metrics import (
    compute_frequency_metrics,
    cross_pol_isolation_db,
    dbi_map,
    equivalent_diameter_deg,
    guaranteed_radius_deg,
    peak_dbi,
    peak_radius_deg,
    radius_deg,
    solid_angle_sr,
    width_3db_deg,
)
from grd_generator.schemas import ComplexField, UVGrid

_SIGMA = 0.05  # écart-type (uv) du lobe gaussien synthétique


def _grid(half_width: float = 0.18, n: int = 181) -> UVGrid:
    # n impair : garantit un nœud exactement en (0, 0).
    return UVGrid(u_min=-half_width, u_max=half_width, v_min=-half_width, v_max=half_width,
                  n_u=n, n_v=n)


def _gaussian_field(
    grid: UVGrid,
    *,
    amplitude: float,
    center: tuple[float, float] = (0.0, 0.0),
    sigma: float = _SIGMA,
) -> ComplexField:
    """Lobe gaussien réel, isotrope en (u,v) : `E = amplitude·exp(-r²/2σ²)`."""
    gu, gv = grid.meshgrid()
    cu, cv = center
    r2 = (gu - cu) ** 2 + (gv - cv) ** 2
    field: ComplexField = (amplitude * np.exp(-r2 / (2.0 * sigma**2))).astype(np.complex128)
    return field


def test_radius_deg_matches_asin_hypot() -> None:
    grid = _grid()
    r = radius_deg(grid)
    gu, gv = grid.meshgrid()
    expected = np.degrees(np.arcsin(np.hypot(gu, gv)))
    np.testing.assert_allclose(r, expected)


def test_peak_dbi_exact_for_gaussian_at_center() -> None:
    grid = _grid()
    amplitude = 3.0
    field = _gaussian_field(grid, amplitude=amplitude)
    assert peak_dbi(field) == pytest.approx(20.0 * math.log10(amplitude), abs=1e-9)


def test_peak_radius_deg_zero_for_centered_lobe() -> None:
    grid = _grid()
    field = _gaussian_field(grid, amplitude=1.0)
    assert peak_radius_deg(field, grid) == pytest.approx(0.0, abs=1e-9)


def test_width_3db_deg_matches_analytic_disk_for_centered_lobe() -> None:
    # -3dB en r_uv tel que 10*log10(exp(-r²/σ²)) = -3 => r = σ*sqrt(3*ln(10)/10).
    grid = _grid()
    sigma = _SIGMA
    field = _gaussian_field(grid, amplitude=1.0, sigma=sigma)
    r_3db_uv = sigma * math.sqrt(3.0 * math.log(10.0) / 10.0)
    expected_diameter_deg = 2.0 * math.degrees(math.asin(r_3db_uv))
    measured = width_3db_deg(field, grid)
    assert measured == pytest.approx(expected_diameter_deg, rel=2e-2)


def test_equivalent_diameter_deg_of_hemisphere_is_180() -> None:
    # Une calotte de demi-angle 90° (hémisphère) a Ω = 2π.
    assert equivalent_diameter_deg(2.0 * math.pi) == pytest.approx(180.0)


def test_equivalent_diameter_deg_zero_solid_angle() -> None:
    assert equivalent_diameter_deg(0.0) == pytest.approx(0.0)


def test_solid_angle_sr_of_full_grid_mask_is_positive_and_bounded() -> None:
    grid = _grid()
    mask = np.ones((grid.n_v, grid.n_u), dtype=bool)
    omega = solid_angle_sr(mask, grid)
    assert 0.0 < omega < 4.0 * math.pi


def test_cross_pol_isolation_db_exact_for_proportional_cross() -> None:
    grid = _grid()
    co = _gaussian_field(grid, amplitude=1.0)
    k = 0.05
    cross = co * k
    isolation = cross_pol_isolation_db(co, cross)
    assert isolation == pytest.approx(-20.0 * math.log10(k), abs=1e-9)


def test_guaranteed_radius_deg_returns_closest_failing_node() -> None:
    grid = _grid(n=21)
    r = radius_deg(grid)
    dbi = np.full((grid.n_v, grid.n_u), 50.0)
    # Un seul nœud en échec, à michemin ; son rayon doit être renvoyé exactement.
    iv, iu = grid.n_v // 2, grid.n_u // 2 + 3
    dbi[iv, iu] = 10.0
    result = guaranteed_radius_deg(dbi, grid, threshold_dbi=39.0)
    assert result == pytest.approx(float(r[iv, iu]))


def test_guaranteed_radius_deg_returns_grid_max_when_nothing_fails() -> None:
    grid = _grid(n=21)
    dbi = np.full((grid.n_v, grid.n_u), 50.0)
    result = guaranteed_radius_deg(dbi, grid, threshold_dbi=39.0)
    assert result == pytest.approx(float(radius_deg(grid).max()))


def test_dbi_map_floor_avoids_log_of_zero() -> None:
    field = np.zeros((3, 3), dtype=np.complex128)
    dbi = dbi_map(field)
    assert np.all(np.isfinite(dbi))


def test_compute_frequency_metrics_aggregates_across_feeds() -> None:
    grid = _grid(n=41)
    # Trois feeds distincts (amplitudes différentes), tous centrés au nadir
    # pour simplifier (un vrai batch les décale, non testé ici).
    amplitudes = [1.0, 2.0, 4.0]
    co_fields = [_gaussian_field(grid, amplitude=a) for a in amplitudes]
    cross_fields = [co * 0.01 for co in co_fields]

    result = compute_frequency_metrics(11400, co_fields, cross_fields, grid)

    assert result.freq_mhz == 11400
    assert len(result.feeds) == 3
    assert [f.feed for f in result.feeds] == [1, 2, 3]
    expected_peaks = [20.0 * math.log10(a) for a in amplitudes]
    for f, expected in zip(result.feeds, expected_peaks, strict=True):
        assert f.peak_dbi == pytest.approx(expected, abs=1e-9)
    assert result.peak_dbi_median == pytest.approx(expected_peaks[1], abs=1e-9)
    assert result.peak_radius_deg_max == pytest.approx(0.0, abs=1e-9)
    # Enveloppe : max_dbi >= le plus fort feed seul (Cauchy-Schwarz, cf. synth.py).
    assert result.envelope.max_dbi >= max(expected_peaks) - 1e-6


def test_frequency_metrics_passes_blocking_criterion() -> None:
    grid = _grid(n=41)
    co_fields = [_gaussian_field(grid, amplitude=1.0, sigma=0.15)]  # lobe large -> grand rayon
    cross_fields = [co_fields[0] * 0.01]
    result = compute_frequency_metrics(11400, co_fields, cross_fields, grid)
    assert result.passes_blocking_criterion(min_radius_3db_deg=0.0) is True
    assert result.passes_blocking_criterion(min_radius_3db_deg=1000.0) is False
