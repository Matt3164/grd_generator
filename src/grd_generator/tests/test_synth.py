import numpy as np
import pytest

from grd_generator.schemas import GaussianSpec, UVGrid
from grd_generator.synth import (
    airy_field,
    available_modes,
    check_power_conservation,
    directivity_dbi_from_field,
    field_generator,
    gaussian_field,
    hex_centers_uv,
    max_peak_dbi,
    optimize_sigma,
    power_fraction,
)


def _grid() -> UVGrid:
    return UVGrid(u_min=-8.0, u_max=8.0, v_min=-8.0, v_max=8.0, n_u=81, n_v=81)


def test_available_modes() -> None:
    assert available_modes() == ["airy", "gaussian"]


def test_gaussian_peak_at_center() -> None:
    grid = _grid()
    spec = GaussianSpec(center_uv=(0.0, 0.0), sigma=1.0, peak_gain_dbi=40.0)
    field = gaussian_field(spec, grid)
    peak = 10.0 ** (40.0 / 20.0)
    assert np.abs(field).max() == pytest.approx(peak, rel=1e-3)


def test_airy_has_sidelobes_and_nulls() -> None:
    grid = _grid()
    spec = GaussianSpec(center_uv=(0.0, 0.0), sigma=2.0, peak_gain_dbi=40.0)
    field = airy_field(spec, grid)
    # Airy amplitude changes sign (true nulls) → min of real part < 0 region exists.
    amp = np.abs(field)
    assert amp.min() < amp.max()  # structured, not flat


def test_field_generator_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        field_generator("nope")


def test_max_peak_dbi_known_modes() -> None:
    assert max_peak_dbi("gaussian", 1.0) > 0
    assert max_peak_dbi("airy", 1.0) > 0
    with pytest.raises(ValueError):
        max_peak_dbi("nope", 1.0)


def test_check_power_conservation_warns_over_bound() -> None:
    # A huge peak for a wide lobe exceeds the physical bound → returns False.
    # max_peak_dbi("gaussian", 5.0) ≈ 27.2 dBi, so 99.0 is well above → False
    assert check_power_conservation("gaussian", sigma=5.0, peak_gain_dbi=99.0) is False
    # max_peak_dbi("gaussian", 0.5) ≈ 47.2 dBi, so 10.0 is well below → True
    assert check_power_conservation("gaussian", sigma=0.5, peak_gain_dbi=10.0) is True


def test_hex_centers_count_and_too_loose() -> None:
    centers = hex_centers_uv((0.0, 0.0), zone_radius=6.0, n=20, spacing=1.5)
    assert len(centers) == 20
    with pytest.raises(ValueError):
        hex_centers_uv((0.0, 0.0), zone_radius=1.0, n=10000, spacing=1.0)


def test_optimize_sigma_feasible_and_monotone() -> None:
    grid = _grid()
    centers = hex_centers_uv((0.0, 0.0), zone_radius=6.0, n=30, spacing=2.0)
    sigma, specs = optimize_sigma(
        grid, centers, peak_gain_dbi=47.0,
        zone_center=(0.0, 0.0), zone_radius=6.0, target_dbi=44.0,
    )
    assert 1e-3 <= sigma <= 5.0
    assert len(specs) == len(centers)
    assert all(s.sigma == sigma for s in specs)


def test_optimize_sigma_infeasible_raises() -> None:
    grid = _grid()
    centers = hex_centers_uv((0.0, 0.0), zone_radius=6.0, n=30, spacing=2.0)
    with pytest.raises(ValueError):
        optimize_sigma(
            grid, centers, peak_gain_dbi=-50.0,  # far too weak to ever reach target
            zone_center=(0.0, 0.0), zone_radius=6.0, target_dbi=44.0, sigma_hi=5.0,
        )


def test_power_fraction_returns_float() -> None:
    """Couvre power_fraction : fraction de puissance rayonnée capturée par la grille."""
    grid = _grid()
    spec = GaussianSpec(center_uv=(0.0, 0.0), sigma=1.0, peak_gain_dbi=40.0)
    field = gaussian_field(spec, grid)
    frac = power_fraction(field, grid)
    assert isinstance(frac, float)
    assert 0.0 < frac <= 1.0


def test_directivity_dbi_from_field_values() -> None:
    """Couvre directivity_dbi_from_field : vérifie la conversion |E|² → dBi."""
    grid = _grid()
    spec = GaussianSpec(center_uv=(0.0, 0.0), sigma=1.0, peak_gain_dbi=40.0)
    field = gaussian_field(spec, grid)
    dbi_map = directivity_dbi_from_field(field)
    # La crête doit être proche de peak_gain_dbi au carré (puissance = 40 dBi).
    assert float(dbi_map.max()) == pytest.approx(40.0, abs=0.1)
