import numpy as np
import pytest

from grd_generator.synth import combined_max_directivity_dbi, hex_centers_uv


def test_hex_centers_count_and_too_loose() -> None:
    centers = hex_centers_uv((0.0, 0.0), zone_radius=6.0, n=20, spacing=1.5)
    assert len(centers) == 20
    with pytest.raises(ValueError):
        hex_centers_uv((0.0, 0.0), zone_radius=1.0, n=10000, spacing=1.0)


def test_combined_max_directivity_dbi_is_rss_of_fields() -> None:
    """Couvre combined_max_directivity_dbi : 10·log10(Σ|Eᵢ|²), RSS (Cauchy-Schwarz)."""
    field_a = np.array([[1.0 + 0j, 0.0 + 0j]])
    field_b = np.array([[0.0 + 0j, 2.0 + 0j]])
    dbi = combined_max_directivity_dbi([field_a, field_b])
    expected = 10.0 * np.log10(np.array([[1.0, 4.0]]))
    np.testing.assert_allclose(dbi, expected)
