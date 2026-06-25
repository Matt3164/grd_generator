import numpy as np
import pytest

from grd_generator.reflector import farfield
from grd_generator.reflector.spec import ReflectorSpec


def test_uniform_aperture_peak_directivity_matches_airy() -> None:
    # Uniform circular aperture (q=0, centered feed), symmetric (no offset to
    # keep it scalar/co-pol): peak directivity ≈ (πD/λ)².
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=1e9,  # ~flat: ψ≈0, n̂≈ẑ
                         offset_clearance_m=0.0, freq_hz=20e9)
    # center the disk on the axis by overriding aperture: build a centered disk.
    n_aperture, pad = 96, 6
    dx = spec.diameter_m / n_aperture
    n = n_aperture * pad
    coords = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(coords, coords)
    inside = (spec.diameter_m / 2.0) ** 2 >= (X**2 + Y**2)
    aperture = inside.astype(np.complex128)

    F, L, M = farfield.far_field_fft(aperture, dx, spec.wavelength_m)
    dL = L[0, 1] - L[0, 0]
    dM = M[1, 0] - M[0, 0]
    norm = farfield.normalize_to_directivity(np.abs(F) ** 2, L, M, dL, dM)
    peak_dbi = 10.0 * np.log10((np.abs(F) * norm).max() ** 2)
    expected = 10.0 * np.log10((np.pi * spec.diameter_m / spec.wavelength_m) ** 2)
    assert peak_dbi == pytest.approx(expected, abs=0.3)
