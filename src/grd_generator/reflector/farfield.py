"""Champ lointain : FFT du champ d'ouverture, normalisé en directivité.

Le far-field est la TF 2D du champ d'ouverture, échantillonné en cosinus
directeurs (l, m) = (sinθcosφ, sinθsinφ). La normalisation impose
|E|² = directivité linéaire (convention isotrope du projet) :
D = 4π·|F|² / ∮|F|² dΩ, dΩ = dl·dm / cosθ sur la région visible l²+m²<1.
"""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def far_field_fft(
    aperture: ComplexArray, dx: float, wavelength_m: float
) -> tuple[ComplexArray, FloatArray, FloatArray]:
    """TF 2D centrée du champ d'ouverture → (F, L, M) en cosinus directeurs."""
    n = aperture.shape[0]
    spec = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(aperture)))
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=dx))  # cycles/m
    lin = wavelength_m * freq  # cosinus directeur l = λ·fx
    L, M = np.meshgrid(lin, lin)
    return spec.astype(np.complex128), L, M


def normalize_to_directivity(
    power: FloatArray, L: FloatArray, M: FloatArray, dL: float, dM: float
) -> float:
    """Facteur d'échelle tel que |F·norm|² soit la directivité (réf. isotrope)."""
    visible = (L**2 + M**2) < 1.0
    cos_theta = np.sqrt(np.clip(1.0 - L**2 - M**2, 1e-12, None))
    radiated = float(np.sum(power[visible] / cos_theta[visible]) * dL * dM)
    return float(np.sqrt(4.0 * np.pi / radiated))
