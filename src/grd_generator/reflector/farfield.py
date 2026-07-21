"""Champ lointain : FFT du champ d'ouverture, normalisé en directivité.

Le far-field est la TF 2D du champ d'ouverture, échantillonné en cosinus
directeurs (l, m) = (sinθcosφ, sinθsinφ). La normalisation impose
|E|² = directivité linéaire (convention isotrope du projet) :
D = 4π·|F|² / ∮|F|² dΩ, dΩ = dl·dm / cosθ sur la région visible l²+m²<1.
"""

import numpy as np
from numpy.typing import NDArray

from grd_generator.schemas import ComplexField, UVGrid

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def far_field_fft(
    aperture: ComplexArray, dx: float, wavelength_m: float
) -> tuple[ComplexArray, FloatArray, FloatArray]:
    """TF 2D centrée du champ d'ouverture → (F, L, M) en cosinus directeurs."""
    n = aperture.shape[0]
    ff = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(aperture)))
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=dx))  # cycles/m
    lin = wavelength_m * freq  # cosinus directeur l = λ·fx
    L, M = np.meshgrid(lin, lin)
    return ff.astype(np.complex128), L, M


def normalize_to_directivity(
    power: FloatArray, L: FloatArray, M: FloatArray, dL: float, dM: float
) -> float:
    """Facteur d'échelle tel que |F·norm|² soit la directivité (réf. isotrope)."""
    visible = (L**2 + M**2) < 1.0
    cos_theta = np.sqrt(np.clip(1.0 - L**2 - M**2, 1e-12, None))
    radiated = float(np.sum(power[visible] / cos_theta[visible]) * dL * dM)
    return float(np.sqrt(4.0 * np.pi / radiated))


def ludwig3_co_cross(
    Fx: ComplexArray, Fy: ComplexArray, L: FloatArray, M: FloatArray
) -> tuple[ComplexArray, ComplexArray]:
    """Décomposition Ludwig-3 (référence x) du far-field cartésien transverse."""
    phi = np.arctan2(M, L)
    sin_t = np.sqrt(np.clip(L**2 + M**2, 0.0, 1.0))
    cos_t = np.sqrt(np.clip(1.0 - sin_t**2, 0.0, 1.0))
    e_theta = (Fx * np.cos(phi) + Fy * np.sin(phi)) * cos_t
    e_phi = -Fx * np.sin(phi) + Fy * np.cos(phi)
    e_co = e_theta * np.cos(phi) - e_phi * np.sin(phi)
    e_cross = e_theta * np.sin(phi) + e_phi * np.cos(phi)
    return e_co.astype(np.complex128), e_cross.astype(np.complex128)


def _bilinear(
    values: ComplexArray,
    axis: FloatArray,
    lq: FloatArray,
    mq: FloatArray,
) -> ComplexArray:
    """Interpolation bilinéaire de `values` (grille régulière `axis`×`axis`).

    Assumes same axis for both dimensions; clamps queries to edge cells.
    """
    ix = np.clip(np.searchsorted(axis, lq) - 1, 0, axis.size - 2)
    iy = np.clip(np.searchsorted(axis, mq) - 1, 0, axis.size - 2)
    x0, x1 = axis[ix], axis[ix + 1]
    y0, y1 = axis[iy], axis[iy + 1]
    tx = (lq - x0) / (x1 - x0)
    ty = (mq - y0) / (y1 - y0)
    v00 = values[iy, ix]
    v01 = values[iy, ix + 1]
    v10 = values[iy + 1, ix]
    v11 = values[iy + 1, ix + 1]
    out: ComplexArray = (
        v00 * (1 - tx) * (1 - ty) + v01 * tx * (1 - ty)
        + v10 * (1 - tx) * ty + v11 * tx * ty
    )
    return out


def resample_to_uvgrid(field: ComplexArray, lin_axis: FloatArray, grid: UVGrid) -> ComplexField:
    """Rééchantillonne le far-field sur la grille (u,v) — reflector only.

    `grid` est déjà exprimée en cosinus directeurs (u,v) = (l,m) : pas de
    conversion `sin` ici, contrairement à l'ancienne convention (u,v) en
    degrés.
    """
    gu, gv = grid.meshgrid()
    out: ComplexField = _bilinear(field, lin_axis, gu, gv).astype(np.complex128)
    return out
