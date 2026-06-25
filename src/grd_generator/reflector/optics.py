"""Optique géométrique : feed → champ d'ouverture du réflecteur offset.

Modèle aperture-field. Un point d'ouverture (X, Y) (m, mesuré depuis l'axe du
paraboloïde parent) est illuminé par le feed sous l'angle polaire
ψ = 2·arctan(ρ/2F) (ρ = ‖(X,Y)‖) ; l'amplitude suit le diagramme de feed cos^q(ψ).
Un feed décalé de (δx, δy) dans le plan focal impose un gradient de phase linéaire
(beam deviation factor) qui dépointe le faisceau secondaire.
"""

import numpy as np
from numpy.typing import NDArray

from grd_generator.reflector.spec import ReflectorSpec

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def aperture_grid(
    spec: ReflectorSpec, n_aperture: int, pad_factor: int
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_], float]:
    """Grille carrée d'ouverture (m) + masque disque + pas dx.

    `n_aperture` échantillons sur le diamètre ; zéro-padding `pad_factor`× pour
    la résolution angulaire du far-field. Disque de diamètre D centré sur l'offset.
    """
    dx = spec.diameter_m / n_aperture
    n = n_aperture * pad_factor
    coords = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(coords, coords)
    r2 = X**2 + (Y - spec.aperture_center_y_m) ** 2
    inside: NDArray[np.bool_] = r2 <= (spec.diameter_m / 2.0) ** 2
    return X, Y, inside, dx


def illumination_angle(X: FloatArray, Y: FloatArray, focal_length_m: float) -> FloatArray:
    """Angle polaire ψ vu du feed pour un point d'ouverture : ψ = 2·arctan(ρ/2F)."""
    rho = np.hypot(X, Y)
    angle: FloatArray = 2.0 * np.arctan(rho / (2.0 * focal_length_m))
    return angle


def aperture_field(
    spec: ReflectorSpec,
    feed_xy: tuple[float, float],
    X: FloatArray,
    Y: FloatArray,
    inside: NDArray[np.bool_],
    q: float,
) -> ComplexArray:
    """Champ scalaire d'ouverture pour un feed : taper cos^q(ψ) + tilt de phase."""
    psi = illumination_angle(X, Y, spec.focal_length_m)
    amp = np.cos(psi) ** q
    k = 2.0 * np.pi / spec.wavelength_m
    dxf, dyf = feed_xy
    tilt = -k * spec.beam_deviation_factor * (dxf * X + dyf * Y) / spec.focal_length_m
    field: ComplexArray = (inside * amp * np.exp(1j * tilt)).astype(np.complex128)
    return field


def aperture_pol_vectors(
    spec: ReflectorSpec, X: FloatArray, Y: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Composantes (ex, ey) du vecteur de polarisation réfléchi (feed x-polarisé).

    Réflexion GO de x̂ sur la normale du paraboloïde : e_r = 2(n̂·x̂)n̂ − x̂.
    Source géométrique de la cross-pol (offset / plans diagonaux).
    """
    f = spec.focal_length_m
    nx = -X / (2.0 * f)
    ny = -Y / (2.0 * f)
    nz = np.ones_like(X)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    ndotx = nx  # n̂ · x̂
    ex: FloatArray = 2.0 * ndotx * nx - 1.0
    ey: FloatArray = 2.0 * ndotx * ny
    return ex, ey
