"""grd_generator — génération du jeu de référence de diagrammes d'antenne (.npz)."""

from grd_generator.schemas import EllipticalSpec, GaussianSpec, UVGrid
from grd_generator.synth import (
    airy_field,
    available_modes,
    elliptical_field,
    field_generator,
    gaussian_field,
    hex_centers_uv,
    optimize_sigma,
    widths_to_sigmas,
)

__all__ = [
    "UVGrid",
    "GaussianSpec",
    "EllipticalSpec",
    "gaussian_field",
    "airy_field",
    "elliptical_field",
    "widths_to_sigmas",
    "available_modes",
    "field_generator",
    "hex_centers_uv",
    "optimize_sigma",
]
