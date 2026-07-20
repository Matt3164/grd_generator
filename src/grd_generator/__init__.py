"""grd_generator — génération du jeu de référence de diagrammes d'antenne (.npz)."""

from grd_generator.generate import (
    auto_spacing,
    generate_reference,
    write_reference,
)
from grd_generator.scenario import (
    FRANCE,
    SatFrameGeometry,
    Scenario,
    sat_frame_geometry,
)
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
    "Scenario",
    "FRANCE",
    "SatFrameGeometry",
    "sat_frame_geometry",
    "gaussian_field",
    "airy_field",
    "elliptical_field",
    "widths_to_sigmas",
    "available_modes",
    "field_generator",
    "hex_centers_uv",
    "optimize_sigma",
    "auto_spacing",
    "write_reference",
    "generate_reference",
]
