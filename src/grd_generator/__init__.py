"""grd_generator — générateur autonome de diagrammes de rayonnement par simulation
de réflecteur alimenté par réseau (AFR, .npz).
"""

from grd_generator.schemas import UVGrid
from grd_generator.synth import combined_max_directivity_dbi, hex_centers_uv

__all__ = [
    "UVGrid",
    "combined_max_directivity_dbi",
    "hex_centers_uv",
]
