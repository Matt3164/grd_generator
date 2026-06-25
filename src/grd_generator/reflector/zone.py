"""Zone de service : disque angulaire (u,v) à couvrir, affiché sur les displays."""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from grd_generator.schemas import UVGrid


class ServiceZone(BaseModel):
    """Disque angulaire centré (par défaut sur le boresight nadir)."""

    radius_deg: float = Field(..., ge=6.0, le=14.0)
    center_uv: tuple[float, float] = (0.0, 0.0)

    def mask(self, grid: UVGrid) -> NDArray[np.bool_]:
        """Masque booléen (n_v, n_u) des points dans le disque."""
        gu, gv = grid.meshgrid()
        cu, cv = self.center_uv
        inside: NDArray[np.bool_] = ((gu - cu) ** 2 + (gv - cv) ** 2) <= self.radius_deg**2
        return inside

    def boundary_uv(self, n: int = 361) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Cercle fermé (u, v) en degrés pour tracer le contour de zone."""
        t = np.linspace(0.0, 2.0 * np.pi, n)
        cu, cv = self.center_uv
        u = cu + self.radius_deg * np.cos(t)
        v = cv + self.radius_deg * np.sin(t)
        return u, v
