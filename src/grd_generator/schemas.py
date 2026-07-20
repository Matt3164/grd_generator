"""Schémas du générateur : diagrammes d'antennes en espace (u, v)."""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

ComplexField = NDArray[np.complex128]  # champ E(u, v) sur la grille
DirectivityMap = NDArray[np.float64]  # directivité en dBi sur la grille


class UVGrid(BaseModel):
    """Maillage angulaire régulier (u, v) vu du satellite.

    `ReflectorStudio` (AFR) l'exprime en cosinus directeurs (u,v) =
    (sinθcosφ, sinθsinφ), conforme GRASP IGRID=1. `UVGrid` lui-même est
    agnostique de l'unité — c'est un maillage régulier générique.
    """

    u_min: float
    u_max: float
    v_min: float
    v_max: float
    n_u: int = Field(..., gt=1)
    n_v: int = Field(..., gt=1)

    def axes(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Axes 1-D (u, v) régulièrement espacés."""
        u = np.linspace(self.u_min, self.u_max, self.n_u)
        v = np.linspace(self.v_min, self.v_max, self.n_v)
        return u, v

    def meshgrid(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Grilles 2-D (gu, gv) de forme (n_v, n_u) ; v croît par ligne."""
        u, v = self.axes()
        gu, gv = np.meshgrid(u, v)
        return gu, gv


__all__ = [
    "ComplexField",
    "DirectivityMap",
    "UVGrid",
]
