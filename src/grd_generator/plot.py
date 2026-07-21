"""Helpers matplotlib partagés par `ReflectorStudio` : cartes (u, v) et zone de service.

Trace des cartes de directivité et de phase en espace angulaire (u, v), ainsi que le
contour de la zone de service. matplotlib est un extra optionnel `viz`.
"""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from grd_generator.reflector.zone import ServiceZone
from grd_generator.schemas import UVGrid

_DYN_RANGE_DB = 40.0  # plage dynamique affichée sous la crête


def _draw_uv_map(ax: Any, grid: UVGrid, dbi: NDArray[np.float64], title: str) -> Any:
    """Carte de directivité (dBi) en (u, v). Renvoie la colorbar (à retirer au redraw)."""
    gu, gv = grid.meshgrid()
    vmax = float(dbi.max())
    mesh = ax.pcolormesh(
        gu, gv, dbi, shading="auto", cmap="viridis", vmin=vmax - _DYN_RANGE_DB, vmax=vmax
    )
    ax.set_aspect("equal")
    ax.set_xlabel("u (deg)")
    ax.set_ylabel("v (deg)")
    ax.set_title(title)
    return ax.figure.colorbar(mesh, ax=ax, label="dBi", shrink=0.8)


def _draw_phase_map(ax: Any, grid: UVGrid, field: NDArray[np.complex128], title: str) -> Any:
    """Carte de phase (rad, −π..π) en (u, v). Renvoie la colorbar (à retirer au redraw)."""
    gu, gv = grid.meshgrid()
    phase = np.angle(field)
    mesh = ax.pcolormesh(gu, gv, phase, shading="auto", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax.set_aspect("equal")
    ax.set_xlabel("u (deg)")
    ax.set_ylabel("v (deg)")
    ax.set_title(title)
    return ax.figure.colorbar(mesh, ax=ax, label="phase (rad)", shrink=0.8)


def draw_service_zone_uv(
    ax: Any,
    zone: ServiceZone,
    *,
    color: str = "w",
    lw: float = 1.5,
    to_direction_cosine: bool = False,
) -> None:
    """Trace le contour du disque de zone de service sur des axes (u, v).

    Par défaut (`to_direction_cosine=False`), le disque est tracé tel quel :
    `zone.radius_deg` est un rayon angulaire en degrés, dans des axes également
    en degrés. `to_direction_cosine=True` (utilisé par `ReflectorStudio`)
    convertit ce rayon en rayon de cosinus directeur `sin(rad(radius_deg))`
    avant traçage, le centre restant inchangé : exact pour un cône de
    demi-angle constant centré à l'origine (seul cas utilisé côté reflector,
    `center_uv` y valant toujours `(0.0, 0.0)`).
    """
    u, v = zone.boundary_uv()
    if to_direction_cosine:
        cu, cv = zone.center_uv
        scale = math.sin(math.radians(zone.radius_deg)) / zone.radius_deg
        u = cu + (u - cu) * scale
        v = cv + (v - cv) * scale
    ax.plot(u, v, color=color, lw=lw, ls="--", label="zone de service")
