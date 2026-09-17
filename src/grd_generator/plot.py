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


def draw_calibration_envelope_map(
    ax: Any,
    grid: UVGrid,
    envelope_dbi: NDArray[np.float64],
    *,
    feed_peaks_uv: NDArray[np.float64] | None = None,
    mission_plan_uv: NDArray[np.float64] | None = None,
    nadir_circle_deg: float = 4.0,
    title: str = "",
) -> Any:
    """Carte de calibrage (lot B3, `examples/example_batch_ku.py --figure`).

    Enveloppe `Σ|co|²` en dBi, isolignes à `max−1 dB`, `max−3 dB` et `39 dBi`,
    crêtes des feeds (`feed_peaks_uv`, un point (u, v) par feed), cercle de
    `nadir_circle_deg` autour du nadir, et, si fourni, centres de faisceaux du
    cas (`mission_plan_uv`, déjà convertis en (u, v) par l'appelant). Renvoie
    la colorbar (à retirer au redraw, comme `_draw_uv_map`).
    """
    gu, gv = grid.meshgrid()
    vmax = float(envelope_dbi.max())
    mesh = ax.pcolormesh(
        gu, gv, envelope_dbi, shading="auto", cmap="viridis", vmin=vmax - _DYN_RANGE_DB, vmax=vmax
    )
    cbar = ax.figure.colorbar(mesh, ax=ax, label="dBi (enveloppe)", shrink=0.8)

    levels = sorted({round(vmax - 3.0, 6), round(vmax - 1.0, 6), 39.0})
    contours = ax.contour(gu, gv, envelope_dbi, levels=levels, colors="white", linewidths=1.0)
    ax.clabel(contours, fmt="%.1f")

    if feed_peaks_uv is not None and len(feed_peaks_uv):
        ax.scatter(
            feed_peaks_uv[:, 0], feed_peaks_uv[:, 1],
            s=10, c="red", marker="+", label="crêtes des feeds",
        )

    r_uv = math.sin(math.radians(nadir_circle_deg))
    t = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(
        r_uv * np.cos(t), r_uv * np.sin(t),
        color="w", lw=1.5, ls="--", label=f"{nadir_circle_deg:.0f}° du nadir",
    )

    if mission_plan_uv is not None and len(mission_plan_uv):
        ax.scatter(
            mission_plan_uv[:, 0], mission_plan_uv[:, 1],
            s=14, c="orange", marker="x", label="centres de faisceaux (cas)",
        )

    ax.set_aspect("equal")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize="x-small")
    return cbar


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
