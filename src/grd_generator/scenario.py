"""Scénario (forme/visée sur la Terre) et repère satellite (nadir) dérivé.

Un `Scenario` est une dataclass autonome : il porte la longitude orbitale, la
visée de l'antenne et la zone de service. `sat_frame_geometry` le traduit en
géométrie de repère SAT prête pour la synthèse : centre de zone rabattu sous le
limbe et grille (u, v) suivant la zone.
"""

import math
from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from grd_generator.geometry import (
    earth_intersection_latlon,
    earth_limb_angle_deg,
    project_point,
)
from grd_generator.schemas import UVGrid

_GRID_MARGIN_DEG = 0.5  # marge de grille autour de la zone de service


@dataclass(frozen=True)
class Scenario:
    """Région de couverture : visée d'antenne et zone de service sur la Terre."""

    name: str
    sat_lon_deg: float  # longitude orbitale du GEO
    antenna_latlon: tuple[float, float]  # visée de l'antenne (lat, lon)
    zone_radius_deg: float = 6.0  # rayon de la zone de service (°), > 0 ; doit tenir sous le limbe
    limb_margin_deg: float = 0.2  # marge gardant la zone strictement sous le limbe

    def __post_init__(self) -> None:
        """Échoue tôt sur un rayon non physique (la tenue sous le limbe est vérifiée
        par `sat_frame_geometry`, qui connaît la géométrie)."""
        if self.zone_radius_deg <= 0.0:
            raise ValueError(
                f"scénario {self.name!r} : zone_radius_deg={self.zone_radius_deg} doit être > 0"
            )


FRANCE = Scenario(
    name="france",
    sat_lon_deg=3.0,
    antenna_latlon=(46.6, 2.5),
    zone_radius_deg=6.0,
    limb_margin_deg=0.2,
)


class RenderBoresight(TypedDict):
    """kwargs de boresight passés tels quels aux fonctions de rendu."""

    sat_lon_deg: float
    boresight_lat_deg: float
    boresight_lon_deg: float


@dataclass(frozen=True)
class SatFrameGeometry:
    """Repère SAT (nadir) d'un scénario, prêt pour la synthèse."""

    sat_lon_deg: float
    zone_center: tuple[float, float]
    zone_radius_deg: float
    grid: UVGrid

    def render_kwargs(self) -> RenderBoresight:
        """kwargs de boresight nadir (repère SAT)."""
        return {
            "sat_lon_deg": self.sat_lon_deg,
            "boresight_lat_deg": 0.0,
            "boresight_lon_deg": self.sat_lon_deg,
        }

    def project_latlon(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        """Projette un point sol (lat, lon) en (u, v) dans ce repère SAT (nadir)."""
        return project_point(lat_deg, lon_deg, self.sat_lon_deg, 0.0, self.sat_lon_deg)

    def ground_latlon(self, u_deg: float, v_deg: float) -> tuple[float, float] | None:
        """Inverse de `project_latlon` : (u, v) → point sol (lat, lon), ou None."""
        lat, lon, hit = earth_intersection_latlon(
            np.array([u_deg]), np.array([v_deg]), self.sat_lon_deg, 0.0, self.sat_lon_deg
        )
        return (float(lat[0]), float(lon[0])) if bool(hit[0]) else None


def sat_frame_geometry(scenario: Scenario, n_u: int = 161, n_v: int = 161) -> SatFrameGeometry:
    """Construit le repère SAT d'un scénario pour la synthèse de diagramme.

    Le centre de zone = projection de la visée antenne dans le repère nadir,
    rabattu vers le nadir le long de cette direction pour que le disque de
    service reste sous le limbe : `dist(centre, nadir) ≤ limbe − marge − rayon`.

    Lève `ValueError` si la zone ne tient pas sous le limbe.
    """
    sat_lon = scenario.sat_lon_deg
    radius = scenario.zone_radius_deg
    lat, lon = scenario.antenna_latlon

    target_uv = project_point(lat, lon, sat_lon, 0.0, sat_lon)
    max_center_dist = earth_limb_angle_deg() - scenario.limb_margin_deg - radius
    if max_center_dist < 0:
        raise ValueError(
            f"scénario {scenario.name!r} : zone_radius_deg={radius} ne tient pas sous "
            f"le limbe ({earth_limb_angle_deg():.2f}°), marge "
            f"{scenario.limb_margin_deg}° comprise"
        )
    target_dist = math.hypot(*target_uv)
    scale = min(1.0, max_center_dist / target_dist) if target_dist > 0.0 else 0.0
    zone_center = (target_uv[0] * scale, target_uv[1] * scale)

    half_span = radius + _GRID_MARGIN_DEG
    grid = UVGrid(
        u_min=zone_center[0] - half_span,
        u_max=zone_center[0] + half_span,
        v_min=zone_center[1] - half_span,
        v_max=zone_center[1] + half_span,
        n_u=n_u,
        n_v=n_v,
    )
    return SatFrameGeometry(
        sat_lon_deg=sat_lon,
        zone_center=zone_center,
        zone_radius_deg=radius,
        grid=grid,
    )


__all__ = [
    "Scenario",
    "FRANCE",
    "RenderBoresight",
    "SatFrameGeometry",
    "sat_frame_geometry",
]
