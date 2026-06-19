"""Géométrie de visée : projection sol → coordonnées angulaires satellite.

On travaille en ECEF (Terre sphérique, suffisant pour une vue). Le satellite
géostationnaire est sur l'équateur. On construit un repère d'antenne aligné sur
l'axe de visée (boresight = direction satellite → point visé), puis on exprime
chaque point sol par ses offsets angulaires (u, v) par rapport au boresight :

    u (azimut)   = atan2(d·u_axis, d·boresight)   → Est positif (vers la droite)
    v (élévation)= atan2(d·v_axis, d·boresight)   → Nord positif (vers le haut)

où d est la direction unitaire satellite → point sol.
"""

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

R_EARTH_KM = 6371.0
R_GEO_KM = 42164.0
_NORTH = np.array([0.0, 0.0, 1.0])


def ground_ecef(lat_deg: Array | float, lon_deg: Array | float) -> Array:
    """Convertit lat/lon (degrés) en coordonnées ECEF (km), Terre sphérique."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    x = R_EARTH_KM * np.cos(lat) * np.cos(lon)
    y = R_EARTH_KM * np.cos(lat) * np.sin(lon)
    z = R_EARTH_KM * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def satellite_ecef(lon_deg: float) -> Array:
    """Position ECEF (km) d'un satellite géostationnaire à la longitude donnée."""
    lon = np.radians(lon_deg)
    return np.array([R_GEO_KM * np.cos(lon), R_GEO_KM * np.sin(lon), 0.0])


def boresight_frame(sat_pos: Array, target_pos: Array) -> tuple[Array, Array, Array]:
    """Repère orthonormé d'antenne (u_axis Est, v_axis Nord, boresight)."""
    boresight = target_pos - sat_pos
    boresight = boresight / np.linalg.norm(boresight)
    # v_axis : composante du Nord géographique orthogonale au boresight (vers le haut).
    v_axis = _NORTH - np.dot(_NORTH, boresight) * boresight
    v_axis = v_axis / np.linalg.norm(v_axis)
    # u_axis : complète le repère pour que l'Est aille vers la droite de l'image.
    u_axis = np.cross(boresight, v_axis)
    u_axis = u_axis / np.linalg.norm(u_axis)
    return u_axis, v_axis, boresight


def project_to_angular(
    lat_deg: Array | float,
    lon_deg: Array | float,
    sat_lon_deg: float,
    boresight_lat_deg: float,
    boresight_lon_deg: float,
) -> tuple[Array, Array]:
    """Projette des points sol en offsets angulaires (u, v) en degrés.

    Retourne deux tableaux (u_deg, v_deg) de même forme que l'entrée.
    """
    sat = satellite_ecef(sat_lon_deg)
    target = ground_ecef(boresight_lat_deg, boresight_lon_deg)
    u_axis, v_axis, boresight = boresight_frame(sat, target)

    points = ground_ecef(lat_deg, lon_deg)
    d = points - sat
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)

    du = d @ u_axis
    dv = d @ v_axis
    dz = d @ boresight
    return np.degrees(np.arctan2(du, dz)), np.degrees(np.arctan2(dv, dz))


def project_point(
    lat_deg: float,
    lon_deg: float,
    sat_lon_deg: float,
    boresight_lat_deg: float,
    boresight_lon_deg: float,
) -> tuple[float, float]:
    """Variante scalaire de `project_to_angular` (retourne deux flottants)."""
    u, v = project_to_angular(
        np.array([lat_deg]), np.array([lon_deg]), sat_lon_deg, boresight_lat_deg, boresight_lon_deg
    )
    return float(u[0]), float(v[0])


def project_to_angular_masked(
    lat_deg: Array | float,
    lon_deg: Array | float,
    sat_lon_deg: float,
    boresight_lat_deg: float,
    boresight_lon_deg: float,
) -> tuple[Array, Array]:
    """Comme `project_to_angular`, mais met à NaN les points cachés (face arrière).

    Un point sol est visible depuis le satellite ssi P·S > R_terre² (il est sur
    l'hémisphère qui fait face au satellite).
    """
    u, v = project_to_angular(lat_deg, lon_deg, sat_lon_deg, boresight_lat_deg, boresight_lon_deg)
    sat = satellite_ecef(sat_lon_deg)
    points = ground_ecef(lat_deg, lon_deg)
    visible = (points @ sat) > R_EARTH_KM**2
    u = np.where(visible, u, np.nan)
    v = np.where(visible, v, np.nan)
    return u, v


def earth_limb_angle_deg() -> float:
    """Rayon angulaire du disque terrestre vu depuis le GEO (≈ 8.7°)."""
    return float(np.degrees(np.arcsin(R_EARTH_KM / R_GEO_KM)))


def angular_to_direction(
    u_deg: Array,
    v_deg: Array,
    sat_lon_deg: float,
    boresight_lat_deg: float,
    boresight_lon_deg: float,
) -> Array:
    """Inverse de la projection : (u, v) → direction unitaire ECEF depuis le sat."""
    sat = satellite_ecef(sat_lon_deg)
    target = ground_ecef(boresight_lat_deg, boresight_lon_deg)
    u_axis, v_axis, boresight = boresight_frame(sat, target)

    tan_u = np.tan(np.radians(u_deg))
    tan_v = np.tan(np.radians(v_deg))
    d = boresight + tan_u[..., None] * u_axis + tan_v[..., None] * v_axis
    return np.asarray(d / np.linalg.norm(d, axis=-1, keepdims=True), dtype=float)


def earth_intersection_latlon(
    u_deg: Array,
    v_deg: Array,
    sat_lon_deg: float,
    boresight_lat_deg: float,
    boresight_lon_deg: float,
) -> tuple[Array, Array, BoolArray]:
    """Lancer de rayon sat → sphère terrestre pour des directions (u, v).

    Retourne (lat, lon, hit) : latitude/longitude du point d'impact (NaN hors
    Terre) et un masque booléen des directions qui touchent la Terre.
    """
    sat = satellite_ecef(sat_lon_deg)
    d = angular_to_direction(u_deg, v_deg, sat_lon_deg, boresight_lat_deg, boresight_lon_deg)

    b = 2.0 * (d @ sat)
    c = float(sat @ sat) - R_EARTH_KM**2
    disc = b**2 - 4.0 * c
    hit = disc >= 0.0
    sqrt_disc = np.sqrt(np.where(hit, disc, 0.0))
    t = (-b - sqrt_disc) / 2.0
    hit = hit & (t > 0.0)

    point = sat + t[..., None] * d
    lat = np.degrees(np.arcsin(np.clip(point[..., 2] / R_EARTH_KM, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(point[..., 1], point[..., 0]))
    lat = np.asarray(np.where(hit, lat, np.nan), dtype=float)
    lon = np.asarray(np.where(hit, lon, np.nan), dtype=float)
    return lat, lon, hit


__all__ = [
    "R_EARTH_KM",
    "R_GEO_KM",
    "ground_ecef",
    "satellite_ecef",
    "boresight_frame",
    "project_to_angular",
    "project_point",
    "project_to_angular_masked",
    "earth_limb_angle_deg",
    "angular_to_direction",
    "earth_intersection_latlon",
]
