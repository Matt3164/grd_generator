"""Rendu matplotlib d'un array généré (.npz) : direct (u, v), phase et vue satellite.

Six panneaux (grille 2×3) pour un `.npz` (référence ou calibré) :
  1. directivité d'un élément en espace angulaire (u, v),
  2. phase de cet élément (u, v),
  3. répartition des barycentres des patterns en (u, v),
  4. enveloppe = directivité **max sur tous les patterns** `10·log10(maxᵢ|Eᵢ|²)`,
  5. enveloppe sur la **vue satellite** (Terre vue du sat, repère (u, v) : disque
     du limbe + côtes projetées), 6. barycentres + empreintes des patterns idem.

Les vues « Terre » sont dans le **repère angulaire de génération** (u, v) : la Terre
y est un disque (limbe ≈ 8,7°), les côtes sont projetées en avant
(`project_to_angular_masked`), et les données (u, v) sont tracées directement —
aucune projection inverse. Côtes Natural Earth (50 m) vendored, hors-ligne.

Usage :
    uv run grd-plot --npz data/processed/reference_array.npz
    uv run grd-plot --npz data/processed/cal0000.npz --sat-lon 3.0
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    # pyplot choisit le backend depuis MPLBACKEND à l'import (Agg en test).
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib requis pour le rendu : installez l'extra viz "
        "(`uv sync --extra viz` ou `pip install 'grd_generator[viz]'`)."
    ) from exc

from grd_generator.geometry import (
    earth_limb_angle_deg,
    project_point,
    project_to_angular_masked,
)
from grd_generator.logger import configure_logging, logger
from grd_generator.schemas import UVGrid

_LAND_GEOJSON = Path(__file__).resolve().parents[2] / "data" / "ne_50m_land.geojson"
_REQUIRED_KEYS = ("u_min", "u_max", "v_min", "v_max", "n_u", "n_v", "fields")
_FLOOR = 1e-12  # plancher de |E|² pour éviter log10(0)
_DYN_RANGE_DB = 40.0  # plage dynamique affichée sous la crête


def load_array(
    path: str | Path,
) -> tuple[UVGrid, NDArray[np.complex128], NDArray[np.float64] | None]:
    """Lit un `.npz` : (grille, champs complexes, centres_uv|None).

    `ValueError` si une clé de grille/champs manque. `centers_uv` est lu s'il est
    présent (référence et calibré l'écrivent), sinon `None`.
    """
    data = np.load(Path(path), allow_pickle=False)
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"{path} : clés absentes du .npz : {missing}")
    grid = UVGrid(
        u_min=float(data["u_min"]),
        u_max=float(data["u_max"]),
        v_min=float(data["v_min"]),
        v_max=float(data["v_max"]),
        n_u=int(data["n_u"]),
        n_v=int(data["n_v"]),
    )
    fields = np.asarray(data["fields"], dtype=np.complex128)
    centers = np.asarray(data["centers_uv"], dtype=float) if "centers_uv" in data else None
    return grid, fields, centers


def coastline_rings() -> list[NDArray[np.float64]]:
    """Anneaux extérieurs (lon, lat) des terres émergées (GeoJSON vendored)."""
    with open(_LAND_GEOJSON) as f:
        collection = json.load(f)
    rings: list[NDArray[np.float64]] = []
    for feature in collection["features"]:
        geom = feature["geometry"]
        polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for polygon in polygons:
            rings.append(np.asarray(polygon[0], dtype=float))
    return rings


def envelope_max_dbi(fields: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Enveloppe = **max sur tous les patterns** : `10·log10(maxᵢ|Eᵢ|²)` en dBi."""
    power = (np.abs(fields) ** 2).max(axis=0)
    return np.asarray(10.0 * np.log10(np.maximum(power, _FLOOR)), dtype=np.float64)


def element_centers(
    grid: UVGrid, fields: NDArray[np.complex128], centers: NDArray[np.float64] | None
) -> NDArray[np.float64]:
    """Barycentres (u, v) des patterns : `centers_uv` du fichier, sinon le pic de |E|."""
    if centers is not None and len(centers) == len(fields):
        return centers
    u, v = grid.axes()
    out = np.empty((len(fields), 2), dtype=float)
    for k, field in enumerate(fields):
        iv, iu = np.unravel_index(int(np.argmax(np.abs(field))), field.shape)
        out[k] = (u[iu], v[iv])
    return out


def element_peaks_dbi(fields: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Crête (dBi) de chaque pattern : `20·log10(max|Eᵢ|)`."""
    peaks = 20.0 * np.log10(np.maximum(np.abs(fields).reshape(len(fields), -1).max(axis=1), 1e-12))
    return np.asarray(peaks, dtype=np.float64)


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


def _draw_centers_uv(ax: Any, centers: NDArray[np.float64], peaks: NDArray[np.float64]) -> None:
    """Répartition des barycentres des patterns en (u, v), colorés par crête."""
    sc = ax.scatter(centers[:, 0], centers[:, 1], c=peaks, s=18, cmap="plasma")
    ax.set_aspect("equal")
    ax.set_xlabel("u (deg)")
    ax.set_ylabel("v (deg)")
    ax.set_title(f"Barycentres des patterns ({len(centers)})")
    ax.figure.colorbar(sc, ax=ax, label="crête (dBi)", shrink=0.8)


def _draw_globe_base(ax: Any, *, sat_lon: float) -> None:
    """Disque terrestre (limbe) + côtes projetées en (u, v) — vue satellite (nadir)."""
    limb = earth_limb_angle_deg()
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(limb * np.cos(theta), limb * np.sin(theta), color="0.2", linewidth=1.0)
    for ring in coastline_rings():
        u, v = project_to_angular_masked(ring[:, 1], ring[:, 0], sat_lon, 0.0, sat_lon)
        ax.plot(u, v, color="0.55", linewidth=0.4)
    margin = limb + 0.5
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.set_xlabel("u (deg)")
    ax.set_ylabel("v (deg)")


def draw_zone_and_antenna(
    ax: Any,
    *,
    sat_lon: float,
    zone_center: tuple[float, float],
    zone_radius: float,
    antenna_lat: float,
    antenna_lon: float,
) -> None:
    """Disque de zone (cercle u, v) + pointage antenne (étoile, direction projetée)."""
    theta = np.linspace(0.0, 2.0 * np.pi, 181)
    ax.plot(
        zone_center[0] + zone_radius * np.cos(theta),
        zone_center[1] + zone_radius * np.sin(theta),
        color="red",
        linewidth=1.3,
    )
    au, av = project_point(antenna_lat, antenna_lon, sat_lon, 0.0, sat_lon)
    ax.plot([au], [av], marker="*", color="red", markersize=13, linestyle="")


def _draw_earth_envelope(
    ax: Any, grid: UVGrid, envelope_dbi: NDArray[np.float64], *, sat_lon: float
) -> None:
    """Enveloppe (dBi) sur la vue satellite (u, v), disque + côtes en fond."""
    gu, gv = grid.meshgrid()
    vmax = float(envelope_dbi.max())
    mesh = ax.pcolormesh(
        gu, gv, envelope_dbi, shading="auto", cmap="viridis", vmin=vmax - _DYN_RANGE_DB, vmax=vmax
    )
    ax.figure.colorbar(mesh, ax=ax, label="dBi", shrink=0.8)
    _draw_globe_base(ax, sat_lon=sat_lon)
    ax.set_title("Enveloppe — vue satellite")


def draw_earth_pattern_footprints(
    ax: Any,
    grid: UVGrid,
    fields: NDArray[np.complex128],
    centers: NDArray[np.float64],
    *,
    sat_lon: float,
    drop_db: float = 1.0,
) -> None:
    """Barycentres + empreinte `−drop_db` dB (du max de chaque pattern), vue satellite."""
    gu, gv = grid.meshgrid()
    for field in fields:
        dbi = 10.0 * np.log10(np.maximum(np.abs(field) ** 2, _FLOOR))
        level = float(dbi.max()) - drop_db
        ax.contour(gu, gv, dbi, levels=[level], colors="0.5", linewidths=0.4)
    ax.plot(centers[:, 0], centers[:, 1], marker=".", color="C0", markersize=4, linestyle="")
    _draw_globe_base(ax, sat_lon=sat_lon)
    ax.set_title(f"Patterns — vue satellite (barycentres + empreintes −{drop_db:g} dB)")


def draw_earth_envelope_contours(
    ax: Any,
    grid: UVGrid,
    envelope_dbi: NDArray[np.float64],
    *,
    sat_lon: float,
    step_db: float = 5.0,
) -> None:
    """Enveloppe en isolignes (tous les `step_db` dB) sur la vue satellite (u, v)."""
    gu, gv = grid.meshgrid()
    vmax = float(envelope_dbi.max())
    levels = np.arange(vmax - _DYN_RANGE_DB, vmax + step_db, step_db)
    cs = ax.contour(gu, gv, envelope_dbi, levels=levels, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    _draw_globe_base(ax, sat_lon=sat_lon)
    ax.set_title(f"Enveloppe — vue satellite, isolignes ({step_db:g} dB)")


def render(
    npz_path: str | Path,
    *,
    out: str | Path | None = None,
    element: int = 0,
    sat_lon: float = 3.0,
    show: bool = True,
) -> Path:
    """Trace 6 panneaux (directivité, phase, barycentres, enveloppe, vue sat) ; écrit un PNG.

    Les vues « Terre » sont en repère angulaire (u, v) à la longitude `sat_lon`.
    `out` défaut = `<npz>.png`.
    """
    grid, fields, centers_uv = load_array(npz_path)
    if not 0 <= element < len(fields):
        raise IndexError(f"élément {element} hors plage (0..{len(fields) - 1})")

    centers = element_centers(grid, fields, centers_uv)
    peaks = element_peaks_dbi(fields)
    element_dbi = np.asarray(10.0 * np.log10(np.maximum(np.abs(fields[element]) ** 2, _FLOOR)))
    envelope_dbi = envelope_max_dbi(fields)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    _draw_uv_map(axes[0, 0], grid, element_dbi, f"Élément #{element} — directivité")
    _draw_phase_map(axes[0, 1], grid, fields[element], f"Élément #{element} — phase")
    _draw_centers_uv(axes[0, 2], centers, peaks)
    _draw_uv_map(axes[1, 0], grid, envelope_dbi, f"Enveloppe — max sur {len(fields)} patterns")
    _draw_earth_envelope(axes[1, 1], grid, envelope_dbi, sat_lon=sat_lon)
    draw_earth_pattern_footprints(axes[1, 2], grid, fields, centers, sat_lon=sat_lon)
    fig.suptitle(Path(npz_path).name)
    fig.tight_layout()

    target = Path(out) if out is not None else Path(npz_path).with_suffix(".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=110)
    logger.info("plot_written", file=str(target), n_elements=len(fields))
    if show:
        plt.show()
    plt.close(fig)
    return target


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Rendu matplotlib d'un array généré (.npz).")
    parser.add_argument("--npz", required=True, help="fichier .npz à tracer")
    parser.add_argument("--out", default=None, help="PNG de sortie (défaut : <npz>.png)")
    parser.add_argument("--element", type=int, default=0, help="indice de l'élément tracé")
    parser.add_argument("--sat-lon", type=float, default=3.0, help="longitude du GEO (deg)")
    parser.add_argument("--no-show", action="store_true", help="ne pas ouvrir de fenêtre")
    args = parser.parse_args()
    render(
        args.npz,
        out=args.out,
        element=args.element,
        sat_lon=args.sat_lon,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
