"""Rendu matplotlib d'un array généré (.npz) : direct (u, v) et sur Terre.

Trois panneaux pour un `.npz` (référence ou calibré) :
  1. directivité d'un élément en espace angulaire (u, v),
  2. enveloppe = directivité max atteignable `10·log10(Σ|Eᵢ|²)` en (u, v),
  3. cette enveloppe projetée sur la Terre, côtes Natural Earth (50 m) en fond.

Hors-ligne : les côtes sont lues depuis le GeoJSON vendored `data/ne_50m_land.geojson`
(aucune dépendance réseau / cartopy). matplotlib est un extra optionnel `viz`.

Usage :
    uv run grd-plot --npz data/processed/reference_array.npz
    uv run grd-plot --npz data/processed/cal0000.npz --boresight-lat 46.6 --boresight-lon 2.5
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

from grd_generator.geometry import earth_intersection_latlon
from grd_generator.logger import configure_logging, logger
from grd_generator.schemas import UVGrid
from grd_generator.synth import combined_max_directivity_dbi, directivity_dbi_from_field

_LAND_GEOJSON = Path(__file__).resolve().parents[2] / "data" / "ne_50m_land.geojson"
_REQUIRED_KEYS = ("u_min", "u_max", "v_min", "v_max", "n_u", "n_v", "fields")


def load_array(path: str | Path) -> tuple[UVGrid, NDArray[np.complex128]]:
    """Lit un `.npz` (grille + champs complexes). `ValueError` si une clé manque."""
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
    return grid, fields


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


def _draw_uv_map(ax: Any, grid: UVGrid, dbi: NDArray[np.float64], title: str) -> None:
    """Carte de directivité (dBi) en espace angulaire (u, v)."""
    gu, gv = grid.meshgrid()
    vmax = float(dbi.max())
    mesh = ax.pcolormesh(gu, gv, dbi, shading="auto", cmap="viridis", vmin=vmax - 40.0, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("u (deg)")
    ax.set_ylabel("v (deg)")
    ax.set_title(title)
    ax.figure.colorbar(mesh, ax=ax, label="dBi", shrink=0.8)


def _draw_earth_map(
    ax: Any,
    grid: UVGrid,
    envelope_dbi: NDArray[np.float64],
    *,
    sat_lon: float,
    boresight_lat: float,
    boresight_lon: float,
) -> None:
    """Enveloppe projetée sur la Terre (lat, lon), côtes en fond."""
    gu, gv = grid.meshgrid()
    lat, lon, hit = earth_intersection_latlon(gu, gv, sat_lon, boresight_lat, boresight_lon)
    hit_lon = lon[hit]
    hit_lat = lat[hit]
    hit_dbi = envelope_dbi[hit]

    if hit_dbi.size:
        vmax = float(hit_dbi.max())
        sc = ax.scatter(
            hit_lon, hit_lat, c=hit_dbi, s=6, cmap="viridis", vmin=vmax - 40.0, vmax=vmax
        )
        ax.figure.colorbar(sc, ax=ax, label="dBi", shrink=0.8)
        # marge autour de l'empreinte, bornée au globe
        pad = 5.0
        lon_lo, lon_hi = float(hit_lon.min()) - pad, float(hit_lon.max()) + pad
        lat_lo, lat_hi = float(hit_lat.min()) - pad, float(hit_lat.max()) + pad
    else:  # pragma: no cover
        lon_lo, lon_hi, lat_lo, lat_hi = -180.0, 180.0, -90.0, 90.0

    for ring in coastline_rings():
        ax.plot(ring[:, 0], ring[:, 1], color="0.35", linewidth=0.4)

    ax.set_xlim(max(-180.0, lon_lo), min(180.0, lon_hi))
    ax.set_ylim(max(-90.0, lat_lo), min(90.0, lat_hi))
    ax.set_aspect("equal")
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title("Enveloppe sur Terre")


def render(
    npz_path: str | Path,
    *,
    out: str | Path | None = None,
    element: int = 0,
    sat_lon: float = 3.0,
    boresight_lat: float = 0.0,
    boresight_lon: float | None = None,
    show: bool = True,
) -> Path:
    """Trace élément + enveloppe (u, v) et enveloppe sur Terre ; écrit un PNG.

    `boresight_lon` défaut = `sat_lon` (visée nadir). `out` défaut = `<npz>.png`.
    """
    grid, fields = load_array(npz_path)
    if not 0 <= element < len(fields):
        raise IndexError(f"élément {element} hors plage (0..{len(fields) - 1})")
    if boresight_lon is None:
        boresight_lon = sat_lon

    element_dbi = directivity_dbi_from_field(fields[element])
    envelope_dbi = combined_max_directivity_dbi(list(fields))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    _draw_uv_map(axes[0], grid, element_dbi, f"Élément #{element}")
    _draw_uv_map(axes[1], grid, envelope_dbi, "Enveloppe (max atteignable)")
    _draw_earth_map(
        axes[2],
        grid,
        envelope_dbi,
        sat_lon=sat_lon,
        boresight_lat=boresight_lat,
        boresight_lon=boresight_lon,
    )
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
    parser.add_argument("--boresight-lat", type=float, default=0.0, help="latitude de visée (deg)")
    parser.add_argument(
        "--boresight-lon", type=float, default=None, help="longitude de visée (défaut : sat-lon)"
    )
    parser.add_argument("--no-show", action="store_true", help="ne pas ouvrir de fenêtre")
    args = parser.parse_args()
    render(
        args.npz,
        out=args.out,
        element=args.element,
        sat_lon=args.sat_lon,
        boresight_lat=args.boresight_lat,
        boresight_lon=args.boresight_lon,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
