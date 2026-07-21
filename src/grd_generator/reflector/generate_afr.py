"""Générateur offline AFR → .npz (script). Couverture exclue : orchestration + I/O."""

from pathlib import Path
from typing import Any

import numpy as np

from grd_generator.logger import configure_logging, logger
from grd_generator.reflector import (
    FeedSpec,
    ReflectorSpec,
    ServiceZone,
    hex_feed_positions,
    synthesize_reflector_fields,
)
from grd_generator.schemas import UVGrid

DEFAULT_OUTPUT = Path("data/processed/reflector_array.npz")


def build_reflector_reference(
    spec: ReflectorSpec, feeds: FeedSpec, zone: ServiceZone, grid: UVGrid
) -> dict[str, Any]:
    """Construit le dict de référence (pur, testable) : champs + métadonnées."""
    co, cross = synthesize_reflector_fields(spec, feeds, grid)
    u, v = grid.axes()
    return {
        "u_axis": u,
        "v_axis": v,
        "fields": np.stack(co),
        "cross": np.stack(cross),
        "feed_positions_m": np.array(feeds.positions_m, dtype=float),
        "diameter_m": spec.diameter_m,
        "focal_length_m": spec.focal_length_m,
        "offset_clearance_m": spec.offset_clearance_m,
        "freq_hz": spec.freq_hz,
        "q": feeds.q,
        "zone_radius_deg": zone.radius_deg,
    }


def write_reflector_reference(path: Path, ref: dict[str, Any]) -> None:  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **ref)
    logger.info("reflector_reference_written", path=str(path), n_feeds=ref["fields"].shape[0])


def main() -> None:  # pragma: no cover
    configure_logging()
    grid = UVGrid(u_min=-0.25, u_max=0.25, v_min=-0.25, v_max=0.25, n_u=161, n_v=161)
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.4, freq_hz=20e9)
    feeds = FeedSpec(positions_m=hex_feed_positions(0.03, 80, 0.3), q=2.0)
    zone = ServiceZone(radius_deg=8.0)
    write_reflector_reference(DEFAULT_OUTPUT, build_reflector_reference(spec, feeds, zone, grid))


if __name__ == "__main__":  # pragma: no cover
    main()
