"""Ingestion d'un rapport grd_analyzer (report.json) → distributions mesurées.

Lit la liste `elements` (statistiques par élément d'un vrai set .grd) et le bloc
`estimated_params`, et en dérive, pour chaque champ utile, une distribution
(moyenne, écart-type) sur les éléments non tronqués. Sert d'entrée à la
calibration du générateur.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from grd_generator.logger import logger


@dataclass(frozen=True)
class FieldDist:
    """Distribution empirique d'un champ : moyenne et écart-type (population)."""

    mean: float
    std: float


@dataclass(frozen=True)
class MeasuredStats:
    """Statistiques mesurées d'un set réel, prêtes pour la calibration."""

    source: str
    mode: str
    n_elements: int
    spacing_deg: float
    peak_dbi: FieldDist
    lobe_width_major_deg: FieldDist
    lobe_width_minor_deg: FieldDist
    orientation_deg: FieldDist
    phase_slope_rad_per_deg: FieldDist
    first_null_radius_deg: FieldDist | None


def _dist(elements: list[dict[str, Any]], key: str) -> FieldDist | None:
    """Distribution d'un champ sur les éléments qui le portent (None si aucun)."""
    vals = [e[key] for e in elements if e.get(key) is not None]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    return FieldDist(mean=float(arr.mean()), std=float(arr.std()))


def _required(elements: list[dict[str, Any]], key: str) -> FieldDist:
    """Comme `_dist` mais lève si le champ est absent de tous les éléments."""
    dist = _dist(elements, key)
    if dist is None:
        raise ValueError(f"champ requis absent du rapport : {key!r}")
    return dist


def read_report(path: str | Path) -> MeasuredStats:
    """Charge un report.json (fichier ou dossier le contenant) en `MeasuredStats`.

    Lève `ValueError` si la liste `elements` est absente/vide, si tous les
    éléments sont tronqués, si le mode estimé n'est pas dans {gaussian, airy},
    si `spacing_deg` est absent ou ≤ 0, ou si un champ requis manque.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "report.json"
    data = json.loads(p.read_text())

    elements = data.get("elements")
    if not elements:
        raise ValueError(f"rapport {p} : liste 'elements' absente ou vide")
    valid = [e for e in elements if not e.get("truncated")]
    if not valid:
        raise ValueError(f"rapport {p} : aucun élément non tronqué")

    est = data.get("estimated_params") or {}
    mode = est.get("mode")
    if mode not in ("gaussian", "airy"):
        raise ValueError(f"rapport {p} : mode estimé invalide ({mode!r})")
    spacing = est.get("spacing_deg")
    if spacing is None or spacing <= 0:
        raise ValueError(f"rapport {p} : spacing_deg invalide ({spacing!r})")
    n_elements = int(data.get("n_elements", len(elements)))

    stats = MeasuredStats(
        source=str(p),
        mode=mode,
        n_elements=n_elements,
        spacing_deg=float(spacing),
        peak_dbi=_required(valid, "peak_dbi"),
        lobe_width_major_deg=_required(valid, "lobe_width_major_deg"),
        lobe_width_minor_deg=_required(valid, "lobe_width_minor_deg"),
        orientation_deg=_required(valid, "lobe_orientation_deg"),
        phase_slope_rad_per_deg=_required(valid, "phase_slope_est_rad_per_deg"),
        first_null_radius_deg=_dist(valid, "first_null_radius_deg"),
    )
    logger.info(
        "report_ingested",
        source=stats.source,
        mode=mode,
        n_elements=n_elements,
        spacing_deg=round(stats.spacing_deg, 6),
    )
    return stats


__all__ = ["FieldDist", "MeasuredStats", "read_report"]
