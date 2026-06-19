# Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest a `grd_analyzer` `report.json` and generate a calibrated, elliptical, dispersed antenna-pattern `.npz` that reproduces the measured statistics — closing the analyzer→generator round-trip.

**Architecture:** Additive modules on top of the existing `grd_generator`. New `EllipticalSpec` (schemas) + `elliptical_field`/`widths_to_sigmas` (synth) extend the pure synthesis layer (covered 100%). A new `report_ingest.py` parses the report into per-field distributions. A new offline `calibrate.py` (CLI `grd-calibrate`, omitted from coverage like `generate.py`) samples per-element specs with a fixed seed, writes the `.npz` via `write_calibrated`, and runs a diagnostic `validate_against`. The existing circular `generate_reference` path is untouched.

**Tech Stack:** Python ≥3.12, numpy, pydantic, loguru. Package manager: `uv`. Test: pytest + pytest-cov (gate 100%).

## Global Constraints

- Python `requires-python = ">=3.12"`. Runtime deps ONLY: `numpy>=2.0`, `pydantic>=2.7`, `loguru>=0.7`. No new deps.
- Namespace `grd_generator`. Structured loguru JSON logging — NO `print()`.
- ruff line-length 100, target py312, lint select `["E","F","I","UP","B","SIM"]`. mypy strict + `pydantic.mypy`. Test functions need `-> None`.
- Coverage gate `--cov-fail-under=100` is ALREADY active. `calibrate.py` and `examples/*` and `tests/*` are OMITTED; `schemas.py`, `synth.py`, `report_ingest.py` must stay at 100%. Add the `calibrate.py` omit entry in the task that creates it.
- Do NOT modify `generate_reference`, `write_reference`, `gaussian_field`, `airy_field`, or the circular `GaussianSpec` — additive only.
- The existing circular reduction must hold: `elliptical_field` with `sigma_major == sigma_minor` equals the circular generator up to the σ convention.
- Shared constant value (must equal `grd_analyzer`): `HALF_POWER_WIDTH_PER_SIGMA = 2.0 * sqrt(ln 2)`.
- Real report fixtures live at `data/reference_reports/0000/report.json` (mode airy) and `0001/report.json` (mode gaussian), already committed.
- Verified fixture statistics over non-truncated elements (use as test expectations):
  - **0000** (airy, n_elements=82, spacing_deg≈0.009391): peak_dbi mean 27.87512 std 0.32509; lobe_width_major_deg mean 0.24326 std 0.00562; lobe_width_minor_deg mean 0.18719 std 0.04419; lobe_orientation_deg mean −2.31124 std 53.23991; phase_slope_est_rad_per_deg mean 120.63267 std 28.84927; first_null_radius_deg mean 0.02974 std 0.01233 (present on 66/82).
  - **0001** (gaussian, n_elements=86, spacing_deg≈0.009925): peak_dbi mean 28.33369 std 0.41035; lobe_width_major_deg mean 0.21207 std 0.02121; lobe_width_minor_deg mean 0.12364 std 0.03708; lobe_orientation_deg mean 0.43397 std 64.80965; phase_slope_est_rad_per_deg mean 126.59554 std 23.79116; first_null_radius_deg ABSENT (None).

---

### Task 1: EllipticalSpec schema

**Files:**
- Modify: `src/grd_generator/schemas.py`
- Test: `src/grd_generator/tests/test_schemas.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `EllipticalSpec(BaseModel)` with `center_uv: tuple[float,float]`, `sigma_major: float (gt=0)`, `sigma_minor: float (gt=0)`, `orientation_deg: float = 0.0`, `peak_gain_dbi: float`, `phase_slope_radial: float = 0.0`.

- [ ] **Step 1: Write the failing test** — append to `src/grd_generator/tests/test_schemas.py`

```python
def test_elliptical_spec_requires_positive_sigmas() -> None:
    from grd_generator.schemas import EllipticalSpec
    with pytest.raises(ValidationError):
        EllipticalSpec(center_uv=(0.0, 0.0), sigma_major=0.0, sigma_minor=0.1, peak_gain_dbi=28.0)
    with pytest.raises(ValidationError):
        EllipticalSpec(center_uv=(0.0, 0.0), sigma_major=0.1, sigma_minor=-0.1, peak_gain_dbi=28.0)


def test_elliptical_spec_defaults() -> None:
    from grd_generator.schemas import EllipticalSpec
    spec = EllipticalSpec(
        center_uv=(1.0, 2.0), sigma_major=0.05, sigma_minor=0.03, peak_gain_dbi=28.0
    )
    assert spec.orientation_deg == 0.0
    assert spec.phase_slope_radial == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/grd_generator/tests/test_schemas.py -k elliptical -v`
Expected: FAIL (`ImportError: cannot import name 'EllipticalSpec'`).

- [ ] **Step 3: Add `EllipticalSpec` to `src/grd_generator/schemas.py`** (after `GaussianSpec`, and add the name to `__all__`)

```python
class EllipticalSpec(BaseModel):
    """Paramètres d'un élément à lobe elliptique (calibration depuis données réelles).

    `sigma_major`/`sigma_minor` sont les demi-largeurs le long des axes propres ;
    `orientation_deg` est l'angle de l'axe majeur. Réduction au cas circulaire
    quand `sigma_major == sigma_minor`.
    """

    center_uv: tuple[float, float]
    sigma_major: float = Field(..., gt=0)
    sigma_minor: float = Field(..., gt=0)
    orientation_deg: float = 0.0
    peak_gain_dbi: float
    phase_slope_radial: float = 0.0
```

Add `"EllipticalSpec",` to the `__all__` list (next to `"GaussianSpec"`).

- [ ] **Step 4: Run tests to verify pass + coverage holds**

Run: `uv run pytest src/grd_generator/tests/test_schemas.py -v && uv run ruff check src/grd_generator/schemas.py && uv run mypy src/grd_generator/schemas.py`
Expected: PASS, ruff/mypy clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add EllipticalSpec schema for calibrated elements"
```

---

### Task 2: Elliptical field synthesis + width→sigma conversion

**Files:**
- Modify: `src/grd_generator/synth.py`
- Test: `src/grd_generator/tests/test_synth.py` (append)

**Interfaces:**
- Consumes: `grd_generator.schemas.EllipticalSpec`, `UVGrid`; existing `_bessel_j1`, `_J1_FIRST_NULL`.
- Produces:
  - `HALF_POWER_WIDTH_PER_SIGMA: float` (== `2.0*sqrt(ln 2)`).
  - `widths_to_sigmas(mode: str, width_major: float, width_minor: float, first_null: float | None = None) -> tuple[float, float]`.
  - `elliptical_field(spec: EllipticalSpec, grid: UVGrid, *, mode: str = "gaussian") -> ComplexField`.

- [ ] **Step 1: Write the failing tests** — append to `src/grd_generator/tests/test_synth.py`

```python
def test_half_power_constant_matches_analyzer() -> None:
    import numpy as np
    from grd_generator.synth import HALF_POWER_WIDTH_PER_SIGMA
    assert HALF_POWER_WIDTH_PER_SIGMA == pytest.approx(2.0 * np.sqrt(np.log(2.0)))


def test_widths_to_sigmas_gaussian() -> None:
    import numpy as np
    from grd_generator.synth import HALF_POWER_WIDTH_PER_SIGMA, widths_to_sigmas
    smaj, smin = widths_to_sigmas("gaussian", 0.24, 0.12)
    assert smaj == pytest.approx(0.24 / HALF_POWER_WIDTH_PER_SIGMA)
    assert smin == pytest.approx(0.12 / HALF_POWER_WIDTH_PER_SIGMA)


def test_widths_to_sigmas_airy_preserves_ellipticity() -> None:
    import numpy as np
    from grd_generator.synth import widths_to_sigmas
    smaj, smin = widths_to_sigmas("airy", 0.30, 0.15, first_null=0.03)
    # geometric mean preserved at first_null; ratio == sqrt(width ratio)
    assert np.sqrt(smaj * smin) == pytest.approx(0.03)
    assert smaj / smin == pytest.approx(np.sqrt(0.30 / 0.15))


def test_widths_to_sigmas_airy_without_null_falls_back_to_gaussian() -> None:
    from grd_generator.synth import widths_to_sigmas
    assert widths_to_sigmas("airy", 0.30, 0.15, None) == widths_to_sigmas("gaussian", 0.30, 0.15)


def test_widths_to_sigmas_unknown_mode_raises() -> None:
    import pytest as _pytest
    from grd_generator.synth import widths_to_sigmas
    with _pytest.raises(ValueError):
        widths_to_sigmas("nope", 0.3, 0.15)


def _ellip_grid():
    from grd_generator.schemas import UVGrid
    return UVGrid(u_min=-1.0, u_max=1.0, v_min=-1.0, v_max=1.0, n_u=101, n_v=101)


def test_elliptical_peak_at_center() -> None:
    import numpy as np
    from grd_generator.schemas import EllipticalSpec
    from grd_generator.synth import elliptical_field
    spec = EllipticalSpec(center_uv=(0.0, 0.0), sigma_major=0.4, sigma_minor=0.2, peak_gain_dbi=40.0)
    field = elliptical_field(spec, _ellip_grid())
    assert np.abs(field).max() == pytest.approx(10.0 ** (40.0 / 20.0), rel=1e-3)


def test_elliptical_circular_reduces_to_gaussian() -> None:
    import numpy as np
    from grd_generator.schemas import EllipticalSpec, GaussianSpec
    from grd_generator.synth import elliptical_field, gaussian_field
    grid = _ellip_grid()
    ell = EllipticalSpec(center_uv=(0.1, -0.1), sigma_major=0.3, sigma_minor=0.3,
                         peak_gain_dbi=35.0, phase_slope_radial=2.0)
    circ = GaussianSpec(center_uv=(0.1, -0.1), sigma=0.3, peak_gain_dbi=35.0, phase_slope_radial=2.0)
    np.testing.assert_allclose(elliptical_field(ell, grid), gaussian_field(circ, grid), rtol=1e-9)


def test_elliptical_orientation_widens_along_major_axis() -> None:
    import numpy as np
    from grd_generator.schemas import EllipticalSpec
    from grd_generator.synth import elliptical_field
    grid = _ellip_grid()
    # major axis along u (orientation 0): amplitude falls off slower along u than v.
    spec = EllipticalSpec(center_uv=(0.0, 0.0), sigma_major=0.5, sigma_minor=0.1,
                          orientation_deg=0.0, peak_gain_dbi=30.0)
    amp = np.abs(elliptical_field(spec, grid))
    cu = grid.n_u // 2
    cv = grid.n_v // 2
    # value at +0.3 along u (row=center) vs +0.3 along v (col=center)
    u, v = grid.axes()
    iu = int(np.argmin(np.abs(u - 0.3)))
    iv = int(np.argmin(np.abs(v - 0.3)))
    assert amp[cv, iu] > amp[iv, cu]  # wider along the major (u) axis


def test_elliptical_airy_has_nulls() -> None:
    import numpy as np
    from grd_generator.schemas import EllipticalSpec
    from grd_generator.synth import elliptical_field
    spec = EllipticalSpec(center_uv=(0.0, 0.0), sigma_major=0.3, sigma_minor=0.2, peak_gain_dbi=30.0)
    amp = np.abs(elliptical_field(spec, _ellip_grid(), mode="airy"))
    assert amp.min() < amp.max() * 0.05  # deep nulls present


def test_elliptical_unknown_mode_raises() -> None:
    import pytest as _pytest
    from grd_generator.schemas import EllipticalSpec
    from grd_generator.synth import elliptical_field
    spec = EllipticalSpec(center_uv=(0.0, 0.0), sigma_major=0.3, sigma_minor=0.2, peak_gain_dbi=30.0)
    with _pytest.raises(ValueError):
        elliptical_field(spec, _ellip_grid(), mode="nope")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/grd_generator/tests/test_synth.py -k "elliptical or widths or half_power" -v`
Expected: FAIL (`ImportError` for the new names).

- [ ] **Step 3: Add the code to `src/grd_generator/synth.py`** (after `airy_field` / near the constants; reuse `_bessel_j1`, `_J1_FIRST_NULL`)

```python
HALF_POWER_WIDTH_PER_SIGMA = 2.0 * float(np.sqrt(np.log(2.0)))


def widths_to_sigmas(
    mode: str,
    width_major: float,
    width_minor: float,
    first_null: float | None = None,
) -> tuple[float, float]:
    """Convertit des largeurs de lobe (deg) en (σ_major, σ_minor) selon le mode.

    `gaussian` : σ = largeur / (2·√ln2) (même convention que grd_analyzer).
    `airy` (avec `first_null`) : on préserve l'ellipticité autour du premier null —
    moyenne géométrique des σ = `first_null`, rapport = √(width_major/width_minor).
    `airy` sans `first_null` : repli sur la conversion gaussian.
    """
    if mode not in ("gaussian", "airy"):
        raise ValueError(f"mode inconnu : {mode!r} (connus : 'gaussian', 'airy')")
    if mode == "airy" and first_null is not None:
        root = float(np.sqrt(width_major / width_minor))
        return first_null * root, first_null / root
    return width_major / HALF_POWER_WIDTH_PER_SIGMA, width_minor / HALF_POWER_WIDTH_PER_SIGMA


def elliptical_field(spec: "EllipticalSpec", grid: UVGrid, *, mode: str = "gaussian") -> ComplexField:
    """Champ complexe d'un élément à lobe elliptique (axes propres tournés).

    Réduction exacte au générateur circulaire correspondant quand
    `sigma_major == sigma_minor`. Phase radiale `phase_slope_radial·‖uv−c‖`.
    """
    gu, gv = grid.meshgrid()
    cu, cv = spec.center_uv
    du = gu - cu
    dv = gv - cv
    theta = np.radians(spec.orientation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    du_p = du * cos_t + dv * sin_t
    dv_p = -du * sin_t + dv * cos_t
    rho = np.sqrt((du_p / spec.sigma_major) ** 2 + (dv_p / spec.sigma_minor) ** 2)
    peak_lin = 10.0 ** (spec.peak_gain_dbi / 20.0)
    if mode == "gaussian":
        amplitude = peak_lin * np.exp(-0.5 * rho**2)
    elif mode == "airy":
        x = _J1_FIRST_NULL * rho
        jinc = np.where(x == 0.0, 1.0, 2.0 * _bessel_j1(x) / np.where(x == 0.0, 1.0, x))
        amplitude = peak_lin * jinc
    else:
        raise ValueError(f"mode de génération inconnu : {mode!r} (connus : {available_modes()})")
    phase = spec.phase_slope_radial * np.sqrt(du**2 + dv**2)
    field: ComplexField = (amplitude * np.exp(1j * phase)).astype(np.complex128)
    return field
```

Add the new import at the top of `synth.py` so the annotation resolves at runtime is unnecessary (string annotation used), but add `EllipticalSpec` to the existing `from grd_generator.schemas import (...)` block for clarity and mypy. Append `HALF_POWER_WIDTH_PER_SIGMA`, `widths_to_sigmas`, `elliptical_field` to `__all__`.

- [ ] **Step 4: Run tests + coverage + types**

Run: `uv run pytest src/grd_generator/tests/test_synth.py -v && uv run ruff check src/grd_generator/synth.py && uv run mypy src/grd_generator/synth.py`
Expected: PASS, clean. Then `uv run pytest` (full) must still be 100%.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add elliptical field synthesis + width->sigma conversion"
```

---

### Task 3: report_ingest.py — parse report.json into MeasuredStats

**Files:**
- Create: `src/grd_generator/report_ingest.py`
- Test: `src/grd_generator/tests/test_report_ingest.py`

**Interfaces:**
- Consumes: `grd_generator.logger.logger`.
- Produces:
  - `FieldDist` frozen dataclass: `mean: float`, `std: float`.
  - `MeasuredStats` frozen dataclass: `source: str`, `mode: str`, `n_elements: int`, `spacing_deg: float`, `peak_dbi: FieldDist`, `lobe_width_major_deg: FieldDist`, `lobe_width_minor_deg: FieldDist`, `orientation_deg: FieldDist`, `phase_slope_rad_per_deg: FieldDist`, `first_null_radius_deg: FieldDist | None`.
  - `read_report(path: str | Path) -> MeasuredStats` (accepts a `report.json` file or a directory containing one).

- [ ] **Step 1: Write the failing tests** `src/grd_generator/tests/test_report_ingest.py`

```python
import json
from pathlib import Path

import pytest

from grd_generator.report_ingest import MeasuredStats, read_report

REPORTS = Path(__file__).resolve().parents[3] / "data" / "reference_reports"


def test_read_report_0000_airy() -> None:
    stats = read_report(REPORTS / "0000")  # directory form
    assert isinstance(stats, MeasuredStats)
    assert stats.mode == "airy"
    assert stats.n_elements == 82
    assert stats.spacing_deg == pytest.approx(0.009391, abs=1e-5)
    assert stats.peak_dbi.mean == pytest.approx(27.87512, abs=1e-4)
    assert stats.peak_dbi.std == pytest.approx(0.32509, abs=1e-4)
    assert stats.lobe_width_major_deg.mean == pytest.approx(0.24326, abs=1e-4)
    assert stats.orientation_deg.std == pytest.approx(53.23991, abs=1e-3)
    assert stats.first_null_radius_deg is not None
    assert stats.first_null_radius_deg.mean == pytest.approx(0.02974, abs=1e-4)


def test_read_report_0001_gaussian_no_null() -> None:
    stats = read_report(REPORTS / "0001" / "report.json")  # file form
    assert stats.mode == "gaussian"
    assert stats.n_elements == 86
    assert stats.peak_dbi.mean == pytest.approx(28.33369, abs=1e-4)
    assert stats.first_null_radius_deg is None  # gaussian set has no nulls


def test_read_report_rejects_empty_elements(tmp_path: Path) -> None:
    p = tmp_path / "report.json"
    p.write_text(json.dumps({"elements": [], "estimated_params": {"mode": "gaussian", "spacing_deg": 1.0}}))
    with pytest.raises(ValueError):
        read_report(p)


def test_read_report_rejects_bad_mode(tmp_path: Path) -> None:
    el = {"peak_dbi": 1.0, "lobe_width_major_deg": 1.0, "lobe_width_minor_deg": 1.0,
          "lobe_orientation_deg": 0.0, "phase_slope_est_rad_per_deg": 1.0, "truncated": False}
    p = tmp_path / "report.json"
    p.write_text(json.dumps({"n_elements": 1, "elements": [el],
                             "estimated_params": {"mode": "weird", "spacing_deg": 1.0}}))
    with pytest.raises(ValueError):
        read_report(p)


def test_read_report_rejects_bad_spacing(tmp_path: Path) -> None:
    el = {"peak_dbi": 1.0, "lobe_width_major_deg": 1.0, "lobe_width_minor_deg": 1.0,
          "lobe_orientation_deg": 0.0, "phase_slope_est_rad_per_deg": 1.0, "truncated": False}
    p = tmp_path / "report.json"
    p.write_text(json.dumps({"n_elements": 1, "elements": [el],
                             "estimated_params": {"mode": "gaussian", "spacing_deg": 0.0}}))
    with pytest.raises(ValueError):
        read_report(p)


def test_read_report_rejects_missing_required_field(tmp_path: Path) -> None:
    el = {"peak_dbi": 1.0, "truncated": False}  # missing widths/orientation/phase
    p = tmp_path / "report.json"
    p.write_text(json.dumps({"n_elements": 1, "elements": [el],
                             "estimated_params": {"mode": "gaussian", "spacing_deg": 1.0}}))
    with pytest.raises(ValueError):
        read_report(p)


def test_read_report_rejects_all_truncated(tmp_path: Path) -> None:
    el = {"peak_dbi": 1.0, "lobe_width_major_deg": 1.0, "lobe_width_minor_deg": 1.0,
          "lobe_orientation_deg": 0.0, "phase_slope_est_rad_per_deg": 1.0, "truncated": True}
    p = tmp_path / "report.json"
    p.write_text(json.dumps({"n_elements": 1, "elements": [el],
                             "estimated_params": {"mode": "gaussian", "spacing_deg": 1.0}}))
    with pytest.raises(ValueError):
        read_report(p)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/grd_generator/tests/test_report_ingest.py -v`
Expected: FAIL (`ModuleNotFoundError: grd_generator.report_ingest`).

- [ ] **Step 3: Create `src/grd_generator/report_ingest.py`**

```python
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
        "report_ingested", source=stats.source, mode=mode, n_elements=n_elements,
        spacing_deg=round(stats.spacing_deg, 6),
    )
    return stats


__all__ = ["FieldDist", "MeasuredStats", "read_report"]
```

- [ ] **Step 4: Run tests + coverage + types**

Run: `uv run pytest src/grd_generator/tests/test_report_ingest.py -v && uv run ruff check src/grd_generator/report_ingest.py && uv run mypy src/grd_generator/report_ingest.py`
Expected: PASS, clean. Then `uv run pytest` (full) — coverage of `report_ingest.py` must be 100% (every error branch + the `_dist` None branch are exercised).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add report.json ingestion into MeasuredStats"
```

---

### Task 4: calibrate.py — calibrated generation, write_calibrated, CLI

**Files:**
- Create: `src/grd_generator/calibrate.py`
- Modify: `pyproject.toml` (add `grd-calibrate` script + omit `calibrate.py` from coverage)
- Test: `src/grd_generator/tests/test_calibrate.py`

**Interfaces:**
- Consumes: `grd_generator.report_ingest.{MeasuredStats, FieldDist, read_report}`; `grd_generator.schemas.{EllipticalSpec, UVGrid}`; `grd_generator.synth.{elliptical_field, widths_to_sigmas}`; `grd_generator.synth.hex_centers_uv`; `grd_generator.logger.{configure_logging, logger}`.
- Produces:
  - `calibrate(stats: MeasuredStats, *, seed: int = 0, n_u: int = 161, n_v: int = 161, grid_margin_sigma: float = 6.0) -> tuple[list[EllipticalSpec], UVGrid]`
  - `write_calibrated(output_path: str | Path, grid: UVGrid, specs: list[EllipticalSpec], *, mode: str) -> Path`
  - `validate_against(stats: MeasuredStats, specs: list[EllipticalSpec], *, tol: float = 0.15) -> list[dict[str, object]]`
  - `generate_calibrated(report_path: str | Path, output_path: str | Path, *, seed: int = 0) -> Path`
  - `main() -> None`

- [ ] **Step 1: Add the `grd-calibrate` script and coverage omit to `pyproject.toml`**

In `[project.scripts]` add:
```toml
grd-calibrate = "grd_generator.calibrate:main"
```
In `[tool.coverage.run] omit` add (next to the `generate.py` line):
```toml
    "*/grd_generator/calibrate.py",      # script offline : sampling + écriture .npz + CLI
```

- [ ] **Step 2: Write the test** `src/grd_generator/tests/test_calibrate.py`

```python
from pathlib import Path

import numpy as np

from grd_generator.calibrate import (
    calibrate,
    generate_calibrated,
    validate_against,
    write_calibrated,
)
from grd_generator.report_ingest import read_report
from grd_generator.schemas import EllipticalSpec

REPORTS = Path(__file__).resolve().parents[3] / "data" / "reference_reports"


def test_calibrate_returns_n_specs_and_is_reproducible() -> None:
    stats = read_report(REPORTS / "0000")
    specs_a, grid_a = calibrate(stats, seed=7)
    specs_b, grid_b = calibrate(stats, seed=7)
    assert len(specs_a) == stats.n_elements
    assert all(isinstance(s, EllipticalSpec) for s in specs_a)
    # reproducible: same seed -> identical first/last spec params
    assert specs_a[0].peak_gain_dbi == specs_b[0].peak_gain_dbi
    assert specs_a[-1].sigma_major == specs_b[-1].sigma_major
    assert grid_a.n_u == 161 and grid_a.n_v == 161


def test_calibrate_orientations_folded() -> None:
    stats = read_report(REPORTS / "0000")
    specs, _ = calibrate(stats, seed=1)
    assert all(-90.0 < s.orientation_deg <= 90.0 for s in specs)


def test_write_calibrated_roundtrip(tmp_path: Path) -> None:
    stats = read_report(REPORTS / "0001")
    specs, grid = calibrate(stats, seed=0)
    out = write_calibrated(tmp_path / "cal.npz", grid, specs, mode=stats.mode)
    data = np.load(out, allow_pickle=False)
    n = stats.n_elements
    assert data["fields"].shape == (n, grid.n_v, grid.n_u)
    assert data["fields"].dtype == np.complex128
    assert data["sigmas_major"].shape == (n,)
    assert data["sigmas_minor"].shape == (n,)
    assert data["orientations_deg"].shape == (n,)
    assert str(data["mode"]) == "gaussian"


def test_validate_against_flags_off_specs() -> None:
    stats = read_report(REPORTS / "0001")
    specs, _ = calibrate(stats, seed=0)
    # conforming specs -> no deviation
    assert validate_against(stats, specs) == []
    # corrupt the peaks far out of tolerance -> deviation reported
    bad = [s.model_copy(update={"peak_gain_dbi": s.peak_gain_dbi + 50.0}) for s in specs]
    devs = validate_against(stats, bad)
    assert any(d["field"] == "peak_dbi" for d in devs)


def test_generate_calibrated_end_to_end(tmp_path: Path) -> None:
    out = generate_calibrated(REPORTS / "0000", tmp_path / "cal0000.npz", seed=0)
    assert out.exists()
    data = np.load(out, allow_pickle=False)
    assert data["fields"].shape[0] == 82
    assert str(data["mode"]) == "airy"
```

- [ ] **Step 3: Create `src/grd_generator/calibrate.py`**

```python
"""Génération calibrée d'un array à partir d'un rapport grd_analyzer.

Offline : tire des éléments à lobe elliptique dans les distributions mesurées
(seed fixe), les place sur une maille hexagonale à l'espacement mesuré, synthétise
et écrit le .npz. `validate_against` re-mesure les stats générées et logue un
warning par champ hors tolérance (diagnostic, jamais bloquant).

Usage :
    uv run grd-calibrate --report data/reference_reports/0000 --out data/processed/cal0000.npz
"""

import argparse
from pathlib import Path

import numpy as np

from grd_generator.logger import configure_logging, logger
from grd_generator.report_ingest import FieldDist, MeasuredStats, read_report
from grd_generator.schemas import EllipticalSpec, UVGrid
from grd_generator.synth import elliptical_field, hex_centers_uv, widths_to_sigmas

_SIGMA_FLOOR = 1e-6  # plancher pour σ/largeurs tirées (strictement > 0)
DEFAULT_OUTPUT = Path("data/processed/calibrated.npz")


def _hex_centers(n: int, spacing: float) -> list[tuple[float, float]]:
    """`n` centres sur maille hex à `spacing` ; élargit le disque jusqu'à loger n."""
    # rayon ~ inverse de auto_spacing ; on élargit si la maille loge trop peu.
    radius = float(spacing * np.sqrt(np.sqrt(3.0) * n / (2.0 * np.pi)))
    while True:
        try:
            return hex_centers_uv((0.0, 0.0), radius, n, spacing)
        except ValueError:
            radius *= 1.3


def _fold_orientation(deg: float) -> float:
    """Replie un angle dans l'intervalle (−90, 90]."""
    return ((deg + 90.0) % 180.0) - 90.0


def _grid_for(
    centers: list[tuple[float, float]],
    specs: list[EllipticalSpec],
    n_u: int,
    n_v: int,
    margin_sigma: float,
) -> UVGrid:
    """Grille couvrant la bbox des centres élargie de `margin_sigma·max(σ_major)`."""
    cu = np.array([c[0] for c in centers])
    cv = np.array([c[1] for c in centers])
    margin = margin_sigma * max(s.sigma_major for s in specs)
    return UVGrid(
        u_min=float(cu.min()) - margin,
        u_max=float(cu.max()) + margin,
        v_min=float(cv.min()) - margin,
        v_max=float(cv.max()) + margin,
        n_u=n_u,
        n_v=n_v,
    )


def calibrate(
    stats: MeasuredStats,
    *,
    seed: int = 0,
    n_u: int = 161,
    n_v: int = 161,
    grid_margin_sigma: float = 6.0,
) -> tuple[list[EllipticalSpec], UVGrid]:
    """Tire `stats.n_elements` specs elliptiques dans les distributions mesurées."""
    rng = np.random.default_rng(seed)
    centers = _hex_centers(stats.n_elements, stats.spacing_deg)

    def draw(dist: FieldDist) -> float:
        return float(rng.normal(dist.mean, dist.std))

    specs: list[EllipticalSpec] = []
    for cu, cv in centers:
        wmaj = max(_SIGMA_FLOOR, draw(stats.lobe_width_major_deg))
        wmin = max(_SIGMA_FLOOR, draw(stats.lobe_width_minor_deg))
        first_null = (
            None
            if stats.first_null_radius_deg is None
            else max(_SIGMA_FLOOR, draw(stats.first_null_radius_deg))
        )
        smaj, smin = widths_to_sigmas(stats.mode, wmaj, wmin, first_null)
        specs.append(
            EllipticalSpec(
                center_uv=(cu, cv),
                sigma_major=max(_SIGMA_FLOOR, smaj),
                sigma_minor=max(_SIGMA_FLOOR, smin),
                orientation_deg=_fold_orientation(draw(stats.orientation_deg)),
                peak_gain_dbi=draw(stats.peak_dbi),
                phase_slope_radial=draw(stats.phase_slope_rad_per_deg),
            )
        )
    grid = _grid_for(centers, specs, n_u, n_v, grid_margin_sigma)
    logger.info("calibrated_specs", n=len(specs), mode=stats.mode)
    return specs, grid


def write_calibrated(
    output_path: str | Path,
    grid: UVGrid,
    specs: list[EllipticalSpec],
    *,
    mode: str,
) -> Path:
    """Synthétise les champs elliptiques et écrit le .npz (clés étendues)."""
    fields = np.stack([elliptical_field(s, grid, mode=mode) for s in specs])
    ids = np.array([f"ant_{k:02d}" for k in range(len(specs))])
    centers_uv = np.array([s.center_uv for s in specs], dtype=float)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        u_min=grid.u_min,
        u_max=grid.u_max,
        v_min=grid.v_min,
        v_max=grid.v_max,
        n_u=grid.n_u,
        n_v=grid.n_v,
        fields=fields,
        antenna_ids=ids,
        centers_uv=centers_uv,
        mode=np.array(mode),
        peaks_dbi=np.array([s.peak_gain_dbi for s in specs], dtype=float),
        phase_slopes=np.array([s.phase_slope_radial for s in specs], dtype=float),
        sigmas_major=np.array([s.sigma_major for s in specs], dtype=float),
        sigmas_minor=np.array([s.sigma_minor for s in specs], dtype=float),
        orientations_deg=np.array([s.orientation_deg for s in specs], dtype=float),
    )
    logger.info("calibrated_written", file=str(out), n_antennas=len(specs))
    return out


def validate_against(
    stats: MeasuredStats,
    specs: list[EllipticalSpec],
    *,
    tol: float = 0.15,
) -> list[dict[str, object]]:
    """Diagnostic : compare stats générées vs mesurées ; warning par champ hors tol.

    Renvoie la liste des écarts détectés (vide si tout est dans la tolérance).
    Ne lève jamais : c'est un diagnostic, pas un gate.
    """
    gen_peak = float(np.mean([s.peak_gain_dbi for s in specs]))
    gen_phase = float(np.mean([s.phase_slope_radial for s in specs]))
    checks: list[tuple[str, float, float]] = [
        ("peak_dbi", stats.peak_dbi.mean, gen_peak),
        ("phase_slope_rad_per_deg", stats.phase_slope_rad_per_deg.mean, gen_phase),
    ]
    deviations: list[dict[str, object]] = []
    for field, measured, generated in checks:
        if measured != 0.0 and abs(generated - measured) / abs(measured) > tol:
            deviations.append({"field": field, "measured": measured, "generated": generated})
            logger.warning(
                "calibration_stat_deviation",
                field=field,
                measured=round(measured, 4),
                generated=round(generated, 4),
                tol=tol,
            )
    return deviations


def generate_calibrated(
    report_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    seed: int = 0,
) -> Path:
    """Bout-en-bout : lit le rapport, calibre, écrit le .npz, logue le diagnostic."""
    stats = read_report(report_path)
    specs, grid = calibrate(stats, seed=seed)
    out = write_calibrated(output_path, grid, specs, mode=stats.mode)
    validate_against(stats, specs)
    return out


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Génère un array calibré depuis un report.json.")
    parser.add_argument("--report", required=True, help="report.json (fichier ou dossier)")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="chemin du .npz de sortie")
    parser.add_argument("--seed", type=int, default=0, help="seed du tirage (reproductibilité)")
    args = parser.parse_args()
    generate_calibrated(args.report, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + CLI end-to-end**

Run: `uv run pytest src/grd_generator/tests/test_calibrate.py -v`
Expected: PASS (5 tests; each calibrate runs full synthesis — a few seconds).

Run: `uv run grd-calibrate --report data/reference_reports/0000 --out data/processed/cal0000.npz && uv run grd-calibrate --report data/reference_reports/0001 --out data/processed/cal0001.npz`
Expected: two `.npz` written; structured JSON logs `report_ingested` / `calibrated_specs` / `calibrated_written`. (`.npz` is git-ignored — do not commit it.)

- [ ] **Step 5: Types + lint**

Run: `uv run mypy src/grd_generator/calibrate.py && uv run ruff check src/grd_generator/calibrate.py`
Expected: clean. (If mypy flags the inner `draw`/`dev` helpers, the `# type: ignore` hints in the code above cover them; adjust only if a different error appears.)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add calibrated generation (calibrate.py) + grd-calibrate CLI"
```

---

### Task 5: Public API, example, and full-suite gate

**Files:**
- Modify: `src/grd_generator/__init__.py`
- Create: `src/grd_generator/examples/example_calibration.py`
- Test: `src/grd_generator/tests/test_example_calibration.py`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: public exports for the calibration path; a runnable example; the full suite green at 100%.

- [ ] **Step 1: Extend `src/grd_generator/__init__.py`** — add the calibration public API

Add these imports and `__all__` entries (keep existing ones):
```python
from grd_generator.calibrate import generate_calibrated
from grd_generator.report_ingest import MeasuredStats, read_report
from grd_generator.schemas import EllipticalSpec
from grd_generator.synth import elliptical_field, widths_to_sigmas
```
Append to `__all__`: `"EllipticalSpec"`, `"elliptical_field"`, `"widths_to_sigmas"`, `"MeasuredStats"`, `"read_report"`, `"generate_calibrated"`.

- [ ] **Step 2: Create the example** `src/grd_generator/examples/example_calibration.py`

```python
"""
Exemple : calibration du générateur depuis un rapport réel
===========================================================
Lit un report.json de grd_analyzer (jeu 0000), génère un array calibré
elliptique dans un .npz temporaire et relit ses métadonnées.

Usage :
    python src/grd_generator/examples/example_calibration.py
"""

import tempfile
from pathlib import Path

import numpy as np

from grd_generator.calibrate import generate_calibrated
from grd_generator.logger import configure_logging, logger

REPORT = Path(__file__).resolve().parents[3] / "data" / "reference_reports" / "0000"


def main() -> None:
    configure_logging()
    logger.info("example_start", feature="calibration")

    with tempfile.TemporaryDirectory() as tmp:
        out = generate_calibrated(REPORT, Path(tmp) / "calibrated.npz", seed=0)
        data = np.load(out, allow_pickle=False)
        logger.info(
            "example_calibrated",
            n_antennas=int(data["fields"].shape[0]),
            mode=str(data["mode"]),
            grid=(int(data["n_u"]), int(data["n_v"])),
        )

    logger.info("example_done", feature="calibration")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write the launch test** `src/grd_generator/tests/test_example_calibration.py`

```python
import subprocess
import sys


def test_example_calibration_runs() -> None:
    """Vérifie que le script d'exemple s'exécute sans erreur."""
    result = subprocess.run(
        [sys.executable, "src/grd_generator/examples/example_calibration.py"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"L'exemple a échoué :\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
```

- [ ] **Step 4: Run the full suite under the gate**

Run: `uv run pytest`
Expected: ALL pass, coverage 100% (`calibrate.py`, `examples/*`, `tests/*` omitted). If any covered line in `schemas.py`/`synth.py`/`report_ingest.py` is uncovered, add a targeted test — do NOT use `# pragma: no cover` unless genuinely unreachable.

- [ ] **Step 5: Import + lint + types**

Run: `uv run python -c "import grd_generator; print(grd_generator.__all__)" && uv run ruff check . && uv run mypy src/grd_generator`
Expected: `__all__` prints with the new symbols; ruff/mypy clean.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: expose calibration API + example, full suite at 100%"
```

---

## Self-Review notes

- **Spec coverage:** EllipticalSpec (T1); elliptical_field + widths_to_sigmas + HALF_POWER_WIDTH_PER_SIGMA (T2); report_ingest MeasuredStats/FieldDist/read_report (T3); calibrate + write_calibrated + validate_against + generate_calibrated + CLI + pyproject omit/script (T4); public API + example + 100% gate (T5). Data copy (reference_reports) already committed with the spec. All spec sections mapped.
- **No regression:** `generate_reference`, `write_reference`, `gaussian_field`, `airy_field`, `GaussianSpec` untouched; all additions are new names.
- **Coverage honesty:** `calibrate.py` omitted (offline script, like `generate.py`); pure logic (`elliptical_field`, `widths_to_sigmas`, `read_report`) lives in covered modules with branch tests for every error path.
- **Type consistency:** `MeasuredStats` field names (`peak_dbi`, `lobe_width_major_deg`, `lobe_width_minor_deg`, `orientation_deg`, `phase_slope_rad_per_deg`, `first_null_radius_deg`) are identical in T3 (definition) and T4 (consumption). `widths_to_sigmas(mode, width_major, width_minor, first_null=None)` and `elliptical_field(spec, grid, *, mode=...)` signatures match between T2 and T4.
```
</content>
