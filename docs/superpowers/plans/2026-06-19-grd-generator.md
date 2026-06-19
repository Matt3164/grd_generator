# grd_generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous Python repo whose only job is to generate the antenna-pattern reference dataset (`.npz`), extracted from the `pattern` feature of `satscope`.

**Architecture:** A single package `grd_generator` with focused modules — schemas (pydantic), ECEF geometry, a minimal scenario dataclass, pure field synthesis + σ optimization, and a generation script that writes the `.npz`. No analysis, plotting, beamforming, or verification.

**Tech Stack:** Python ≥3.12, numpy, pydantic, loguru. Dev: pytest, pytest-cov, ruff, mypy. Package manager: `uv`.

## Global Constraints

- Python `requires-python = ">=3.12"`.
- Runtime deps ONLY: `numpy>=2.0`, `pydantic>=2.7`, `loguru>=0.7`. No matplotlib, no pandas.
- Package namespace: `grd_generator` (all `satscope.*` imports rewritten to `grd_generator.*`).
- Structured JSON logging via loguru — NO `print()`.
- Final coverage gate: `--cov-fail-under=100` over `src/grd_generator`, with `generate.py`, `examples/*`, and `tests/*` omitted. The gate is added in the LAST task; earlier tasks run pytest without the fail-under threshold.
- ruff line-length 100, target py312, lint select `["E","F","I","UP","B","SIM"]`.
- mypy strict, plugin `pydantic.mypy`.
- Source of truth for copied code: `~/Documents/satscope-weekend-recovered/src/satscope/`.
- French docstrings/comments preserved from source (user works in French).
- Reference output path: `data/processed/reference_array.npz`.
- Generation parameters (unchanged from source): `N_ANTENNAS=80`, `PEAK_GAIN_DBI=47.0`, `PHASE_SLOPE_RADIAL=3.0`, grid `161×161`, `SIGMA_LO=1e-3`, `SIGMA_HI=5.0`, `TARGET_DBI=44.0`.

---

### Task 1: Project scaffold, tooling, and logger

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `.claude/settings.json`
- Create: `data/processed/.gitkeep`
- Create: `src/grd_generator/__init__.py` (empty for now)
- Create: `src/grd_generator/logger.py`
- Test: `src/grd_generator/tests/__init__.py` (empty), `src/grd_generator/tests/test_logger.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `grd_generator.logger.logger` (loguru logger), `grd_generator.logger.configure_logging(level: str = "INFO") -> None`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "grd_generator"
version = "0.1.0"
description = "Générateur autonome du jeu de référence de diagrammes de rayonnement d'antenne (.npz)."
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.0",
    "pydantic>=2.7",
    "loguru>=0.7",
]

[dependency-groups]
dev = [
    "pytest>=8.2",
    "pytest-cov>=5.0",
    "ruff>=0.5",
    "mypy>=1.10",
]

[project.scripts]
grd-generate = "grd_generator.generate:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/grd_generator"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.12"
strict = true
plugins = ["pydantic.mypy"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["src"]
addopts = "-q --import-mode=importlib"

[tool.coverage.run]
source = ["src/grd_generator"]
branch = true
omit = [
    "*/grd_generator/generate.py",       # script offline : orchestration + écriture .npz
    "*/grd_generator/examples/*",        # scripts d'exemple : testés en lancement
    "*/tests/*",
]

[tool.coverage.report]
show_missing = true
exclude_also = ["if __name__ == .__main__.:"]
```

Note: the `[project.scripts]` entry and the `generate.py` omit reference a module created in Task 6 — harmless until then.

- [ ] **Step 2: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
.venv*/
*.egg-info/
.mypy_cache/
.ruff_cache/
.pytest_cache/
.coverage
.coverage.*
htmlcov/

# Exports de debug et données intermédiaires
debug_export/

# Données (jamais versionnées : volumineuses)
data/processed/*
!data/processed/.gitkeep

# Outils
.idea/
.vscode/
```

- [ ] **Step 3: Create `.claude/settings.json`** (permissions block, no hooks)

```json
{
  "permissions": {
    "allow": [
      "Bash(uv:*)",
      "Bash(uv run pytest:*)",
      "Bash(uv run ruff:*)",
      "Bash(uv run mypy:*)",
      "Bash(python:*)",
      "Bash(python3:*)",
      "Bash(pytest:*)",
      "Bash(ruff:*)",
      "Bash(mypy:*)",
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(git status)",
      "Bash(git diff:*)",
      "Bash(git log:*)"
    ],
    "deny": []
  }
}
```

- [ ] **Step 4: Create `README.md`, `data/processed/.gitkeep`, empty `src/grd_generator/__init__.py`, empty `src/grd_generator/tests/__init__.py`**

`README.md`:
```markdown
# grd_generator

Générateur autonome du jeu de référence de diagrammes de rayonnement d'antenne
satellite (fichier `.npz`).

## Usage

```bash
uv sync
uv run grd-generate            # écrit data/processed/reference_array.npz
```

Voir `src/grd_generator/examples/example_generation.py` pour un exemple.
```

`data/processed/.gitkeep`: empty file.
`src/grd_generator/__init__.py`: empty file.
`src/grd_generator/tests/__init__.py`: empty file.

- [ ] **Step 5: Create `src/grd_generator/logger.py`** (copy of `satscope/shared/logger.py`)

```python
"""Configuration centralisée du logging structuré (loguru, sortie JSON)."""

import sys

from loguru import logger

_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Configure loguru pour émettre des logs JSON structurés sur stderr.

    À appeler une seule fois depuis un point d'entrée (CLI, notebook, exemple).
    Idempotent : les appels suivants sont ignorés.
    """
    global _configured
    if _configured:
        return
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time} {level} {message}",
        serialize=True,
    )
    _configured = True


__all__ = ["logger", "configure_logging"]
```

- [ ] **Step 6: Write the failing test** `src/grd_generator/tests/test_logger.py`

```python
from grd_generator.logger import configure_logging, logger


def test_configure_logging_is_idempotent():
    configure_logging()
    configure_logging()  # second call must be a no-op, not raise
    logger.info("test_event", key="value")  # must not raise
```

- [ ] **Step 7: Initialize uv, install, run test to verify it passes**

Run: `uv sync && uv run pytest src/grd_generator/tests/test_logger.py -v`
Expected: PASS. (`uv sync` creates the venv and installs deps.)

- [ ] **Step 8: Lint check**

Run: `uv run ruff check . && uv run mypy src/grd_generator/logger.py`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat: scaffold grd_generator (pyproject, settings, logger)"
```

---

### Task 2: schemas.py (pruned)

**Files:**
- Create: `src/grd_generator/schemas.py`
- Test: `src/grd_generator/tests/test_schemas.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `UVGrid(BaseModel)` with fields `u_min, u_max, v_min, v_max: float`, `n_u, n_v: int (>1)`; methods `axes() -> tuple[NDArray, NDArray]`, `meshgrid() -> tuple[NDArray, NDArray]` (shape `(n_v, n_u)`).
  - `GaussianSpec(BaseModel)` with `center_uv: tuple[float, float]`, `sigma: float (>0)`, `peak_gain_dbi: float`, `phase_slope_radial: float = 0.0`.
  - Aliases `ComplexField = NDArray[np.complex128]`, `DirectivityMap = NDArray[np.float64]`.

- [ ] **Step 1: Write the failing test** `src/grd_generator/tests/test_schemas.py`

```python
import numpy as np
import pytest
from pydantic import ValidationError

from grd_generator.schemas import GaussianSpec, UVGrid


def test_uvgrid_axes_lengths():
    grid = UVGrid(u_min=-1.0, u_max=1.0, v_min=-2.0, v_max=2.0, n_u=5, n_v=9)
    u, v = grid.axes()
    assert u.shape == (5,) and v.shape == (9,)
    assert u[0] == -1.0 and u[-1] == 1.0


def test_uvgrid_meshgrid_shape_is_nv_by_nu():
    grid = UVGrid(u_min=0.0, u_max=1.0, v_min=0.0, v_max=1.0, n_u=4, n_v=7)
    gu, gv = grid.meshgrid()
    assert gu.shape == (7, 4) and gv.shape == (7, 4)


def test_uvgrid_rejects_degenerate_axis():
    with pytest.raises(ValidationError):
        UVGrid(u_min=0.0, u_max=1.0, v_min=0.0, v_max=1.0, n_u=1, n_v=5)


def test_gaussian_spec_requires_positive_sigma():
    with pytest.raises(ValidationError):
        GaussianSpec(center_uv=(0.0, 0.0), sigma=0.0, peak_gain_dbi=40.0)


def test_gaussian_spec_defaults_phase_slope_zero():
    spec = GaussianSpec(center_uv=(1.0, 2.0), sigma=0.5, peak_gain_dbi=40.0)
    assert spec.phase_slope_radial == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/grd_generator/tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: grd_generator.schemas`.

- [ ] **Step 3: Create `src/grd_generator/schemas.py`**

```python
"""Schémas du générateur : diagrammes d'antennes en espace (u, v)."""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

ComplexField = NDArray[np.complex128]  # champ E(u, v) sur la grille
DirectivityMap = NDArray[np.float64]  # directivité en dBi sur la grille


class UVGrid(BaseModel):
    """Maillage angulaire régulier (u, v) vu du satellite, en degrés."""

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


class GaussianSpec(BaseModel):
    """Paramètres synthétiques d'un élément (entrée du générateur)."""

    center_uv: tuple[float, float]
    sigma: float = Field(..., gt=0)
    peak_gain_dbi: float
    phase_slope_radial: float = 0.0


__all__ = [
    "ComplexField",
    "DirectivityMap",
    "UVGrid",
    "GaussianSpec",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/grd_generator/tests/test_schemas.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add pruned pattern schemas (UVGrid, GaussianSpec)"
```

---

### Task 3: geometry.py (ECEF projection)

**Files:**
- Create: `src/grd_generator/geometry.py` — verbatim copy of `~/Documents/satscope-weekend-recovered/src/satscope/satview/geometry.py` (it has NO `satscope.*` imports, so no rewrite needed).
- Test: `src/grd_generator/tests/test_geometry.py`

**Interfaces:**
- Consumes: nothing.
- Produces (exact signatures from source):
  - `R_EARTH_KM = 6371.0`, `R_GEO_KM = 42164.0`
  - `ground_ecef(lat_deg, lon_deg) -> Array`
  - `satellite_ecef(lon_deg: float) -> Array`
  - `boresight_frame(sat_pos, target_pos) -> tuple[Array, Array, Array]`
  - `project_to_angular(lat_deg, lon_deg, sat_lon_deg, boresight_lat_deg, boresight_lon_deg) -> tuple[Array, Array]`
  - `project_point(lat_deg, lon_deg, sat_lon_deg, boresight_lat_deg, boresight_lon_deg) -> tuple[float, float]`
  - `project_to_angular_masked(...) -> tuple[Array, Array]`
  - `earth_limb_angle_deg() -> float`
  - `angular_to_direction(u_deg, v_deg, sat_lon_deg, boresight_lat_deg, boresight_lon_deg) -> Array`
  - `earth_intersection_latlon(u_deg, v_deg, sat_lon_deg, boresight_lat_deg, boresight_lon_deg) -> tuple[Array, Array, BoolArray]`

- [ ] **Step 1: Copy the source file verbatim**

```bash
cp ~/Documents/satscope-weekend-recovered/src/satscope/satview/geometry.py \
   src/grd_generator/geometry.py
```

Verify with `grep satscope src/grd_generator/geometry.py` → expected: no output (the module is numpy-only).

- [ ] **Step 2: Write the failing test** `src/grd_generator/tests/test_geometry.py`

```python
import numpy as np

from grd_generator.geometry import (
    earth_intersection_latlon,
    earth_limb_angle_deg,
    project_point,
    project_to_angular_masked,
)


def test_limb_angle_about_8_7_degrees():
    assert abs(earth_limb_angle_deg() - 8.7) < 0.1


def test_boresight_point_projects_to_origin():
    # A point at the boresight (sat sub-point) projects to (u, v) ≈ (0, 0).
    u, v = project_point(0.0, 3.0, 3.0, 0.0, 3.0)
    assert abs(u) < 1e-6 and abs(v) < 1e-6


def test_project_then_raytrace_roundtrip():
    # Project a ground point to (u, v), ray-trace back, recover lat/lon.
    sat_lon = 3.0
    lat0, lon0 = 46.6, 2.5
    u, v = project_point(lat0, lon0, sat_lon, 0.0, sat_lon)
    lat, lon, hit = earth_intersection_latlon(
        np.array([u]), np.array([v]), sat_lon, 0.0, sat_lon
    )
    assert bool(hit[0])
    assert abs(float(lat[0]) - lat0) < 1e-3
    assert abs(float(lon[0]) - lon0) < 1e-3


def test_backface_point_is_nan():
    # A point on the far side of Earth (opposite the sat) is hidden → NaN.
    u, v = project_to_angular_masked(
        np.array([0.0]), np.array([183.0]), 3.0, 0.0, 3.0
    )
    assert np.isnan(u[0]) and np.isnan(v[0])


def test_raytrace_miss_off_disk():
    # A direction well outside the limb misses Earth → hit False, NaN lat/lon.
    lat, lon, hit = earth_intersection_latlon(
        np.array([45.0]), np.array([0.0]), 3.0, 0.0, 3.0
    )
    assert not bool(hit[0])
    assert np.isnan(lat[0])
```

- [ ] **Step 3: Run test to verify it passes**

Run: `uv run pytest src/grd_generator/tests/test_geometry.py -v`
Expected: PASS (5 tests). If `test_raytrace_miss_off_disk` does not actually miss (limb ≈ 8.7°, 45° is far outside), it will pass; if numerics surprise you, debug with the systematic-debugging skill — do NOT weaken the assertion blindly.

- [ ] **Step 4: Type-check**

Run: `uv run mypy src/grd_generator/geometry.py`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add ECEF viewing geometry (project, limb, ray-trace)"
```

---

### Task 4: scenario.py (minimal dataclass + FRANCE + sat_frame_geometry)

**Files:**
- Create: `src/grd_generator/scenario.py`
- Test: `src/grd_generator/tests/test_scenario.py`

**Interfaces:**
- Consumes: `grd_generator.schemas.UVGrid`; `grd_generator.geometry.{project_point, earth_limb_angle_deg, earth_intersection_latlon}`.
- Produces:
  - `Scenario` frozen dataclass: `name: str`, `sat_lon_deg: float`, `antenna_latlon: tuple[float, float]`, `zone_radius_deg: float = 6.0`, `limb_margin_deg: float = 0.2`.
  - `FRANCE: Scenario`.
  - `SatFrameGeometry` frozen dataclass: `sat_lon_deg: float`, `zone_center: tuple[float, float]`, `zone_radius_deg: float`, `grid: UVGrid`; methods `render_kwargs() -> RenderBoresight`, `project_latlon(lat, lon) -> tuple[float, float]`, `ground_latlon(u, v) -> tuple[float, float] | None`.
  - `sat_frame_geometry(scenario: Scenario, n_u: int = 161, n_v: int = 161) -> SatFrameGeometry`.

- [ ] **Step 1: Write the failing test** `src/grd_generator/tests/test_scenario.py`

```python
import math

import pytest

from grd_generator.scenario import FRANCE, Scenario, sat_frame_geometry
from grd_generator.geometry import earth_limb_angle_deg


def test_france_defaults():
    assert FRANCE.name == "france"
    assert FRANCE.sat_lon_deg == 3.0
    assert FRANCE.antenna_latlon == (46.6, 2.5)
    assert FRANCE.zone_radius_deg == 6.0
    assert FRANCE.limb_margin_deg == 0.2


def test_scenario_rejects_subregulatory_radius():
    with pytest.raises(ValueError):
        Scenario(name="x", sat_lon_deg=0.0, antenna_latlon=(0.0, 0.0), zone_radius_deg=5.0)


def test_sat_frame_geometry_zone_under_limb():
    geo = sat_frame_geometry(FRANCE)
    # The whole service disk stays under the limb (with margin).
    center_dist = math.hypot(*geo.zone_center)
    limb = earth_limb_angle_deg()
    assert center_dist + geo.zone_radius_deg + FRANCE.limb_margin_deg <= limb + 1e-9
    assert geo.grid.n_u == 161 and geo.grid.n_v == 161


def test_sat_frame_geometry_grid_spans_zone():
    geo = sat_frame_geometry(FRANCE)
    half_span = geo.zone_radius_deg + 0.5  # _GRID_MARGIN_DEG
    assert geo.grid.u_min == pytest.approx(geo.zone_center[0] - half_span)
    assert geo.grid.u_max == pytest.approx(geo.zone_center[0] + half_span)


def test_zone_too_wide_raises():
    # radius ≥ limb − margin cannot fit under the limb.
    big = Scenario(name="big", sat_lon_deg=0.0, antenna_latlon=(0.0, 0.0), zone_radius_deg=9.0)
    with pytest.raises(ValueError):
        sat_frame_geometry(big)


def test_render_kwargs_and_roundtrip_helpers():
    geo = sat_frame_geometry(FRANCE)
    kw = geo.render_kwargs()
    assert kw["boresight_lat_deg"] == 0.0
    assert kw["sat_lon_deg"] == FRANCE.sat_lon_deg
    # ground_latlon of the zone center is on Earth; a far off-disk point is None.
    assert geo.ground_latlon(*geo.zone_center) is not None
    assert geo.ground_latlon(45.0, 0.0) is None
    # project_latlon of the antenna aim lands near the zone center direction.
    u, v = geo.project_latlon(*FRANCE.antenna_latlon)
    assert isinstance(u, float) and isinstance(v, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/grd_generator/tests/test_scenario.py -v`
Expected: FAIL with `ModuleNotFoundError: grd_generator.scenario`.

- [ ] **Step 3: Create `src/grd_generator/scenario.py`**

```python
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
    zone_radius_deg: float = 6.0  # rayon de la zone de service (min réglementaire 6°)
    limb_margin_deg: float = 0.2  # marge gardant la zone strictement sous le limbe

    def __post_init__(self) -> None:
        """Échoue tôt sur un scénario mal formé."""
        if self.zone_radius_deg < 6.0:
            raise ValueError(
                f"scénario {self.name!r} : zone_radius_deg={self.zone_radius_deg} "
                f"sous le minimum réglementaire de 6°"
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/grd_generator/tests/test_scenario.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Type-check**

Run: `uv run mypy src/grd_generator/scenario.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add minimal Scenario dataclass + FRANCE + sat_frame_geometry"
```

---

### Task 5: synth.py (field synthesis + σ optimization)

**Files:**
- Create: `src/grd_generator/synth.py` — copy of `~/Documents/satscope-weekend-recovered/src/satscope/pattern/synth.py` with the two import lines rewritten.
- Test: `src/grd_generator/tests/test_synth.py`

**Interfaces:**
- Consumes: `grd_generator.schemas.{ComplexField, DirectivityMap, GaussianSpec, UVGrid}`; `grd_generator.logger.logger`.
- Produces (exact names from source):
  - `gaussian_field(spec, grid) -> ComplexField`, `airy_field(spec, grid) -> ComplexField`
  - `available_modes() -> list[str]`, `field_generator(mode: str) -> FieldGenerator`, `FieldGenerator` type
  - `power_fraction(field, grid) -> float`, `max_peak_dbi(mode, sigma) -> float`, `check_power_conservation(mode, sigma, peak_gain_dbi) -> bool`
  - `directivity_dbi_from_field(field) -> DirectivityMap`, `combined_max_directivity_dbi(fields) -> DirectivityMap`
  - `max_envelope(specs, grid, *, mode="gaussian") -> DirectivityMap`
  - `envelope_min_dbi(specs, grid, zone_center, zone_radius, *, mode="gaussian") -> float`
  - `hex_centers_uv(zone_center, zone_radius, n, spacing) -> list[tuple[float, float]]`
  - `optimize_sigma(grid, centers, peak_gain_dbi, zone_center, zone_radius, target_dbi=45.0, sigma_lo=1e-3, sigma_hi=5.0, tol=1e-3, phase_slope_radial=0.0, *, mode="gaussian") -> tuple[float, list[GaussianSpec]]`

- [ ] **Step 1: Copy the source file**

```bash
cp ~/Documents/satscope-weekend-recovered/src/satscope/pattern/synth.py \
   src/grd_generator/synth.py
```

- [ ] **Step 2: Rewrite the two import lines**

In `src/grd_generator/synth.py`, change:
```python
from satscope.pattern.schemas import (
    ComplexField,
    DirectivityMap,
    GaussianSpec,
    UVGrid,
)
from satscope.shared.logger import logger
```
to:
```python
from grd_generator.schemas import (
    ComplexField,
    DirectivityMap,
    GaussianSpec,
    UVGrid,
)
from grd_generator.logger import logger
```

Verify: `grep satscope src/grd_generator/synth.py` → expected: no output.

- [ ] **Step 3: Write the failing test** `src/grd_generator/tests/test_synth.py`

```python
import numpy as np
import pytest

from grd_generator.schemas import GaussianSpec, UVGrid
from grd_generator.synth import (
    airy_field,
    available_modes,
    check_power_conservation,
    field_generator,
    gaussian_field,
    hex_centers_uv,
    max_peak_dbi,
    optimize_sigma,
)


def _grid() -> UVGrid:
    return UVGrid(u_min=-8.0, u_max=8.0, v_min=-8.0, v_max=8.0, n_u=81, n_v=81)


def test_available_modes():
    assert available_modes() == ["airy", "gaussian"]


def test_gaussian_peak_at_center():
    grid = _grid()
    spec = GaussianSpec(center_uv=(0.0, 0.0), sigma=1.0, peak_gain_dbi=40.0)
    field = gaussian_field(spec, grid)
    peak = 10.0 ** (40.0 / 20.0)
    assert np.abs(field).max() == pytest.approx(peak, rel=1e-3)


def test_airy_has_sidelobes_and_nulls():
    grid = _grid()
    spec = GaussianSpec(center_uv=(0.0, 0.0), sigma=2.0, peak_gain_dbi=40.0)
    field = airy_field(spec, grid)
    # Airy amplitude changes sign (true nulls) → min of real part < 0 region exists.
    amp = np.abs(field)
    assert amp.min() < amp.max()  # structured, not flat


def test_field_generator_unknown_mode_raises():
    with pytest.raises(ValueError):
        field_generator("nope")


def test_max_peak_dbi_known_modes():
    assert max_peak_dbi("gaussian", 1.0) > 0
    assert max_peak_dbi("airy", 1.0) > 0
    with pytest.raises(ValueError):
        max_peak_dbi("nope", 1.0)


def test_check_power_conservation_warns_over_bound(caplog):
    # A huge peak for a wide lobe exceeds the physical bound → returns False.
    assert check_power_conservation("gaussian", sigma=5.0, peak_gain_dbi=99.0) is False
    assert check_power_conservation("gaussian", sigma=0.5, peak_gain_dbi=10.0) is True


def test_hex_centers_count_and_too_loose():
    centers = hex_centers_uv((0.0, 0.0), zone_radius=6.0, n=20, spacing=1.5)
    assert len(centers) == 20
    with pytest.raises(ValueError):
        hex_centers_uv((0.0, 0.0), zone_radius=1.0, n=10000, spacing=1.0)


def test_optimize_sigma_feasible_and_monotone():
    grid = _grid()
    centers = hex_centers_uv((0.0, 0.0), zone_radius=6.0, n=30, spacing=2.0)
    sigma, specs = optimize_sigma(
        grid, centers, peak_gain_dbi=47.0,
        zone_center=(0.0, 0.0), zone_radius=6.0, target_dbi=44.0,
    )
    assert 1e-3 <= sigma <= 5.0
    assert len(specs) == len(centers)
    assert all(s.sigma == sigma for s in specs)


def test_optimize_sigma_infeasible_raises():
    grid = _grid()
    centers = hex_centers_uv((0.0, 0.0), zone_radius=6.0, n=30, spacing=2.0)
    with pytest.raises(ValueError):
        optimize_sigma(
            grid, centers, peak_gain_dbi=-50.0,  # far too weak to ever reach target
            zone_center=(0.0, 0.0), zone_radius=6.0, target_dbi=44.0, sigma_hi=5.0,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/grd_generator/tests/test_synth.py -v`
Expected: PASS. If `test_check_power_conservation_warns_over_bound` boundary values don't straddle the bound, adjust the test inputs using `max_peak_dbi("gaussian", sigma)` as the reference — pick a `peak_gain_dbi` clearly above/below it. Do not change `synth.py`.

- [ ] **Step 5: Type-check**

Run: `uv run mypy src/grd_generator/synth.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add pure field synthesis and sigma optimization"
```

---

### Task 6: generate.py (write the .npz) + CLI

**Files:**
- Create: `src/grd_generator/generate.py` — adapted from `~/Documents/satscope-weekend-recovered/src/satscope/pattern/generation/generate_reference.py`.
- Test: `src/grd_generator/tests/test_generate.py`

**Interfaces:**
- Consumes: `grd_generator.scenario.{FRANCE, Scenario, sat_frame_geometry}`; `grd_generator.schemas.{GaussianSpec, UVGrid}`; `grd_generator.synth.{check_power_conservation, field_generator, hex_centers_uv, optimize_sigma}`; `grd_generator.logger.{configure_logging, logger}`.
- Produces:
  - `auto_spacing(zone_radius: float, n: int) -> float`
  - `write_reference(output_path, grid: UVGrid, specs: list[GaussianSpec], *, mode: str = "gaussian") -> Path`
  - `generate_reference(output_path=DEFAULT_OUTPUT, *, mode="gaussian", scenario: Scenario = FRANCE) -> Path`
  - `main() -> None`
  - Module constants: `N_ANTENNAS=80`, `PEAK_GAIN_DBI=47.0`, `TARGET_DBI=44.0`, `PHASE_SLOPE_RADIAL=3.0`, `GRID_N_U=161`, `GRID_N_V=161`, `SIGMA_LO=1e-3`, `SIGMA_HI=5.0`, `DEFAULT_OUTPUT=Path("data/processed/reference_array.npz")`.

- [ ] **Step 1: Create `src/grd_generator/generate.py`** (adapted — TARGET_DBI is a local constant, NO verify guard-rail)

```python
"""Générateur offline du jeu de référence pattern (80 éléments) → .npz.

Autonome et séparé de toute analyse : porte l'optimisation σ (bissection sous
contrainte enveloppe ≥ 44 dBi) puis écrit grille + 80 champs complexes +
métadonnées dans `data/processed/reference_array.npz`. Lancé à la main.

Usage :
    uv run grd-generate
    python src/grd_generator/generate.py
"""

from pathlib import Path

import numpy as np

from grd_generator.logger import configure_logging, logger
from grd_generator.scenario import FRANCE, Scenario, sat_frame_geometry
from grd_generator.schemas import GaussianSpec, UVGrid
from grd_generator.synth import (
    check_power_conservation,
    field_generator,
    hex_centers_uv,
    optimize_sigma,
)

# Paramètres d'array de référence.
N_ANTENNAS = 80
PEAK_GAIN_DBI = 47.0  # gain crête par élément (dBi)
# Plancher d'enveloppe imposé dans la zone = seuil de couverture arbitré (44 dBi).
TARGET_DBI = 44.0
# Pente de phase **radiale** (rad/°) commune à tous les éléments :
# φ = pente·‖uv−centre‖. N'affecte pas |E| ni la couverture, mais rend les poids
# de combinaison complexes et fait apparaître des ripples d'interférence.
PHASE_SLOPE_RADIAL = 3.0
GRID_N_U = 161
GRID_N_V = 161
SIGMA_LO = 1e-3
SIGMA_HI = 5.0
DEFAULT_OUTPUT = Path("data/processed/reference_array.npz")


def auto_spacing(zone_radius: float, n: int) -> float:
    """Pas hexagonal pour loger ~`n` points dans le disque (aire = π r²)."""
    return float(np.sqrt(2.0 * np.pi * zone_radius**2 / (np.sqrt(3.0) * n)))


def write_reference(
    output_path: str | Path,
    grid: UVGrid,
    specs: list[GaussianSpec],
    *,
    mode: str = "gaussian",
) -> Path:
    """Synthétise les champs des `specs` (selon le `mode`) et écrit le .npz."""
    generate = field_generator(mode)
    fields = np.stack([generate(s, grid) for s in specs])
    ids = np.array([f"ant_{k:02d}" for k in range(len(specs))])
    centers_uv = np.array([s.center_uv for s in specs], dtype=float)
    sigmas = np.array([s.sigma for s in specs], dtype=float)
    peaks = np.array([s.peak_gain_dbi for s in specs], dtype=float)
    phase_slopes = np.array([s.phase_slope_radial for s in specs], dtype=float)

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
        sigmas=sigmas,
        peaks_dbi=peaks,
        phase_slopes=phase_slopes,
    )
    logger.info("pattern_gen_done", file=str(out), n_antennas=len(specs))
    return out


def generate_reference(
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    mode: str = "gaussian",
    scenario: Scenario = FRANCE,
) -> Path:
    """Optimise la largeur, synthétise les 80 champs et écrit le .npz de référence.

    `mode` sélectionne le générateur d'éléments (`gaussian` par défaut, `airy`
    pour des éléments à lobes secondaires). `scenario` fixe la géométrie
    (défaut FRANCE). Géométrie en repère SAT (nadir) dérivée du scénario.
    """
    geo = sat_frame_geometry(scenario, n_u=GRID_N_U, n_v=GRID_N_V)
    spacing = auto_spacing(geo.zone_radius_deg, N_ANTENNAS)
    centers = hex_centers_uv(geo.zone_center, geo.zone_radius_deg, N_ANTENNAS, spacing)

    logger.info("pattern_gen_start", n_antennas=N_ANTENNAS, spacing_deg=round(spacing, 4))
    sigma, specs = optimize_sigma(
        geo.grid,
        centers,
        peak_gain_dbi=PEAK_GAIN_DBI,
        zone_center=geo.zone_center,
        zone_radius=geo.zone_radius_deg,
        target_dbi=TARGET_DBI,
        sigma_lo=SIGMA_LO,
        sigma_hi=SIGMA_HI,
        phase_slope_radial=PHASE_SLOPE_RADIAL,
        mode=mode,
    )
    logger.info("pattern_gen_sigma", sigma_deg=round(sigma, 4))

    # Conservation de la puissance : crête et largeur sont liées (D·Ω ≈ 4π).
    # Warning seulement.
    check_power_conservation(mode, sigma, PEAK_GAIN_DBI)

    return write_reference(output_path, geo.grid, specs, mode=mode)


def main() -> None:
    configure_logging()
    generate_reference()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the test** `src/grd_generator/tests/test_generate.py`

```python
import numpy as np

from grd_generator.generate import auto_spacing, generate_reference


def test_auto_spacing_positive():
    assert auto_spacing(6.0, 80) > 0


def test_generate_reference_writes_valid_npz(tmp_path):
    out = generate_reference(tmp_path / "ref.npz")
    assert out.exists()
    data = np.load(out, allow_pickle=False)
    # Grid + 80 fields with the right shape and metadata.
    assert int(data["n_u"]) == 161 and int(data["n_v"]) == 161
    assert data["fields"].shape == (80, 161, 161)
    assert data["fields"].dtype == np.complex128
    assert data["antenna_ids"].shape == (80,)
    assert data["centers_uv"].shape == (80, 2)
    assert data["sigmas"].shape == (80,)
    assert str(data["mode"]) == "gaussian"


def test_generate_reference_airy_mode(tmp_path):
    out = generate_reference(tmp_path / "airy.npz", mode="airy")
    data = np.load(out, allow_pickle=False)
    assert str(data["mode"]) == "airy"
    assert data["fields"].shape == (80, 161, 161)
```

Note: `generate.py` is omitted from coverage (Global Constraints), so this test asserts behavior, not coverage.

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest src/grd_generator/tests/test_generate.py -v`
Expected: PASS (3 tests). Note: each generation runs the full 161×161 × 80-element synthesis with σ bisection — may take a few seconds per test.

- [ ] **Step 4: Run the CLI end-to-end**

Run: `uv run grd-generate && ls -la data/processed/reference_array.npz`
Expected: file written; structured JSON logs `pattern_gen_start` / `pattern_gen_sigma` / `pattern_gen_done` on stderr.

- [ ] **Step 5: Type-check**

Run: `uv run mypy src/grd_generator/generate.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add reference .npz generator and CLI entry point"
```

---

### Task 7: Public API, example script, and 100% coverage gate

**Files:**
- Modify: `src/grd_generator/__init__.py` (fill the public API)
- Create: `src/grd_generator/examples/__init__.py` (empty)
- Create: `src/grd_generator/examples/example_generation.py`
- Test: `src/grd_generator/tests/test_example_generation.py`
- Modify: `pyproject.toml` (add `--cov` flags to `addopts`)

**Interfaces:**
- Consumes: all prior modules.
- Produces: importable public API from `grd_generator`; runnable example; enforced 100% coverage gate.

- [ ] **Step 1: Fill `src/grd_generator/__init__.py`**

```python
"""grd_generator — génération du jeu de référence de diagrammes d'antenne (.npz)."""

from grd_generator.generate import (
    auto_spacing,
    generate_reference,
    write_reference,
)
from grd_generator.scenario import (
    FRANCE,
    SatFrameGeometry,
    Scenario,
    sat_frame_geometry,
)
from grd_generator.schemas import GaussianSpec, UVGrid
from grd_generator.synth import (
    airy_field,
    available_modes,
    field_generator,
    gaussian_field,
    hex_centers_uv,
    optimize_sigma,
)

__all__ = [
    "UVGrid",
    "GaussianSpec",
    "Scenario",
    "FRANCE",
    "SatFrameGeometry",
    "sat_frame_geometry",
    "gaussian_field",
    "airy_field",
    "available_modes",
    "field_generator",
    "hex_centers_uv",
    "optimize_sigma",
    "auto_spacing",
    "write_reference",
    "generate_reference",
]
```

- [ ] **Step 2: Create the example** `src/grd_generator/examples/example_generation.py`

```python
"""
Exemple : génération du jeu de référence de patterns
=====================================================
Génère les 80 champs de référence (mode gaussian) dans un fichier .npz et
relit ses métadonnées.

Usage :
    python src/grd_generator/examples/example_generation.py
"""

import tempfile
from pathlib import Path

import numpy as np

from grd_generator.generate import generate_reference
from grd_generator.logger import configure_logging, logger


def main() -> None:
    configure_logging()
    logger.info("example_start", feature="generation")

    with tempfile.TemporaryDirectory() as tmp:
        out = generate_reference(Path(tmp) / "reference_array.npz")
        data = np.load(out, allow_pickle=False)
        logger.info(
            "example_generated",
            n_antennas=int(data["fields"].shape[0]),
            grid=(int(data["n_u"]), int(data["n_v"])),
            mode=str(data["mode"]),
        )

    logger.info("example_done", feature="generation")


if __name__ == "__main__":
    main()
```

Also create empty `src/grd_generator/examples/__init__.py`.

- [ ] **Step 3: Write the launch test** `src/grd_generator/tests/test_example_generation.py`

```python
import subprocess
import sys


def test_example_generation_runs():
    """Vérifie que le script d'exemple s'exécute sans erreur."""
    result = subprocess.run(
        [sys.executable, "src/grd_generator/examples/example_generation.py"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"L'exemple a échoué :\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
```

- [ ] **Step 4: Run the example test and import check**

Run: `uv run pytest src/grd_generator/tests/test_example_generation.py -v && uv run python -c "import grd_generator; print(grd_generator.__all__)"`
Expected: PASS; the `__all__` list prints.

- [ ] **Step 5: Enable the 100% coverage gate** — edit `pyproject.toml` `[tool.pytest.ini_options]` `addopts`:

```toml
addopts = "-q --import-mode=importlib --cov=src/grd_generator --cov-report=term-missing --cov-fail-under=100"
```

- [ ] **Step 6: Run the full suite under the gate**

Run: `uv run pytest`
Expected: ALL tests PASS and coverage == 100% (`generate.py`, `examples/*`, `tests/*` omitted per config). If any line in `schemas.py`, `geometry.py`, `scenario.py`, or `synth.py` is uncovered, add a targeted test for that branch — do NOT add `# pragma: no cover` to reach the number unless the line is genuinely unreachable (and justify it in the commit).

- [ ] **Step 7: Full lint + type pass**

Run: `uv run ruff check . && uv run mypy src/grd_generator`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: public API, example script, and 100% coverage gate"
```

---

## Coverage closure note

`geometry.py` is copied verbatim and includes `project_to_angular_masked` and the full ray-tracing path. The Task 3 tests cover the limb angle, projection, masked back-face, ray-trace hit, and ray-trace miss. If, under the Task 7 gate, any geometry/synth branch remains uncovered (e.g. the `target_dist == 0` branch in `scenario.sat_frame_geometry`, or the `x == 0` jinc center in `airy_field`), add the specific test that exercises it:
- `sat_frame_geometry` `target_dist == 0`: a scenario whose `antenna_latlon` is the sub-satellite point — `Scenario(name="nadir", sat_lon_deg=0.0, antenna_latlon=(0.0, 0.0))` → `zone_center == (0.0, 0.0)`.
- `airy_field` center: any grid that includes the element center (the default grids do) covers the `x == 0` branch.
- `combined_max_directivity_dbi` / `directivity_dbi_from_field`: exercised indirectly by `optimize_sigma`/`max_envelope`; add a direct unit test if a line shows uncovered.

## Self-Review notes

- **Spec coverage:** scaffold+settings+logger (T1), schemas pruned (T2), geometry (T3), scenario minimal + FRANCE + sat_frame_geometry (T4), synth gaussian+airy+optimize (T5), generate without verify guard-rail + TARGET_DBI local (T6), public API + example + 100% gate (T7). All spec sections mapped.
- **No verify/service/plot/pandas/matplotlib** anywhere — matches "génération seule".
- **Type consistency:** `generate_reference`, `write_reference`, `sat_frame_geometry`, `optimize_sigma`, `hex_centers_uv`, `field_generator` signatures are identical across the tasks that define and consume them.
</content>
