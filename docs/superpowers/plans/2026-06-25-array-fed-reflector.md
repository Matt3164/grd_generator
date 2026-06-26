# Array-Fed Reflector (AFR) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a physically-grounded array-fed reflector model (offset paraboloid + cos^q feeds + aperture-field/FFT far-field with Ludwig-3 co/cross-pol) producing per-feed complex secondary patterns, plus a displayed service zone, integrated into generation and the GUI.

**Architecture:** A new isolated `reflector/` submodule computes, for each feed, a complex co-pol field `E(u,v)` (and a cross-pol field) on the existing `UVGrid`. Each feed is mapped to an aperture field by geometric optics (cos^q illumination taper + linear phase tilt for beam scan), the far-field is the FFT of the aperture field, normalized so that `|E|²` is the linear directivity (matching the existing isotropic-referenced convention), then resampled onto the degree-based `UVGrid`. The existing envelope/plot/GUI consume these fields unchanged; a `ServiceZone` disk is overlaid on every display.

**Tech Stack:** Python 3, numpy (incl. `numpy.fft`), pydantic v2, loguru, matplotlib + PyQt6 (GUI). No scipy.

## Global Constraints

- Python package layout `src/grd_generator/...`; new module under `src/grd_generator/reflector/`.
- Runtime deps limited to `numpy`, `pydantic>=2.7`, `loguru` (+ `matplotlib`/`PyQt6` for GUI). **No scipy.**
- Test gate is strict: `pytest --cov-fail-under=100`. Every non-script line must be covered. Generation **scripts** are excluded from coverage exactly like `generate.py` (via `# pragma: no cover` on the `main()`/IO-write entrypoint and the `[tool.coverage]` omit list).
- Angular grids (`UVGrid`) are in **degrees**; `meshgrid()` returns shape `(n_v, n_u)` with `v` increasing per row.
- Field convention (from `synth.py`): a complex field `E` is isotropic-referenced so that `|E|²` **is the linear directivity** `D(u,v)`; dBi `= 10·log10(|E|²)`. Power conservation: `(1/4π)·Σ|E|²·dΩ ≤ 1`.
- Lengths in **meters**, frequencies in **Hz** internally; `c = 299_792_458.0 m/s`.
- Lint/type: code must pass `ruff` and `mypy` as configured. Match the existing French docstring style.
- Spec: `docs/superpowers/specs/2026-06-25-array-fed-reflector-design.md`.

---

### Task 1: Reflector & feed specs (`reflector/spec.py`)

**Files:**
- Create: `src/grd_generator/reflector/__init__.py`
- Create: `src/grd_generator/reflector/spec.py`
- Test: `src/grd_generator/tests/test_reflector_spec.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `C_LIGHT: float = 299_792_458.0`
  - `ReflectorSpec(diameter_m, focal_length_m, offset_clearance_m=0.0, freq_hz, beam_deviation_factor=1.0)` (pydantic) with read-only properties `wavelength_m -> float`, `f_over_d -> float`, `aperture_center_y_m -> float` (`= offset_clearance_m + diameter_m/2`).
  - `FeedSpec(positions_m: list[tuple[float,float]], q: float)` (pydantic) with property `n_feeds -> int`.

- [ ] **Step 1: Write the failing test**

```python
# src/grd_generator/tests/test_reflector_spec.py
import math
import pytest
from pydantic import ValidationError
from grd_generator.reflector.spec import C_LIGHT, ReflectorSpec, FeedSpec


def test_reflector_spec_derived_quantities():
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.4, freq_hz=20e9)
    assert spec.wavelength_m == pytest.approx(C_LIGHT / 20e9)
    assert spec.f_over_d == pytest.approx(1.2)
    assert spec.aperture_center_y_m == pytest.approx(1.0)  # 0 + D/2


def test_reflector_spec_rejects_nonpositive():
    with pytest.raises(ValidationError):
        ReflectorSpec(diameter_m=0.0, focal_length_m=2.4, freq_hz=20e9)


def test_feed_spec_counts():
    feeds = FeedSpec(positions_m=[(0.0, 0.0), (0.01, 0.0)], q=2.0)
    assert feeds.n_feeds == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/grd_generator/tests/test_reflector_spec.py -q`
Expected: FAIL — `ModuleNotFoundError: grd_generator.reflector.spec`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/grd_generator/reflector/__init__.py
"""Modèle physique array-fed reflector (offset paraboloïde + feeds cos^q)."""
```

```python
# src/grd_generator/reflector/spec.py
"""Spécifications physiques du réflecteur offset et du réseau de feeds."""

from pydantic import BaseModel, Field

C_LIGHT = 299_792_458.0  # vitesse de la lumière (m/s)


class ReflectorSpec(BaseModel):
    """Réflecteur parabolique offset simple + fréquence d'exploitation."""

    diameter_m: float = Field(..., gt=0)
    focal_length_m: float = Field(..., gt=0)
    offset_clearance_m: float = Field(0.0, ge=0)
    freq_hz: float = Field(..., gt=0)
    beam_deviation_factor: float = Field(1.0, gt=0)

    @property
    def wavelength_m(self) -> float:
        return C_LIGHT / self.freq_hz

    @property
    def f_over_d(self) -> float:
        return self.focal_length_m / self.diameter_m

    @property
    def aperture_center_y_m(self) -> float:
        """Décalage du centre de l'ouverture projetée par rapport à l'axe."""
        return self.offset_clearance_m + self.diameter_m / 2.0


class FeedSpec(BaseModel):
    """Réseau de feeds dans le plan focal : positions (m) et taper cos^q."""

    positions_m: list[tuple[float, float]]
    q: float = Field(..., ge=0.0)

    @property
    def n_feeds(self) -> int:
        return len(self.positions_m)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/grd_generator/tests/test_reflector_spec.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/grd_generator/reflector/__init__.py src/grd_generator/reflector/spec.py src/grd_generator/tests/test_reflector_spec.py
git commit -m "feat(reflector): ReflectorSpec & FeedSpec specs"
```

---

### Task 2: Service zone (`reflector/zone.py`)

**Files:**
- Create: `src/grd_generator/reflector/zone.py`
- Test: `src/grd_generator/tests/test_reflector_zone.py`

**Interfaces:**
- Consumes: `UVGrid` (`grd_generator.schemas`); `geometry.earth_intersection_latlon`, `geometry.angular_to_direction` if needed (see Step 3 — we reuse `project`-style helpers already present).
- Produces:
  - `ServiceZone(radius_deg: float, center_uv: tuple[float,float] = (0.0, 0.0))` (pydantic), validated `6.0 <= radius_deg <= 14.0`.
  - `ServiceZone.mask(grid: UVGrid) -> NDArray[np.bool_]` shape `(n_v, n_u)`, True inside the disk.
  - `ServiceZone.boundary_uv(n: int = 361) -> tuple[NDArray, NDArray]` — closed circle `(u[], v[])` in degrees.

- [ ] **Step 1: Write the failing test**

```python
# src/grd_generator/tests/test_reflector_zone.py
import numpy as np
import pytest
from pydantic import ValidationError
from grd_generator.schemas import UVGrid
from grd_generator.reflector.zone import ServiceZone


def _grid():
    return UVGrid(u_min=-14, u_max=14, v_min=-14, v_max=14, n_u=29, n_v=29)


def test_zone_radius_bounds():
    with pytest.raises(ValidationError):
        ServiceZone(radius_deg=5.0)
    with pytest.raises(ValidationError):
        ServiceZone(radius_deg=14.1)
    ServiceZone(radius_deg=6.0)
    ServiceZone(radius_deg=14.0)


def test_zone_mask_is_disk():
    zone = ServiceZone(radius_deg=8.0)
    grid = _grid()
    mask = zone.mask(grid)
    gu, gv = grid.meshgrid()
    expected = (gu**2 + gv**2) <= 8.0**2
    assert mask.shape == (29, 29)
    np.testing.assert_array_equal(mask, expected)


def test_zone_boundary_on_circle():
    zone = ServiceZone(radius_deg=10.0, center_uv=(1.0, -2.0))
    u, v = zone.boundary_uv(n=180)
    r = np.hypot(u - 1.0, v + 2.0)
    np.testing.assert_allclose(r, 10.0, atol=1e-9)
    assert u[0] == pytest.approx(u[-1]) and v[0] == pytest.approx(v[-1])  # closed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/grd_generator/tests/test_reflector_zone.py -q`
Expected: FAIL — `ModuleNotFoundError: grd_generator.reflector.zone`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/grd_generator/reflector/zone.py
"""Zone de service : disque angulaire (u,v) à couvrir, affiché sur les displays."""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from grd_generator.schemas import UVGrid


class ServiceZone(BaseModel):
    """Disque angulaire centré (par défaut sur le boresight nadir)."""

    radius_deg: float = Field(..., ge=6.0, le=14.0)
    center_uv: tuple[float, float] = (0.0, 0.0)

    def mask(self, grid: UVGrid) -> NDArray[np.bool_]:
        """Masque booléen (n_v, n_u) des points dans le disque."""
        gu, gv = grid.meshgrid()
        cu, cv = self.center_uv
        inside: NDArray[np.bool_] = ((gu - cu) ** 2 + (gv - cv) ** 2) <= self.radius_deg**2
        return inside

    def boundary_uv(self, n: int = 361) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Cercle fermé (u, v) en degrés pour tracer le contour de zone."""
        t = np.linspace(0.0, 2.0 * np.pi, n)
        cu, cv = self.center_uv
        u = cu + self.radius_deg * np.cos(t)
        v = cv + self.radius_deg * np.sin(t)
        return u, v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/grd_generator/tests/test_reflector_zone.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/grd_generator/reflector/zone.py src/grd_generator/tests/test_reflector_zone.py
git commit -m "feat(reflector): ServiceZone disk (mask + boundary, 6-14 deg)"
```

---

### Task 3: Aperture grid & feed illumination (`reflector/optics.py` — part 1)

**Files:**
- Create: `src/grd_generator/reflector/optics.py`
- Test: `src/grd_generator/tests/test_reflector_optics.py`

**Interfaces:**
- Consumes: `ReflectorSpec`, `FeedSpec` (Task 1).
- Produces:
  - `aperture_grid(spec, n_aperture, pad_factor) -> tuple[X, Y, inside, dx]` where `X,Y` are `(N,N)` meshgrids in meters (`N = n_aperture * pad_factor`), `inside` is the bool disk mask of diameter `D` centered at `(0, aperture_center_y_m)`, `dx` is the sample spacing (m), `dx = D / n_aperture`.
  - `illumination_angle(X, Y, focal_length_m) -> NDArray` — feed polar angle `ψ = 2·arctan(ρ/(2F))`, `ρ = hypot(X,Y)`.
  - `aperture_field(spec, feed_xy, X, Y, inside, q) -> NDArray[complex128]` — `inside · cos(ψ)^q · exp(j·tilt)`, `tilt = -k·BDF·(δx·X + δy·Y)/F`.

- [ ] **Step 1: Write the failing test**

```python
# src/grd_generator/tests/test_reflector_optics.py
import numpy as np
import pytest
from grd_generator.reflector.spec import ReflectorSpec
from grd_generator.reflector import optics


def _spec(offset=0.0):
    return ReflectorSpec(diameter_m=2.0, focal_length_m=2.0,
                         offset_clearance_m=offset, freq_hz=20e9)


def test_aperture_grid_disk_and_spacing():
    X, Y, inside, dx = optics.aperture_grid(_spec(), n_aperture=64, pad_factor=2)
    assert X.shape == (128, 128)
    assert dx == pytest.approx(2.0 / 64)
    # disk centered on (0, D/2)=(0,1): a point at (0,1) is inside, (0,3) outside.
    assert inside.sum() > 0
    r2 = X**2 + (Y - 1.0) ** 2
    np.testing.assert_array_equal(inside, r2 <= 1.0**2)


def test_illumination_angle_zero_on_axis():
    X = np.array([[0.0]])
    Y = np.array([[0.0]])
    assert optics.illumination_angle(X, Y, 2.0)[0, 0] == pytest.approx(0.0)


def test_aperture_field_uniform_when_q0_and_centered_feed():
    spec = _spec()
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture=32, pad_factor=2)
    field = optics.aperture_field(spec, (0.0, 0.0), X, Y, inside, q=0.0)
    # q=0 → cos^0 = 1 inside; centered feed → zero phase tilt → real & flat.
    assert np.allclose(field[inside], 1.0)
    assert np.allclose(field[~inside], 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/grd_generator/tests/test_reflector_optics.py -q`
Expected: FAIL — `ModuleNotFoundError` / attribute errors.

- [ ] **Step 3: Write minimal implementation**

```python
# src/grd_generator/reflector/optics.py
"""Optique géométrique : feed → champ d'ouverture du réflecteur offset.

Modèle aperture-field. Un point d'ouverture (X, Y) (m, mesuré depuis l'axe du
paraboloïde parent) est illuminé par le feed sous l'angle polaire
ψ = 2·arctan(ρ/2F) (ρ = ‖(X,Y)‖) ; l'amplitude suit le diagramme de feed cos^q(ψ).
Un feed décalé de (δx, δy) dans le plan focal impose un gradient de phase linéaire
(beam deviation factor) qui dépointe le faisceau secondaire.
"""

import numpy as np
from numpy.typing import NDArray

from grd_generator.reflector.spec import ReflectorSpec

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def aperture_grid(
    spec: ReflectorSpec, n_aperture: int, pad_factor: int
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_], float]:
    """Grille carrée d'ouverture (m) + masque disque + pas dx.

    `n_aperture` échantillons sur le diamètre ; zéro-padding `pad_factor`× pour
    la résolution angulaire du far-field. Disque de diamètre D centré sur l'offset.
    """
    dx = spec.diameter_m / n_aperture
    n = n_aperture * pad_factor
    coords = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(coords, coords)
    r2 = X**2 + (Y - spec.aperture_center_y_m) ** 2
    inside: NDArray[np.bool_] = r2 <= (spec.diameter_m / 2.0) ** 2
    return X, Y, inside, dx


def illumination_angle(X: FloatArray, Y: FloatArray, focal_length_m: float) -> FloatArray:
    """Angle polaire ψ vu du feed pour un point d'ouverture : ψ = 2·arctan(ρ/2F)."""
    rho = np.hypot(X, Y)
    angle: FloatArray = 2.0 * np.arctan(rho / (2.0 * focal_length_m))
    return angle


def aperture_field(
    spec: ReflectorSpec,
    feed_xy: tuple[float, float],
    X: FloatArray,
    Y: FloatArray,
    inside: NDArray[np.bool_],
    q: float,
) -> ComplexArray:
    """Champ scalaire d'ouverture pour un feed : taper cos^q(ψ) + tilt de phase."""
    psi = illumination_angle(X, Y, spec.focal_length_m)
    amp = np.cos(psi) ** q
    k = 2.0 * np.pi / spec.wavelength_m
    dxf, dyf = feed_xy
    tilt = -k * spec.beam_deviation_factor * (dxf * X + dyf * Y) / spec.focal_length_m
    field: ComplexArray = (inside * amp * np.exp(1j * tilt)).astype(np.complex128)
    return field
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/grd_generator/tests/test_reflector_optics.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/grd_generator/reflector/optics.py src/grd_generator/tests/test_reflector_optics.py
git commit -m "feat(reflector): aperture grid + cos^q feed illumination"
```

---

### Task 4: Aperture polarization vectors (`reflector/optics.py` — part 2)

**Files:**
- Modify: `src/grd_generator/reflector/optics.py`
- Test: `src/grd_generator/tests/test_reflector_optics.py` (add)

**Interfaces:**
- Consumes: `ReflectorSpec`, `aperture_grid` outputs.
- Produces: `aperture_pol_vectors(spec, X, Y) -> tuple[ex, ey]` — real `(N,N)` arrays giving the x and y components of the GO-reflected polarization unit vector for an x-polarized feed. Mechanism: reflect `x̂` about the paraboloid surface normal `n̂ = (-X/2F, -Y/2F, 1)/‖·‖` via `e_r = 2(n̂·x̂)n̂ − x̂`; keep transverse `(x,y)` parts. Near the vertex `n̂→ẑ` so `e_r→(-1,0,0)` (pure co-pol, zero cross). Cross-pol (`ey≠0`) appears where `X·Y≠0` (offset/diagonal regions).

- [ ] **Step 1: Write the failing test**

```python
# add to src/grd_generator/tests/test_reflector_optics.py

def test_pol_vectors_pure_copol_on_axis():
    spec = _spec()
    X = np.array([[0.0]])
    Y = np.array([[0.0]])
    ex, ey = optics.aperture_pol_vectors(spec, X, Y)
    assert abs(ey[0, 0]) < 1e-12          # no cross-pol on axis
    assert abs(abs(ex[0, 0]) - 1.0) < 1e-12


def test_pol_vectors_cross_in_diagonal():
    spec = _spec()
    X = np.array([[0.4]])
    Y = np.array([[0.4]])
    ex, ey = optics.aperture_pol_vectors(spec, X, Y)
    assert abs(ey[0, 0]) > 1e-6           # diagonal region → cross-pol present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/grd_generator/tests/test_reflector_optics.py -q`
Expected: FAIL — `AttributeError: ... aperture_pol_vectors`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/grd_generator/reflector/optics.py

def aperture_pol_vectors(
    spec: ReflectorSpec, X: FloatArray, Y: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Composantes (ex, ey) du vecteur de polarisation réfléchi (feed x-polarisé).

    Réflexion GO de x̂ sur la normale du paraboloïde : e_r = 2(n̂·x̂)n̂ − x̂.
    Source géométrique de la cross-pol (offset / plans diagonaux).
    """
    f = spec.focal_length_m
    nx = -X / (2.0 * f)
    ny = -Y / (2.0 * f)
    nz = np.ones_like(X)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    ndotx = nx  # n̂ · x̂
    ex: FloatArray = 2.0 * ndotx * nx - 1.0
    ey: FloatArray = 2.0 * ndotx * ny
    return ex, ey
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/grd_generator/tests/test_reflector_optics.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/grd_generator/reflector/optics.py src/grd_generator/tests/test_reflector_optics.py
git commit -m "feat(reflector): GO aperture polarization vectors (cross-pol source)"
```

---

### Task 5: Far-field FFT + directivity normalization (`reflector/farfield.py` — part 1)

**Files:**
- Create: `src/grd_generator/reflector/farfield.py`
- Test: `src/grd_generator/tests/test_reflector_farfield.py`

**Interfaces:**
- Consumes: aperture complex fields + `dx`, `wavelength_m`.
- Produces:
  - `far_field_fft(aperture, dx, wavelength_m) -> tuple[F, L, M]` — `F` complex `(N,N)` far-field amplitude (fftshifted), `L,M` `(N,N)` direction-cosine grids (`l=sinθcosφ`), `L = λ·fftshift(fftfreq(N, dx))` broadcast.
  - `normalize_to_directivity(power_components, L, M, dL, dM) -> float` — returns the scalar `norm = sqrt(4π / ∮(Σ|F|²)dΩ)` integrating `dΩ = dL·dM/cosθ` over the visible region `L²+M²<1`. `power_components` is the summed `|F|²` array.

- [ ] **Step 1: Write the failing test (Airy oracle)**

```python
# src/grd_generator/tests/test_reflector_farfield.py
import numpy as np
import pytest
from grd_generator.reflector.spec import ReflectorSpec
from grd_generator.reflector import optics, farfield


def test_uniform_aperture_peak_directivity_matches_airy():
    # Uniform circular aperture (q=0, centered feed), symmetric (no offset to
    # keep it scalar/co-pol): peak directivity ≈ (πD/λ)².
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=1e9,  # ~flat: ψ≈0, n̂≈ẑ
                         offset_clearance_m=0.0, freq_hz=20e9)
    # center the disk on the axis by overriding aperture: build a centered disk.
    n_aperture, pad = 96, 6
    dx = spec.diameter_m / n_aperture
    n = n_aperture * pad
    coords = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(coords, coords)
    inside = (X**2 + Y**2) <= (spec.diameter_m / 2.0) ** 2
    aperture = inside.astype(np.complex128)

    F, L, M = farfield.far_field_fft(aperture, dx, spec.wavelength_m)
    dL = L[0, 1] - L[0, 0]
    dM = M[1, 0] - M[0, 0]
    norm = farfield.normalize_to_directivity(np.abs(F) ** 2, L, M, dL, dM)
    peak_dbi = 10.0 * np.log10((np.abs(F) * norm).max() ** 2)
    expected = 10.0 * np.log10((np.pi * spec.diameter_m / spec.wavelength_m) ** 2)
    assert peak_dbi == pytest.approx(expected, abs=0.3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/grd_generator/tests/test_reflector_farfield.py -q`
Expected: FAIL — `ModuleNotFoundError: grd_generator.reflector.farfield`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/grd_generator/reflector/farfield.py
"""Champ lointain : FFT du champ d'ouverture, normalisé en directivité.

Le far-field est la TF 2D du champ d'ouverture, échantillonné en cosinus
directeurs (l, m) = (sinθcosφ, sinθsinφ). La normalisation impose
|E|² = directivité linéaire (convention isotrope du projet) :
D = 4π·|F|² / ∮|F|² dΩ, dΩ = dl·dm / cosθ sur la région visible l²+m²<1.
"""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def far_field_fft(
    aperture: ComplexArray, dx: float, wavelength_m: float
) -> tuple[ComplexArray, FloatArray, FloatArray]:
    """TF 2D centrée du champ d'ouverture → (F, L, M) en cosinus directeurs."""
    n = aperture.shape[0]
    spec = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(aperture)))
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=dx))  # cycles/m
    lin = wavelength_m * freq  # cosinus directeur l = λ·fx
    L, M = np.meshgrid(lin, lin)
    return spec.astype(np.complex128), L, M


def normalize_to_directivity(
    power: FloatArray, L: FloatArray, M: FloatArray, dL: float, dM: float
) -> float:
    """Facteur d'échelle tel que |F·norm|² soit la directivité (réf. isotrope)."""
    visible = (L**2 + M**2) < 1.0
    cos_theta = np.sqrt(np.clip(1.0 - L**2 - M**2, 1e-12, None))
    radiated = float(np.sum(power[visible] / cos_theta[visible]) * dL * dM)
    return float(np.sqrt(4.0 * np.pi / radiated))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/grd_generator/tests/test_reflector_farfield.py -q`
Expected: PASS — peak directivity within 0.3 dB of `(πD/λ)²`.

- [ ] **Step 5: Commit**

```bash
git add src/grd_generator/reflector/farfield.py src/grd_generator/tests/test_reflector_farfield.py
git commit -m "feat(reflector): far-field FFT + directivity normalization (Airy oracle)"
```

---

### Task 6: Ludwig-3 co/cross-pol + resample to UVGrid (`reflector/farfield.py` — part 2)

**Files:**
- Modify: `src/grd_generator/reflector/farfield.py`
- Test: `src/grd_generator/tests/test_reflector_farfield.py` (add)

**Interfaces:**
- Consumes: Cartesian far-field components `Fx, Fy` + `L, M`.
- Produces:
  - `ludwig3_co_cross(Fx, Fy, L, M) -> tuple[E_co, E_cross]` — exact Ludwig-3 (x-reference). At `L=M=0`: `E_co=Fx`, `E_cross=Fy`.
  - `resample_to_uvgrid(field, lin_axis, grid) -> ComplexField` — bilinear-interpolate complex `field` (defined on the regular ascending `lin_axis` for both axes) onto `grid` direction cosines `l=sin(deg2rad(u))`, `m=sin(deg2rad(v))`; output shape `(n_v, n_u)`.

- [ ] **Step 1: Write the failing test**

```python
# add to src/grd_generator/tests/test_reflector_farfield.py
from grd_generator.schemas import UVGrid


def test_ludwig3_identity_at_boresight():
    L = np.array([[0.0]]); M = np.array([[0.0]])
    Fx = np.array([[3.0 + 1j]]); Fy = np.array([[1.0 - 2j]])
    co, cross = farfield.ludwig3_co_cross(Fx, Fy, L, M)
    assert co[0, 0] == pytest.approx(3.0 + 1j)
    assert cross[0, 0] == pytest.approx(1.0 - 2j)


def test_ludwig3_pure_copol_has_negligible_cross_near_axis():
    lin = np.linspace(-0.05, 0.05, 11)
    L, M = np.meshgrid(lin, lin)
    Fx = np.ones_like(L, dtype=complex)
    Fy = np.zeros_like(L, dtype=complex)
    _, cross = farfield.ludwig3_co_cross(Fx, Fy, L, M)
    assert np.max(np.abs(cross)) < 1e-2


def test_resample_picks_nearest_grid_values():
    lin = np.linspace(-0.2, 0.2, 41)
    field = np.tile(lin, (41, 1)).astype(complex)  # value == l
    grid = UVGrid(u_min=-5, u_max=5, v_min=-5, v_max=5, n_u=11, n_v=11)
    out = farfield.resample_to_uvgrid(field, lin, grid)
    u, _ = grid.axes()
    np.testing.assert_allclose(out[5, :].real, np.sin(np.radians(u)), atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/grd_generator/tests/test_reflector_farfield.py -q`
Expected: FAIL — `AttributeError: ludwig3_co_cross` / `resample_to_uvgrid`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/grd_generator/reflector/farfield.py
from grd_generator.schemas import ComplexField, UVGrid


def ludwig3_co_cross(
    Fx: ComplexArray, Fy: ComplexArray, L: FloatArray, M: FloatArray
) -> tuple[ComplexArray, ComplexArray]:
    """Décomposition Ludwig-3 (référence x) du far-field cartésien transverse."""
    phi = np.arctan2(M, L)
    sin_t = np.sqrt(np.clip(L**2 + M**2, 0.0, 1.0))
    cos_t = np.sqrt(np.clip(1.0 - sin_t**2, 0.0, 1.0))
    e_theta = (Fx * np.cos(phi) + Fy * np.sin(phi)) * cos_t
    e_phi = -Fx * np.sin(phi) + Fy * np.cos(phi)
    e_co = e_theta * np.cos(phi) - e_phi * np.sin(phi)
    e_cross = e_theta * np.sin(phi) + e_phi * np.cos(phi)
    return e_co.astype(np.complex128), e_cross.astype(np.complex128)


def _bilinear(values: ComplexArray, axis: FloatArray, lq: FloatArray, mq: FloatArray) -> ComplexArray:
    """Interpolation bilinéaire de `values` (grille régulière `axis`×`axis`)."""
    ix = np.clip(np.searchsorted(axis, lq) - 1, 0, axis.size - 2)
    iy = np.clip(np.searchsorted(axis, mq) - 1, 0, axis.size - 2)
    x0, x1 = axis[ix], axis[ix + 1]
    y0, y1 = axis[iy], axis[iy + 1]
    tx = (lq - x0) / (x1 - x0)
    ty = (mq - y0) / (y1 - y0)
    v00 = values[iy, ix]; v01 = values[iy, ix + 1]
    v10 = values[iy + 1, ix]; v11 = values[iy + 1, ix + 1]
    out: ComplexArray = (
        v00 * (1 - tx) * (1 - ty) + v01 * tx * (1 - ty)
        + v10 * (1 - tx) * ty + v11 * tx * ty
    )
    return out


def resample_to_uvgrid(field: ComplexArray, lin_axis: FloatArray, grid: UVGrid) -> ComplexField:
    """Rééchantillonne le far-field (cosinus directeurs) sur la grille (u,v) en degrés."""
    gu, gv = grid.meshgrid()
    lq = np.sin(np.radians(gu))
    mq = np.sin(np.radians(gv))
    out: ComplexField = _bilinear(field, lin_axis, lq, mq).astype(np.complex128)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/grd_generator/tests/test_reflector_farfield.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/grd_generator/reflector/farfield.py src/grd_generator/tests/test_reflector_farfield.py
git commit -m "feat(reflector): Ludwig-3 co/cross-pol + UVGrid resample"
```

---

### Task 7: Assembly — `synthesize_reflector_fields` (`reflector/synth_afr.py`)

**Files:**
- Create: `src/grd_generator/reflector/synth_afr.py`
- Modify: `src/grd_generator/reflector/__init__.py` (export public API)
- Test: `src/grd_generator/tests/test_synth_afr.py`

**Interfaces:**
- Consumes: `ReflectorSpec`, `FeedSpec`, `UVGrid`, plus `optics.*` and `farfield.*`.
- Produces:
  - `synthesize_reflector_fields(spec, feeds, grid, *, n_aperture=128, pad_factor=4) -> tuple[list[ComplexField], list[ComplexField]]` — `(co_fields, cross_fields)`, one per feed, on `grid`. Each `co`/`cross` pair is normalized jointly (shared `norm`).
  - `hex_feed_positions(pitch_m, n_feeds, focal_radius_m) -> list[tuple[float,float]]` — hex lattice of feed offsets in the focal plane (reuses `synth.hex_centers_uv` with center `(0,0)`).
  - re-export `ReflectorSpec`, `FeedSpec`, `ServiceZone`, `synthesize_reflector_fields`, `hex_feed_positions` from `reflector/__init__.py`.

- [ ] **Step 1: Write the failing test**

```python
# src/grd_generator/tests/test_synth_afr.py
import numpy as np
import pytest
from grd_generator.schemas import UVGrid
from grd_generator.reflector import (
    ReflectorSpec, FeedSpec, synthesize_reflector_fields, hex_feed_positions,
)
from grd_generator.synth import combined_max_directivity_dbi


def _grid():
    return UVGrid(u_min=-14, u_max=14, v_min=-14, v_max=14, n_u=81, n_v=81)


def test_centered_feed_beam_points_at_boresight():
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.0, freq_hz=20e9)
    feeds = FeedSpec(positions_m=[(0.0, 0.0)], q=2.0)
    co, cross = synthesize_reflector_fields(spec, feeds, _grid(), n_aperture=64, pad_factor=4)
    assert len(co) == 1 and len(cross) == 1
    dbi = 10.0 * np.log10(np.abs(co[0]) ** 2)
    iv, iu = np.unravel_index(np.argmax(dbi), dbi.shape)
    gu, gv = _grid().meshgrid()
    assert abs(gu[iv, iu]) < 1.0 and abs(gv[iv, iu]) < 1.0  # peak near (0,0)


def test_displaced_feed_beam_scans(monkeypatch):
    # A feed displaced +x in the focal plane scans the beam off boresight.
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.0, freq_hz=20e9)
    feeds = FeedSpec(positions_m=[(0.05, 0.0)], q=2.0)
    co, _ = synthesize_reflector_fields(spec, feeds, _grid(), n_aperture=64, pad_factor=4)
    dbi = 10.0 * np.log10(np.abs(co[0]) ** 2)
    iv, iu = np.unravel_index(np.argmax(dbi), dbi.shape)
    gu, _ = _grid().meshgrid()
    assert abs(gu[iv, iu]) > 1.0  # beam moved away from boresight in u


def test_hex_feed_positions_count_and_centering():
    pos = hex_feed_positions(pitch_m=0.03, n_feeds=7, focal_radius_m=0.2)
    assert len(pos) == 7
    assert pos[0] == pytest.approx((0.0, 0.0))  # closest to center first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/grd_generator/tests/test_synth_afr.py -q`
Expected: FAIL — import errors (`synthesize_reflector_fields` not exported).

- [ ] **Step 3: Write minimal implementation**

```python
# src/grd_generator/reflector/synth_afr.py
"""Assemblage AFR : feeds → champs secondaires complexes (co + cross) sur (u,v)."""

import numpy as np

from grd_generator.schemas import ComplexField, UVGrid
from grd_generator.synth import hex_centers_uv
from grd_generator.reflector.spec import FeedSpec, ReflectorSpec
from grd_generator.reflector import optics, farfield


def hex_feed_positions(pitch_m: float, n_feeds: int, focal_radius_m: float) -> list[tuple[float, float]]:
    """Lattice hexagonal de positions de feeds (m) dans le plan focal, centré."""
    return hex_centers_uv((0.0, 0.0), focal_radius_m, n_feeds, pitch_m)


def synthesize_reflector_fields(
    spec: ReflectorSpec,
    feeds: FeedSpec,
    grid: UVGrid,
    *,
    n_aperture: int = 128,
    pad_factor: int = 4,
) -> tuple[list[ComplexField], list[ComplexField]]:
    """Pour chaque feed : champ co-pol et cross-pol (u,v), normalisés en directivité."""
    X, Y, inside, dx = optics.aperture_grid(spec, n_aperture, pad_factor)
    ex, ey = optics.aperture_pol_vectors(spec, X, Y)
    co_fields: list[ComplexField] = []
    cross_fields: list[ComplexField] = []
    for feed_xy in feeds.positions_m:
        scalar = optics.aperture_field(spec, feed_xy, X, Y, inside, feeds.q)
        Fx, L, M = farfield.far_field_fft(scalar * ex, dx, spec.wavelength_m)
        Fy, _, _ = farfield.far_field_fft(scalar * ey, dx, spec.wavelength_m)
        e_co, e_cross = farfield.ludwig3_co_cross(Fx, Fy, L, M)
        dL = float(L[0, 1] - L[0, 0])
        dM = float(M[1, 0] - M[0, 0])
        norm = farfield.normalize_to_directivity(
            np.abs(e_co) ** 2 + np.abs(e_cross) ** 2, L, M, dL, dM
        )
        lin = L[0, :]
        co_fields.append(farfield.resample_to_uvgrid(e_co * norm, lin, grid))
        cross_fields.append(farfield.resample_to_uvgrid(e_cross * norm, lin, grid))
    return co_fields, cross_fields
```

```python
# src/grd_generator/reflector/__init__.py  (replace contents)
"""Modèle physique array-fed reflector (offset paraboloïde + feeds cos^q)."""

from grd_generator.reflector.spec import C_LIGHT, FeedSpec, ReflectorSpec
from grd_generator.reflector.zone import ServiceZone
from grd_generator.reflector.synth_afr import (
    hex_feed_positions,
    synthesize_reflector_fields,
)

__all__ = [
    "C_LIGHT",
    "ReflectorSpec",
    "FeedSpec",
    "ServiceZone",
    "hex_feed_positions",
    "synthesize_reflector_fields",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/grd_generator/tests/test_synth_afr.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Run full suite + coverage gate**

Run: `uv run pytest -q`
Expected: PASS, coverage 100% (or note which `# pragma: no cover` is needed; no script added yet).

- [ ] **Step 6: Commit**

```bash
git add src/grd_generator/reflector/synth_afr.py src/grd_generator/reflector/__init__.py src/grd_generator/tests/test_synth_afr.py
git commit -m "feat(reflector): synthesize_reflector_fields assembly + hex feed lattice"
```

---

### Task 8: Service-zone overlay in `plot.py`

**Files:**
- Modify: `src/grd_generator/plot.py`
- Test: `src/grd_generator/tests/test_plot_contours.py` (add) or new `test_plot_zone.py`

**Interfaces:**
- Consumes: `ServiceZone` (Task 2), matplotlib `Axes`.
- Produces:
  - `draw_service_zone_uv(ax, zone, *, color="w", lw=1.5) -> None` — plots the zone boundary circle on a (u,v) axes using `zone.boundary_uv()`.

- [ ] **Step 1: Write the failing test**

```python
# src/grd_generator/tests/test_plot_zone.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grd_generator.reflector.zone import ServiceZone
from grd_generator.plot import draw_service_zone_uv


def test_draw_service_zone_adds_line():
    fig, ax = plt.subplots()
    before = len(ax.lines)
    draw_service_zone_uv(ax, ServiceZone(radius_deg=8.0))
    assert len(ax.lines) == before + 1
    plt.close(fig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/grd_generator/tests/test_plot_zone.py -q`
Expected: FAIL — `ImportError: cannot import name 'draw_service_zone_uv'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/grd_generator/plot.py
from grd_generator.reflector.zone import ServiceZone


def draw_service_zone_uv(ax, zone: ServiceZone, *, color: str = "w", lw: float = 1.5) -> None:
    """Trace le contour du disque de zone de service sur des axes (u, v)."""
    u, v = zone.boundary_uv()
    ax.plot(u, v, color=color, lw=lw, ls="--", label="zone de service")
```

(If `plot.py` has a typed signature convention for `ax`, match it — e.g. `matplotlib.axes.Axes`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/grd_generator/tests/test_plot_zone.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/grd_generator/plot.py src/grd_generator/tests/test_plot_zone.py
git commit -m "feat(plot): service-zone overlay on (u,v) displays"
```

---

### Task 9: AFR generation script (`reflector/generate_afr.py`)

**Files:**
- Create: `src/grd_generator/reflector/generate_afr.py`
- Modify: `pyproject.toml` (coverage omit + optional console-script `grd-generate-afr`)
- Test: `src/grd_generator/tests/test_generate_afr.py`

**Interfaces:**
- Consumes: `ReflectorSpec`, `FeedSpec`, `ServiceZone`, `synthesize_reflector_fields`, `hex_feed_positions`, `UVGrid`.
- Produces:
  - `build_reflector_reference(spec, feeds, zone, grid) -> dict` — returns the dict of arrays/metadata to save (pure, testable): keys `u_axis, v_axis, fields (N,n_v,n_u complex), cross (N,n_v,n_u complex), feed_positions_m, diameter_m, focal_length_m, offset_clearance_m, freq_hz, q, zone_radius_deg`.
  - `write_reflector_reference(path, ...) -> None` and `main()` — IO/script, `# pragma: no cover`.

- [ ] **Step 1: Write the failing test**

```python
# src/grd_generator/tests/test_generate_afr.py
import numpy as np
from grd_generator.schemas import UVGrid
from grd_generator.reflector import ReflectorSpec, FeedSpec, ServiceZone, hex_feed_positions
from grd_generator.reflector.generate_afr import build_reflector_reference


def test_build_reflector_reference_shapes():
    grid = UVGrid(u_min=-14, u_max=14, v_min=-14, v_max=14, n_u=41, n_v=41)
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.0, freq_hz=20e9)
    feeds = FeedSpec(positions_m=hex_feed_positions(0.03, 7, 0.2), q=2.0)
    zone = ServiceZone(radius_deg=8.0)
    ref = build_reflector_reference(spec, feeds, zone, grid)
    assert ref["fields"].shape == (7, 41, 41)
    assert ref["cross"].shape == (7, 41, 41)
    assert ref["zone_radius_deg"] == 8.0
    assert ref["freq_hz"] == 20e9
    assert np.iscomplexobj(ref["fields"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/grd_generator/tests/test_generate_afr.py -q`
Expected: FAIL — `ModuleNotFoundError: grd_generator.reflector.generate_afr`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/grd_generator/reflector/generate_afr.py
"""Générateur offline AFR → .npz (script). Couverture exclue comme generate.py."""

from pathlib import Path
from typing import Any

import numpy as np

from grd_generator.logger import configure_logging, logger
from grd_generator.schemas import UVGrid
from grd_generator.reflector import (
    FeedSpec,
    ReflectorSpec,
    ServiceZone,
    hex_feed_positions,
    synthesize_reflector_fields,
)

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
    grid = UVGrid(u_min=-14, u_max=14, v_min=-14, v_max=14, n_u=161, n_v=161)
    spec = ReflectorSpec(diameter_m=2.0, focal_length_m=2.4, freq_hz=20e9)
    feeds = FeedSpec(positions_m=hex_feed_positions(0.03, 80, 0.3), q=2.0)
    zone = ServiceZone(radius_deg=8.0)
    write_reflector_reference(DEFAULT_OUTPUT, build_reflector_reference(spec, feeds, zone, grid))


if __name__ == "__main__":  # pragma: no cover
    main()
```

Update `pyproject.toml` coverage omit (match the existing `generate.py` entry):

```toml
[tool.coverage.run]
omit = [
    "src/grd_generator/generate.py",
    "src/grd_generator/reflector/generate_afr.py",
]
```

(If a `[project.scripts]` table exists, add `grd-generate-afr = "grd_generator.reflector.generate_afr:main"` alongside `grd-generate`.)

- [ ] **Step 4: Run test + full suite**

Run: `uv run pytest -q`
Expected: PASS, coverage 100% (the `# pragma: no cover` IO/`main` lines excluded; `build_reflector_reference` covered by the test).

- [ ] **Step 5: Commit**

```bash
git add src/grd_generator/reflector/generate_afr.py src/grd_generator/tests/test_generate_afr.py pyproject.toml
git commit -m "feat(reflector): AFR generation script + reference builder"
```

---

### Task 10: GUI integration — reflector params, frequency, zone overlay

**Files:**
- Modify: `src/grd_generator/gui.py`
- Test: `src/grd_generator/tests/test_gui.py` (add a headless `build_result`-style test)

**Interfaces:**
- Consumes: `ReflectorSpec`, `FeedSpec`, `ServiceZone`, `synthesize_reflector_fields`, `hex_feed_positions`, `draw_service_zone_uv`.
- Produces:
  - A pure helper `build_reflector_result(*, diameter_m, f_over_d, offset_clearance_m, freq_ghz, q, pitch_m, n_feeds, zone_radius_deg, grid) -> ReflectorResult` (a small dataclass holding `co_fields`, `cross_fields`, `zone`, `grid`) — pure, no Qt, fully testable.
  - GUI wiring: spin boxes for diameter, F/D, offset, **frequency (GHz)**, q, feed pitch, N feeds, **zone radius**; the existing element selector/phase slider reused; `draw_service_zone_uv` called on every (u,v) panel after redraw.

- [ ] **Step 1: Write the failing test**

```python
# add to src/grd_generator/tests/test_gui.py
from grd_generator.gui import build_reflector_result
from grd_generator.schemas import UVGrid


def test_build_reflector_result_pure():
    grid = UVGrid(u_min=-14, u_max=14, v_min=-14, v_max=14, n_u=41, n_v=41)
    res = build_reflector_result(
        diameter_m=2.0, f_over_d=1.2, offset_clearance_m=0.0, freq_ghz=20.0,
        q=2.0, pitch_m=0.03, n_feeds=7, zone_radius_deg=8.0, grid=grid,
    )
    assert len(res.co_fields) == 7
    assert res.zone.radius_deg == 8.0
    assert res.co_fields[0].shape == (41, 41)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/grd_generator/tests/test_gui.py::test_build_reflector_result_pure -q`
Expected: FAIL — `ImportError: cannot import name 'build_reflector_result'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/grd_generator/gui.py (near the existing build_result)
from dataclasses import dataclass

from grd_generator.schemas import UVGrid
from grd_generator.reflector import (
    FeedSpec, ReflectorSpec, ServiceZone, hex_feed_positions, synthesize_reflector_fields,
)


@dataclass
class ReflectorResult:
    co_fields: list
    cross_fields: list
    zone: ServiceZone
    grid: UVGrid


def build_reflector_result(
    *,
    diameter_m: float,
    f_over_d: float,
    offset_clearance_m: float,
    freq_ghz: float,
    q: float,
    pitch_m: float,
    n_feeds: int,
    zone_radius_deg: float,
    grid: UVGrid,
) -> ReflectorResult:
    """Construit le résultat AFR (pur, sans Qt) depuis les paramètres de l'IHM."""
    spec = ReflectorSpec(
        diameter_m=diameter_m,
        focal_length_m=f_over_d * diameter_m,
        offset_clearance_m=offset_clearance_m,
        freq_hz=freq_ghz * 1e9,
    )
    feeds = FeedSpec(positions_m=hex_feed_positions(pitch_m, n_feeds, focal_radius_m=10.0 * pitch_m), q=q)
    co, cross = synthesize_reflector_fields(spec, feeds, grid)
    return ReflectorResult(co_fields=co, cross_fields=cross,
                           zone=ServiceZone(radius_deg=zone_radius_deg), grid=grid)
```

Then wire the Qt widget: add the spin boxes (diameter, F/D, offset, frequency GHz, q, pitch, N feeds, zone radius) in the parameters form, call `build_reflector_result` from the redraw path for the reflector mode, and after each (u,v) raster/phase/envelope redraw call `draw_service_zone_uv(ax, result.zone)`. Reuse the existing element selector to index `co_fields`/`cross_fields` and the existing envelope (`combined_max_directivity_dbi`) over `co_fields`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/grd_generator/tests/test_gui.py::test_build_reflector_result_pure -q`
Expected: PASS.

- [ ] **Step 5: Manual GUI smoke check**

Run: `uv run python -m grd_generator.examples.example_gui` (or the GUI entrypoint). Confirm: frequency & reflector params present; zone circle drawn on every (u,v) panel; element slider scans feeds; envelope shows zone coverage visually.

- [ ] **Step 6: Run full suite + commit**

```bash
uv run pytest -q   # expect PASS + 100% coverage
git add src/grd_generator/gui.py src/grd_generator/tests/test_gui.py
git commit -m "feat(gui): reflector params, frequency input, service-zone overlay"
```

---

## Self-Review

**Spec coverage:**
- Real array-fed reflector model → Tasks 3–7.
- Aperture-field + FFT far-field → Tasks 5–6.
- Offset paraboloid geometry → Tasks 3–4 (`aperture_grid`, `aperture_pol_vectors`).
- Separate `reflector/` module → all tasks.
- cos^q feed → Task 3.
- Vectorial co/cross-pol (Ludwig-3) → Tasks 4 & 6.
- Single frequency input → Task 1 (`wavelength_m`), Task 10 (GHz GUI input).
- Service zone (disk, [6,14]°, displayed everywhere) → Task 2 (model), Task 8 (plot overlay), Task 10 (GUI).
- Default boresight = nadir, feeds scan to tile zone → Task 7 (`center_uv=(0,0)`, displaced-feed beam-scan test).
- Coverage validated visually, no threshold → no calibration module added; Task 10 displays envelope + zone.
- `.npz` carries cross-pol + reflector metadata → Task 9.
- 100% coverage gate, scripts excluded → Tasks 7/8/9 run full suite; Task 9 uses `# pragma: no cover` + omit list.
- `.grd` export, PO, multi-frequency → out of scope (spec notes); no tasks, intentionally.

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Every code step shows complete code. The only deferred detail is the Qt-widget *wiring* prose in Task 10 Step 3 — the pure, testable `build_reflector_result` is fully specified; the widget layout follows the existing `PatternStudio` pattern in `gui.py` (acceptable: it mirrors established code and is verified by the manual smoke check).

**Type consistency:** `ReflectorSpec`/`FeedSpec`/`ServiceZone` names and fields are stable across Tasks 1–10. `synthesize_reflector_fields` returns `(co_fields, cross_fields)` consistently consumed by Tasks 9–10. `far_field_fft` returns `(F, L, M)`; `ludwig3_co_cross(Fx, Fy, L, M)`; `resample_to_uvgrid(field, lin_axis, grid)` — call sites in Task 7 match. `draw_service_zone_uv(ax, zone)` consistent in Tasks 8 & 10.

**Open implementation notes (validate during execution, not blockers):**
- The Airy oracle tolerance (0.3 dB) in Task 5 depends on `n_aperture`/`pad_factor`; raise resolution if it fails.
- `beam_deviation_factor` default 1.0 (paraxial). Refine via research R2 if scan position needs calibration — adjust only `ReflectorSpec` default, no API change.
- Cross-pol sign conventions (Task 4/6) are validated qualitatively; absolute polarity is not asserted.
