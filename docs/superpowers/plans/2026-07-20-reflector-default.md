# Reflector par défaut — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** faire de la simulation réflecteur AFR l'unique mode du projet (GUI et génération), supprimer le pipeline pattern, tracer ses valeurs de référence.

**Architecture :** suppression de modules entiers + élagage des modules partagés (`synth.py`, `plot.py`, `schemas.py`) aux seules fonctions atteignables depuis le code réflecteur. Aucune nouvelle fonctionnalité. Spec : `docs/superpowers/specs/2026-07-20-reflector-default-design.md`.

**Tech stack :** Python 3.12, uv, pytest (couverture 100 % sur les modules non-omis), ruff, mypy strict, PyQt6/matplotlib (extra viz).

## Global Constraints

- Après CHAQUE tâche : `uv run pytest` passe (couverture 100 % maintenue), `uv run ruff check src` et `uv run mypy src` passent. Un commit par tâche.
- Répondre en français dans les messages de commit (style existant : `feat(...)`, `refactor(...)`, `chore(...)`).
- Ne PAS toucher : `src/grd_generator/reflector/` (tout le package), `logger.py`, `examples/example_reflector_roundtrip.py`, `docs/pattern_reference_values.json`, `docs/superpowers/`.
- Ne PAS supprimer `data/processed/*` du disque (gitignoré, données locales de l'utilisateur).
- `synth.py` : supprimer des fonctions entières, ne pas remanier le style interne de celles qui restent (extraction verbatim satscope).
- Travailler dans le worktree courant (branche `feat/reflector-default`), ne pas pousser ni créer de PR (fait par l'orchestrateur).

---

### Task 1 : GUI reflector-only + entry points

**Files:**
- Modify: `src/grd_generator/gui.py`
- Modify: `pyproject.toml` (sections `[project]` description, `[project.scripts]`)
- Delete: `src/grd_generator/examples/example_gui.py`, `src/grd_generator/tests/test_example_gui.py`
- Modify: `src/grd_generator/tests/test_gui.py`

**Interfaces:**
- Produces : `grd_generator.gui.main` lance ReflectorStudio sans argument ; `build_reflector_result` et `ReflectorStudio` restent exportés à l'identique (consommés par `example_reflector_roundtrip.py` et `test_gui.py`).

- [ ] **Step 1 : supprimer PatternStudio de `gui.py`**

Supprimer la classe `PatternStudio` et tout ce qui n'est utilisé que par elle : constante `REPORTS_DIR`, helpers earth/rapports, et les imports devenus inutiles (`calibrate`, `geometry`, `report_ingest`, `scenario`, `elliptical_field`, `EllipticalSpec`, et les helpers plot earth : `_draw_earth_envelope`, `draw_earth_envelope_contours`, `draw_earth_pattern_footprints`, `draw_zone_and_antenna`). Vérifier par grep qu'aucun symbole supprimé n'est utilisé par ReflectorStudio :

```bash
grep -n "PatternStudio\|REPORTS_DIR\|earth\|calibrate\|scenario\|report_ingest\|elliptical" src/grd_generator/gui.py
```

- [ ] **Step 2 : réécrire `main()` sans `--mode`**

```python
def main() -> None:  # pragma: no cover
    configure_logging()
    app = QApplication.instance() or QApplication(sys.argv)
    studio = ReflectorStudio()
    studio.show()
    app.exec()
```

Retirer `import argparse` s'il n'est plus utilisé.

- [ ] **Step 3 : mettre à jour `pyproject.toml`**

```toml
[project]
name = "grd_generator"
version = "0.1.0"
description = "Générateur autonome de diagrammes de rayonnement par simulation de réflecteur alimenté par réseau (AFR, .npz)."
```

```toml
[project.scripts]
grd-generate = "grd_generator.reflector.generate_afr:main"
grd-gui = "grd_generator.gui:main"
```

(Les entry points `grd-calibrate`, `grd-plot`, `grd-generate-afr` disparaissent.)

- [ ] **Step 4 : purger `test_gui.py` et l'exemple GUI**

Supprimer `examples/example_gui.py` et `tests/test_example_gui.py` (ils ciblent PatternStudio). Dans `test_gui.py`, supprimer les tests de PatternStudio, garder tous les tests ReflectorStudio/`build_reflector_result` ; adapter les imports.

- [ ] **Step 5 : valider**

```bash
uv run pytest && uv run ruff check src && uv run mypy src
uv run grd-gui --help >/dev/null 2>&1 || true  # option supprimée : vérifier que l'entry point résout
uv run python -c "from grd_generator.gui import main, ReflectorStudio, build_reflector_result"
```

Attendu : suite verte, couverture 100 %, imports OK.

- [ ] **Step 6 : commit**

```bash
git add -A && git commit -m "feat(gui): ReflectorStudio unique — suppression de PatternStudio et du sélecteur --mode"
```

---

### Task 2 : suppression du pipeline calibration

**Files:**
- Delete: `src/grd_generator/calibrate.py`, `src/grd_generator/report_ingest.py`
- Delete: `src/grd_generator/examples/example_calibration.py`
- Delete: `src/grd_generator/tests/test_calibrate.py`, `src/grd_generator/tests/test_report_ingest.py`, `src/grd_generator/tests/test_example_calibration.py`
- Delete: `data/reference_reports/` (après vérification)
- Modify: `pyproject.toml` (retirer `calibrate.py` de `[tool.coverage.run] omit`)

**Interfaces:**
- Consumes : Task 1 a retiré les derniers usages GUI de `calibrate`/`report_ingest`.

- [ ] **Step 1 : vérifier l'absence d'usages restants**

```bash
grep -rn "calibrate\|report_ingest\|reference_reports" src/ pyproject.toml README.md --include="*.py" --include="*.toml" --include="*.md"
```

Attendu : seuls les fichiers à supprimer (et le README, traité en Task 5) apparaissent. Si `example_reflector_roundtrip.py` référence `data/reference_reports/`, NE PAS supprimer ce dossier et le signaler dans le rapport final.

- [ ] **Step 2 : supprimer les fichiers listés + l'entrée omit**

```bash
git rm src/grd_generator/calibrate.py src/grd_generator/report_ingest.py \
  src/grd_generator/examples/example_calibration.py \
  src/grd_generator/tests/test_calibrate.py src/grd_generator/tests/test_report_ingest.py \
  src/grd_generator/tests/test_example_calibration.py
git rm -r data/reference_reports   # seulement si Step 1 n'a trouvé aucun usage restant
```

Dans `pyproject.toml`, retirer la ligne `"*/grd_generator/calibrate.py",` du bloc `omit`.

- [ ] **Step 3 : valider puis committer**

```bash
uv run pytest && uv run ruff check src && uv run mypy src
git add -A && git commit -m "refactor: suppression du pipeline de calibration pattern (calibrate, report_ingest)"
```

---

### Task 3 : suppression generate/scenario/geometry + élagage plot.py

**Files:**
- Delete: `src/grd_generator/generate.py`, `src/grd_generator/scenario.py`, `src/grd_generator/geometry.py`
- Delete: `src/grd_generator/examples/example_generation.py`, `src/grd_generator/examples/example_plot.py`
- Delete: `src/grd_generator/tests/test_generate.py`, `test_scenario.py`, `test_geometry.py`, `test_example_generation.py`, `test_example_plot.py`, `test_plot_contours.py`
- Modify: `src/grd_generator/plot.py`, `src/grd_generator/tests/test_plot_zone.py`
- Delete: `data/ne_50m_land.geojson`
- Modify: `pyproject.toml` (omit : retirer `generate.py`)

**Interfaces:**
- Produces : `plot.py` réduit exporte exactement `envelope_max_dbi`, `_draw_uv_map`, `_draw_phase_map`, `draw_service_zone_uv` (+ leurs dépendances privées), signatures inchangées — consommés par `gui.py`.

- [ ] **Step 1 : élaguer `plot.py`**

Garder uniquement : `envelope_max_dbi`, `_draw_uv_map`, `_draw_phase_map`, `draw_service_zone_uv`, leurs imports et éventuelles dépendances privées. Supprimer tout le reste (`main`, `render`, `load_array`, `coastline_rings`, `element_centers`, `element_peaks_dbi`, `_draw_centers_uv`, `_project`, `_draw_coastlines`, `_draw_earth_envelope`, `_draw_earth_centers`, `draw_zone_and_antenna`, `draw_earth_pattern_footprints`, `draw_earth_envelope_contours`) et l'import `geometry`. Vérifier ensuite :

```bash
grep -rn "from grd_generator.plot import\|from grd_generator import plot" src/
```

Attendu : seuls `gui.py` et `tests/test_plot_zone.py` importent `plot`, uniquement des symboles conservés.

- [ ] **Step 2 : supprimer modules, exemples, tests et données**

```bash
git rm src/grd_generator/generate.py src/grd_generator/scenario.py src/grd_generator/geometry.py \
  src/grd_generator/examples/example_generation.py src/grd_generator/examples/example_plot.py \
  src/grd_generator/tests/test_generate.py src/grd_generator/tests/test_scenario.py \
  src/grd_generator/tests/test_geometry.py src/grd_generator/tests/test_example_generation.py \
  src/grd_generator/tests/test_example_plot.py src/grd_generator/tests/test_plot_contours.py \
  data/ne_50m_land.geojson
```

Dans `pyproject.toml`, retirer `"*/grd_generator/generate.py",` du bloc `omit`. Dans `test_plot_zone.py`, garder les tests des symboles conservés, supprimer ceux des symboles disparus.

- [ ] **Step 3 : valider puis committer**

```bash
uv run pytest && uv run ruff check src && uv run mypy src
git add -A && git commit -m "refactor: suppression du générateur pattern et du rendu earth (generate, scenario, geometry, plot élagué)"
```

---

### Task 4 : élagage synth.py et schemas.py

**Files:**
- Modify: `src/grd_generator/synth.py`, `src/grd_generator/schemas.py`
- Modify: `src/grd_generator/tests/test_synth.py`, `src/grd_generator/tests/test_schemas.py`

**Interfaces:**
- Produces : `synth.py` exporte exactement `hex_centers_uv` et `combined_max_directivity_dbi` (+ dépendances privées, style interne intact) ; `schemas.py` garde `UVGrid` et `ComplexField` (+ tout type encore réellement importé).

- [ ] **Step 1 : établir la liste exacte des symboles utilisés**

```bash
grep -rn "from grd_generator.synth import\|from grd_generator.schemas import" src/ | grep -v tests/
```

Garder dans `synth.py`/`schemas.py` exactement ces symboles plus leurs dépendances internes (fermeture transitive). Supprimer le reste (`elliptical_field`, `field_generator`, `optimize_sigma`, `check_power_conservation`, `pattern_envelope_min_in_zone`, `GaussianSpec`, `EllipticalSpec`… selon le grep). Couverture 100 % exigée : ces deux modules ne sont PAS dans `omit`, toute fonction gardée doit rester testée.

- [ ] **Step 2 : adapter `test_synth.py` et `test_schemas.py`**

Supprimer les tests des symboles disparus, garder intacts ceux des symboles conservés.

- [ ] **Step 3 : valider puis committer**

```bash
uv run pytest && uv run ruff check src && uv run mypy src
git add -A && git commit -m "refactor(synth,schemas): élagage aux seuls types et fonctions du chemin réflecteur"
```

---

### Task 5 : README, docs, vérification de bout en bout

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml` (relire le bloc `omit` : il ne doit plus rester que gui.py, plot.py, generate_afr.py, examples, tests)

**Interfaces:**
- Consumes : l'état final des tasks 1-4.

- [ ] **Step 1 : réécrire le README**

```markdown
# grd_generator

Générateur autonome de diagrammes de rayonnement d'antenne satellite par
simulation physique d'un réflecteur alimenté par réseau (AFR) — fichier `.npz`.

## Usage

```bash
uv sync
uv run grd-generate            # écrit data/processed/reflector_array.npz
uv run grd-gui                 # ReflectorStudio (GUI interactive)
```

## Boucle round-trip

Génère un AFR, l'exporte en `.grd`, l'analyse avec le projet frère
`grd_analyzer` et compare le rapport obtenu à un rapport de référence
(nécessite `grd_analyzer` en sibling du repo, ou `$GRD_ANALYZER_DIR`) :

```bash
uv run python src/grd_generator/examples/example_reflector_roundtrip.py
```

## Historique

L'ancien pipeline « pattern » (jeu de référence gaussien calibré sur rapports
mesurés) a été retiré ; ses valeurs de référence sont tracées dans
`docs/pattern_reference_values.json`. `synth.py` est un extrait du projet
amont `satscope`, réduit aux fonctions utilisées — ne pas remanier le style
interne des fonctions conservées.
```

Adapter la section round-trip si le Step 1 de la Task 2 a conservé `data/reference_reports/`.

- [ ] **Step 2 : validation de bout en bout**

```bash
uv run pytest && uv run ruff check src && uv run mypy src
uv run grd-generate            # doit écrire data/processed/reflector_array.npz et logger reflector_reference_written
uv run python -c "from grd_generator.gui import main"
grep -rn "PatternStudio\|calibrate\|scenario\|geometry\|report_ingest" src/ README.md pyproject.toml || echo "OK: aucun reliquat"
```

- [ ] **Step 3 : commit**

```bash
git add -A && git commit -m "docs: README recentré sur la simulation réflecteur + trace des valeurs pattern"
```
