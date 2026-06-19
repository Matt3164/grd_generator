# grd_generator — générateur de patterns autonome

**Date :** 2026-06-19
**Statut :** validé, prêt pour plan d'implémentation

## Objectif

Repo Python **autonome** dont l'unique rôle est de **générer le jeu de référence
de diagrammes de rayonnement d'antenne** (fichier `.npz`), extrait de la feature
`pattern` de `satscope` (`~/Documents/satscope-weekend-recovered`).

Le générateur :
1. dérive une géométrie de repère satellite (nadir) à partir d'un scénario
   (forme/visée sur la Terre) ;
2. place N=80 centres d'éléments sur une maille hexagonale dans la zone de
   service ;
3. optimise la largeur σ par bissection sous la contrainte « enveloppe de
   directivité ≥ 44 dBi dans la zone » ;
4. synthétise N champs complexes (modes `gaussian` ou `airy`) ;
5. écrit grille + champs + métadonnées dans un `.npz`.

**Hors périmètre** (présents dans satscope, volontairement exclus) : analyse,
plot/rendu matplotlib, combinaison/beamforming (`service.py`), vérification
(`verify.py`), benchmark, GUI, ingestion, le reste de `satview`.

## Décisions de cadrage (arbitrées avec l'utilisateur)

| Sujet | Décision |
|-------|----------|
| Périmètre | **Génération seule, sans verify** : on retire le garde-fou post-écriture `verify_set`/`load_array`. On garde `check_power_conservation` (warning, dans `synth`). |
| Scénario | **Dataclass autonome minimale** — « une forme/polygone sur la Terre ». Plus de chargement JSON ni de machinerie `satview` (stress/cities/border/beams). `FRANCE` est un littéral en code. |
| Namespace | Package `grd_generator` (`src/grd_generator/...`). Tous les imports `satscope.*` réécrits. |
| Settings `.claude` | academic-pcs n'a **pas** de bloc `permissions`, seulement des hooks projet-spécifiques. On n'importe **aucun hook** ; on écrit un `settings.json` avec un bloc `permissions` (autoriser python/pytest/uv/ruff/mypy). |
| Couverture tests | **Gate strict `--cov-fail-under=100`** comme satscope. `generate.py` exclu de couverture (script d'écriture, comme `generate_reference.py` dans la source). |
| Modes de génération | **`gaussian` + `airy`** conservés (registre enfichable, Bessel J₁ inclus). |

## Architecture

```
grd_generator/
├── pyproject.toml          # name=grd_generator ; deps runtime: numpy, pydantic, loguru
├── README.md
├── .gitignore              # inclut debug_export/, data/processed/, .venv, caches
├── .claude/
│   └── settings.json       # bloc permissions, pas de hooks
├── docs/
│   └── superpowers/specs/  # ce document
└── src/grd_generator/
    ├── __init__.py         # API publique
    ├── logger.py           # configure_logging() + logger (loguru JSON)
    ├── schemas.py          # UVGrid, GaussianSpec, alias ComplexField/DirectivityMap
    ├── geometry.py         # projection ECEF sol→(u,v), limbe, lancer de rayon
    ├── scenario.py         # Scenario (dataclass) + FRANCE + SatFrameGeometry + sat_frame_geometry
    ├── synth.py            # générateurs de champ, Bessel J₁, power checks, hex_centers_uv, optimize_sigma
    ├── generate.py         # auto_spacing, write_reference, generate_reference, main()
    ├── examples/
    │   └── example_generation.py
    └── tests/
        ├── test_schemas.py
        ├── test_geometry.py
        ├── test_scenario.py
        ├── test_synth.py
        ├── test_generate.py
        └── test_example_generation.py
```

### Modules — rôle, source, adaptations

**`logger.py`** — copie de `satscope/shared/logger.py` (loguru JSON, idempotent).
Aucune adaptation hors namespace.

**`schemas.py`** — extrait de `pattern/schemas.py`, **élagué** aux seuls types
consommés par la génération : `UVGrid`, `GaussianSpec`, alias `ComplexField`,
`DirectivityMap`. On retire `AntennaPattern`, `AntennaArray`, `BeamRegion`,
`BeamWeights`, `ComplexMatrix` (servaient à service/plot/verify).

**`geometry.py`** — copie de `satscope/satview/geometry.py` (déjà autonome,
numpy seul). On garde tout (le module est cohérent et entièrement utile/testable) :
`ground_ecef`, `satellite_ecef`, `boresight_frame`, `project_to_angular`,
`project_point`, `project_to_angular_masked`, `earth_limb_angle_deg`,
`angular_to_direction`, `earth_intersection_latlon`.

**`scenario.py`** — fusionne `pattern/scenario.py` et la partie utile de
`satview/scenarios.py`. `Scenario` devient une dataclass `frozen` minimale :

```python
@dataclass(frozen=True)
class Scenario:
    name: str
    sat_lon_deg: float
    antenna_latlon: tuple[float, float]   # visée (lat, lon)
    zone_radius_deg: float = 6.0          # min réglementaire 6°
    limb_margin_deg: float = 0.2
```

`__post_init__` conserve la validation utile : `zone_radius_deg >= 6.0`.
`FRANCE = Scenario(name="france", sat_lon_deg=3.0, antenna_latlon=(46.6, 2.5),
zone_radius_deg=6.0, limb_margin_deg=0.2)`.
On garde `SatFrameGeometry` et `sat_frame_geometry()` à l'identique, sauf que
`antenna_latlon` n'est plus `Optional` (toujours requis ici) — la branche
`ValueError(antenna_latlon=None)` disparaît.

**`synth.py`** — copie quasi à l'identique (déjà self-contained : schemas +
logger). Contient les deux modes (`gaussian`, `airy` + `_bessel_j1`), les bornes
de puissance, `power_fraction`, `check_power_conservation`, `directivity_dbi_from_field`,
`combined_max_directivity_dbi`, `max_envelope`, `envelope_min_dbi`,
`hex_centers_uv`, `optimize_sigma`.

**`generate.py`** — adapté de `pattern/generation/generate_reference.py` :
- `TARGET_DBI = 44.0` constante locale (ex-`DEFAULT_COVERAGE_THRESHOLD_DBI`) ;
- **suppression** du bloc garde-fou final (`verify_set`/`load_array`) ;
- conserve `auto_spacing`, `write_reference`, `generate_reference`, `main()` ;
- conserve l'appel `check_power_conservation` (warning).
Paramètres inchangés : `N_ANTENNAS=80`, `PEAK_GAIN_DBI=47.0`,
`PHASE_SLOPE_RADIAL=3.0`, grille 161×161, σ∈[1e-3, 5.0],
sortie `data/processed/reference_array.npz`.

**`__init__.py`** — expose : `UVGrid`, `GaussianSpec`, les générateurs/optimiseur
de `synth`, `Scenario`/`FRANCE`/`sat_frame_geometry`/`SatFrameGeometry`,
`generate_reference`/`write_reference`.

**`examples/example_generation.py`** — script autonome (convention CLAUDE.md) :
`configure_logging()` puis `generate_reference()` vers un chemin temporaire,
log structuré début/fin. Exécutable seul et testé par `test_example_generation.py`.

## Flux de données

```
FRANCE (Scenario)
   │  sat_frame_geometry()           geometry.py (ECEF, limbe)
   ▼
SatFrameGeometry { sat_lon, zone_center, zone_radius, grid:UVGrid }
   │  auto_spacing + hex_centers_uv
   ▼
centres (list[(u,v)])
   │  optimize_sigma (bissection, envelope_min_dbi ≥ 44 dBi)
   ▼
(σ, specs:list[GaussianSpec])
   │  check_power_conservation (warning)
   │  field_generator(mode) → champs complexes
   ▼
write_reference → data/processed/reference_array.npz
   { grille, fields[N], antenna_ids, centers_uv, mode, sigmas, peaks_dbi, phase_slopes }
```

## Gestion d'erreurs

- `optimize_sigma` lève `ValueError` si la contrainte est infaisable à `sigma_hi`.
- `hex_centers_uv` lève `ValueError` si la maille loge moins de N centres.
- `sat_frame_geometry` lève `ValueError` si la zone ne tient pas sous le limbe.
- `field_generator`/`max_peak_dbi` lèvent `ValueError` sur mode inconnu.
- Dépassement de puissance (`check_power_conservation`) : **warning structuré**,
  pas d'exception (politique d'origine conservée).

## Tests (gate 100 %)

- `test_schemas.py` : `UVGrid.axes/meshgrid`, validation `GaussianSpec` (σ>0).
- `test_geometry.py` : `earth_limb_angle_deg` (~8.7°), aller-retour
  `project_point`/`earth_intersection_latlon`, masque `project_to_angular_masked`.
- `test_scenario.py` : `sat_frame_geometry(FRANCE)` (centre sous le limbe, grille
  cohérente), validation `zone_radius_deg < 6`, zone trop large → `ValueError`.
- `test_synth.py` : `gaussian_field`/`airy_field` (crête, symétrie), `_bessel_j1`
  vs valeurs connues, `optimize_sigma` (faisabilité, monotonie gaussienne),
  `hex_centers_uv` (compte, tri), `check_power_conservation` (warning).
- `test_generate.py` : `generate_reference` vers un `tmp_path`, relecture `.npz`
  (formes, clés, N=80), `auto_spacing`. (Module `generate` exclu de couverture
  via `# pragma: no cover` / config — comme la source — ou couvert par ce test.)
- `test_example_generation.py` : lance le script via `subprocess`, code retour 0.

## Dépendances

Runtime : `numpy>=2.0`, `pydantic>=2.7`, `loguru>=0.7`.
Dev : `pytest`, `pytest-cov`, `ruff`, `mypy`.
Pas de `matplotlib`/`pandas` (servaient à plot/service/debug_export, retirés).

## `.claude/settings.json`

Bloc `permissions` autorisant les commandes du flux dev (python, pytest, uv,
ruff, mypy, git). Aucun hook (les hooks academic-pcs sont projet-spécifiques).
</content>
</invoke>
