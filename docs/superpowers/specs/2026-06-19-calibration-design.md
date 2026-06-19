# Calibration du générateur depuis des données réelles

**Date :** 2026-06-19
**Statut :** validé, prêt pour plan d'implémentation

## Objectif

Boucler le round-trip `grd_analyzer → grd_generator`. À partir d'un `report.json`
produit par `grd_analyzer` (statistiques par élément d'un vrai set `.grd`),
générer un array de référence **calibré** dont les diagrammes reproduisent les
statistiques mesurées — au lieu des constantes figées de `generate_reference`
(N=80, peak 47 dBi, σ optimisé, phase 3.0 rad/°, lobes circulaires).

Le contrat est direct : `grd_analyzer` émet déjà un bloc `estimated_params`
(`mode, peak_gain_dbi, sigma, phase_slope_radial, spacing_deg`) qui mappe sur les
paramètres du générateur — les deux outils sont issus de `satscope`, mêmes
sémantiques (notamment : pour `airy`, `sigma` = rayon du premier null).

## Décisions de cadrage (arbitrées)

| Sujet | Décision |
|-------|----------|
| Périmètre | **Calibration + lobes elliptiques.** On ingère les distributions mesurées et on ajoute un générateur de champ elliptique. Pas de cross-pol (hors périmètre). |
| Variabilité | **Array dispersé.** Chaque élément reçoit des paramètres tirés des distributions mesurées (moyenne+écart-type par champ), **seed fixe** pour reproductibilité. |
| Données | **Copier les `report.json`** des jeux `0000`/`0001` dans `data/reference_reports/<id>/report.json` (versionnés, ~60 KB chacun ; PNG non copiés). Repo autonome. |
| Validation | **Diagnostic (warning)**, pas de test dur : on régénère, on re-mesure les stats et on logue un warning structuré si l'écart dépasse une tolérance. Robuste au bruit du tirage. |
| Écriture `.npz` | **Variante `write_calibrated`** dédiée aux arrays elliptiques. `write_reference` (circulaire) reste intact — zéro régression sur `generate_reference`. |
| Couverture | Gate **100 %** conservé. Les nouveaux scripts/CLI (`calibrate.py`) suivent la même règle d'omission que `generate.py` (offline, orchestration + écriture). |

## Architecture

Modules **additifs** ; le chemin `generate_reference` existant n'est pas modifié.

```
src/grd_generator/
├── schemas.py          (+ EllipticalSpec)
├── synth.py            (+ elliptical_field, + HALF_POWER_WIDTH_PER_SIGMA)
├── report_ingest.py    (NOUVEAU) report.json → MeasuredStats
├── calibrate.py        (NOUVEAU) MeasuredStats → array elliptique → .npz (+ CLI grd-calibrate)
├── examples/
│   └── example_calibration.py   (NOUVEAU)
└── tests/
    ├── test_report_ingest.py
    ├── test_elliptical.py       (extension de test_synth ou fichier dédié)
    └── test_calibrate.py
data/reference_reports/0000/report.json   (copie)
data/reference_reports/0001/report.json   (copie)
```

### `schemas.py` — `EllipticalSpec`

```python
class EllipticalSpec(BaseModel):
    center_uv: tuple[float, float]
    sigma_major: float = Field(..., gt=0)   # demi-largeur axe majeur (deg)
    sigma_minor: float = Field(..., gt=0)   # demi-largeur axe mineur (deg)
    orientation_deg: float = 0.0            # angle de l'axe majeur
    peak_gain_dbi: float
    phase_slope_radial: float = 0.0
```

Le `GaussianSpec` circulaire reste tel quel (utilisé par `generate_reference`).

### `synth.py` — générateur elliptique

`elliptical_field(spec: EllipticalSpec, grid: UVGrid, *, mode: str = "gaussian")`:
1. `du = gu - cu`, `dv = gv - cv`.
2. Rotation dans le repère propre par `-orientation_deg` :
   `du' = du·cosθ + dv·sinθ`, `dv' = −du·sinθ + dv·cosθ`.
3. Rayon elliptique `ρ = √((du'/σ_major)² + (dv'/σ_minor)²)`.
4. Amplitude selon le mode :
   - `gaussian` : `peak_lin · exp(−ρ²/2)`.
   - `airy` : `peak_lin · jinc(J1_FIRST_NULL · ρ)` (réutilise `_bessel_j1`).
   Réduction au cas circulaire si `σ_major == σ_minor` (identique aux générateurs
   existants à la convention de σ près).
5. Phase **radiale** inchangée : `phase_slope_radial · √(du²+dv²)`.
6. `E = amplitude · e^{jφ}`, complex128.

Constante partagée (même valeur que `grd_analyzer`) :
`HALF_POWER_WIDTH_PER_SIGMA = 2.0 * sqrt(ln 2)`.

### `report_ingest.py` — `MeasuredStats`

Lit un `report.json`. Produit :

```python
@dataclass(frozen=True)
class FieldDist:        # distribution d'un champ
    mean: float
    std: float

@dataclass(frozen=True)
class MeasuredStats:
    source: str                 # chemin/id du rapport
    mode: str                   # "gaussian" | "airy" (depuis estimated_params)
    n_elements: int
    spacing_deg: float
    peak_dbi: FieldDist
    lobe_width_major_deg: FieldDist
    lobe_width_minor_deg: FieldDist
    orientation_deg: FieldDist
    phase_slope_rad_per_deg: FieldDist
    first_null_radius_deg: FieldDist | None   # présent pour airy, None sinon
```

- `mean`/`std` calculés sur la liste `elements` (population std). Les éléments
  `truncated` sont exclus.
- Champs `null` tolérés : si un champ manque pour tous les éléments, sa `FieldDist`
  vaut `None` (et la calibration choisit une voie de repli — voir conversion).
- `mode`, `n_elements`, `spacing_deg` lus depuis `estimated_params` / top-level.
- Erreurs explicites (`ValueError`) si `elements` absent/vide ou `mode` inconnu.

### `calibrate.py` — génération calibrée

`calibrate(stats: MeasuredStats, *, seed: int = 0) -> tuple[list[EllipticalSpec], UVGrid]`
puis écriture.

1. **Centres** : maille hexagonale de `n_elements` points à l'espacement
   `spacing_deg`, centrée sur l'origine. Réutilise `hex_centers_uv` avec un
   `zone_radius` calculé pour loger `n_elements` (inverse de `auto_spacing`).
2. **Tirage par élément** (générateur `numpy` à `seed` fixe) : pour chaque champ,
   `value = clip(normal(mean, std), bornes physiques)`. Bornes : peak libre ;
   largeurs/σ strictement > 0 ; orientation repliée dans (−90, 90].
3. **Conversion largeur → σ** via une fonction **pure dans `synth.py`** (couverte
   par les tests), `widths_to_sigmas(mode, width_major, width_minor, first_null)`:
   - `gaussian` : `σ_axis = width_axis / HALF_POWER_WIDTH_PER_SIGMA`.
   - `airy` : on préserve l'ellipticité autour du premier null mesuré —
     `e = width_major / width_minor` ; `σ_major = first_null · √e`,
     `σ_minor = first_null / √e`. (Si `first_null` absent/None, repli sur la
     conversion gaussian.)
4. **Grille** : bbox des centres élargie d'une marge `k·max(σ_major)` (k≈6),
   résolution `n_u = n_v = 161` (réglable).
5. **Synthèse + écriture** via `write_calibrated`.

`write_calibrated(output_path, grid, specs, *, mode) -> Path` : `np.savez` avec
les clés de `write_reference` (grille, `fields`, `antenna_ids`, `centers_uv`,
`mode`, `peaks_dbi`, `phase_slopes`) **plus** `sigmas_major`, `sigmas_minor`,
`orientations_deg`. L'array reste relisible et ré-analysable.

CLI `grd-calibrate --report <report.json|dir> --out <npz> [--seed N]`.

### Boucle de validation (diagnostic)

Après écriture, fonction `validate_against(stats, specs)` qui recalcule les stats
agrégées des specs générées (peak moyen, ellipticité moyenne, orientation,
phase_slope) et **logue un warning structuré** par champ dont l'écart à la
moyenne mesurée dépasse une tolérance relative (par défaut 15 %). Aucune
exception : c'est un diagnostic, pas un gate.

## Flux de données

```
report.json (grd_analyzer)
   │ read_report()                      report_ingest.py
   ▼
MeasuredStats { mode, n, spacing, FieldDist par champ }
   │ calibrate(seed)                    calibrate.py
   │   ├─ hex_centers_uv (spacing mesuré)
   │   ├─ tirage N(mean,std) par élément  → EllipticalSpec[]
   │   └─ conversion largeur→σ (par mode)
   ▼
list[EllipticalSpec] + UVGrid
   │ elliptical_field(mode)             synth.py
   │ write_calibrated                   → calibrated_<id>.npz
   │ validate_against (warn)
   ▼
data/processed/calibrated_<id>.npz
```

## Gestion d'erreurs

- `read_report` : `ValueError` si `elements` vide/absent, `mode` ∉ {gaussian, airy},
  `n_elements` ≤ 0 ou `spacing_deg` ≤ 0.
- `calibrate` : `ValueError` si la maille ne loge pas `n_elements` (propagé par
  `hex_centers_uv`).
- Tirage : σ et largeurs reclippés à un plancher > 0 ; pas d'exception sur un
  std nul (distribution dégénérée → toutes valeurs = mean).
- Validation : warnings uniquement.

## Tests (gate 100 %)

- `test_report_ingest.py` : parsing des deux rapports réels copiés (mean/std
  cohérents, `mode` correct, `first_null` None pour le gaussian) ; champs `null`
  tolérés ; erreurs (`elements` vide, mode inconnu).
- `test_elliptical.py` : `elliptical_field` — pic au centre ; un lobe
  `σ_major > σ_minor` orienté à θ produit une coupe plus large le long de θ
  (ellipticité et orientation retrouvées) ; réduction au circulaire quand
  `σ_major == σ_minor` ; mode `airy` présente des nulls.
- `test_calibrate.py` : `calibrate` rend `n_elements` specs ; **reproductibilité**
  (même seed → mêmes specs) ; `write_calibrated` → `.npz` relisible avec les clés
  elliptiques ; `validate_against` logue un warning quand on lui passe des specs
  hors tolérance et rien quand elles sont conformes.
- `test_example_calibration.py` : lancement du script d'exemple, code retour 0.

`calibrate.py` est **exclu de couverture** (script offline + orchestration +
écriture + CLI + `validate_against`, comme `generate.py`). Les logiques pures
testables — `EllipticalSpec`, `elliptical_field`, `widths_to_sigmas`,
`read_report`/`MeasuredStats` — vivent dans `schemas.py`/`synth.py`/
`report_ingest.py` et restent **couvertes à 100 %**.

## Dépendances

Inchangées : `numpy`, `pydantic`, `loguru`. Pas de nouvelle dépendance.
```
</content>
