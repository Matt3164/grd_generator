# Interface interactive (GUI) de calibration & visualisation

**Date :** 2026-06-22
**Statut :** validé, prêt pour plan d'implémentation

## Objectif

Une application interactive (PyQt6 + matplotlib embarqué) pour piloter la
génération calibrée depuis une **zone-disque** et un **rapport**, et visualiser le
résultat en direct : barycentres, directivité et phase d'un pattern (n° réglable),
et l'enveloppe max projetée sur Terre en **float** et en **isolignes (tous les
5 dB)**. Un bouton **Export** écrit le `.npz`.

## Décisions de cadrage (arbitrées)

| Sujet | Décision |
|-------|----------|
| Techno | **PyQt6** (déjà dans l'extra `viz`) + `FigureCanvasQTAgg`. |
| Zone | **Disque** : `antenna_latlon` (lat, lon), `sat_lon_deg`, `zone_radius_deg`. Pas de polygone. |
| Caractère d'élément | **Menu des rapports embarqués** (`data/reference_reports/*`). |
| Couverture | `gui.py` + `examples/*` **exclus de couverture** (effets de bord Qt/graphiques) ; gate 100 % préservé sur le reste. Test de lancement offscreen. |
| Intégration | Construit sur la branche `feat/gui` (redesign calibration zone-driven + visualisation fusionnés ; conflit `pyproject` résolu). |

## Architecture

```
src/grd_generator/
├── gui.py                  (NOUVEAU, exclu) widget Qt + CLI grd-gui
├── plot.py                 (réutilisé) helpers de rendu (directivité, phase, côtes, earth)
├── calibrate.py            (réutilisé) generate calibré zone-driven
└── examples/
    └── example_gui.py      (NOUVEAU, exclu) instancie l'app offscreen et ferme
```

**Séparation logique pure / widget.** La logique de calcul est isolée du Qt :

- `@dataclass CalibrationResult { grid, fields, centers_uv, peaks_dbi, mode, specs }`.
- `build_result(antenna_lat, antenna_lon, sat_lon, zone_radius_deg, report_path,
  n_elements, seed, coverage_margin_db, n_u, n_v) -> CalibrationResult` :
  construit `Scenario(name="custom", sat_lon_deg=…, antenna_latlon=(lat,lon),
  zone_radius_deg=…)`, `read_report`, `calibrate`, synthétise les champs
  (`elliptical_field`). Pure, sans Qt — directement testable.

**Helpers de rendu** (réutilisés/ajoutés dans `plot.py`, **module entièrement omis
de couverture** — effets de bord graphiques, exercés par le test de lancement) :
- directivité élément, phase élément, barycentres (u,v) — déjà dans `plot.py` ;
- enveloppe max sur Terre **float** — déjà dans `plot.py` (scatter projeté + côtes) ;
- enveloppe max sur Terre **isolignes** — NOUVEAU `draw_earth_envelope_contours(ax,
  grid, envelope_dbi, …, step_db=5.0)` : projette la grille (u,v)→(lat,lon),
  `contour` sur la grille projetée 2-D (cellules hors-disque masquées via
  `np.ma.masked_invalid`, coordonnées NaN remplacées par un filler ignoré car
  Z masqué), niveaux tous les `step_db`, + côtes.

### Widget Qt — `PatternStudio(QMainWindow)`

**Contrôles (panneau gauche, `QFormLayout` dans des `QGroupBox`) :**
- *Disque (zone)* : `antenna_lat`, `antenna_lon`, `sat_lon`, `zone_radius_deg`
  (`QDoubleSpinBox`, bornes physiques : lat ∈ [−90, 90], lon ∈ [−180, 180],
  zone_radius ≥ 6).
- *Simulation* : `report` (`QComboBox`, items = dossiers de `data/reference_reports`),
  `n_elements` (`QSpinBox`), `seed` (`QSpinBox`), `coverage_margin_db`
  (`QDoubleSpinBox`).
- *Pattern affiché* : `element` (`QSpinBox`, max = n_elements − 1).
- Boutons : **Générer** (`_on_generate`), **Export…** (`_on_export`).

**Canvas (droite, `FigureCanvasQTAgg`)** — `Figure` 2×3 (5 vues + 1 cellule libre) :
1. Barycentres des patterns (u,v).
2. Directivité du pattern `element` (u,v).
3. Phase du pattern `element` (u,v).
4. Enveloppe max sur Terre — float.
5. Enveloppe max sur Terre — isolignes (tous les 5 dB).

**Comportement :**
- `_on_generate` : lit les contrôles → `build_result(...)` → stocke le résultat →
  borne le spinbox `element` à `[0, n−1]` → redessine les 5 vues.
- changement de `element` (`valueChanged`) : ne redessine que les vues 2 et 3
  (directivité + phase) à partir du résultat en cache — pas de recalcul.
- `_on_export` : `QFileDialog` → `write_calibrated(path, grid, specs, mode)`.
- Au démarrage : une génération initiale avec les valeurs par défaut (FRANCE :
  lat 46.6, lon 2.5, sat_lon 3.0, zone 6.0 ; rapport 0000 ; n=80, seed 0).

### CLI / entrée

`grd-gui` → `main()` : `configure_logging()`, crée `QApplication`, instancie
`PatternStudio`, `show()`, `app.exec()`. Backend Qt natif (le test force offscreen).

## Gestion d'erreurs

- `build_result` : `ValueError` propagés de `read_report` (rapport invalide) et de
  `calibrate`/`sat_frame_geometry` (zone ne tenant pas sous le limbe — ex.
  `zone_radius` trop grand). Dans le widget, ces erreurs sont attrapées et
  affichées dans une `QMessageBox` (pas de crash).
- Export sans génération préalable : bouton désactivé tant qu'aucun résultat.

## Tests

- `gui.py`, `plot.py` et `examples/*` sont **omis de couverture** (Qt / rendu
  matplotlib). Aucun code 100 %-couvert nouveau : les modules couverts (`schemas`,
  `synth`, `scenario`, `report_ingest`) sont inchangés ; le gate 100 % tient.
- `test_example_gui.py` : lance `examples/example_gui.py` en sous-processus avec
  `QT_QPA_PLATFORM=offscreen` et `MPLBACKEND=Agg` ; code retour 0. L'exemple
  instancie `PatternStudio`, déclenche une génération (`_on_generate`), change
  l'`element`, appelle `_on_export` vers un `.npz` temporaire, puis ferme — **sans
  `app.exec()`** (pas de boucle d'événements bloquante).
- `test_gui.py` (optionnel, comportemental) : `build_result(...)` à petites tailles
  rend un `CalibrationResult` aux bonnes formes (gui.py omis de la *mesure* de
  couverture, mais la fonction reste importable et testable).

## Dépendances

Extra `viz` (déjà : `matplotlib`, `pyqt6`). Aucune nouvelle dépendance.
</content>
