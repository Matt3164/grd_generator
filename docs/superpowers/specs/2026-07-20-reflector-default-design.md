# Design — Le mode réflecteur devient l'unique mode

Date : 2026-07-20 · Branche : `feat/reflector-default` · Approche retenue : A (suppression franche + trace JSON)

## Objectif

La simulation physique de réflecteur alimenté par réseau (AFR, modèle defocus
simple issu de la PR #14) devient le **seul** mode du projet. Le pipeline
« pattern » (génération du jeu de référence gaussien, calibration depuis les
rapports mesurés, PatternStudio) est supprimé. Les valeurs de référence du jeu
pattern sont conservées dans un fichier de trace versionné.

## Décisions validées

- **Trace des valeurs de ref** : `docs/pattern_reference_values.json` (déjà créé)
  — constantes de génération (N=80, crête 47 dBi, seuil 44 dBi, pente de phase
  radiale 3.0 rad/°, bissection σ), scénario FRANCE, grille exacte du `.npz`,
  et valeurs par élément (ids, centres (u,v), σ, crêtes, pentes de phase)
  extraites du `reference_array.npz` local. Référencé dans le README.
- **`grd-generate`** pointe désormais sur la génération AFR
  (`reflector/generate_afr.py`, sortie `data/processed/reflector_array.npz`).
  L'alias `grd-generate-afr` disparaît.
- **`grd-gui`** lance ReflectorStudio directement ; l'option `--mode` et
  PatternStudio disparaissent.
- **`grd-calibrate` et `grd-plot`** disparaissent (entry points et modules).
- Le fichier local `data/processed/reference_array.npz` (gitignoré) est laissé
  sur disque, non touché.

## Suppressions

Modules : `calibrate.py`, `generate.py`, `geometry.py`, `scenario.py`,
`report_ingest.py`. Dans `gui.py` : classe PatternStudio et tout ce qui n'est
utilisé que par elle (`REPORTS_DIR`, onglets earth/rapports…). Exemples
`example_calibration.py`, `example_generation.py`, `example_gui.py`,
`example_plot.py` et leurs tests de lancement. Tests unitaires des modules
supprimés. Données `data/reference_reports/` et `data/ne_50m_land.geojson` si
plus référencées.

## Élagages (contrainte : couverture 100 %, zéro code mort)

- `synth.py` (extraction verbatim satscope) : ne garder que les fonctions
  atteignables depuis le code réflecteur (`hex_centers_uv`,
  `combined_max_directivity_dbi` et leurs dépendances internes). On supprime
  des fonctions entières sans remanier le style de celles qui restent —
  amendement assumé de la consigne « verbatim » du README, qui est mise à jour.
- `plot.py` : ne garder que les helpers utilisés par ReflectorStudio
  (`_draw_uv_map`, `_draw_phase_map`, `draw_service_zone_uv`,
  `envelope_max_dbi` et dépendances). Le rendu « earth » (traits de côte,
  footprints, limbe) part avec le mode pattern.
- `schemas.py` : retirer les specs devenues orphelines (`GaussianSpec`,
  `EllipticalSpec`…) si plus utilisées ; garder `UVGrid`, `ComplexField`.

## Interfaces conservées (inchangées)

`reflector/` au complet (spec, optics, farfield, synth_afr, zone, grd_export,
generate_afr), la boucle round-trip `example_reflector_roundtrip.py`, le
logger loguru, l'export GRD/params de ReflectorStudio.

## Tests et validation

- Adapter la suite : suppression des tests du code retiré, conservation de
  tous les tests réflecteur, GUI incluse. Couverture 100 % exigée par la CI.
- `uv run grd-generate` écrit `reflector_array.npz` ; `uv run grd-gui` ouvre
  ReflectorStudio ; le round-trip tourne si `grd_analyzer` est disponible.
- README et `pyproject.toml` (entry points) mis à jour en conséquence.
