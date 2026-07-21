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
`grd_analyzer` et compare le rapport obtenu à un rapport de référence stocké
dans `data/reference_reports/` (nécessite `grd_analyzer` en sibling du repo,
ou `$GRD_ANALYZER_DIR`) :

```bash
uv run python src/grd_generator/examples/example_reflector_roundtrip.py
```

## Historique

L'ancien pipeline « pattern » (jeu de référence gaussien calibré sur rapports
mesurés) a été retiré ; ses valeurs de référence sont tracées dans
`docs/pattern_reference_values.json`. `synth.py` est un extrait du projet
amont `satscope`, réduit aux fonctions utilisées — ne pas remanier le style
interne des fonctions conservées.
