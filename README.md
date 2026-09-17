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

## Lot HDF5 — jeu multi-fréquence

`grd-batch` génère, sans interface graphique et de façon rejouable, un jeu de
diagrammes **HDF5** (structure de type TICRA Tools 25.0, grille sphérique uv) :
une famille de fichiers par fréquence, H simulé + V fabriqué par échange exact
des deux composantes (voir `reflector/hdf5_export.py`, `reflector/batch.py`).

```bash
uv run grd-batch --config batch-ku.json --out <dossier>          # grd/, meta/, MANIFEST.sha256
uv run grd-batch --config batch-ku.json --out <dossier> --check  # + code 1 si critère bloquant KO
```

Le calibrage (choix de `defocus_m`, `pitch_m`, `q`, `n_feeds` avant de figer
`batch-ku.json`) passe par un script d'exemple dédié, qui balaie une petite
grille de paramètres à une fréquence et imprime/logue le tableau de métriques
(`reflector/metrics.py`), avec une figure optionnelle (`--figure`) :
[`src/grd_generator/examples/example_batch_ku.py`](src/grd_generator/examples/example_batch_ku.py) :

```bash
uv run python src/grd_generator/examples/example_batch_ku.py \
    --defocus-m 0.53 0.75 0.93 1.1 --pitch-m 0.032 0.036 0.040 \
    --q 2.0 3.0 --n-feeds 91 127
uv run python src/grd_generator/examples/example_batch_ku.py \
    --defocus-m 0.53 --pitch-m 0.032 --q 2.0 --n-feeds 91 \
    --figure calibrage.png --mission-plan <mission_plan.json>
```

Détails : `docs/superpowers/specs/2026-09-17-batch-ku-hdf5-design.md`.

## Historique

L'ancien pipeline « pattern » (jeu de référence gaussien calibré sur rapports
mesurés) a été retiré ; ses valeurs de référence sont tracées dans
`docs/pattern_reference_values.json`. `synth.py` est un extrait du projet
amont `satscope`, réduit aux fonctions utilisées — ne pas remanier le style
interne des fonctions conservées.
