# grd_generator

Générateur autonome du jeu de référence de diagrammes de rayonnement d'antenne
satellite (fichier `.npz`).

## Usage

```bash
uv sync
uv run grd-generate            # écrit data/processed/reference_array.npz
```

Voir `src/grd_generator/examples/example_generation.py` pour un exemple.

## Interfaces graphiques

```bash
uv run grd-gui                    # PatternStudio (calibration de patterns, défaut)
uv run grd-gui --mode reflector   # ReflectorStudio (réflecteur alimenté par réseau, AFR)
```

## Autres commandes

```bash
uv run grd-calibrate       # calibration en ligne de commande
uv run grd-plot            # tracé des diagrammes
uv run grd-generate-afr    # génération AFR (réflecteur)

# Boucle round-trip : génère un AFR, l'exporte en .grd, l'analyse avec le
# projet frère grd_analyzer et compare le rapport obtenu à un rapport de
# référence (nécessite grd_analyzer en sibling du repo, ou $GRD_ANALYZER_DIR)
uv run python src/grd_generator/examples/example_reflector_roundtrip.py
```

## Notes

`geometry.py` et `synth.py` sont des extractions verbatim du projet amont `satscope` — ne pas remanier leur style interne.
