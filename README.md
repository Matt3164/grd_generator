# grd_generator

Générateur autonome du jeu de référence de diagrammes de rayonnement d'antenne
satellite (fichier `.npz`).

## Usage

```bash
uv sync
uv run grd-generate            # écrit data/processed/reference_array.npz
```

Voir `src/grd_generator/examples/example_generation.py` pour un exemple.
