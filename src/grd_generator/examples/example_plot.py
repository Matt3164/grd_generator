"""
Exemple : rendu matplotlib d'un array de référence
===================================================
Génère le jeu de référence (géométrie FRANCE) dans un fichier temporaire, puis
trace élément + enveloppe (u, v) et l'enveloppe sur Terre (côtes) dans un PNG.

Usage :
    python src/grd_generator/examples/example_plot.py
    (nécessite l'extra viz : uv sync --extra viz)
"""

import tempfile
from pathlib import Path

from grd_generator.generate import generate_reference
from grd_generator.logger import configure_logging, logger
from grd_generator.plot import render


def main() -> None:
    configure_logging()
    logger.info("example_start", feature="plot")

    with tempfile.TemporaryDirectory() as tmp:
        npz = generate_reference(Path(tmp) / "reference_array.npz")
        png = render(npz, out=Path(tmp) / "reference_plot.png", show=False)
        logger.info("example_plotted", png=str(png), exists=png.exists())

    logger.info("example_done", feature="plot")


if __name__ == "__main__":
    main()
