"""
Exemple : génération du jeu de référence de patterns
=====================================================
Génère les 80 champs de référence (mode gaussian) dans un fichier .npz et
relit ses métadonnées.

Usage :
    python src/grd_generator/examples/example_generation.py
"""

import tempfile
from pathlib import Path

import numpy as np

from grd_generator.generate import generate_reference
from grd_generator.logger import configure_logging, logger


def main() -> None:
    configure_logging()
    logger.info("example_start", feature="generation")

    with tempfile.TemporaryDirectory() as tmp:
        out = generate_reference(Path(tmp) / "reference_array.npz")
        data = np.load(out, allow_pickle=False)
        logger.info(
            "example_generated",
            n_antennas=int(data["fields"].shape[0]),
            grid=(int(data["n_u"]), int(data["n_v"])),
            mode=str(data["mode"]),
        )

    logger.info("example_done", feature="generation")


if __name__ == "__main__":
    main()
