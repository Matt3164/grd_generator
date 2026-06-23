"""
Exemple : studio interactif de patterns (offscreen)
====================================================
Instancie PatternStudio (génération initiale), change le pattern affiché, exporte
un .npz temporaire, puis ferme — sans boucle d'événements (utilisable headless).

Usage :
    QT_QPA_PLATFORM=offscreen python src/grd_generator/examples/example_gui.py
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from grd_generator.gui import PatternStudio  # noqa: E402
from grd_generator.logger import configure_logging, logger  # noqa: E402


def main() -> None:
    configure_logging()
    logger.info("example_start", feature="gui")
    app = QApplication.instance() or QApplication([])  # noqa: F841  # keeps QApp alive
    studio = PatternStudio(n_u=41, n_v=41)  # petite grille = rapide
    studio.set_element(1)
    with tempfile.TemporaryDirectory() as tmp:
        studio.export_to(Path(tmp) / "gui.npz")
    studio.close()
    logger.info("example_done", feature="gui")


if __name__ == "__main__":
    main()
