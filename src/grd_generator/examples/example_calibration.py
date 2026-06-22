"""
Exemple : calibration du générateur depuis un rapport réel
===========================================================
Lit un report.json de grd_analyzer (jeu 0000), génère un array calibré
elliptique dans un .npz temporaire et relit ses métadonnées.

Usage :
    python src/grd_generator/examples/example_calibration.py
"""

import tempfile
from pathlib import Path

import numpy as np

from grd_generator.calibrate import generate_calibrated
from grd_generator.logger import configure_logging, logger

REPORT = Path(__file__).resolve().parents[3] / "data" / "reference_reports" / "0000"


def main() -> None:
    configure_logging()
    logger.info("example_start", feature="calibration")

    with tempfile.TemporaryDirectory() as tmp:
        out = generate_calibrated(REPORT, Path(tmp) / "calibrated.npz", seed=0)
        data = np.load(out, allow_pickle=False)
        logger.info(
            "example_calibrated",
            n_antennas=int(data["fields"].shape[0]),
            mode=str(data["mode"]),
            grid=(int(data["n_u"]), int(data["n_v"])),
        )

    logger.info("example_done", feature="calibration")


if __name__ == "__main__":
    main()
