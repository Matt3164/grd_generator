import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GRD_ANALYZER_DIR = Path(os.environ.get("GRD_ANALYZER_DIR", REPO_ROOT.parent / "grd_analyzer"))


def test_example_reflector_roundtrip_runs(tmp_path: Path) -> None:
    """Vérifie que la boucle round-trip (génération -> .grd -> grd_analyzer ->
    comparaison) s'exécute sans erreur et produit un comparison.json.

    Skip si le projet frère grd_analyzer n'est pas disponible en local.
    """
    if not GRD_ANALYZER_DIR.is_dir():
        pytest.skip(f"grd_analyzer introuvable : {GRD_ANALYZER_DIR}")

    out_dir = tmp_path / "roundtrip"
    result = subprocess.run(
        [
            sys.executable,
            "src/grd_generator/examples/example_reflector_roundtrip.py",
            "--n-grid",
            "41",
            "--n-feeds",
            "7",
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"L'exemple round-trip a échoué :\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert (out_dir / "comparison.json").exists()
