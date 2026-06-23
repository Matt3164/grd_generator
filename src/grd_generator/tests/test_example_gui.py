import os
import subprocess
import sys


def test_example_gui_runs() -> None:
    """Vérifie que le studio s'instancie, génère, exporte et ferme (offscreen)."""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, "src/grd_generator/examples/example_gui.py"],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert result.returncode == 0, (
        f"L'exemple GUI a échoué :\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
