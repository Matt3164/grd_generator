import os
import subprocess
import sys


def test_example_plot_runs() -> None:
    """Vérifie que le script d'exemple de rendu s'exécute sans erreur (backend Agg)."""
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, "src/grd_generator/examples/example_plot.py"],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert result.returncode == 0, (
        f"L'exemple a échoué :\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
