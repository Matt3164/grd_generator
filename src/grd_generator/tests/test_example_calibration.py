import subprocess
import sys


def test_example_calibration_runs() -> None:
    """Vérifie que le script d'exemple s'exécute sans erreur."""
    result = subprocess.run(
        [sys.executable, "src/grd_generator/examples/example_calibration.py"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"L'exemple a échoué :\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
