import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Grille réduite (n_feeds/n_grid/n_aperture/pad_factor petits) : cf.
# `docs/superpowers/specs/2026-09-17-batch-ku-hdf5-design.md`, § Tests
# (« test de lancement de l'exemple, grille réduite, < 30 s »). Ce ne sont PAS
# les valeurs de calibrage réel (voir les défauts du script lui-même).
_FAST_ARGS = [
    "--defocus-m", "0.53",
    "--pitch-m", "0.032",
    "--q", "2.0",
    "--n-feeds", "7",
    "--n-grid", "21",
    "--n-aperture", "16",
    "--pad-factor", "4",
]

# Plan de mission SYNTHÉTIQUE minimal, même forme que celle lue par
# `load_mission_plan_uv` (`beams[].coverages[0].geometry.center`, en degrés) :
# 4 faisceaux proches du nadir, pour exercer systématiquement la surimpression
# de `--figure`, sans dépendre d'un fichier externe au dépôt.
_SYNTHETIC_MISSION_PLAN = {
    "beams": [
        {"coverages": [{"geometry": {"center": [0.5, 0.5]}}]},
        {"coverages": [{"geometry": {"center": [-0.5, 0.5]}}]},
        {"coverages": [{"geometry": {"center": [0.5, -0.5]}}]},
        {"coverages": [{"geometry": {"center": [-0.5, -0.5]}}]},
    ]
}


def test_example_batch_ku_sweep_runs() -> None:
    """Balayage seul (pas de figure) : doit tourner en quelques secondes."""
    result = subprocess.run(
        [sys.executable, "src/grd_generator/examples/example_batch_ku.py", *_FAST_ARGS],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"L'exemple a échoué :\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "criteres" in result.stdout  # en-tête du tableau imprimé


def test_example_batch_ku_figure_runs(tmp_path: Path) -> None:
    """Avec `--figure` et `--mission-plan` (surimpression toujours exercée) :
    doit rester < 30 s."""
    figure_path = tmp_path / "calibrage.png"
    mission_plan_path = tmp_path / "mission_plan.json"
    mission_plan_path.write_text(json.dumps(_SYNTHETIC_MISSION_PLAN), encoding="utf-8")

    args = [
        *_FAST_ARGS,
        "--figure", str(figure_path),
        "--mission-plan", str(mission_plan_path),
    ]
    result = subprocess.run(
        [sys.executable, "src/grd_generator/examples/example_batch_ku.py", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"L'exemple a échoué :\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
