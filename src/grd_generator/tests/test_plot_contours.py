import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402

from grd_generator.generate import generate_reference  # noqa: E402
from grd_generator.plot import (  # noqa: E402
    draw_earth_envelope_contours,
    envelope_max_dbi,
    load_array,
)


def test_draw_earth_envelope_contours_renders(tmp_path: Path) -> None:
    npz = generate_reference(tmp_path / "ref.npz")
    grid, fields, _ = load_array(npz)
    env = envelope_max_dbi(fields)
    fig, ax = plt.subplots()
    draw_earth_envelope_contours(ax, grid, env, sat_lon=3.0, step_db=5.0)
    out = tmp_path / "contours.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 0
