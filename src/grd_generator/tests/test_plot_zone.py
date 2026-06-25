import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from grd_generator.plot import draw_service_zone_uv
from grd_generator.reflector.zone import ServiceZone


def test_draw_service_zone_adds_line() -> None:
    fig, ax = plt.subplots()
    before = len(ax.lines)
    draw_service_zone_uv(ax, ServiceZone(radius_deg=8.0))
    assert len(ax.lines) == before + 1
    plt.close(fig)
