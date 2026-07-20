"""Export TICRA GRASP `.grd` ASCII des patterns AFR (co + cross) + params JSON.

Format `.grd` (un jeu de champ par fichier) : bloc d'en-tête texte libre, séparateur
`++++`, puis KTYPE, (NSET, ICOMP, NCOMP, IGRID), (IX, IY), (XS, YS, XE, YE),
(NX, NY, KLIMIT), et NX·NY lignes `Re(co) Im(co) Re(cross) Im(cross)`.

Conventions retenues : ICOMP=3 (co/cross Ludwig-3), NCOMP=2, IGRID=1 (grille en
cosinus directeurs u=sinθcosφ, v=sinθsinφ) — la grille `UVGrid` du reflector est
déjà exprimée en cosinus directeurs, donc les bornes XS/YS/XE/YE sont les valeurs
d'axe telles quelles (pas de `sin` ici : elle a déjà été appliquée en amont, lors
du rééchantillonnage du far-field sur la grille).
Les champs sont ceux stockés par la synthèse : amplitude normalisée en directivité.
"""

import io
import zipfile
from typing import Any

from grd_generator.schemas import ComplexField, UVGrid


def pattern_to_grd(
    co: ComplexField,
    cross: ComplexField,
    grid: UVGrid,
    *,
    freq_hz: float,
    title: str = "grd_generator AFR pattern",
) -> str:
    """Sérialise un pattern (co + cross) au format TICRA GRASP `.grd` ASCII."""
    u_axis, v_axis = grid.axes()
    xs = float(u_axis[0])
    ys = float(v_axis[0])
    xe = float(u_axis[-1])
    ye = float(v_axis[-1])
    nx, ny = grid.n_u, grid.n_v

    lines = [
        title,
        f"Champ co/cross (Ludwig-3), normalise en directivite, freq = {freq_hz:.6e} Hz",
        "++++",
        f"{1:>8d}",  # KTYPE
        f"{1:>8d}{3:>9d}{2:>9d}{1:>9d}",  # NSET ICOMP NCOMP IGRID
        f"{0:>8d}{0:>9d}",  # IX IY (centre de faisceau)
        " ".join(f"{val:.10E}" for val in (xs, ys, xe, ye)),  # XS YS XE YE (cosinus directeurs)
        f"{nx:>8d}{ny:>9d}{0:>9d}",  # NX NY KLIMIT
    ]
    for iv in range(ny):
        for iu in range(nx):
            c = co[iv, iu]
            x = cross[iv, iu]
            lines.append(" ".join(f"{val:.10E}" for val in (c.real, c.imag, x.real, x.imag)))
    return "\n".join(lines) + "\n"


def patterns_to_zip_bytes(
    co_fields: list[ComplexField],
    cross_fields: list[ComplexField],
    grid: UVGrid,
    *,
    freq_hz: float,
    prefix: str = "pattern",
) -> bytes:
    """Zippe un `.grd` par pattern (co + cross) ; renvoie les octets du ZIP en mémoire."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (co, cross) in enumerate(zip(co_fields, cross_fields, strict=True)):
            text = pattern_to_grd(co, cross, grid, freq_hz=freq_hz, title=f"{prefix} #{i}")
            zf.writestr(f"{prefix}_{i:03d}.grd", text)
    return buf.getvalue()


def simulation_params_dict(
    *,
    diameter_m: float,
    f_over_d: float,
    offset_clearance_m: float,
    freq_ghz: float,
    q: float,
    pitch_m: float,
    n_feeds: int,
    zone_radius_deg: float,
    defocus_m: float = 0.0,
    centered_aperture: bool = False,
) -> dict[str, Any]:
    """Dict sérialisable JSON des paramètres de simulation (focale dérivée incluse)."""
    return {
        "diameter_m": diameter_m,
        "f_over_d": f_over_d,
        "focal_length_m": f_over_d * diameter_m,
        "offset_clearance_m": offset_clearance_m,
        "freq_ghz": freq_ghz,
        "q": q,
        "pitch_m": pitch_m,
        "n_feeds": n_feeds,
        "zone_radius_deg": zone_radius_deg,
        "defocus_m": defocus_m,
        "centered_aperture": centered_aperture,
    }
