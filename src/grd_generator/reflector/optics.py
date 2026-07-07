"""Optique géométrique : feed → champ d'ouverture du réflecteur offset.

Modèle aperture-field. Un point d'ouverture (X, Y) (m, mesuré depuis l'axe du
paraboloïde parent) est illuminé par le feed sous l'angle polaire
ψ = 2·arctan(ρ/2F) (ρ = ‖(X,Y)‖) ; l'amplitude suit le diagramme de feed cos^q(ψ).
Un feed décalé de (δx, δy) dans le plan focal impose un gradient de phase linéaire
(beam deviation factor) qui dépointe le faisceau secondaire.

Un feed défocalisé de δz le long de l'axe focal (>0 = feed reculé, éloigné du
réflecteur) ajoute une phase d'erreur quadratique en ouverture :

    φ_defocus(ψ) = k·δz·(1 − cosψ),  k = 2π/λ

Justification : pour un paraboloïde idéal, le trajet feed→ouverture est
constant (propriété de foyer) uniquement si le feed est au foyer exact. Un
déplacement axial δz du feed introduit une différence de marche
≈ −δz·cosψ (mesurée depuis le foyer, le long du rayon d'angle ψ) ; à une
constante additive près référencée à ψ=0 (le long de l'axe, différence de
marche nulle par convention), la phase correspondante est k·δz·(1 − cosψ).
Ce terme s'ajoute au tilt de dépointage existant. Il ne dépend que de ψ (pas
de la position du feed dans le plan focal) : autour du centre de l'ouverture,
il étale symétriquement le pattern par feed autour de sa direction pointée,
sans la déplacer.

Squint sur ouverture offset : ψ dépend de ρ = ‖(X,Y)‖ mesuré depuis l'axe du
paraboloïde parent, pas depuis le centre du disque d'ouverture, qui est lui
centré en Y₀ = `aperture_center_y_m` ≠ 0. Le développement de (1 − cosψ)
autour de Y₀ contient donc, en plus du terme quadratique attendu, une
composante plane a + b·X + c·Y (essentiellement un gradient en Y
proportionnel à Y₀) qui se comporte exactement comme le tilt de dépointage :
le faisceau part hors zone au lieu de s'étaler. Ce squint est un effet
géométrique réel d'un feed axialement déplacé devant un réflecteur offset,
mais ce n'est pas l'effet modélisé ici : `defocus_m` est un bouton
phénoménologique d'étalement pur, et le pointage doit rester gouverné par la
seule position transverse (δx, δy) du feed (dans un vrai design, le cluster
défocalisé est repositionné/repointé pour recentrer la couverture).
`aperture_field` retire donc, aux moindres carrés sur le
disque d'ouverture, le plan a + b·X + c·Y ajusté à `k·δz·(1 − cosψ)` avant de
l'ajouter à la phase ; le résidu ne contient plus que l'étalement symétrique
recherché.

Écran de phase aléatoire corrélé (erreurs type Ruze) : `random_phase_screen`
modélise les erreurs de surface/alignement statistiques d'un feed par un
champ de phase aléatoire mais spatialement corrélé sur l'ouverture. Le champ
brut est un bruit blanc gaussien N(0,1) échantillonné sur la grille complète
(RNG dédié, un par feed) ; il est lissé par un filtre passe-bas gaussien de
longueur caractéristique `corr_length_m`, réalisé en numpy pur par
convolution dans le domaine spectral (`np.fft.rfft2`/`irfft2`, sans
dépendance scipy) — le pas physique `dx` de la grille convertit cette
longueur en pixels. Le résultat est recentré (moyenne nulle sur `inside`)
puis renormalisé pour que son écart-type sur `inside` vaille exactement
`rms_rad` (radians). Contrairement au defocus, aucune composante plane n'est
retirée : l'errance de pointage aléatoire par feed qu'introduit le bruit
basse fréquence est un effet recherché, qui reproduit la dispersion des
centres de faisceau observée sur données réelles (le pitch, lui, ne pilote
que l'espacement nominal des faisceaux).
"""

import numpy as np
from numpy.typing import NDArray

from grd_generator.reflector.spec import ReflectorSpec

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def aperture_grid(
    spec: ReflectorSpec, n_aperture: int, pad_factor: int
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_], float]:
    """Grille carrée d'ouverture (m) + masque disque + pas dx.

    `n_aperture` échantillons sur le diamètre ; zéro-padding `pad_factor`× pour
    la résolution angulaire du far-field. Disque de diamètre D centré sur l'offset.
    """
    dx = spec.diameter_m / n_aperture
    n = n_aperture * pad_factor
    coords = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(coords, coords)
    r2 = X**2 + (Y - spec.aperture_center_y_m) ** 2
    inside: NDArray[np.bool_] = r2 <= (spec.diameter_m / 2.0) ** 2
    return X, Y, inside, dx


def illumination_angle(X: FloatArray, Y: FloatArray, focal_length_m: float) -> FloatArray:
    """Angle polaire ψ vu du feed pour un point d'ouverture : ψ = 2·arctan(ρ/2F)."""
    rho = np.hypot(X, Y)
    angle: FloatArray = 2.0 * np.arctan(rho / (2.0 * focal_length_m))
    return angle


def _fit_plane(
    X: FloatArray, Y: FloatArray, values: FloatArray, inside: NDArray[np.bool_]
) -> FloatArray:
    """Plan a + b·X + c·Y ajusté aux moindres carrés sur `inside`, évalué sur toute la grille.

    Utilisé pour retirer la composante de dépointage artificielle qu'introduit
    le terme de defocus sur une ouverture offset (voir docstring du module).
    """
    ones = np.ones(int(inside.sum()))
    design = np.column_stack((ones, X[inside], Y[inside]))
    coeffs, *_ = np.linalg.lstsq(design, values[inside], rcond=None)
    a, b, c = coeffs
    plane: FloatArray = a + b * X + c * Y
    return plane


def random_phase_screen(
    X: FloatArray,
    Y: FloatArray,
    inside: NDArray[np.bool_],
    dx: float,
    rms_rad: float,
    corr_length_m: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Écran de phase aléatoire corrélé (erreurs type Ruze) sur la grille d'ouverture.

    Bruit blanc gaussien N(0,1) (`rng`, dédié à l'appelant) sur la grille
    complète, lissé par un filtre passe-bas gaussien de longueur
    caractéristique `corr_length_m` (m) — convolution réalisée dans le
    domaine spectral (FFT 2D réelle, noyau gaussien périodique construit
    dans le domaine spatial), numpy pur, sans dépendance scipy. `dx` (m) est
    le pas physique de la grille, nécessaire pour convertir `corr_length_m`
    en écart-type du noyau exprimé en pixels.

    La moyenne sur `inside` est retirée puis le champ est renormalisé pour
    que son écart-type sur `inside` vaille exactement `rms_rad` (radians).
    Aucune composante plane n'est retirée (contrairement à `defocus_m`) :
    l'errance de pointage aléatoire par feed que cela introduit est un effet
    recherché (voir docstring du module).
    """
    noise = rng.standard_normal(X.shape)
    n_y, n_x = X.shape
    sigma_px = corr_length_m / dx
    iy = (np.arange(n_y) + n_y // 2) % n_y - n_y // 2
    ix = (np.arange(n_x) + n_x // 2) % n_x - n_x // 2
    grid_ix, grid_iy = np.meshgrid(ix, iy)
    kernel = np.exp(-(grid_ix**2 + grid_iy**2) / (2.0 * sigma_px**2))
    kernel /= kernel.sum()
    smoothed: FloatArray = np.fft.irfft2(
        np.fft.rfft2(noise) * np.fft.rfft2(kernel), s=X.shape
    )
    centered = smoothed - smoothed[inside].mean()
    std_inside = centered[inside].std()
    screen: FloatArray = centered * (rms_rad / std_inside)
    return screen


def aperture_field(
    spec: ReflectorSpec,
    feed_xy: tuple[float, float],
    X: FloatArray,
    Y: FloatArray,
    inside: NDArray[np.bool_],
    q: float,
    defocus_m: float = 0.0,
    extra_phase: FloatArray | None = None,
) -> ComplexArray:
    """Champ scalaire d'ouverture pour un feed : taper cos^q(ψ) + tilt + defocus.

    `defocus_m` (δz, >0 = feed reculé) ajoute la phase quadratique
    `k·δz·(1 − cosψ)` (voir docstring du module) à la phase de tilt existante,
    après en avoir retiré la composante plane a + b·X + c·Y ajustée aux
    moindres carrés sur le disque d'ouverture (`inside`). Sans ce retrait, le
    terme de defocus dépointe le faisceau sur une ouverture offset au lieu de
    seulement l'étaler symétriquement (squint, choix de modélisation — cf.
    docstring du module) ; la constante `a` du plan est sans effet physique
    (elle disparaît dans `exp(1j·phase)` relatif) mais la retirer avec le
    reste du plan ne coûte rien.
    Avec `defocus_m=0.0`, le champ est strictement inchangé (early-out : pas
    de résolution moindres carrés à payer).

    `extra_phase`, si fourni, est simplement ajouté à la phase totale (tilt +
    defocus) avant l'exponentielle complexe — typiquement l'écran de phase
    aléatoire construit par l'appelant via `random_phase_screen` (l'appelant
    porte le contexte par-feed : RNG seedé, paramètres). Avec
    `extra_phase=None`, le champ est strictement inchangé.
    """
    psi = illumination_angle(X, Y, spec.focal_length_m)
    amp = np.cos(psi) ** q
    k = 2.0 * np.pi / spec.wavelength_m
    dxf, dyf = feed_xy
    tilt = -k * spec.beam_deviation_factor * (dxf * X + dyf * Y) / spec.focal_length_m
    if defocus_m == 0.0:
        defocus_phase = np.zeros_like(X)
    else:
        raw_defocus_phase = k * defocus_m * (1.0 - np.cos(psi))
        defocus_phase = raw_defocus_phase - _fit_plane(X, Y, raw_defocus_phase, inside)
    total_phase = tilt + defocus_phase
    if extra_phase is not None:
        total_phase = total_phase + extra_phase
    field: ComplexArray = (inside * amp * np.exp(1j * total_phase)).astype(np.complex128)
    return field


def aperture_pol_vectors(
    spec: ReflectorSpec, X: FloatArray, Y: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Composantes (ex, ey) du vecteur de polarisation réfléchi (feed x-polarisé).

    Réflexion GO de x̂ sur la normale du paraboloïde : e_r = 2(n̂·x̂)n̂ − x̂.
    Source géométrique de la cross-pol (offset / plans diagonaux).
    """
    f = spec.focal_length_m
    nx = -X / (2.0 * f)
    ny = -Y / (2.0 * f)
    nz = np.ones_like(X)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx, ny = nx / norm, ny / norm
    ndotx = nx  # n̂ · x̂
    ex: FloatArray = 2.0 * ndotx * nx - 1.0
    ey: FloatArray = 2.0 * ndotx * ny
    return ex, ey
