"""Spécifications physiques du réflecteur offset et du réseau de feeds."""

from pydantic import BaseModel, Field

C_LIGHT = 299_792_458.0  # vitesse de la lumière (m/s)


class ReflectorSpec(BaseModel):
    """Réflecteur parabolique offset simple + fréquence d'exploitation."""

    diameter_m: float = Field(..., gt=0)
    focal_length_m: float = Field(..., gt=0)
    offset_clearance_m: float = Field(0.0, ge=0)
    freq_hz: float = Field(..., gt=0)
    beam_deviation_factor: float = Field(1.0, gt=0)

    @property
    def wavelength_m(self) -> float:
        return C_LIGHT / self.freq_hz

    @property
    def f_over_d(self) -> float:
        return self.focal_length_m / self.diameter_m

    @property
    def aperture_center_y_m(self) -> float:
        """Décalage du centre de l'ouverture projetée par rapport à l'axe."""
        return self.offset_clearance_m + self.diameter_m / 2.0


class FeedSpec(BaseModel):
    """Réseau de feeds dans le plan focal : positions (m) et taper cos^q."""

    positions_m: list[tuple[float, float]]
    q: float = Field(..., ge=0.0)
    # Déplacement axial du plan des feeds (m), le long de l'axe focal
    # (>0 = feed reculé, éloigné du réflecteur). Pas de contrainte de signe.
    defocus_m: float = Field(0.0)
    # Écran de phase aléatoire corrélé par feed (erreurs type Ruze), voir
    # `optics.random_phase_screen`. rms=0 -> pas d'écran (comportement inchangé).
    phase_error_rms_rad: float = Field(0.0, ge=0.0)
    phase_corr_length_m: float = Field(0.05, gt=0.0)
    phase_error_seed: int = Field(0)
    # Écran de phase aléatoire corrélé COMMUN à tous les feeds (erreurs de
    # surface du réflecteur, par opposition aux erreurs propres au feed
    # ci-dessus) : même longueur de corrélation `phase_corr_length_m`,
    # construit une seule fois (voir `synth_afr._SHARED_SEED_OFFSET` pour le
    # RNG dédié, distinct de tous les seeds par-feed `phase_error_seed + i`,
    # i >= 0) et ajouté à l'écran de chaque feed. rms=0 -> pas d'écran commun.
    phase_error_shared_rms_rad: float = Field(0.0, ge=0.0)
    # Modèle deux échelles : le réflecteur (`diameter_m`) redevient un vrai
    # disque physique ; chaque feed n'en illumine qu'une empreinte gaussienne
    # mobile (voir `optics.footprint_amplitude_mask`). Diamètre FWHM (en
    # amplitude) de cette empreinte, en m. 0 = désactivé (pleine ouverture,
    # comportement actuel strictement inchangé).
    footprint_m: float = Field(0.0, ge=0.0)
    # Déplacement du centre de l'empreinte par mètre de décalage transverse
    # du feed (m/m, signe libre) : centre_i = (0, aperture_center_y_m) +
    # magnification·(δx_i, δy_i). 0 = empreinte centrée pour tous les feeds
    # (pas de balayage). Sans effet si `footprint_m == 0`.
    footprint_magnification: float = Field(0.0)

    @property
    def n_feeds(self) -> int:
        return len(self.positions_m)
