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
    centered_aperture: bool = Field(
        False,
        description=(
            "Ouverture centrée sur l'axe (réflecteur front-fed axisymétrique) : "
            "patterns radiaux symétriques. Quand True, `offset_clearance_m` est ignoré."
        ),
    )

    @property
    def wavelength_m(self) -> float:
        return C_LIGHT / self.freq_hz

    @property
    def f_over_d(self) -> float:
        return self.focal_length_m / self.diameter_m

    @property
    def aperture_center_y_m(self) -> float:
        """Décalage du centre de l'ouverture projetée par rapport à l'axe.

        `0.0` si `centered_aperture` (ouverture axisymétrique, front-fed) ;
        sinon `offset_clearance_m + diameter_m/2` (ouverture offset réelle).
        """
        if self.centered_aperture:
            return 0.0
        return self.offset_clearance_m + self.diameter_m / 2.0


class FeedSpec(BaseModel):
    """Réseau de feeds dans le plan focal : positions (m) et taper cos^q."""

    positions_m: list[tuple[float, float]]
    q: float = Field(..., ge=0.0)
    # Déplacement axial du plan des feeds (m), le long de l'axe focal
    # (>0 = feed reculé, éloigné du réflecteur). Pas de contrainte de signe.
    defocus_m: float = Field(0.0)

    @property
    def n_feeds(self) -> int:
        return len(self.positions_m)
