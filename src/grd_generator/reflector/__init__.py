"""Modèle physique array-fed reflector (offset paraboloïde + feeds cos^q)."""

from grd_generator.reflector.spec import C_LIGHT, FeedSpec, ReflectorSpec
from grd_generator.reflector.synth_afr import (
    beamform_weights,
    dereference_phase,
    directivity_barycenters,
    form_beam,
    hex_feed_positions,
    synthesize_reflector_fields,
)
from grd_generator.reflector.zone import ServiceZone

__all__ = [
    "C_LIGHT",
    "ReflectorSpec",
    "FeedSpec",
    "ServiceZone",
    "beamform_weights",
    "dereference_phase",
    "directivity_barycenters",
    "form_beam",
    "hex_feed_positions",
    "synthesize_reflector_fields",
]
