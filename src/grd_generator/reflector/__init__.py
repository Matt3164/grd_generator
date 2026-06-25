"""Modèle physique array-fed reflector (offset paraboloïde + feeds cos^q)."""

from grd_generator.reflector.spec import C_LIGHT, FeedSpec, ReflectorSpec
from grd_generator.reflector.synth_afr import (
    hex_feed_positions,
    synthesize_reflector_fields,
)
from grd_generator.reflector.zone import ServiceZone

__all__ = [
    "C_LIGHT",
    "ReflectorSpec",
    "FeedSpec",
    "ServiceZone",
    "hex_feed_positions",
    "synthesize_reflector_fields",
]
