"""Configuration centralisée du logging structuré (loguru, sortie JSON)."""

import sys

from loguru import logger

_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Configure loguru pour émettre des logs JSON structurés sur stderr.

    À appeler une seule fois depuis un point d'entrée (CLI, notebook, exemple).
    Idempotent : les appels suivants sont ignorés.
    """
    global _configured
    if _configured:
        return
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time} {level} {message}",
        serialize=True,
    )
    _configured = True


__all__ = ["logger", "configure_logging"]
