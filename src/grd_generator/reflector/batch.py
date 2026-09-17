"""`grd-batch` : jeu multi-fréquence HDF5, structure de type TICRA Tools 25.0
(grille sphérique uv) (lot B2).

Génère, sans interface graphique et de façon rejouable, un jeu de diagrammes
HDF5 (grille (u, v), une famille de fichiers par fréquence, H simulé + V
fabriqué par échange de composantes — voir `hdf5_export.py`) à partir d'une
configuration `BatchConfig`. Voir
`docs/superpowers/specs/2026-09-17-batch-ku-hdf5-design.md`, sections
« Conteneur HDF5 » et « Commande `grd-batch` ».

Usage :
    uv run grd-batch --config batch-ku.json --out <dossier>
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from grd_generator.logger import configure_logging, logger
from grd_generator.reflector.grd_export import simulation_params_dict
from grd_generator.reflector.hdf5_export import farfield_name, swap_components, write_grd_hdf5
from grd_generator.reflector.metrics import FrequencyMetrics, compute_frequency_metrics
from grd_generator.reflector.spec import FeedSpec, ReflectorSpec
from grd_generator.reflector.synth_afr import hex_feed_positions, synthesize_reflector_fields
from grd_generator.schemas import UVGrid

# Version du générateur, écrite dans `/info` de chaque fichier HDF5 (voir
# `hdf5_export.write_grd_hdf5`) et dans `meta/NOTICE.md` — dérivée de la version
# du paquet installé (`pyproject.toml`), jamais dupliquée en dur ici.
GENERATOR_VERSION = _package_version("grd_generator")

# Rayon (m) de la zone où chercher des centres de feed hexagonaux, très généreux
# devant le pas du réseau : même convention que `gui.build_reflector_result`
# (`focal_radius_m=10.0 * pitch_m`), qui contient largement les 169 feeds max
# envisagés par le calibrage (§ Décisions validées de la spec).
_FOCAL_RADIUS_PITCH_FACTOR = 10.0

_HZ_PER_MHZ = 1.0e6
_MHZ_PER_GHZ = 1000.0

_NOTICE_TEMPLATE = """\
# NOTICE — polarisation V synthétique (`{set_id}`)

Le fichier V de chaque (feed, fréquence) est **fabriqué**, pas mesuré : une
seule excitation est simulée (feed polarisé x) ; `V[..., 0] = H[..., 1]` et
`V[..., 1] = H[..., 0]` (échange exact des deux composantes de champ, mêmes
octets flottants, aucune reconversion — `hdf5_export.swap_components`), dans
la même base fixe (`E_Lx`/`E_Ly`) que le fichier H.

## Ce que ce n'est PAS

**V n'est jamais une excitation V réellement simulée.** C'est une
approximation de symétrie : la réponse croisée mesurée pour l'excitation H
est supposée approcher la réponse co-polaire qu'aurait une excitation V dans
la même base fixe — plausible pour une antenne bien polarisée, non vérifiée
indépendamment ici. Ne rien conclure de ce jeu sur la polarisation croisée
réelle de l'antenne modélisée, ni sur sa discrimination H/V effective.

## Provenance

Généré par `grd-batch` (`grd_generator {generator_version}`,
`reflector/batch.py`) depuis `meta/batch-config.json` (copie exacte de la
configuration utilisée). Voir
`docs/superpowers/specs/2026-09-17-batch-ku-hdf5-design.md`.
"""


class BatchOutputNotEmptyError(Exception):
    """Le dossier de sortie existe déjà et n'est pas vide."""


class ReflectorConfig(BaseModel):
    """Réflecteur (constant sur toutes les fréquences du batch)."""

    diameter_m: float = Field(..., gt=0)
    f_over_d: float = Field(..., gt=0)
    offset_clearance_m: float = Field(0.0, ge=0)
    centered_aperture: bool = False


class FeedsConfig(BaseModel):
    """Réseau de feeds (constant sur toutes les fréquences du batch).

    `n_feeds` est un paramètre de calibrage comme les autres (voir spec,
    § Métriques et calibrage) : la maille hexagonale (`hex_feed_positions`)
    retient les `n_feeds` centres les plus proches du centre de la zone,
    quel que soit `n_feeds`. Pour un **nombre hexagonal centré** exact
    (1, 7, 19, 37, 61, 91, 127, 169, … = 3k² − 3k + 1), ce sont exactement les
    `k` anneaux complets de la maille : l'amas obtenu est symétrique par
    rotation de 60° et par tout axe miroir passant par le centre (donc en
    particulier par v → −v, la symétrie invoquée au point ouvert n°1 de la
    spec). Pour toute autre valeur, le dernier anneau est tronqué : les points
    retenus parmi les ex-æquo à distance égale du centre dépendent de l'ordre
    d'énumération de `hex_centers_uv` (balayage ligne par ligne), pas d'un
    critère de symétrie — l'amas résultant n'est alors garanti ni symétrique
    par v → −v, ni par rotation. Accepté par ce schéma (aucune contrainte de
    valeur), mais toute valeur ≠ {91, 127, 169} doit être présentée avec sa
    figure de calibrage et n'est retenue qu'après accord (spec, § Calibrage).
    """

    n_feeds: int = Field(..., gt=0)
    pitch_m: float = Field(..., gt=0)
    q: float = Field(..., ge=0.0)
    defocus_m: float = 0.0


class GridConfig(BaseModel):
    """Grille (u, v) de sortie, en cosinus directeurs."""

    u_min: float
    u_max: float
    v_min: float
    v_max: float
    n_u: int = Field(..., gt=1)
    n_v: int = Field(..., gt=1)

    @model_validator(mode="after")
    def _check_not_degenerate(self) -> "GridConfig":
        if self.u_max <= self.u_min:
            raise ValueError(f"grille dégénérée : u_max ({self.u_max}) <= u_min ({self.u_min})")
        if self.v_max <= self.v_min:
            raise ValueError(f"grille dégénérée : v_max ({self.v_max}) <= v_min ({self.v_min})")
        return self

    def to_uv_grid(self) -> UVGrid:
        return UVGrid(
            u_min=self.u_min,
            u_max=self.u_max,
            v_min=self.v_min,
            v_max=self.v_max,
            n_u=self.n_u,
            n_v=self.n_v,
        )


class SamplingConfig(BaseModel):
    """Résolution de la synthèse AFR (voir `synthesize_reflector_fields`)."""

    n_aperture: int = Field(..., gt=0)
    pad_factor: int = Field(..., gt=0)


class PolarizationsConfig(BaseModel):
    """Mode de production de chaque polarisation.

    Un seul mode est supporté par lot (voir spec, § Décisions validées) : H
    simulé, V fabriqué par échange. Toute autre valeur (`Literal`) est un
    refus pydantic — « mode de polarisation inconnu ».
    """

    H: Literal["simulated"] = "simulated"
    V: Literal["swap_of_H"] = "swap_of_H"


class Hdf5Config(BaseModel):
    icomp: int = -3


class BatchConfig(BaseModel):
    """Configuration `grd-batch` — voir spec, section « Commande `grd-batch` ».

    Recopiée telle quelle (octets bruts, pas re-sérialisée) dans
    `meta/batch-config.json` de la sortie : c'est la provenance du jeu.
    Les champs additionnels (ex. un bloc `_provenance` figé après calibrage,
    voir spec § Calibrage) sont ignorés par ce schéma (comportement par
    défaut de pydantic) mais conservés dans la copie recopiée.
    """

    set_id: str
    generated_on: str
    reflector: ReflectorConfig
    feeds: FeedsConfig
    frequencies_mhz: list[int] = Field(..., min_length=1)
    grid: GridConfig
    sampling: SamplingConfig
    polarizations: PolarizationsConfig = PolarizationsConfig()
    hdf5: Hdf5Config = Hdf5Config()
    name_pattern: str = "Tx_{feed:04d}_{freq_mhz:d}_{pol}P.grd"

    @field_validator("frequencies_mhz")
    @classmethod
    def _check_frequencies(cls, value: list[int]) -> list[int]:
        # La liste vide est déjà refusée par `Field(min_length=1)` ci-dessus.
        if len(set(value)) != len(value):
            raise ValueError("frequencies_mhz contient des fréquences dupliquées")
        return value

    def file_name(self, *, feed: int, freq_mhz: int, pol: Literal["H", "V"]) -> str:
        return self.name_pattern.format(feed=feed, freq_mhz=freq_mhz, pol=pol)


def _feed_positions(cfg: BatchConfig) -> list[tuple[float, float]]:
    return hex_feed_positions(
        cfg.feeds.pitch_m,
        cfg.feeds.n_feeds,
        focal_radius_m=_FOCAL_RADIUS_PITCH_FACTOR * cfg.feeds.pitch_m,
    )


def _zone_radius_deg(grid: GridConfig) -> float:
    """Rayon angulaire (deg) informatif couvert par la grille (bord le plus proche
    du centre), pour `simulation_params_dict` — pas une contrainte du batch."""
    half_width = min(abs(grid.u_min), abs(grid.u_max), abs(grid.v_min), abs(grid.v_max))
    return math.degrees(math.asin(min(1.0, half_width)))


def _sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _maybe_export(export_dir: Path | None, name: str, data: object) -> None:
    """Écrit `data` en JSON sous `export_dir/<name>.json`, seulement si
    `DEBUG_EXPORT=true` (voir `export_dir`, calculé une fois par `run_batch`).
    Zéro coût hors debug : `export_dir` vaut `None` par défaut."""
    if export_dir is None:
        return
    export_dir.mkdir(parents=True, exist_ok=True)
    target = export_dir / f"{name}.json"
    target.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.debug("debug_export_written", file=str(target))


def run_batch(config: BatchConfig, out_dir: Path, *, config_text: str) -> list[FrequencyMetrics]:
    """Génère le jeu complet dans `out_dir` — échoue sans rien écrire si le
    dossier existe et n'est pas vide."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise BatchOutputNotEmptyError(f"dossier de sortie non vide : {out_dir}")

    logger.info(
        "batch_start",
        set_id=config.set_id,
        n_feeds=config.feeds.n_feeds,
        n_frequencies=len(config.frequencies_mhz),
        out_dir=str(out_dir),
    )

    grd_dir = out_dir / "grd"
    meta_dir = out_dir / "meta"
    grd_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    positions = _feed_positions(config)
    feeds = FeedSpec(positions_m=positions, q=config.feeds.q, defocus_m=config.feeds.defocus_m)
    grid = config.grid.to_uv_grid()
    written_paths: list[Path] = []
    sim_params: dict[str, object] = {}
    all_metrics: list[FrequencyMetrics] = []
    zone_radius_deg = _zone_radius_deg(config.grid)
    export_dir = (
        Path("debug_export") / datetime.now().strftime("%Y%m%d_%H%M%S")
        if os.getenv("DEBUG_EXPORT") == "true"
        else None
    )

    for freq_mhz in config.frequencies_mhz:
        start = time.monotonic()
        freq_hz = freq_mhz * _HZ_PER_MHZ
        spec = ReflectorSpec(
            diameter_m=config.reflector.diameter_m,
            focal_length_m=config.reflector.f_over_d * config.reflector.diameter_m,
            offset_clearance_m=config.reflector.offset_clearance_m,
            freq_hz=freq_hz,
            centered_aperture=config.reflector.centered_aperture,
        )
        co_fields, cross_fields = synthesize_reflector_fields(
            spec,
            feeds,
            grid,
            n_aperture=config.sampling.n_aperture,
            pad_factor=config.sampling.pad_factor,
        )
        for i, (co, cross) in enumerate(zip(co_fields, cross_fields, strict=True)):
            feed = i + 1
            h_path = grd_dir / config.file_name(feed=feed, freq_mhz=freq_mhz, pol="H")
            write_grd_hdf5(
                h_path,
                component0=co,
                component1=cross,
                grid=grid,
                freq_hz=freq_hz,
                name=farfield_name(config.set_id, "H"),
                generator_version=GENERATOR_VERSION,
                generated_on=config.generated_on,
                icomp=config.hdf5.icomp,
            )
            written_paths.append(h_path)

            comp0_v, comp1_v = swap_components(co, cross)
            v_path = grd_dir / config.file_name(feed=feed, freq_mhz=freq_mhz, pol="V")
            write_grd_hdf5(
                v_path,
                component0=comp0_v,
                component1=comp1_v,
                grid=grid,
                freq_hz=freq_hz,
                name=farfield_name(config.set_id, "V"),
                generator_version=GENERATOR_VERSION,
                generated_on=config.generated_on,
                icomp=config.hdf5.icomp,
            )
            written_paths.append(v_path)

        sim_params[str(freq_mhz)] = simulation_params_dict(
            diameter_m=config.reflector.diameter_m,
            f_over_d=config.reflector.f_over_d,
            offset_clearance_m=config.reflector.offset_clearance_m,
            freq_ghz=freq_mhz / _MHZ_PER_GHZ,
            q=config.feeds.q,
            pitch_m=config.feeds.pitch_m,
            n_feeds=config.feeds.n_feeds,
            zone_radius_deg=zone_radius_deg,
            defocus_m=config.feeds.defocus_m,
            centered_aperture=config.reflector.centered_aperture,
        )

        freq_metrics = compute_frequency_metrics(freq_mhz, co_fields, cross_fields, grid)
        all_metrics.append(freq_metrics)
        _maybe_export(export_dir, f"metrics_{freq_mhz}", freq_metrics.model_dump())

        duration_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "frequency_done",
            freq_mhz=freq_mhz,
            n_files=2 * len(co_fields),
            duration_ms=round(duration_ms, 1),
        )

    (meta_dir / "batch-config.json").write_text(config_text, encoding="utf-8")
    written_paths.append(meta_dir / "batch-config.json")

    sim_params_path = meta_dir / "sim_params.json"
    sim_params_path.write_text(
        json.dumps(sim_params, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    written_paths.append(sim_params_path)

    metrics_path = meta_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps([fm.model_dump() for fm in all_metrics], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    written_paths.append(metrics_path)

    notice_path = meta_dir / "NOTICE.md"
    notice_path.write_text(
        _NOTICE_TEMPLATE.format(set_id=config.set_id, generator_version=GENERATOR_VERSION),
        encoding="utf-8",
    )
    written_paths.append(notice_path)

    manifest_lines = [
        f"{_sha256_hex(path)}  {path.relative_to(out_dir).as_posix()}"
        for path in sorted(written_paths, key=lambda p: p.relative_to(out_dir).as_posix())
    ]
    (out_dir / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    logger.info(
        "batch_done",
        set_id=config.set_id,
        n_files=len(written_paths),
        out_dir=str(out_dir),
    )
    return all_metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="fichier BatchConfig JSON")
    parser.add_argument("--out", type=Path, required=True, help="dossier de sortie")
    parser.add_argument(
        "--check",
        action="store_true",
        help="code retour 1 si le critère bloquant (rayon d'enveloppe à -3 dB) "
        "n'est pas tenu à une fréquence (le jeu est écrit dans tous les cas)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)

    try:
        config_text = args.config.read_text(encoding="utf-8")
        config = BatchConfig.model_validate_json(config_text)
    except (OSError, ValueError) as exc:
        logger.error("batch_config_invalid", config=str(args.config), error=str(exc))
        return 1

    try:
        all_metrics = run_batch(config, args.out, config_text=config_text)
    except BatchOutputNotEmptyError as exc:
        logger.error("batch_output_not_empty", out_dir=str(args.out), error=str(exc))
        return 1

    if args.check:
        failing = [fm.freq_mhz for fm in all_metrics if not fm.passes_blocking_criterion()]
        if failing:
            logger.error("batch_check_failed", failing_freq_mhz=failing)
            return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
