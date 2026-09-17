"""Export HDF5 des patterns AFR (co + cross), structure de type TICRA Tools 25.0
(grille sphérique uv).

Reproduit la structure d'un jeu de référence au même format : `/data` complex128
de forme `(1, 1, n_v, n_u, 2)`, `/parameters/{frequency,beams,u,v,polarisation}`,
attributs `/object` et `/tt_display_hints`. Voir
`docs/superpowers/specs/2026-09-17-batch-ku-hdf5-design.md`, section
« Conteneur HDF5 », pour le tableau complet.

Deux différences volontaires avec un jeu produit par l'outillage d'origine :
- **honnêteté de la provenance** : `/info` ne porte pas les attributs propres au
  producteur d'origine (licence, numéro de série) ; `comment`/`timestamp` disent
  explicitement « SYNTHETIQUE ». `generated_on` est fourni par l'appelant (config,
  pas l'horloge) pour que deux exécutions produisent des fichiers identiques octet
  pour octet.
- **polarisation V par échange** (`swap_components`) : une seule excitation est
  simulée (feed x-polarisé) ; le fichier V est fabriqué en échangeant les deux
  composantes du fichier H, sans resimuler quoi que ce soit.
"""

from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from grd_generator.schemas import ComplexField, UVGrid

_POLARISATION_LABELS = ("E_Lx", "E_Ly")
_HZ_PER_GHZ = 1.0e9


def farfield_name(set_id: str, pol: Literal["H", "V"]) -> str:
    """Nom `/object@name` (structure de type TICRA Tools 25.0) :
    `FarField_<set_id>_HComb`/`_VComb`."""
    return f"FarField_{set_id}_{pol}Comb"


def swap_components(
    component0: ComplexField, component1: ComplexField
) -> tuple[ComplexField, ComplexField]:
    """Échange exact des deux composantes : fabrique (comp0, comp1) du fichier V
    à partir de (comp0, comp1) = (co, cross) du fichier H, sans recalcul."""
    return component1, component0


def write_grd_hdf5(
    path: Path,
    *,
    component0: ComplexField,
    component1: ComplexField,
    grid: UVGrid,
    freq_hz: float,
    name: str,
    generator_version: str,
    generated_on: str,
    icomp: int = -3,
) -> None:
    """Écrit un fichier `.grd` HDF5, structure de type TICRA Tools 25.0.

    `component0`/`component1` remplissent `/data[..., 0]`/`[..., 1]` tels quels :
    pour un fichier H, ce sont `(co, cross)` (champs bruts de
    `synthesize_reflector_fields`) ; pour un fichier V, `swap_components(co, cross)`.
    `name` est le `/object@name` complet (voir `farfield_name`).
    """
    u_axis, v_axis = grid.axes()
    n_v, n_u = grid.n_v, grid.n_u
    data = np.empty((1, 1, n_v, n_u, 2), dtype=np.complex128)
    data[0, 0, :, :, 0] = component0
    data[0, 0, :, :, 1] = component1

    path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(path, "w", libver="earliest") as h5:
        h5.create_dataset("data", data=data, track_times=False)
        h5.create_dataset(
            "parameterlist",
            data=np.array(["frequency", "beams", "v", "u", "polarisation"], dtype=object),
            dtype=str_dtype,
            track_times=False,
        )

        parameters = h5.create_group("parameters", track_order=False)
        freq_ds = parameters.create_dataset(
            "frequency",
            data=np.array([freq_hz / _HZ_PER_GHZ], dtype=np.float64),
            track_times=False,
        )
        freq_ds.attrs["unit"] = "GHz"
        beams_ds = parameters.create_dataset(
            "beams", data=np.array([""], dtype=object), dtype=str_dtype, track_times=False
        )
        beams_ds.attrs["num_named_beams"] = 0
        u_ds = parameters.create_dataset(
            "u", data=u_axis.astype(np.float64), track_times=False
        )
        u_ds.attrs["unit"] = ""
        v_ds = parameters.create_dataset(
            "v", data=v_axis.astype(np.float64), track_times=False
        )
        v_ds.attrs["unit"] = ""
        pol_ds = parameters.create_dataset(
            "polarisation",
            data=np.array(_POLARISATION_LABELS, dtype=object),
            dtype=str_dtype,
            track_times=False,
        )
        pol_ds.attrs["unit"] = np.array(["N/A", "N/A"], dtype=object)

        obj = h5.create_group("object", track_order=False)
        obj.attrs["class"] = "spherical_grid"
        obj.attrs["grid_type"] = "uv"
        obj.attrs["igrid"] = 1
        obj.attrs["ncomp"] = 2
        obj.attrs["icomp"] = icomp
        obj.attrs["polarisation"] = "linear"
        obj.attrs["field_region"] = "far"
        obj.attrs["field_type"] = "E field"
        obj.attrs["field_distance"] = float("inf")
        obj.attrs["truncation"] = "rectangular"
        obj.attrs["beam_centres"] = np.array([[0, 0]], dtype=np.int32)
        obj.attrs["name"] = name

        hints = h5.create_group("tt_display_hints", track_order=False)
        hints.attrs["type"] = "Grid:Spherical"
        hints.attrs["x_axis"] = "u"
        hints.attrs["y_axis"] = "v"
        hints.attrs["data_parameter"] = "polarisation"
        hints.attrs["db_factor"] = 20.0

        # /info écrit en dernier : aucun attribut `licensee`/`serial_number` (ceux du
        # fournisseur réel) — provenance honnête, cf. docstring du module.
        info = h5.create_group("info", track_order=False)
        info.attrs["version"] = 2
        info.attrs["timestamp"] = (
            f"Generated by grd_generator {generator_version} on {generated_on}"
        )
        info.attrs["comment"] = (
            f"SYNTHETIQUE - grd_generator {generator_version}, structure TICRA Tools 25.0"
        )


__all__ = [
    "farfield_name",
    "swap_components",
    "write_grd_hdf5",
]
