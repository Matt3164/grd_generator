import time
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from grd_generator.reflector.hdf5_export import (  # noqa: E402
    farfield_name,
    swap_components,
    write_grd_hdf5,
)
from grd_generator.schemas import ComplexField, UVGrid  # noqa: E402


def _grid() -> UVGrid:
    # Grille rectangulaire (n_u != n_v) volontaire : anti-transposition (une
    # grille carrée ne détecterait pas une transposition accidentelle des axes).
    return UVGrid(u_min=-0.1, u_max=0.1, v_min=-0.05, v_max=0.05, n_u=5, n_v=3)


def _asymmetric_fields(grid: UVGrid) -> tuple[ComplexField, ComplexField]:
    """Champ co/cross non symétrique : `f(i, j) != f(j, i)`, pour détecter une
    transposition (i, j) accidentelle des axes u/v."""
    co = np.zeros((grid.n_v, grid.n_u), dtype=np.complex128)
    for iv in range(grid.n_v):
        for iu in range(grid.n_u):
            co[iv, iu] = complex(iv * 10 + iu, -(iv + 1) * (iu + 1))
    cross = co * 0.01j + 0.001
    return co, cross


def test_write_grd_hdf5_structure_matches_ticra25(tmp_path: Path) -> None:
    grid = _grid()
    co, cross = _asymmetric_fields(grid)
    path = tmp_path / "Tx_0001_11400_HP.grd"
    write_grd_hdf5(
        path,
        component0=co,
        component1=cross,
        grid=grid,
        freq_hz=11.4e9,
        name=farfield_name("gen-ku-91f", "H"),
        generator_version="0.2.0",
        generated_on="2026-09-17",
        icomp=-3,
    )
    with h5py.File(path, "r") as h5:
        assert h5["data"].shape == (1, 1, grid.n_v, grid.n_u, 2)
        assert h5["data"].dtype == np.complex128
        assert h5["data"].compression is None  # non compressé
        assert h5["data"].chunks is None  # contigu

        parameterlist = [s.decode() for s in h5["parameterlist"][:]]
        assert parameterlist == ["frequency", "beams", "v", "u", "polarisation"]

        freq_ds = h5["parameters/frequency"]
        np.testing.assert_allclose(freq_ds[:], [11.4])
        assert freq_ds.attrs["unit"] == "GHz"

        beams_ds = h5["parameters/beams"]
        assert [s.decode() for s in beams_ds[:]] == [""]
        assert int(beams_ds.attrs["num_named_beams"]) == 0

        u_ds = h5["parameters/u"]
        v_ds = h5["parameters/v"]
        np.testing.assert_allclose(u_ds[:], grid.axes()[0])
        np.testing.assert_allclose(v_ds[:], grid.axes()[1])
        assert u_ds.attrs["unit"] == ""
        assert v_ds.attrs["unit"] == ""

        pol_ds = h5["parameters/polarisation"]
        assert [s.decode() for s in pol_ds[:]] == ["E_Lx", "E_Ly"]
        unit_attr = [s.decode() if isinstance(s, bytes) else s for s in pol_ds.attrs["unit"]]
        assert unit_attr == ["N/A", "N/A"]

        obj = h5["object"]
        assert obj.attrs["class"] == "spherical_grid"
        assert obj.attrs["grid_type"] == "uv"
        assert int(obj.attrs["igrid"]) == 1
        assert int(obj.attrs["ncomp"]) == 2
        assert int(obj.attrs["icomp"]) == -3
        assert obj.attrs["polarisation"] == "linear"
        assert obj.attrs["field_region"] == "far"
        assert obj.attrs["field_type"] == "E field"
        assert obj.attrs["field_distance"] == float("inf")
        assert obj.attrs["truncation"] == "rectangular"
        np.testing.assert_array_equal(obj.attrs["beam_centres"], [[0, 0]])
        assert obj.attrs["name"] == "FarField_gen-ku-91f_HComb"

        hints = h5["tt_display_hints"]
        assert hints.attrs["type"] == "Grid:Spherical"
        assert hints.attrs["x_axis"] == "u"
        assert hints.attrs["y_axis"] == "v"
        assert hints.attrs["data_parameter"] == "polarisation"
        assert float(hints.attrs["db_factor"]) == 20.0

        info = h5["info"]
        assert int(info.attrs["version"]) == 2
        assert "0.2.0" in info.attrs["timestamp"]
        assert "2026-09-17" in info.attrs["timestamp"]
        assert "SYNTHETIQUE" in info.attrs["comment"]
        assert "licensee" not in info.attrs
        assert "serial_number" not in info.attrs


def test_write_grd_hdf5_icomp_configurable(tmp_path: Path) -> None:
    grid = _grid()
    co, cross = _asymmetric_fields(grid)
    path = tmp_path / "Tx_0001_11400_HP.grd"
    write_grd_hdf5(
        path,
        component0=co,
        component1=cross,
        grid=grid,
        freq_hz=11.4e9,
        name=farfield_name("gen-ku-91f", "H"),
        generator_version="0.2.0",
        generated_on="2026-09-17",
        icomp=3,
    )
    with h5py.File(path, "r") as h5:
        assert int(h5["object"].attrs["icomp"]) == 3


def test_write_grd_hdf5_axis_order_no_transposition(tmp_path: Path) -> None:
    """`data[0, 0, iv, iu, c]` : aucune transposition x/y (grille rectangulaire,
    champ asymétrique en (iv, iu))."""
    grid = _grid()
    co, cross = _asymmetric_fields(grid)
    path = tmp_path / "Tx_0001_11400_HP.grd"
    write_grd_hdf5(
        path,
        component0=co,
        component1=cross,
        grid=grid,
        freq_hz=11.4e9,
        name=farfield_name("gen-ku-91f", "H"),
        generator_version="0.2.0",
        generated_on="2026-09-17",
    )
    with h5py.File(path, "r") as h5:
        data = h5["data"][0, 0, :, :, :]
        np.testing.assert_allclose(data[:, :, 0], co)
        np.testing.assert_allclose(data[:, :, 1], cross)


def test_swap_components_is_exact_exchange() -> None:
    grid = _grid()
    co, cross = _asymmetric_fields(grid)
    comp0_v, comp1_v = swap_components(co, cross)
    np.testing.assert_array_equal(comp0_v, cross)
    np.testing.assert_array_equal(comp1_v, co)


def test_h_and_v_files_are_exact_component_exchange(tmp_path: Path) -> None:
    grid = _grid()
    co, cross = _asymmetric_fields(grid)
    h_path = tmp_path / "Tx_0001_11400_HP.grd"
    v_path = tmp_path / "Tx_0001_11400_VP.grd"
    write_grd_hdf5(
        h_path,
        component0=co,
        component1=cross,
        grid=grid,
        freq_hz=11.4e9,
        name=farfield_name("gen-ku-91f", "H"),
        generator_version="0.2.0",
        generated_on="2026-09-17",
    )
    comp0_v, comp1_v = swap_components(co, cross)
    write_grd_hdf5(
        v_path,
        component0=comp0_v,
        component1=comp1_v,
        grid=grid,
        freq_hz=11.4e9,
        name=farfield_name("gen-ku-91f", "V"),
        generator_version="0.2.0",
        generated_on="2026-09-17",
    )
    with h5py.File(h_path, "r") as hf, h5py.File(v_path, "r") as vf:
        h_data = hf["data"][0, 0, :, :, :]
        v_data = vf["data"][0, 0, :, :, :]
        np.testing.assert_array_equal(v_data[:, :, 0], h_data[:, :, 1])
        np.testing.assert_array_equal(v_data[:, :, 1], h_data[:, :, 0])
        assert hf["object"].attrs["name"] != vf["object"].attrs["name"]
        assert hf["object"].attrs["name"] == "FarField_gen-ku-91f_HComb"
        assert vf["object"].attrs["name"] == "FarField_gen-ku-91f_VComb"


def test_farfield_name_pattern() -> None:
    assert farfield_name("gen-ku-91f", "H") == "FarField_gen-ku-91f_HComb"
    assert farfield_name("gen-ku-91f", "V") == "FarField_gen-ku-91f_VComb"


def test_write_grd_hdf5_is_byte_for_byte_reproducible(tmp_path: Path) -> None:
    """Deux écritures de la même configuration (même `generated_on`, fourni par
    l'appelant) produisent des octets identiques — pas d'horodatage d'horloge."""
    grid = _grid()
    co, cross = _asymmetric_fields(grid)
    path_a = tmp_path / "a.grd"
    path_b = tmp_path / "b.grd"
    common: dict[str, object] = {
        "component0": co,
        "component1": cross,
        "grid": grid,
        "freq_hz": 11.4e9,
        "name": farfield_name("gen-ku-91f", "H"),
        "generator_version": "0.2.0",
        "generated_on": "2026-09-17",
    }
    write_grd_hdf5(path_a, **common)  # type: ignore[arg-type]
    time.sleep(1.1)  # traverse une seconde d'horloge système
    write_grd_hdf5(path_b, **common)  # type: ignore[arg-type]
    assert path_a.read_bytes() == path_b.read_bytes()
