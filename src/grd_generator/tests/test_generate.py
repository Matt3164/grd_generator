import numpy as np

from grd_generator.generate import auto_spacing, generate_reference


def test_auto_spacing_positive() -> None:
    assert auto_spacing(6.0, 80) > 0


def test_generate_reference_writes_valid_npz(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = generate_reference(tmp_path / "ref.npz")
    assert out.exists()
    data = np.load(out, allow_pickle=False)
    # Grid + 80 fields with the right shape and metadata.
    assert int(data["n_u"]) == 161 and int(data["n_v"]) == 161
    assert data["fields"].shape == (80, 161, 161)
    assert data["fields"].dtype == np.complex128
    assert data["antenna_ids"].shape == (80,)
    assert data["centers_uv"].shape == (80, 2)
    assert data["sigmas"].shape == (80,)
    assert str(data["mode"]) == "gaussian"


def test_generate_reference_airy_mode(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = generate_reference(tmp_path / "airy.npz", mode="airy")
    data = np.load(out, allow_pickle=False)
    assert str(data["mode"]) == "airy"
    assert data["fields"].shape == (80, 161, 161)
