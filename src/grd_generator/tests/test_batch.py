import json
import re
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

import grd_generator.reflector.batch as batch_module
from grd_generator.reflector.batch import (
    BatchConfig,
    BatchOutputNotEmptyError,
    main,
    run_batch,
)
from grd_generator.reflector.metrics import EnvelopeMetrics, FeedMetrics, FrequencyMetrics

_INVENTORY_PATTERN = re.compile(r"^Tx_(?P<feed>\d+)_(?P<freq>\d+)_(?P<pol>[HV])P\.grd$")


def _config_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "set_id": "gen-ku-test",
        "generated_on": "2026-09-17",
        "reflector": {
            "diameter_m": 1.3,
            "f_over_d": 1.5,
            "offset_clearance_m": 0.2,
            "centered_aperture": True,
        },
        "feeds": {"n_feeds": 7, "pitch_m": 0.032, "q": 2.0, "defocus_m": 0.53},
        "frequencies_mhz": [10700, 11400],
        "grid": {"u_min": -0.18, "u_max": 0.18, "v_min": -0.18, "v_max": 0.18, "n_u": 9, "n_v": 9},
        "sampling": {"n_aperture": 32, "pad_factor": 4},
        "polarizations": {"H": "simulated", "V": "swap_of_H"},
        "hdf5": {"icomp": -3},
        "name_pattern": "Tx_{feed:04d}_{freq_mhz:d}_{pol}P.grd",
    }
    base.update(overrides)
    return base


def _config_text(**overrides: Any) -> str:
    return json.dumps(_config_dict(**overrides), indent=2)


def test_generator_version_matches_installed_package() -> None:
    from importlib.metadata import version

    assert version("grd_generator") == batch_module.GENERATOR_VERSION


def test_batch_config_parses_example() -> None:
    config = BatchConfig.model_validate_json(_config_text())
    assert config.set_id == "gen-ku-test"
    assert config.feeds.n_feeds == 7
    assert config.frequencies_mhz == [10700, 11400]


def test_batch_config_tolerates_unknown_extra_fields() -> None:
    # Un bloc `_provenance` figé après calibrage (spec, § Calibrage) doit être
    # accepté sans erreur (il est de toute façon recopié tel quel, pas relu).
    text = json.dumps({**_config_dict(), "_provenance": {"chosen": "whatever"}})
    BatchConfig.model_validate_json(text)


def test_batch_config_rejects_empty_frequencies() -> None:
    with pytest.raises(ValidationError):
        BatchConfig.model_validate_json(_config_text(frequencies_mhz=[]))


def test_batch_config_rejects_duplicate_frequencies() -> None:
    with pytest.raises(ValidationError):
        BatchConfig.model_validate_json(_config_text(frequencies_mhz=[11400, 11400]))


def test_batch_config_rejects_degenerate_grid_u() -> None:
    grid = _config_dict()["grid"] | {"u_min": 0.2, "u_max": 0.1}
    with pytest.raises(ValidationError):
        BatchConfig.model_validate_json(_config_text(grid=grid))


def test_batch_config_rejects_degenerate_grid_v() -> None:
    grid = _config_dict()["grid"] | {"v_min": 0.2, "v_max": 0.1}
    with pytest.raises(ValidationError):
        BatchConfig.model_validate_json(_config_text(grid=grid))


def test_batch_config_rejects_unknown_polarization_mode() -> None:
    with pytest.raises(ValidationError):
        BatchConfig.model_validate_json(
            _config_text(polarizations={"H": "measured", "V": "swap_of_H"})
        )


def test_batch_config_rejects_non_integer_frequency_mhz() -> None:
    with pytest.raises(ValidationError):
        BatchConfig.model_validate_json(_config_text(frequencies_mhz=[10700, 11400.5]))


def test_run_batch_writes_expected_tree_and_names(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config_text = _config_text()
    config = BatchConfig.model_validate_json(config_text)
    run_batch(config, out_dir, config_text=config_text)

    grd_files = sorted((out_dir / "grd").glob("*.grd"))
    # 7 feeds x 2 fréquences x 2 polarisations
    assert len(grd_files) == 7 * 2 * 2
    feeds_seen: set[int] = set()
    freqs_seen: set[int] = set()
    for path in grd_files:
        m = _INVENTORY_PATTERN.match(path.name)
        assert m is not None, path.name
        feeds_seen.add(int(m.group("feed")))
        freqs_seen.add(int(m.group("freq")))
        assert len(m.group("feed")) == 4  # zero-padded sur 4 chiffres
    assert feeds_seen == {1, 2, 3, 4, 5, 6, 7}
    assert freqs_seen == {10700, 11400}

    assert (out_dir / "meta" / "batch-config.json").read_text(encoding="utf-8") == config_text
    sim_params = json.loads((out_dir / "meta" / "sim_params.json").read_text(encoding="utf-8"))
    assert set(sim_params.keys()) == {"10700", "11400"}
    assert sim_params["11400"]["n_feeds"] == 7
    notice = (out_dir / "meta" / "NOTICE.md").read_text(encoding="utf-8")
    assert "SYNTHETIQUE" not in notice  # NOTICE V, pas le disclaimer HDF5 (hdf5_export)
    assert "jamais une excitation V réellement simulée" in notice

    metrics_json = json.loads((out_dir / "meta" / "metrics.json").read_text(encoding="utf-8"))
    assert [m["freq_mhz"] for m in metrics_json] == [10700, 11400]
    assert len(metrics_json[0]["feeds"]) == 7

    manifest_lines = (out_dir / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    # + batch-config/sim_params/metrics/NOTICE
    assert len(manifest_lines) == len(grd_files) + 4
    for line in manifest_lines:
        digest, _, rel_path = line.partition("  ")
        assert len(digest) == 64
        assert (out_dir / rel_path).is_file()


def test_run_batch_h_and_v_are_swapped_components(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config_text = _config_text()
    config = BatchConfig.model_validate_json(config_text)
    run_batch(config, out_dir, config_text=config_text)

    h_path = out_dir / "grd" / "Tx_0001_11400_HP.grd"
    v_path = out_dir / "grd" / "Tx_0001_11400_VP.grd"
    with h5py.File(h_path, "r") as hf, h5py.File(v_path, "r") as vf:
        h_data = hf["data"][0, 0, :, :, :]
        v_data = vf["data"][0, 0, :, :, :]
        np.testing.assert_array_equal(v_data[:, :, 0], h_data[:, :, 1])
        np.testing.assert_array_equal(v_data[:, :, 1], h_data[:, :, 0])


def test_run_batch_refuses_non_empty_output_dir_without_writing(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sentinel = out_dir / "already-here.txt"
    sentinel.write_text("do not touch")

    config_text = _config_text()
    config = BatchConfig.model_validate_json(config_text)
    with pytest.raises(BatchOutputNotEmptyError):
        run_batch(config, out_dir, config_text=config_text)

    assert list(out_dir.iterdir()) == [sentinel]  # rien d'autre écrit


def test_run_batch_accepts_existing_empty_output_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    config_text = _config_text()
    config = BatchConfig.model_validate_json(config_text)
    run_batch(config, out_dir, config_text=config_text)
    assert (out_dir / "MANIFEST.sha256").exists()


def test_run_batch_is_reproducible_manifest(tmp_path: Path) -> None:
    config_text = _config_text()
    config = BatchConfig.model_validate_json(config_text)
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    run_batch(config, out_a, config_text=config_text)
    time.sleep(1.1)  # traverse une seconde d'horloge système
    run_batch(config, out_b, config_text=config_text)

    manifest_a = (out_a / "MANIFEST.sha256").read_text(encoding="utf-8")
    manifest_b = (out_b / "MANIFEST.sha256").read_text(encoding="utf-8")
    assert manifest_a == manifest_b


def test_main_cli_runs_end_to_end(tmp_path: Path) -> None:
    config_path = tmp_path / "batch-ku.json"
    config_path.write_text(_config_text(), encoding="utf-8")
    out_dir = tmp_path / "out"
    code = main(["--config", str(config_path), "--out", str(out_dir)])
    assert code == 0
    assert (out_dir / "MANIFEST.sha256").exists()


def test_main_cli_rejects_invalid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "batch-ku.json"
    config_path.write_text(_config_text(frequencies_mhz=[]), encoding="utf-8")
    out_dir = tmp_path / "out"
    code = main(["--config", str(config_path), "--out", str(out_dir)])
    assert code == 1
    assert not out_dir.exists()


def test_main_cli_rejects_non_empty_output_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "batch-ku.json"
    config_path.write_text(_config_text(), encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "already-here.txt").write_text("x")
    code = main(["--config", str(config_path), "--out", str(out_dir)])
    assert code == 1


def _fake_metrics(radius_3db_deg: float) -> FrequencyMetrics:
    feed = FeedMetrics(
        feed=1, peak_dbi=40.0, peak_radius_deg=0.0, width_3db_deg=5.0, cross_pol_isolation_db=20.0
    )
    envelope = EnvelopeMetrics(
        max_dbi=44.0, radius_1db_deg=3.0, radius_3db_deg=radius_3db_deg, radius_39dbi_deg=5.0
    )
    return FrequencyMetrics(
        freq_mhz=11400,
        feeds=[feed],
        envelope=envelope,
        peak_dbi_median=40.0,
        width_3db_deg_median=5.0,
        peak_radius_deg_max=0.0,
    )


def test_main_cli_check_flag_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Le critère bloquant (rayon d'enveloppe à -3dB) est testé isolément dans
    # test_metrics.py ; ici on vérifie seulement le câblage --check du CLI,
    # en substituant compute_frequency_metrics pour ne pas dépendre d'un
    # calibrage physique réaliste dans ce test rapide.
    monkeypatch.setattr(
        batch_module, "compute_frequency_metrics", lambda *a, **kw: _fake_metrics(5.0)
    )
    config_path = tmp_path / "batch-ku.json"
    config_path.write_text(_config_text(frequencies_mhz=[11400]), encoding="utf-8")
    out_dir = tmp_path / "out"
    code = main(["--config", str(config_path), "--out", str(out_dir), "--check"])
    assert code == 0


def test_main_cli_check_flag_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        batch_module, "compute_frequency_metrics", lambda *a, **kw: _fake_metrics(1.0)
    )
    config_path = tmp_path / "batch-ku.json"
    config_path.write_text(_config_text(frequencies_mhz=[11400]), encoding="utf-8")
    out_dir = tmp_path / "out"
    code = main(["--config", str(config_path), "--out", str(out_dir), "--check"])
    assert code == 1


def test_run_batch_debug_export_writes_metrics_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEBUG_EXPORT", "true")
    out_dir = tmp_path / "out"
    config_text = _config_text(frequencies_mhz=[11400])
    config = BatchConfig.model_validate_json(config_text)
    run_batch(config, out_dir, config_text=config_text)

    exported = list((tmp_path / "debug_export").glob("*/metrics_11400.json"))
    assert len(exported) == 1
    payload = json.loads(exported[0].read_text(encoding="utf-8"))
    assert payload["freq_mhz"] == 11400


def test_run_batch_debug_export_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEBUG_EXPORT", raising=False)
    out_dir = tmp_path / "out"
    config_text = _config_text()
    config = BatchConfig.model_validate_json(config_text)
    run_batch(config, out_dir, config_text=config_text)
    assert not (tmp_path / "debug_export").exists()
