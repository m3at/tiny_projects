import json

import numpy as np
import pytest

from shodo.config import SimConfig
from shodo.contracts import ActionContract, SensorConfig, sensor_contract
from shodo.dataset import EpisodeRecorder
from shodo.vla_data import (
    PreparedDataset,
    prepare_dataset,
    preprocess_image,
    state_from_observation,
)


def native(path, char="一", first_camera=0.0, history=2, offset=0, action=0.5):
    sensors = SensorConfig(history=history)
    recorder = EpisodeRecorder(
        {
            "char": char,
            "sensors": sensors.to_dict(),
            "config": SimConfig().to_dict(),
            "observation_contract": sensor_contract(sensors),
            "action_contract": ActionContract().to_dict(),
            "attribution": "KanjiVG CC BY-SA 3.0",
        }
    )
    obs = np.tile(np.arange(39), history).astype(np.float32) + offset
    recorder.start(obs)
    recorder.add_camera_frame(np.full((2, 4, 3), 32, np.uint8), timestamp_s=first_camera)
    for index in range(3):
        recorder.append(
            np.full(6, -0.9),
            np.full(6, action + index * 0.1),
            obs + index + 1,
            timestamp_s=(index + 1) * 0.02,
            terminated=index == 2,
            privileged={"secret": [1e8]},
        )
    recorder.add_camera_frame(np.full((2, 4, 3), 128, np.uint8), timestamp_s=0.03)
    recorder.add_camera_frame(np.full((2, 4, 3), 255, np.uint8), timestamp_s=0.06)
    return recorder.save(path)


def test_prepared_roundtrip_alignment_stats_and_padding(tmp_path):
    paths = [native(tmp_path / "one.npz"), native(tmp_path / "two.npz", char="二", offset=10)]
    root = prepare_dataset(paths, tmp_path / "prepared", image_size=4)
    dataset = PreparedDataset(root, chunk_size=4)
    assert len(dataset) == 6
    assert isinstance(dataset.episodes[0][1]["images"], np.memmap)
    assert dataset.manifest["chars"] == ["一", "二"]
    assert dataset.manifest["episodes"][0]["attribution"] == "KanjiVG CC BY-SA 3.0"
    np.testing.assert_array_equal(dataset.episodes[0][1]["camera_index"], [0, 0, 1])
    assert dataset[0]["image"][0, 1, 0] == np.float32(32 / 255)
    assert dataset[2]["image"][0, 1, 0] == np.float32(128 / 255)
    assert dataset[0]["image"][0, 0, 0] == 0
    np.testing.assert_array_equal(dataset[2]["action_is_pad"], [False, True, True, True])
    np.testing.assert_array_equal(dataset[2]["actions"][1:], 0)
    np.testing.assert_allclose(
        dataset[2]["actions"][0] * dataset.action_std + dataset.action_mean, 0.7
    )
    states = np.stack([dataset[i]["state"] for i in range(len(dataset))])
    np.testing.assert_allclose(states.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(states.std(axis=0), 1, atol=1e-6)
    assert dataset[-1]["task"].startswith("Draw 二")
    with pytest.raises(IndexError):
        dataset[6]


def test_only_latest_sensor_features_are_selected():
    obs = np.r_[np.full(39, -999), np.arange(39)]
    state = state_from_observation(obs, 2)
    np.testing.assert_array_equal(state, np.r_[np.arange(25), np.arange(32, 39)])
    with pytest.raises(ValueError, match="sensor-history"):
        state_from_observation(np.zeros(40), 1)


def test_no_future_camera_and_failed_publication_leaves_no_output(tmp_path):
    path = native(tmp_path / "late.npz", first_camera=0.01)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="at or before"):
        prepare_dataset([path], output)
    assert not output.exists()


def test_no_overwrite_and_heldout_guard(tmp_path):
    path = native(tmp_path / "heldout.npz", char="永")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Held-out"):
        prepare_dataset([path], output)
    prepare_dataset([path], output, allow_heldout=True, image_size=4)
    assert PreparedDataset(output).manifest["allow_heldout"] is True
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_dataset([path], output, allow_heldout=True)
    assert (output / "manifest.json").read_bytes() == before


def test_contract_configuration_mismatch_and_action_bounds(tmp_path):
    a = native(tmp_path / "a.npz", history=1)
    b = native(tmp_path / "b.npz", history=2)
    with pytest.raises(ValueError, match="matching configurations"):
        prepare_dataset([a, b], tmp_path / "mixed", image_size=4)
    invalid = native(tmp_path / "invalid.npz", action=1.1)
    with pytest.raises(ValueError, match="bounds"):
        prepare_dataset([invalid], tmp_path / "invalid", image_size=4)


def test_reader_rejects_contract_and_camera_mapping_corruption(tmp_path):
    path = native(tmp_path / "a.npz")
    output = prepare_dataset([path], tmp_path / "prepared", image_size=4)
    indices = np.load(output / "episode-000000/camera_index.npy", mmap_mode="r+")
    indices[0] = 1
    indices.flush()
    with pytest.raises(ValueError, match="causal"):
        PreparedDataset(output)
    indices[0] = 0
    indices.flush()
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["state_contract"]["privileged_inputs"] = True
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="state contract"):
        PreparedDataset(output)


def test_image_dtype_and_resize_validation():
    with pytest.raises(ValueError, match="uint8"):
        preprocess_image(np.zeros((4, 4, 3)))
    with pytest.raises(ValueError, match="positive integer"):
        preprocess_image(np.zeros((4, 4, 3), np.uint8), 0)


@pytest.mark.parametrize("corruption", ["no_camera", "privileged", "action_frame"])
def test_invalid_native_inputs_rejected(tmp_path, corruption):
    path = native(tmp_path / "native.npz")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if corruption == "no_camera":
        arrays.pop("camera_frames")
        arrays.pop("camera_times_s")
    elif corruption == "privileged":
        metadata["observation_contract"]["privileged_inputs"] = True
    else:
        metadata["action_contract"]["frame"] = "tool"
    arrays["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="camera|contract"):
        prepare_dataset([path], tmp_path / "prepared", image_size=4)


def test_constant_features_use_finite_floor_and_duplicate_sources_rejected(tmp_path):
    path = native(tmp_path / "native.npz")
    with pytest.raises(ValueError, match="distinct"):
        prepare_dataset([path, path], tmp_path / "duplicate", image_size=4)
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    arrays["observations"][:] = 5
    arrays["applied_actions"][:] = 0
    np.savez_compressed(path, **arrays)
    dataset = PreparedDataset(prepare_dataset([path], tmp_path / "constant", image_size=4))
    np.testing.assert_array_equal(dataset[0]["state"], 0)
    np.testing.assert_array_equal(dataset[0]["actions"], 0)
    np.testing.assert_allclose(dataset.state_std, 1)
    np.testing.assert_allclose(dataset.action_std, 1e-6)


def test_constant_sensor_flags_and_age_have_bounded_unseen_events(tmp_path):
    path = native(tmp_path / "native.npz")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    arrays["observations"][:, -2] = 0  # nominal sensor age
    arrays["observations"][:, -1] = 1  # nominal fresh flag
    np.savez_compressed(path, **arrays)
    dataset = PreparedDataset(prepare_dataset([path], tmp_path / "prepared", image_size=4))
    assert dataset.manifest["version"] == 2
    nominal = state_from_observation(arrays["observations"][0], 2)
    dropped = nominal.copy()
    dropped[-2:] = [0.02, 0]
    normalized = (dropped - dataset.state_mean) / dataset.state_std
    np.testing.assert_allclose(normalized[-2:], [0.02, -1])


def test_legacy_manifest_keeps_recorded_normalization(tmp_path):
    path = native(tmp_path / "native.npz")
    root = prepare_dataset([path], tmp_path / "prepared", image_size=4)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = 1
    manifest.pop("normalization")
    manifest.pop("supervision")
    manifest["stats"]["state"]["std"][0] = 1e-6
    manifest_path.write_text(json.dumps(manifest))
    dataset = PreparedDataset(root)
    assert dataset.state_std[0] == np.float32(1e-6)
    manifest["stats_std_floor"] = 0.1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="normalization floor"):
        PreparedDataset(root)


def test_explicit_expert_targets_preserve_behavior_and_forbid_chunks(tmp_path):
    path = native(tmp_path / "native.npz")
    with pytest.raises(ValueError, match="counterfactual labels"):
        prepare_dataset([path], tmp_path / "missing", image_size=4, supervision="expert")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    behavior = arrays["applied_actions"].copy()
    arrays["privileged_expert_applied_action"] = -behavior
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["recovery_supervision"] = {
        "version": 1,
        "teacher": "privileged oracle",
        "noise_std": 0.08,
        "label_timing": "pre-action decision boundary",
        "action_semantics": "counterfactual effective clipped command increment; not executed robot displacement",
        "coherent_action_chunks": False,
    }
    arrays["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    root = prepare_dataset([path], tmp_path / "expert", image_size=4, supervision="expert")
    with pytest.raises(ValueError, match="chunk_size=1"):
        PreparedDataset(root, chunk_size=2)
    expert = PreparedDataset(root, chunk_size=1)
    np.testing.assert_allclose(
        expert[0]["actions"][0] * expert.action_std + expert.action_mean, -behavior[0]
    )
    np.testing.assert_allclose(expert.action_mean, -behavior.mean(axis=0))
    with np.load(path, allow_pickle=False) as source:
        np.testing.assert_array_equal(source["applied_actions"], behavior)
    applied = PreparedDataset(prepare_dataset([path], tmp_path / "applied", image_size=4))
    np.testing.assert_allclose(applied.action_mean, behavior.mean(axis=0))
