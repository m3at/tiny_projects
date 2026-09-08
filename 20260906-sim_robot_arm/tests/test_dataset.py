import json

import numpy as np
import pytest

from shodo.dataset import EpisodeRecorder, _numeric, load_episode


def test_numeric_validation_can_borrow_without_weakening_ownership_or_checks():
    array = np.arange(12.0).reshape(3, 4)
    owned = _numeric(array, "owned")
    borrowed = _numeric(array, "borrowed", copy=False)
    assert borrowed is array
    assert not np.shares_memory(owned, array)
    array[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _numeric(array, "invalid", copy=False)
    assert np.isfinite(owned).all()


def episode():
    recorder = EpisodeRecorder(
        {"observation_contract": "test", "units": {"position": "m"}, "provenance": {"seed": 7}}
    )
    recorder.start(np.array([1, 2], dtype=np.float32), timestamp_s=1)
    recorder.append(
        [2.0], [1.0], [3.0, 4.0], timestamp_s=1.02, reward=0.25, privileged={"force": [0, 0, 1]}
    )
    recorder.append(
        [-2.0],
        [-1.0],
        [5.0, 6.0],
        timestamp_s=1.04,
        truncated=True,
        privileged={"force": [0, 0, 2]},
    )
    return recorder


def test_roundtrip_preserves_alignment_raw_frames_and_separate_diagnostics(tmp_path):
    recorder = episode()
    frame = np.full((3, 5, 3), 27, dtype=np.uint8)
    recorder.add_camera_frame(frame, timestamp_s=1.01)
    frame[:] = 91
    recorder.add_camera_frame(frame, timestamp_s=1.035)
    frame[:] = 0
    path = recorder.save(tmp_path / "episode.npz")
    loaded = load_episode(path)
    assert len(loaded) == 2
    assert loaded.metadata["complete"] is True
    assert loaded.metadata["provenance"] == {"seed": 7}
    np.testing.assert_array_equal(loaded.arrays["observations"], [[1, 2], [3, 4], [5, 6]])
    np.testing.assert_array_equal(loaded.arrays["requested_actions"], [[2], [-2]])
    np.testing.assert_array_equal(loaded.arrays["applied_actions"], [[1], [-1]])
    np.testing.assert_array_equal(loaded.arrays["privileged_force"], [[0, 0, 1], [0, 0, 2]])
    assert np.all(loaded.arrays["camera_frames"][0] == 27)
    assert np.all(loaded.arrays["camera_frames"][1] == 91)
    assert len(list(tmp_path.iterdir())) == 1
    window = loaded.window(1, 1)
    np.testing.assert_array_equal(window["observations"], [[3, 4], [5, 6]])
    np.testing.assert_array_equal(window["requested_actions"], [[-2]])
    np.testing.assert_array_equal(window["camera_times_s"], [1.035])
    window["observations"][:] = 0
    assert loaded.arrays["observations"][1, 0] == 3
    for start, length in ((-1, 1), (0, 0), (1, 2), (0.5, 1)):
        with pytest.raises((ValueError, TypeError), match="window"):
            loaded.window(start, length)


def test_save_refuses_overwrite_and_cleans_staging_file(tmp_path):
    path = tmp_path / "episode.npz"
    episode().save(path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        episode().save(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_failed_publication_leaves_no_partial_episode(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("publication failed")

    monkeypatch.setattr("shodo.dataset.os.link", fail)
    with pytest.raises(OSError, match="publication failed"):
        episode().save(tmp_path / "episode.npz")
    assert not list(tmp_path.iterdir())


def test_recorder_copies_inputs_and_marks_incomplete_episode(tmp_path):
    recorder = EpisodeRecorder({})
    obs = np.zeros(2)
    recorder.start(obs)
    obs[:] = 1
    action = np.array([0.3])
    recorder.append(action, action, obs, timestamp_s=0.02)
    action[:] = 9
    obs[:] = 9
    loaded = load_episode(recorder.save(tmp_path / "partial.npz"))
    assert loaded.metadata["complete"] is False
    np.testing.assert_array_equal(loaded.arrays["observations"], [[0, 0], [1, 1]])
    assert loaded.arrays["applied_actions"][0, 0] == 0.3
    assert "camera_frames" not in loaded.arrays


@pytest.mark.parametrize(
    "changes",
    [
        {"timestamp_s": 0},
        {"timestamp_s": np.nan},
        {"requested_action": [np.nan]},
        {"applied_action": [1, 2]},
        {"next_observation": [1]},
        {"reward": np.inf},
        {"terminated": "false"},
        {"privileged": {"invalid/name": 1}},
    ],
)
def test_invalid_transition_is_rejected_without_mutation(tmp_path, changes):
    recorder = EpisodeRecorder({})
    recorder.start([0, 0])
    kwargs = {
        "requested_action": [1],
        "applied_action": [1],
        "next_observation": [1, 1],
        "timestamp_s": 0.02,
    }
    with pytest.raises((ValueError, TypeError)):
        recorder.append(**(kwargs | changes))
    recorder.append(**kwargs)
    assert len(load_episode(recorder.save(tmp_path / "episode.npz"))) == 1


def test_lifecycle_and_diagnostic_contract(tmp_path):
    recorder = EpisodeRecorder({})
    with pytest.raises(ValueError, match="start"):
        recorder.append([0], [0], [0], timestamp_s=1)
    recorder.start([0])
    with pytest.raises(ValueError, match="started"):
        recorder.start([0])
    with pytest.raises(ValueError, match="transition"):
        recorder.save(tmp_path / "empty.npz")
    recorder.append([0], [0], [0], timestamp_s=1, privileged={"force": 1})
    with pytest.raises(ValueError, match="diagnostic"):
        recorder.append([0], [0], [0], timestamp_s=2)
    recorder.append([0], [0], [0], timestamp_s=2, privileged={"force": 2}, terminated=True)
    with pytest.raises(ValueError, match="after episode"):
        recorder.append([0], [0], [0], timestamp_s=3)


def test_camera_validation_and_bounds(tmp_path):
    recorder = episode()
    with pytest.raises(ValueError, match="uint8"):
        recorder.add_camera_frame(np.zeros((2, 2, 3)), timestamp_s=1)
    recorder.add_camera_frame(np.zeros((2, 2, 3), np.uint8), timestamp_s=1.05)
    with pytest.raises(ValueError, match="increase"):
        recorder.add_camera_frame(np.zeros((2, 2, 3), np.uint8), timestamp_s=1.05)
    with pytest.raises(ValueError, match="bounds"):
        recorder.save(tmp_path / "bad-camera.npz")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("corruption", ["time", "action", "termination", "schema", "camera"])
def test_reader_rejects_misaligned_or_unsupported_archive(tmp_path, corruption):
    loaded = load_episode(episode().save(tmp_path / "good.npz"))
    arrays, metadata = loaded.arrays, loaded.metadata
    if corruption == "time":
        arrays["observation_times_s"][1] = 0
    elif corruption == "action":
        arrays["applied_actions"] = np.zeros((1, 1))
    elif corruption == "termination":
        arrays["terminated"][0] = True
    elif corruption == "schema":
        metadata["schema_version"] = 100
    else:
        arrays["camera_times_s"] = np.array([1.0])
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, metadata_json=np.array(json.dumps(metadata)), **arrays)
    with pytest.raises(ValueError):
        load_episode(path)


def test_raw_input_channels_preserve_decision_alignment_copies_and_windows(tmp_path):
    recorder = EpisodeRecorder({"input_units": {"pose": "m", "acquisition_time_s": "s"}})
    pose = np.array([0.5, 0, 0.03])
    recorder.start(
        [1, 2],
        timestamp_s=1.0,
        inputs={"pose": pose, "acquisition_time_s": 1.0, "fresh": True},
    )
    pose[:] = [0.5, 0, 0.02]
    recorder.append(
        [1],
        [1],
        [3, 4],
        timestamp_s=1.02,
        inputs={"pose": pose, "acquisition_time_s": 1.0, "fresh": False},
        privileged={"actual_pose": [0.5, 0, 0.01]},
    )
    pose[:] = [0.5, 0, 0.01]
    recorder.append(
        [1],
        [1],
        [5, 6],
        timestamp_s=1.04,
        terminated=True,
        inputs={"pose": pose, "acquisition_time_s": 1.02, "fresh": True},
        privileged={"actual_pose": [0.5, 0, 0]},
    )
    pose[:] = 999
    loaded = load_episode(recorder.save(tmp_path / "raw-inputs.npz"))
    assert loaded.metadata["schema_version"] == 1
    np.testing.assert_array_equal(
        loaded.arrays["input_pose"], [[0.5, 0, 0.03], [0.5, 0, 0.02], [0.5, 0, 0.01]]
    )
    np.testing.assert_array_equal(loaded.arrays["input_acquisition_time_s"], [1.0, 1.0, 1.02])
    np.testing.assert_array_equal(loaded.arrays["input_fresh"], [True, False, True])
    assert loaded.arrays["input_fresh"].dtype == np.bool_
    assert loaded.arrays["privileged_actual_pose"].shape == (2, 3)
    window = loaded.window(1, 1)
    np.testing.assert_array_equal(window["input_pose"], [[0.5, 0, 0.02], [0.5, 0, 0.01]])
    np.testing.assert_array_equal(window["input_acquisition_time_s"], [1.0, 1.02])
    assert window["privileged_actual_pose"].shape == (1, 3)
    window["input_pose"][:] = -999
    assert loaded.arrays["input_pose"][1, 0] == 0.5


@pytest.mark.parametrize(
    "inputs",
    [None, {}, {"other": [0, 0]}, {"pose": [0]}, {"pose": [0, np.nan]}, {"bad/name": [0, 0]}],
)
def test_input_contract_changes_rejected_without_mutating_episode(tmp_path, inputs):
    recorder = EpisodeRecorder({})
    recorder.start([0], inputs={"pose": [0, 0]})
    with pytest.raises(ValueError):
        recorder.append([0], [0], [1], timestamp_s=0.02, inputs=inputs)
    recorder.append([0], [0], [1], timestamp_s=0.02, inputs={"pose": [1, 1]})
    loaded = load_episode(recorder.save(tmp_path / "inputs.npz"))
    assert len(loaded) == 1
    np.testing.assert_array_equal(loaded.arrays["input_pose"], [[0, 0], [1, 1]])


def test_inputs_cannot_appear_after_reset_and_invalid_start_does_not_start_episode(tmp_path):
    recorder = EpisodeRecorder({})
    with pytest.raises(ValueError, match="identifiers"):
        recorder.start([0], inputs={"bad/name": 0})
    recorder.start([0])
    with pytest.raises(ValueError, match="keys and shapes"):
        recorder.append([0], [0], [1], timestamp_s=0.02, inputs={"pose": [0]})
    recorder.append([0], [0], [1], timestamp_s=0.02)
    loaded = load_episode(recorder.save(tmp_path / "no-inputs.npz"))
    assert not any(name.startswith("input_") for name in loaded.arrays)


@pytest.mark.parametrize(
    "name,array",
    [
        ("input_pose", np.zeros((2, 3))),
        ("input_pose", np.array(0)),
        ("input_force", np.array([0, np.inf, 0])),
        ("input_", np.zeros(3)),
    ],
)
def test_reader_validates_optional_input_alignment_and_values(tmp_path, name, array):
    loaded = load_episode(episode().save(tmp_path / "good.npz"))
    arrays = {**loaded.arrays, name: array}
    path = tmp_path / "bad-inputs.npz"
    np.savez_compressed(path, metadata_json=np.array(json.dumps(loaded.metadata)), **arrays)
    with pytest.raises(ValueError):
        load_episode(path)
