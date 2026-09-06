import json
from contextlib import ExitStack

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from scipy.spatial.transform import Rotation

from shodo.brush import Brush
from shodo.config import SimConfig
from shodo.contracts import (
    SENSOR_FEATURES,
    ActionContract,
    ReferenceSample,
    SensorConfig,
    SensorSample,
    classical_action,
    sensor_contract,
    sensor_features,
)
from shodo.dataset import EpisodeRecorder, load_episode
from shodo.env import HISTORY_COLUMNS, OBSERVATION_VERSION, ShodoEnv
from shodo.runtime import SensorEnv, execute_step


def test_recovery_labels_are_preaction_counterfactuals_and_noise_is_separate(sensor_env):
    observation, _ = sensor_env.reset(seed=7)
    expected = sensor_env.unwrapped.expert().copy()
    command = sensor_env.unwrapped.command.copy()
    _, clipped = sensor_env.action_contract.apply(command, expected)
    noise = np.random.default_rng(11).normal(0, 0.08, 6)
    transition = execute_step(
        sensor_env,
        observation,
        "oracle",
        expert_noise=0.08,
        noise_rng=np.random.default_rng(11),
    )
    np.testing.assert_array_equal(transition.expert_requested_action, expected)
    np.testing.assert_array_equal(transition.expert_applied_action, clipped)
    np.testing.assert_allclose(transition.requested_action, expected + noise)
    np.testing.assert_array_equal(transition.observation, observation)
    _, behavior_applied = sensor_env.action_contract.apply(command, expected + noise)
    np.testing.assert_array_equal(transition.applied_action, behavior_applied)


def test_clean_recovery_labels_preserve_oracle_behavior(sensor_env):
    observation, _ = sensor_env.reset(seed=7)
    baseline = execute_step(sensor_env, observation, "oracle")
    observation, _ = sensor_env.reset(seed=7)
    labeled = execute_step(
        sensor_env,
        observation,
        "oracle",
        expert_noise=0,
        noise_rng=np.random.default_rng(11),
    )
    assert baseline.expert_requested_action is None
    np.testing.assert_array_equal(baseline.next_observation, labeled.next_observation)
    np.testing.assert_array_equal(baseline.applied_action, labeled.expert_applied_action)


@pytest.mark.parametrize("noise,policy", [(-1, "oracle"), (np.nan, "oracle"), (0, "zero")])
def test_invalid_recovery_collection_is_rejected_before_output(tmp_path, noise, policy):
    from shodo.runtime import record_episodes

    with pytest.raises(ValueError, match="noise|oracle"):
        record_episodes(tmp_path, chars="一", policy=policy, expert_noise=noise)
    assert not list(tmp_path.iterdir())


def test_recovery_recording_keeps_labels_separate_and_reproducible(tmp_path):
    from shodo.runtime import record_episodes

    episodes = [
        load_episode(
            record_episodes(tmp_path / str(i), chars="一", policy="oracle", expert_noise=0.08)[0]
        )
        for i in range(2)
    ]
    first = episodes[0]
    assert first.metadata["recovery_supervision"]["coherent_action_chunks"] is False
    assert first.metadata["recovery_supervision"]["label_timing"] == "pre-action decision boundary"
    for key in first.arrays:
        np.testing.assert_array_equal(first.arrays[key], episodes[1].arrays[key])
    assert first.arrays["privileged_expert_applied_action"].shape == (len(first), 6)
    assert not np.array_equal(
        first.arrays["privileged_expert_applied_action"], first.arrays["applied_actions"]
    )
    assert first.arrays["observations"].shape[1] == SENSOR_FEATURES * 4
    window = first.window(0, 2)
    assert window["privileged_expert_applied_action"].shape == (2, 6)
    replay = SensorEnv("一")
    try:
        replay.reset(seed=7)
        for index in range(12):
            teacher = replay.unwrapped.expert()
            _, effective = replay.action_contract.apply(replay.unwrapped.command, teacher)
            np.testing.assert_array_equal(
                first.arrays["privileged_expert_requested_action"][index], teacher
            )
            np.testing.assert_array_equal(
                first.arrays["privileged_expert_applied_action"][index], effective
            )
            replay.step(first.arrays["requested_actions"][index])
    finally:
        replay.close()


@pytest.fixture
def sensor_env():
    env = SensorEnv("一")
    try:
        yield env
    finally:
        env.close()


def test_sensor_history_contract_reset_and_returned_array_independence(sensor_env):
    obs, info = sensor_env.reset(seed=7)
    assert info["char"] == "一"
    assert SENSOR_FEATURES == 39
    assert obs.shape == (39 * 4,)
    assert obs.dtype == np.float32
    assert sensor_env.observation_space.contains(obs)
    features = obs.reshape(4, 39)
    for row in features:
        np.testing.assert_array_equal(row, features[0])
    assert features[-1, 37] == 0
    assert features[-1, 38] == 1
    initial = obs.copy()
    obs[:] = 999
    np.testing.assert_array_equal(sensor_env.observation, initial)
    next_obs, *_ = sensor_env.step(np.zeros(6))
    np.testing.assert_array_equal(next_obs.reshape(4, 39)[:-1], initial.reshape(4, 39)[1:])
    reset, _ = sensor_env.reset(seed=7)
    np.testing.assert_array_equal(reset, initial)
    assert len(sensor_env.samples) == 1
    contract = sensor_contract(sensor_env.sensors)
    assert contract["privileged_inputs"] is False
    assert contract["features_per_sample"] * contract["history"] == len(obs)


def test_sensor_encoding_does_not_read_contact_center_deflection_or_ink(sensor_env, monkeypatch):
    sensor_env.reset(seed=7)

    def encode():
        return sensor_features(
            sensor_env._acquire(),
            sensor_env._reference(),
            sensor_env.env.command,
            sensor_env.env.data.time,
            True,
            sensor_env.action_contract,
        )

    expected = encode()
    force = sensor_env.env.brush.force.copy()

    def forbidden(_):
        raise AssertionError("privileged channel accessed")

    monkeypatch.setattr(ShodoEnv, "tracking_pose", property(forbidden))
    monkeypatch.setattr(Brush, "deflection", property(forbidden))
    monkeypatch.setattr(Brush, "ink_positions", property(forbidden))
    monkeypatch.setattr(Brush, "ink_loads", property(forbidden))
    sensor_env.env.brush.contact[:] = 100
    sensor_env.env.brush.normal[:] = 999
    sensor_env.env.brush.tangent[:] = -100
    sensor_env.env.brush.touching[:] = True
    sensor_env.env.paper.fixed[:] = 100
    np.testing.assert_array_equal(sensor_env.env.brush.force, force)
    np.testing.assert_array_equal(encode(), expected)


def test_sensor_noise_rng_is_reproducible_and_independent_of_plant_draws():
    config = SimConfig(randomize=True)
    sensors = SensorConfig(position_noise=0.001, force_noise=0.01, dropout=0.5)
    with ExitStack() as stack:
        raw = ShodoEnv(chars="一二", config=config)
        measured = SensorEnv("一二", config=config, sensors=sensors)
        other = SensorEnv("一二", config=config, sensors=SensorConfig())
        for env in (raw, measured, other):
            stack.callback(env.close)
        first = None
        for seed in (7, None, 7):
            _, raw_info = raw.reset(seed=seed)
            obs, measured_info = measured.reset(seed=seed)
            _, other_info = other.reset(seed=seed)
            assert raw_info == measured_info == other_info
            assert raw.brush.config == measured.env.brush.config == other.env.brush.config
            np.testing.assert_array_equal(raw.paper.fibers, measured.env.paper.fibers)
            np.testing.assert_array_equal(raw.paper.fibers, other.env.paper.fibers)
            np.testing.assert_array_equal(raw.data.qpos, measured.env.data.qpos)
            if first is None:
                first = obs.copy()
            elif seed == 7:
                np.testing.assert_array_equal(obs, first)
            for _ in range(3):
                raw.step(np.zeros(6))
                measured.step(np.zeros(6))
                other.step(np.zeros(6))
            np.testing.assert_array_equal(raw.data.qpos, measured.env.data.qpos)


@pytest.mark.parametrize("latency,dropout", [(0, 0), (2, 0), (2, 1)])
def test_sensor_age_latency_dropout_and_bootstrap(latency, dropout):
    env = SensorEnv("一", sensors=SensorConfig(history=1, latency_steps=latency, dropout=dropout))
    try:
        obs, _ = env.reset(seed=7)
        assert obs[37] == 0 and obs[38] == 1
        previous_acquisition = 0
        for step in range(1, 6):
            obs, _, _, _, info = env.step(np.zeros(6))
            expected_acquisition = 0 if dropout else max(0, step - latency) * 0.02
            expected_fresh = expected_acquisition > previous_acquisition
            assert env.sample.timestamp_s == pytest.approx(expected_acquisition)
            assert info["sensor_age_s"] == pytest.approx(step * 0.02 - expected_acquisition)
            assert obs[37] == pytest.approx(info["sensor_age_s"])
            assert bool(obs[38]) == expected_fresh
            assert bool(info["sensor_fresh"]) == expected_fresh
            previous_acquisition = expected_acquisition
    finally:
        env.close()


def test_calibration_offsets_transform_reference_and_measurement_not_physics():
    cfg = SensorConfig(
        history=1,
        tool_offset=(0.001, -0.002, 0.003),
        reference_offset=(0.004, 0.005, -0.006),
        reference_yaw=0.17,
        force_bias=(0.1, -0.2, 0.3),
    )
    env = SensorEnv("永", sensors=cfg)
    try:
        env.reset(seed=7)
        np.testing.assert_allclose(env.sample.pose[:3], env.env.pose[:3] + cfg.tool_offset)
        np.testing.assert_allclose(env.sample.force, env.env.brush.force + cfg.force_bias)
        rotation = Rotation.from_rotvec([0, 0, cfg.reference_yaw])
        origin = np.array([env.env.config.paper_x, 0, env.env.config.paper_z])
        for pose, index in ((env.reference.pose, 0), (env.reference.preview, 3)):
            expected_xyz = (
                rotation.apply(env.env.path[index] - origin) + origin + cfg.reference_offset
            )
            np.testing.assert_allclose(pose[:3], expected_xyz)
            np.testing.assert_allclose(
                Rotation.from_rotvec(pose[3:]).as_matrix(),
                rotation.as_matrix() @ Rotation.from_rotvec(env.env.tilts[index]).as_matrix(),
                atol=1e-14,
            )
        # Calibration error must not relocate the simulated paper or actuator command.
        np.testing.assert_array_equal(env.env.command[:3], env.env.path[0])
        assert env.env.config.paper_x == 0.5 and env.env.config.paper_z == 0
    finally:
        env.close()


def test_classical_controller_is_pure_measured_force_feedback():
    action = ActionContract()
    pose = np.array([0.5, 0, 0.0, 0, 0, 0], dtype=float)
    sample = SensorSample(0, pose.copy(), np.zeros(7), np.zeros(7), np.array([0, 0, 0.4]))
    reference = ReferenceSample(pose.copy(), pose.copy(), 0.5, True)
    command = pose.copy()
    result = classical_action(sample, reference, command, action, force_gain=0.02)
    np.testing.assert_allclose(result, [0, 0, -0.5, 0, 0, 0])
    airborne = ReferenceSample(pose.copy(), pose.copy(), 0.5, False)
    np.testing.assert_array_equal(classical_action(sample, airborne, command, action), np.zeros(6))
    np.testing.assert_array_equal(sample.pose, pose)
    np.testing.assert_array_equal(reference.pose, pose)
    np.testing.assert_array_equal(command, pose)


@pytest.mark.parametrize("yaw", [0.0, 0.17])
def test_reference_cache_matches_per_sample_calibration_and_refreshes_on_reset(yaw):
    cfg = SensorConfig(reference_yaw=yaw, reference_offset=(0.001, -0.002, 0.003))
    env = SensorEnv("一永", sensors=cfg)
    try:
        for char in "一永":
            env.reset(seed=7, options={"char": char})
            assert not env._reference_poses.flags.writeable
            rotation = Rotation.from_rotvec([0, 0, yaw])
            origin = np.array([env.env.config.paper_x, 0, env.env.config.paper_z])
            for index in (0, 10, len(env.env.path) - 1, len(env.env.path)):
                env.env.index = index
                i = min(index, len(env.env.path) - 1)
                j = min(i + 3, len(env.env.path) - 1)
                expected = np.c_[env.env.path[[i, j]], env.env.tilts[[i, j]]]
                expected[:, :3] = (
                    rotation.apply(expected[:, :3] - origin) + origin + cfg.reference_offset
                )
                expected[:, 3:] = (rotation * Rotation.from_rotvec(expected[:, 3:])).as_rotvec()
                actual = env._reference()
                np.testing.assert_array_equal(actual.pose, expected[0])
                np.testing.assert_array_equal(actual.preview, expected[1])
                # Public samples are independent of both the cache and each other.
                actual.pose[:] = 123
                actual.preview[:] = -456
                np.testing.assert_array_equal(env._reference().pose, expected[0])
                np.testing.assert_array_equal(env._reference().preview, expected[1])
    finally:
        env.close()


def test_sensor_encoder_matches_contract_concatenation_exactly():
    rng = np.random.default_rng(7)
    action = ActionContract(translation_step=0.002, rotation_step=0.05)
    sample = SensorSample(
        0.2, rng.normal(size=6), rng.normal(size=7), rng.normal(size=7), rng.normal(size=3)
    )
    reference = ReferenceSample(rng.normal(size=6), rng.normal(size=6), 0.35, True)
    command = rng.normal(size=6)
    expected = np.r_[
        (reference.pose - sample.pose) / action.scales,
        (command - sample.pose) / action.scales,
        (reference.preview - reference.pose) / action.scales,
        sample.joints / 3,
        sample.velocities / 5,
        sample.force,
        reference.force_n,
        float(reference.drawing),
        0.3 - sample.timestamp_s,
        1.0,
    ].astype(np.float32)
    np.testing.assert_array_equal(
        sensor_features(sample, reference, command, 0.3, True, action), expected
    )


def test_action_contract_clips_input_and_workspace_without_mutation():
    contract = ActionContract()
    command = np.array([0.604, 0, 0, 0.299, 0, 0])
    requested = np.array([2, -2, 0, 2, 0, 0], dtype=float)
    before_command, before_action = command.copy(), requested.copy()
    target, applied = contract.apply(command, requested)
    np.testing.assert_allclose(target, [0.605, -0.004, 0, 0.3, 0, 0])
    np.testing.assert_allclose(applied, [0.25, -1, 0, 1 / 30, 0, 0])
    np.testing.assert_array_equal(command, before_command)
    np.testing.assert_array_equal(requested, before_action)
    assert contract.to_dict()["frame"] == "world"


@pytest.mark.parametrize("bad", [np.zeros(5), np.full(6, np.nan), np.full(6, np.inf)])
def test_invalid_actions_rejected_before_physics(sensor_env, bad):
    obs, _ = sensor_env.reset(seed=7)
    initial_time, command = sensor_env.env.data.time, sensor_env.env.command.copy()
    with pytest.raises(ValueError, match="finite 6-vector"):
        execute_step(sensor_env, obs, lambda _: bad)
    assert sensor_env.env.data.time == initial_time
    np.testing.assert_array_equal(sensor_env.env.command, command)


def test_execute_step_records_requested_and_effective_increment_as_copies(sensor_env):
    obs, _ = sensor_env.reset(seed=7)
    requested = np.full(6, 3.0)
    command = sensor_env.env.command.copy()
    before_obs = obs.copy()

    def policy(inputs):
        inputs[:] = 123
        return requested

    transition = execute_step(sensor_env, obs, policy)
    np.testing.assert_array_equal(obs, before_obs)
    np.testing.assert_array_equal(transition.observation, before_obs)
    np.testing.assert_array_equal(transition.requested_action, requested)
    np.testing.assert_allclose(
        transition.applied_action,
        (sensor_env.env.command - command) / sensor_env.action_contract.scales,
    )
    assert transition.timestamp_s == 0
    assert transition.next_timestamp_s == pytest.approx(0.02)
    requested[:] = -3
    sensor_env.env.last_applied_action[:] = -4
    obs[:] = -9
    assert np.all(transition.requested_action == 3)
    assert np.all(transition.applied_action >= 0)
    np.testing.assert_array_equal(transition.observation, before_obs)


def test_sensor_wrapper_identical_actions_preserve_legacy_plant_and_reward():
    config = SimConfig(randomize=True, record=True)
    with ExitStack() as stack:
        raw = ShodoEnv(chars="一", config=config)
        wrapped = SensorEnv("一", config=config, sensors=SensorConfig(position_noise=0.002))
        stack.callback(raw.close)
        stack.callback(wrapped.close)
        raw.reset(seed=7)
        wrapped.reset(seed=7)
        for _ in range(60):
            action = raw.expert()
            previous = raw.command.copy()
            expected = np.clip(
                previous + raw.scales * np.clip(action, -1, 1),
                [0.395, -0.105, -0.007, -0.3, -0.3, -0.3],
                [0.605, 0.105, 0.05, 0.3, 0.3, 0.3],
            )
            _, reward, term, trunc, _ = raw.step(action)
            _, other_reward, other_term, other_trunc, _ = wrapped.step(action)
            np.testing.assert_array_equal(raw.command, expected)
            np.testing.assert_array_equal(raw.data.qpos, wrapped.env.data.qpos)
            np.testing.assert_array_equal(raw.paper.fixed, wrapped.env.paper.fixed)
            assert (reward, term, trunc) == (other_reward, other_term, other_trunc)
        np.testing.assert_array_equal(raw.history, wrapped.env.history)
        assert raw.paper.deposited_pigment > 0
        np.testing.assert_array_equal(raw.paper.mobile, wrapped.env.paper.mobile)
        np.testing.assert_array_equal(raw.paper.water, wrapped.env.paper.water)


def test_sensor_environment_gymnasium_contract(sensor_env):
    check_env(sensor_env, skip_render_check=True)


@pytest.mark.parametrize(
    "options",
    [
        {"history": 0},
        {"history": True},
        {"latency_steps": -1},
        {"dropout": 1.01},
        {"position_noise": -1},
        {"force_noise": float("nan")},
        {"tool_offset": [1, 2]},
        {"reference_offset": [0, float("inf"), 0]},
        {"reference_yaw": float("nan")},
    ],
)
def test_invalid_sensor_configuration(options):
    with pytest.raises(ValueError):
        SensorConfig(**options)


def _assert_raw_inputs_reconstruct_policy_history(dataset, history):
    # Deliberately pass only input_* channels and timestamps to the reconstruction.
    # No simulator or privileged history is needed to recover the policy features.
    inputs = {
        name.removeprefix("input_"): array
        for name, array in dataset.arrays.items()
        if name.startswith("input_")
    }
    times = dataset.arrays["observation_times_s"]
    expected = []
    for index, timestamp in enumerate(times):
        sample = SensorSample(
            float(inputs["acquisition_time_s"][index]),
            inputs["pose"][index],
            inputs["joints"][index],
            inputs["velocities"][index],
            inputs["force"][index],
        )
        reference = ReferenceSample(
            inputs["reference_pose"][index],
            inputs["reference_preview"][index],
            float(inputs["reference_force_n"][index]),
            bool(inputs["reference_drawing"][index]),
        )
        features = sensor_features(
            sample,
            reference,
            inputs["command"][index],
            timestamp,
            bool(inputs["fresh"][index]),
            ActionContract(),
        )
        expected.append(features)
        np.testing.assert_array_equal(dataset.arrays["observations"][index, -39:], features)
        stacked = np.concatenate(
            [expected[max(0, i)] for i in range(index - history + 1, index + 1)]
        )
        np.testing.assert_array_equal(dataset.arrays["observations"][index], stacked)


def test_sensor_rollout_records_raw_camera_transitions_and_replayable_history(tmp_path):
    from shodo.runtime import sensor_rollout

    sensors = SensorConfig(history=4)
    recorder = EpisodeRecorder({"observation_contract": sensor_contract(sensors)})
    metrics = sensor_rollout("一", sensors=sensors, recorder=recorder, camera_every=30)
    dataset = load_episode(recorder.save(tmp_path / "sensor.npz"))
    arrays = dataset.arrays
    count = metrics["steps"]
    assert count == 247
    assert arrays["observations"].shape == (count + 1, 156)
    assert arrays["requested_actions"].shape == arrays["applied_actions"].shape == (count, 6)
    assert arrays["privileged_history"].shape == (count, len(HISTORY_COLUMNS))
    assert arrays["terminated"][-1] and not arrays["truncated"].any()
    assert dataset.metadata["complete"] is True
    _assert_raw_inputs_reconstruct_policy_history(dataset, sensors.history)
    np.testing.assert_allclose(arrays["observation_times_s"], np.arange(count + 1) * 0.02)
    camera_steps = [*range(0, count + 1, 30), count]
    assert arrays["camera_frames"].shape == (len(camera_steps), 480, 640, 3)
    assert arrays["camera_frames"].dtype == np.uint8
    np.testing.assert_allclose(arrays["camera_times_s"], np.asarray(camera_steps) * 0.02)
    assert np.any(arrays["camera_frames"][0] != arrays["camera_frames"][-1])
    # Recorded requested actions reproduce the plant and observations without rendering.
    replay = SensorEnv("一", config=SimConfig(record=True), sensors=sensors)
    try:
        obs, _ = replay.reset(seed=7)
        np.testing.assert_array_equal(obs, arrays["observations"][0])
        for index, action in enumerate(arrays["requested_actions"]):
            obs, reward, terminated, truncated, _ = replay.step(action)
            np.testing.assert_array_equal(obs, arrays["observations"][index + 1])
            np.testing.assert_array_equal(
                replay.env.last_applied_action, arrays["applied_actions"][index]
            )
            assert reward == arrays["rewards"][index]
            assert terminated == arrays["terminated"][index]
            assert truncated == arrays["truncated"][index]
        np.testing.assert_array_equal(replay.env.history, arrays["privileged_history"])
    finally:
        replay.close()


def test_raw_input_recording_reconstructs_delayed_dropped_and_noisy_measurements(tmp_path):
    sensors = SensorConfig(
        history=3,
        latency_steps=2,
        dropout=0.35,
        position_noise=0.0002,
        force_noise=0.01,
        joint_noise=0.0001,
        velocity_noise=0.001,
        force_bias=(0.01, -0.02, 0.03),
        tool_offset=(0.001, 0, 0.0005),
        reference_offset=(0, 0.001, -0.0005),
        reference_yaw=0.01,
    )
    env = SensorEnv("一", sensors=sensors)
    recorder = EpisodeRecorder({"observation_contract": sensor_contract(sensors)})
    try:
        obs, _ = env.reset(seed=7)
        recorder.start(obs, timestamp_s=env.env.data.time, inputs=env.input_channels())
        for _ in range(80):
            transition = execute_step(env, obs, "classical")
            assert not transition.terminated and not transition.truncated
            obs = transition.next_observation
            channels = env.input_channels()
            recorder.append(
                transition.requested_action,
                transition.applied_action,
                obs,
                timestamp_s=transition.next_timestamp_s,
                reward=transition.reward,
                inputs=channels,
            )
            # Neither recorder storage nor live sensor state may alias the exported channels.
            channels["force"][:] = 999
            assert not np.any(env.sample.force == 999)
    finally:
        env.close()
    dataset = load_episode(recorder.save(tmp_path / "delayed-inputs.npz"))
    assert len(dataset) == 80 and dataset.metadata["complete"] is False
    assert not any(name.startswith("privileged_") for name in dataset.arrays)
    _assert_raw_inputs_reconstruct_policy_history(dataset, sensors.history)
    arrays = dataset.arrays
    acquisition = arrays["input_acquisition_time_s"]
    times = arrays["observation_times_s"]
    fresh = arrays["input_fresh"]
    ages = times - acquisition
    assert np.max(ages) > sensors.latency_steps * 0.02 + 0.01
    assert fresh[1:].any() and not fresh[1:].all()
    assert fresh[0] and acquisition[0] == 0
    np.testing.assert_array_equal(fresh[1:], np.diff(acquisition) > 0)
    np.testing.assert_allclose(arrays["observations"][:, -2], ages, atol=1e-8)
    for index in np.flatnonzero(~fresh):
        for name in ("pose", "force", "joints", "velocities"):
            np.testing.assert_array_equal(
                arrays[f"input_{name}"][index], arrays[f"input_{name}"][index - 1]
            )


def test_record_episodes_metadata_and_existing_destination_preflight(tmp_path, monkeypatch):
    from shodo.runtime import record_episodes

    sensors = SensorConfig(history=2, latency_steps=1)
    paths = record_episodes(tmp_path, chars="一", sensors=sensors)
    assert paths == [tmp_path / "episode-000000.npz"]
    dataset = load_episode(paths[0])
    metadata = dataset.metadata
    assert metadata["observation_contract"] == sensor_contract(sensors)
    assert SensorConfig(**metadata["sensors"]) == sensors
    assert metadata["action_contract"] == ActionContract().to_dict()
    assert metadata["privileged_history_columns"] == list(HISTORY_COLUMNS)
    assert "CC BY-SA 3.0" in metadata["attribution"]
    assert "source_sha256" in metadata["provenance"]
    assert not metadata["camera"]["enabled"]
    assert "camera_frames" not in dataset.arrays
    assert metadata["metrics"]["steps"] == len(dataset)
    original = paths[0].read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("existing paths must be rejected before running an episode")

    monkeypatch.setattr("shodo.runtime.sensor_rollout", forbidden)
    with pytest.raises(FileExistsError, match="new directory"):
        record_episodes(tmp_path, chars="一", sensors=sensors)
    assert paths[0].read_bytes() == original


def test_sensor_bc_training_loading_and_observation_contract_rejection(tmp_path):
    import torch

    from shodo.learning import load_policy, network, rollout, train
    from shodo.runtime import sensor_rollout

    sensors = SensorConfig(history=2)
    path = tmp_path / "sensor.pt"
    metadata = train(episodes=1, epochs=1, chars="一", sensors=sensors, output=path)
    assert metadata["samples"] == 247
    assert metadata["sensor_contract"] == sensor_contract(sensors)
    policy = load_policy(path)
    assert policy.sensor_config == sensors
    output = policy(np.zeros(78, dtype=np.float32))
    assert output.shape == (6,) and np.isfinite(output).all()
    recorder = EpisodeRecorder({})
    metrics = sensor_rollout("一", policy, recorder=recorder)
    recorded = load_episode(recorder.save(tmp_path / "learned.npz"))
    assert recorded.arrays["observations"].shape == (metrics["steps"] + 1, 78)
    assert recorded.metadata["complete"] is True
    with pytest.raises(ValueError, match="history differ"):
        sensor_rollout("一", policy, sensors=SensorConfig(history=4))
    legacy_path = tmp_path / "legacy.pt"
    torch.save(
        {"state_dict": network().state_dict(), "observation_version": OBSERVATION_VERSION},
        legacy_path,
    )
    legacy = load_policy(legacy_path)
    with pytest.raises(ValueError, match="observation contracts differ"):
        sensor_rollout("一", legacy)
    with pytest.raises(ValueError, match="observation contracts differ"):
        rollout("一", legacy, sensors=sensors)


def test_evaluation_summary_exposes_partial_episode_counts_and_report_path(
    tmp_path, monkeypatch, capsys
):
    from shodo import learning

    calls = []

    def partial(char, policy, **kwargs):
        calls.append((char, policy, kwargs))
        assert isinstance(kwargs["sensors"], SensorConfig)
        return (
            {
                "char": char,
                "steps": 2,
                "expected_steps": 247,
                "terminated": False,
                "truncated": True,
                "rmse_mm": 0.001,
                "ink_rmse_mm": None,
                "max_force_n": 9.0,
            },
            None,
            None,
            [],
        )

    monkeypatch.setattr(learning, "rollout", partial)
    path = tmp_path / "evaluation.json"
    results = learning.evaluate(path, algorithm="classical", chars="一二")
    assert set(results) == {"classical", "oracle", "zero"}
    assert len(calls) == 6
    printed = capsys.readouterr().out
    for controller in results:
        summary = next(line for line in printed.splitlines() if line.startswith(f"{controller}:"))
        assert "RMSE 0.001 mm" in summary
        assert "complete 0/2" in summary
        assert "truncated 2" in summary
        assert "missing ink 2" in summary
    assert f"Evaluation report: {path.resolve()}" in printed
    assert json.loads(path.read_text()) == results


def test_raw_camera_request_without_recorder_fails_before_constructing_environment(monkeypatch):
    from shodo import learning, runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("missing recorder must be rejected before constructing physics")

    monkeypatch.setattr(learning, "ShodoEnv", forbidden)
    monkeypatch.setattr(runtime, "SensorEnv", forbidden)
    with pytest.raises(ValueError, match="requires a training-data recorder"):
        learning.rollout("一", camera_every=10)
    with pytest.raises(ValueError, match="requires a training-data recorder"):
        runtime.sensor_rollout("一", camera_every=10)
