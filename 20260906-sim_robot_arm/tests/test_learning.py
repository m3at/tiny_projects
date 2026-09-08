import json

import numpy as np
import pytest

from shodo.config import SimConfig
from shodo.data import TRAIN
from shodo.env import OBSERVATION_VERSION, ShodoEnv
from shodo.rl import InkObjective, train_ppo


def test_ink_objective_changes_only_drawing_reward(short_stroke):
    plain, shaped = ShodoEnv(chars="一"), InkObjective(ShodoEnv(chars="一"))
    bonuses = []
    try:
        a, _ = plain.reset(seed=7)
        b, _ = shaped.reset(seed=7)
        np.testing.assert_array_equal(a, b)
        for _ in range(short_stroke):
            action = plain.expert()
            a, reward, done, truncated, _ = plain.step(action)
            b, shaped_reward, shaped_done, shaped_truncated, info = shaped.step(action)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(plain.data.qpos, shaped.unwrapped.data.qpos)
            np.testing.assert_array_equal(plain.paper.mobile, shaped.unwrapped.paper.mobile)
            assert (done, truncated) == (shaped_done, shaped_truncated)
            bonus = shaped_reward - reward
            assert -1e-12 <= bonus <= 0.65 + 1e-12
            if plain.stroke_ids[plain.index - 1] < 0:
                assert bonus == 0
                assert "ink_error_m" not in info
            bonuses.append(bonus)
        assert max(bonuses) > 0.3
    finally:
        plain.close()
        shaped.close()


def test_invalid_learning_objective_is_rejected_before_output(tmp_path):
    destination = tmp_path / "should-not-exist"
    with pytest.raises(ValueError, match="objective"):
        train_ppo(destination, objective="unknown")
    assert not destination.exists()


def test_resume_cannot_silently_change_unspecified_reward_objective(tmp_path):
    metadata = {
        "observation_version": OBSERVATION_VERSION,
        "residual_scale": 0.0,
        "config": SimConfig(randomize=True).to_dict(),
        "seed": 7,
        "train_chars": TRAIN,
    }
    (tmp_path / "ppo.json").write_text(json.dumps(metadata))
    checkpoint = tmp_path / "ppo.zip"
    checkpoint.write_bytes(b"unchanged checkpoint sentinel")
    with pytest.raises(ValueError, match="changing objective"):
        train_ppo(tmp_path, resume=True, objective="ink")
    assert checkpoint.read_bytes() == b"unchanged checkpoint sentinel"
