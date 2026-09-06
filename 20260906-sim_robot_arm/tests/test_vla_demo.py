from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from shodo import vla_demo
from shodo.config import SimConfig
from shodo.contracts import SensorConfig


@pytest.fixture
def demo_dependencies(monkeypatch):
    report = {
        "chunk_size": 16,
        "dataset": {"chars": ["一"], "config": SimConfig().to_dict()},
    }
    calls = {"loaded": [], "rollouts": [], "saved": [], "closed": 0}
    policy = SimpleNamespace(
        report=report,
        sensor_config=SensorConfig(),
        inference_seconds=[0.1],
        decision_seconds=[0.12],
        planning_seconds=[0.12],
    )

    def close():
        calls["closed"] += 1

    policy.close = close

    def controller(adapter, **kwargs):
        calls["loaded"].append((adapter, kwargs))
        return policy

    def rollout(char, actor, **kwargs):
        calls["rollouts"].append((char, actor, kwargs))
        frame = Image.new("RGB", (8, 8), "white")
        return (
            {
                "char": char,
                "ink_rmse_mm": None if char == "一" else 1.25,
                "max_force_n": 0.75,
                "truncated": False,
                "frame_times_s": [0.0, 0.1],
            },
            np.zeros((1, 22)),
            frame,
            [frame, frame],
        )

    monkeypatch.setattr(vla_demo, "require_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(vla_demo, "_read_adapter_report", lambda *args, **kwargs: report)
    monkeypatch.setattr(vla_demo, "SmolController", controller)
    monkeypatch.setattr(vla_demo, "rollout", rollout)
    monkeypatch.setattr(
        vla_demo, "save_rollout", lambda base, result: calls["saved"].append((base, result))
    )
    return report, calls, policy


@pytest.mark.parametrize(
    "objective,chunk,execute,denoise",
    [("flow_matching", 16, 4, 10), ("action_regression", 1, 1, 1)],
)
def test_demo_reuses_sensor_rollout_and_artifacts_with_objective_defaults(
    tmp_path, demo_dependencies, capsys, objective, chunk, execute, denoise
):
    report, calls, policy = demo_dependencies
    report.update(objective=objective, chunk_size=chunk)
    rows = vla_demo.demo_adapter(
        "adapter",
        tmp_path,
        chars="一永",
        optimize_inference=True,
        trim_language_padding=True,
        cache_static_inputs=True,
    )
    assert len(calls["loaded"]) == 1
    settings = calls["loaded"][0][1]
    assert settings["execute_steps"] == execute
    assert settings["denoise_steps"] == denoise
    assert settings["cache_static_inputs"] is True
    assert calls["closed"] == 1
    assert [row[0] for row in calls["rollouts"]] == ["一", "永"]
    for _, actor, kwargs in calls["rollouts"]:
        assert actor is policy
        assert kwargs == {"seed": 7, "frames": True, "config": SimConfig()}
    assert [base.name for base, _ in calls["saved"]] == ["smolvla-04e00", "smolvla-06c38"]
    for _, (metrics, history, paper, frames) in calls["saved"]:
        assert metrics["smolvla"]["objective"] == objective
        assert metrics["smolvla"]["denoise_steps"] == denoise
        assert metrics["smolvla"]["execute_steps"] == execute
        assert metrics["smolvla"]["optimize_inference"] is True
        assert metrics["smolvla"]["trim_language_padding"] is True
        assert metrics["smolvla"]["cache_static_inputs"] is True
        assert metrics["frame_times_s"] == [0.0, 0.1]
        assert metrics["inference_seconds"] == [0.1]
        assert history.shape == (1, 22) and len(frames) == 2
        assert isinstance(paper, Image.Image)
    assert rows[0]["metrics"]["smolvla"]["training_character"] is True
    assert rows[1]["metrics"]["smolvla"]["training_character"] is False
    output = capsys.readouterr().out
    assert "ink=missing" in output and "ink=1.250 mm" in output
    assert "peak force=0.750 N" in output and "Video:" in output
    assert "smolvla-06c38.mp4" in output
    assert "frame_times_s" not in output and "inference_seconds" not in output


@pytest.mark.parametrize("chars", ["", " ", "一 一", "永永", None])
def test_bad_characters_fail_before_model_loading(tmp_path, demo_dependencies, chars):
    _, calls, _ = demo_dependencies
    with pytest.raises(ValueError):
        vla_demo.demo_adapter("adapter", tmp_path, chars=chars)
    assert not calls["loaded"]


@pytest.mark.parametrize("suffix", [".mp4", ".png", ".npz", ".json", "-scene.png"])
def test_existing_artifact_refused_before_model_loading(tmp_path, demo_dependencies, suffix):
    _, calls, _ = demo_dependencies
    original = tmp_path / f"smolvla-06c38{suffix}"
    original.write_bytes(b"preserve me")
    with pytest.raises(FileExistsError, match="exists"):
        vla_demo.demo_adapter("adapter", tmp_path)
    assert original.read_bytes() == b"preserve me"
    assert not calls["loaded"]


def test_missing_ffmpeg_fails_before_output_or_model_loading(
    tmp_path, demo_dependencies, monkeypatch
):
    _, calls, _ = demo_dependencies

    def missing():
        raise RuntimeError("FFmpeg unavailable")

    monkeypatch.setattr(vla_demo, "require_ffmpeg", missing)
    output = tmp_path / "new-demo"
    with pytest.raises(RuntimeError, match="FFmpeg"):
        vla_demo.demo_adapter("adapter", output)
    assert not output.exists() and not calls["loaded"]


@pytest.mark.parametrize("stage", ["rollout", "save_rollout"])
def test_controller_closed_when_rollout_or_export_fails(
    tmp_path, demo_dependencies, monkeypatch, stage
):
    _, calls, _ = demo_dependencies

    def fail(*args, **kwargs):
        raise RuntimeError("test failure")

    monkeypatch.setattr(vla_demo, stage, fail)
    with pytest.raises(RuntimeError, match="test failure"):
        vla_demo.demo_adapter("adapter", tmp_path)
    assert calls["closed"] == 1


def test_bad_action_objective_sampler_fails_before_model_loading(tmp_path, demo_dependencies):
    report, calls, _ = demo_dependencies
    report.update(objective="action_regression", chunk_size=1)
    with pytest.raises(ValueError, match="denois"):
        vla_demo.demo_adapter("adapter", tmp_path, denoise_steps=2)
    assert not calls["loaded"]


def test_output_file_refused_before_model_loading(tmp_path, demo_dependencies):
    _, calls, _ = demo_dependencies
    path = tmp_path / "not-a-directory"
    path.write_text("preserve me")
    with pytest.raises(NotADirectoryError):
        vla_demo.demo_adapter("adapter", path)
    assert path.read_text() == "preserve me" and not calls["loaded"]


@pytest.mark.parametrize(
    "options",
    [{"cache_static_inputs": True}, {"cache_static_inputs": 1, "optimize_inference": True}],
)
def test_static_cache_option_guard_precedes_model_loading(tmp_path, demo_dependencies, options):
    _, calls, _ = demo_dependencies
    with pytest.raises((TypeError, ValueError)):
        vla_demo.demo_adapter("adapter", tmp_path, **options)
    assert not calls["loaded"]
