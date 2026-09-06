import json
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from shodo.artifacts import save_rollout
from shodo.env import HISTORY_COLUMNS
from shodo.video import save_video


@pytest.fixture
def video_tools():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Video integration checks require ffmpeg and ffprobe")


def decode(path, width=8, height=8):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True,
        check=True,
    )
    return stream, np.frombuffer(raw.stdout, dtype=np.uint8).reshape(-1, height, width, 3)


def test_decimal_experiment_names_do_not_overwrite_each_other(tmp_path, video_tools):
    frame = Image.new("RGB", (8, 8), "white")
    history = np.zeros((1, len(HISTORY_COLUMNS)))
    for name in ("dt-0.0001-04e00", "dt-0.0002-06c38"):
        save_rollout(tmp_path / name, ({"case": name}, history, frame, [frame]))
        for suffix in (".png", ".npz", ".mp4", "-scene.png", ".json"):
            assert (tmp_path / f"{name}{suffix}").is_file()
        assert json.loads((tmp_path / f"{name}.json").read_text())["metrics"]["case"] == name
        with np.load(tmp_path / f"{name}.npz") as archive:
            np.testing.assert_array_equal(archive["history"], history)
            assert archive["columns"].tolist() == list(HISTORY_COLUMNS)
    assert len(list(tmp_path.iterdir())) == 10


@pytest.mark.parametrize(
    "times,starts",
    [([0, 0.04, 0.08, 0.096], [0, 2, 4, 5]), ([0, 0.033, 0.066, 0.099], [0, 2, 3, 5])],
)
def test_video_preserves_state_timing_without_accumulating_drift(
    tmp_path, video_tools, times, starts
):
    frames = [Image.new("RGB", (8, 8), color) for color in ("white", "gray", "red", "black")]
    history = np.zeros((12, len(HISTORY_COLUMNS)))
    save_rollout(tmp_path / "timed", ({"frame_times_s": times}, history, frames[-1], frames))
    stream, decoded = decode(tmp_path / "timed.mp4")
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert stream["r_frame_rate"] == "50/1"
    assert float(stream["duration"]) == pytest.approx(0.2)
    assert len(decoded) == 10  # 100 ms quantized simulation plus a 100 ms final hold.
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(decoded)
        expected = np.asarray(frames[i])[0, 0].astype(float)
        assert np.max(np.abs(decoded[start:end].astype(float) - expected)) < 8
    payload = (tmp_path / "timed.mp4").read_bytes()
    assert payload.index(b"moov") < payload.index(b"mdat"), "Playback index must be at the front"
    saved = json.loads((tmp_path / "timed.json").read_text())["metrics"]
    assert saved["recording_final_hold_s"] == 0.1
    assert saved["video"]["duration_s"] == pytest.approx(0.2)


@pytest.mark.parametrize("exit_code", [0, 1])
def test_encoder_failure_preserves_existing_video_and_cleans_temporary_files(
    tmp_path, monkeypatch, exit_code
):
    encoder = tmp_path / "failed-encoder"
    encoder.write_text(f"#!/bin/sh\necho 'test encoder unavailable' >&2\nexit {exit_code}\n")
    encoder.chmod(0o755)
    monkeypatch.setattr("shodo.video.require_ffmpeg", lambda: str(encoder))
    output = tmp_path / "saved.mp4"
    output.write_bytes(b"existing video")
    with pytest.raises(RuntimeError, match="test encoder unavailable"):
        save_video(output, [Image.new("RGB", (960, 480))])
    assert output.read_bytes() == b"existing video"
    assert not list(tmp_path.glob(".video-*"))


@pytest.mark.parametrize("times", [[1], [0, 0], [0, float("nan")], [0, 0.001]])
def test_bad_timestamps_do_not_write_artifacts(tmp_path, monkeypatch, times):
    monkeypatch.setattr("shodo.video.require_ffmpeg", lambda: "unused")
    frame = Image.new("RGB", (8, 8))
    with pytest.raises(ValueError, match="Frame"):
        save_rollout(
            tmp_path / "bad",
            ({"frame_times_s": times}, np.empty((0, 22)), frame, [frame] * len(times)),
        )
    assert not list(tmp_path.iterdir())


def test_video_pads_odd_dimensions_for_compatible_playback(tmp_path, video_tools):
    output = tmp_path / "odd.mp4"
    save_video(output, [Image.new("RGB", (9, 7), "white")])
    stream, _ = decode(output, width=10, height=8)
    assert (stream["width"], stream["height"]) == (10, 8)


def test_numerical_artifacts_do_not_require_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr("shodo.video.shutil.which", lambda _: None)
    frame = Image.new("RGB", (8, 8))
    save_rollout(tmp_path / "data", ({}, np.empty((0, 22)), frame, []))
    assert {p.suffix for p in tmp_path.iterdir()} == {".png", ".npz", ".json"}
