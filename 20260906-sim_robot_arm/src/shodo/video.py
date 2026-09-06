"""Portable H.264 recordings of timestamped RGB frames, encoded without image intermediates."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

FPS = 50  # Preserve the simulator's 20 ms controller boundaries.
FINAL_HOLD = 0.1


def require_ffmpeg():
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("MP4 recording requires FFmpeg with libx264; install ffmpeg and retry.")
    return executable


def save_video(path, frames, times=None):
    """Hold each sampled state until the next timestamp; publish only a complete MP4."""
    executable = require_ffmpeg()
    times = np.arange(len(frames)) * 0.1 if times is None else np.asarray(times, dtype=float)
    if (
        not frames
        or times.shape != (len(frames),)
        or not np.isfinite(times).all()
        or times[0] != 0
        or (np.diff(times) <= 0).any()
    ):
        raise ValueError("Frame times must start at zero and strictly increase, one per frame")
    ticks = np.rint(times * FPS).astype(int)
    repeats = np.r_[np.diff(ticks), round(FINAL_HOLD * FPS)]
    if (repeats < 1).any():
        raise ValueError("Frame intervals must remain at least 20 ms after video quantization")
    size = frames[0].size
    if any(frame.size != size for frame in frames):
        raise ValueError("Video frames must all have the same dimensions")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".video-", dir=path.parent) as temporary:
        output = Path(temporary) / "recording.mp4"
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{size[0]}x{size[1]}",
            "-framerate",
            str(FPS),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-threads",
            "2",
            "-movflags",
            "+faststart",
            str(output),
        ]
        # A file drains diagnostics without a second pipe that could block the encoder.
        with tempfile.TemporaryFile() as errors:
            pipe_closed = False
            try:
                with subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors
                ) as encoder:
                    for frame, count in zip(frames, repeats, strict=True):
                        pixels = frame.convert("RGB").tobytes()
                        for _ in range(count):
                            encoder.stdin.write(pixels)
            except BrokenPipeError:
                pipe_closed = True
            except OSError as error:
                raise RuntimeError(f"Could not run FFmpeg: {error}") from error
            if encoder.returncode or pipe_closed:
                errors.seek(0)
                detail = errors.read().decode(errors="replace").strip()
                raise RuntimeError(
                    f"FFmpeg could not encode MP4: {detail or 'encoder closed input early'}"
                )
        output.replace(path)
    return {
        "codec": "h264",
        "encoder": "libx264",
        "preset": "medium",
        "pixel_format": "yuv420p",
        "fps": FPS,
        "frames": int(repeats.sum()),
        "duration_s": float(repeats.sum() / FPS),
        "crf": 18,
        "faststart": True,
    }
