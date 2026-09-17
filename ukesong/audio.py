"""Audio loading: anything in -> mono float32 samples at a working sample rate."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

WORKING_SR = 22050


def _decode_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        packed = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16))
        packed = np.where(packed & 0x800000, packed - 0x1000000, packed)
        data = packed.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data.astype(np.float32), sr


def _decode_with_ffmpeg(path: str) -> tuple[np.ndarray, int]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            f"{os.path.basename(path)} is not a WAV file and ffmpeg was not found. "
            "Install ffmpeg, or convert the song to WAV first."
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", path,
             "-ac", "1", "-ar", str(WORKING_SR), "-f", "wav", tmp.name],
            check=True,
        )
        return _decode_wav(tmp.name)
    finally:
        os.unlink(tmp.name)


def resample(x: np.ndarray, sr: int, target_sr: int = WORKING_SR) -> np.ndarray:
    """Linear-interpolation resample. Good enough: pitch tracking runs on a
    log-frequency grid whose resolution is far coarser than the artifacts this
    introduces."""
    if sr == target_sr or x.size == 0:
        return x
    duration = x.size / sr
    n_out = max(1, int(round(duration * target_sr)))
    src_idx = np.linspace(0.0, x.size - 1, n_out)
    return np.interp(src_idx, np.arange(x.size), x).astype(np.float32)


def load(path: str, target_sr: int = WORKING_SR) -> tuple[np.ndarray, int]:
    """Load `path` as mono float32 in [-1, 1] at `target_sr`."""
    try:
        data, sr = _decode_wav(path)
    except (wave.Error, EOFError, ValueError):
        data, sr = _decode_with_ffmpeg(path)

    data = resample(data, sr, target_sr)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 0:
        data = data / peak
    return data, target_sr
