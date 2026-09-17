"""Synthesise test tones so the pipeline can be checked without real recordings."""

from __future__ import annotations

import wave

import numpy as np

SR = 22050


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def tone(midi: int, duration: float, sr: int = SR, harmonics: int = 4) -> np.ndarray:
    """A note with overtones and a soft attack/release, like a plucked string."""
    t = np.arange(int(duration * sr)) / sr
    wave_out = np.zeros_like(t)
    for k in range(1, harmonics + 1):
        wave_out += (1.0 / k) * np.sin(2 * np.pi * midi_to_hz(midi) * k * t)
    envelope = np.minimum(1.0, np.minimum(t / 0.01, (duration - t) / 0.02))
    return (wave_out * np.clip(envelope, 0, 1)).astype(np.float32)


def melody(notes: list[tuple[int, float]], sr: int = SR) -> np.ndarray:
    out = np.concatenate([tone(m, d, sr) for m, d in notes])
    return out / np.max(np.abs(out))


def write_wav(path: str, samples: np.ndarray, sr: int = SR) -> None:
    pcm = np.clip(samples, -1, 1)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((pcm * 32767).astype("<i2").tobytes())
