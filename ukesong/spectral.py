"""The sound graph: audio -> a semitone-by-time energy map.

This is the representation everything downstream reads. Rows are notes on the
chromatic scale (one row per semitone), columns are time frames, and each cell
holds how much energy that pitch had in that frame. A plain FFT spectrogram is
linear in Hz, which smears low notes together and wastes resolution up high;
binning into semitones puts the grid where music actually lives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_MIDI = 36   # C2
MAX_MIDI = 96   # C7
N_FFT = 4096
HOP = 512


def midi_to_hz(midi: np.ndarray | float) -> np.ndarray | float:
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)


def hz_to_midi(hz: np.ndarray | float) -> np.ndarray | float:
    hz = np.asarray(hz, dtype=np.float64)
    return 69.0 + 12.0 * np.log2(np.maximum(hz, 1e-9) / 440.0)


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[int(midi) % 12]}{int(midi) // 12 - 1}"


@dataclass
class SoundGraph:
    """`energy` is (n_notes, n_frames); row i is MIDI note `min_midi + i`."""

    energy: np.ndarray
    times: np.ndarray       # seconds, one per frame
    rms: np.ndarray         # per-frame loudness of the raw signal
    min_midi: int
    max_midi: int
    sr: int
    hop: int
    n_fft: int

    @property
    def midi_range(self) -> np.ndarray:
        return np.arange(self.min_midi, self.max_midi + 1)

    @property
    def frame_seconds(self) -> float:
        return self.hop / self.sr

    @property
    def half_window_frames(self) -> int:
        """How many frames a sound leads its own analysis window by."""
        return int(round((self.n_fft / 2) / self.hop))


def _semitone_filterbank(sr: int, n_fft: int, min_midi: int, max_midi: int) -> np.ndarray:
    """Triangular weights mapping FFT bins onto semitone rows.

    Each row spans one semitone (+/- 50 cents around its centre) and is
    triangle-weighted, so energy that sits between two notes is split rather
    than being forced into the wrong one.
    """
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    bin_midi = hz_to_midi(np.maximum(freqs, 1e-9))
    midis = np.arange(min_midi, max_midi + 1)

    distance = np.abs(bin_midi[None, :] - midis[:, None])  # in semitones
    bank = np.clip(1.0 - distance, 0.0, 1.0)

    # A single FFT bin is wider than a semitone down low, so some rows catch no
    # bin centre at all. Give those rows the nearest bin so they are not silent.
    for i, row in enumerate(bank):
        if not row.any():
            bank[i, int(np.argmin(distance[i]))] = 1.0

    norm = bank.sum(axis=1, keepdims=True)
    return bank / np.maximum(norm, 1e-12)


def build(
    samples: np.ndarray,
    sr: int,
    n_fft: int = N_FFT,
    hop: int = HOP,
    min_midi: int = MIN_MIDI,
    max_midi: int = MAX_MIDI,
) -> SoundGraph:
    """Turn samples into a SoundGraph."""
    if samples.size < n_fft:
        samples = np.pad(samples, (0, n_fft - samples.size))

    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (samples.size - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(n_frames, n_fft),
        strides=(samples.strides[0] * hop, samples.strides[0]),
    )

    spectrum = np.abs(np.fft.rfft(frames * window, axis=1))  # (frames, bins)
    bank = _semitone_filterbank(sr, n_fft, min_midi, max_midi)
    energy = (bank @ spectrum.T)  # (notes, frames)

    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    # Time-stamp a frame by where its window opens: a note is audible from the
    # moment it enters the window, and stamping the centre instead reports every
    # onset half a window late.
    times = np.arange(n_frames) * hop / sr

    return SoundGraph(
        energy=energy,
        times=times,
        rms=rms,
        min_midi=min_midi,
        max_midi=max_midi,
        sr=sr,
        hop=hop,
        n_fft=n_fft,
    )


_RAMP = " .:-=+*#%@"


def render_ascii(graph: SoundGraph, width: int = 100, low: int = 55, high: int = 88) -> str:
    """Print the sound graph as a piano roll you can actually read in a terminal."""
    lo = max(low, graph.min_midi)
    hi = min(high, graph.max_midi)
    block = graph.energy[lo - graph.min_midi: hi - graph.min_midi + 1]

    if block.shape[1] > width:
        edges = np.linspace(0, block.shape[1], width + 1).astype(int)
        block = np.stack([block[:, a:b].max(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)

    peak = float(block.max()) if block.size else 0.0
    lines = []
    for row_idx in range(block.shape[0] - 1, -1, -1):
        midi = lo + row_idx
        scaled = block[row_idx] / peak if peak > 0 else block[row_idx]
        cells = "".join(_RAMP[min(len(_RAMP) - 1, int(v * len(_RAMP)))] for v in scaled)
        lines.append(f"{note_name(midi):>4} |{cells}")
    total = graph.times[-1] if graph.times.size else 0.0
    lines.append(f"{'':>4} +{'-' * block.shape[1]}")
    lines.append(f"{'':>4}  0.0s{' ' * max(0, block.shape[1] - 10)}{total:.1f}s")
    return "\n".join(lines)
