"""Sound graph -> a melody: a list of note events with pitch, start and duration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral import SoundGraph, note_name

# Harmonics 1..6 land these many semitones above the fundamental. A real note
# lights up all of them, so summing along this comb scores the fundamental far
# above its own overtones (which is what makes a plain "loudest bin" reading
# jump an octave).
HARMONIC_OFFSETS = [0, 12, 19, 24, 28, 31]
HARMONIC_WEIGHTS = [1.0, 0.55, 0.4, 0.28, 0.2, 0.15]


@dataclass
class NoteEvent:
    midi: int
    start: float      # seconds
    duration: float   # seconds
    confidence: float

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def name(self) -> str:
        return note_name(self.midi)


def harmonic_salience(graph: SoundGraph) -> np.ndarray:
    """Score every candidate fundamental by summing its harmonic series."""
    energy = graph.energy
    n_notes = energy.shape[0]
    salience = np.zeros_like(energy)
    for offset, weight in zip(HARMONIC_OFFSETS, HARMONIC_WEIGHTS):
        if offset >= n_notes:
            break
        shifted = np.zeros_like(energy)
        if offset == 0:
            shifted = energy
        else:
            shifted[: n_notes - offset] = energy[offset:]
        salience += weight * shifted
    return salience


def onset_frames(
    graph: SoundGraph,
    min_separation: float = 0.1,
    sensitivity: float = 1.0,
) -> np.ndarray:
    """Frames where a new note is struck.

    Pitch alone cannot tell one long note from the same note played twice, so a
    repeated note would otherwise collapse into a single event. Spectral flux -
    how much energy *rose* between frames - spikes at every fresh attack,
    including one on a pitch that is already sounding.
    """
    energy = graph.energy
    if energy.shape[1] < 3:
        return np.zeros(0, dtype=int)

    flux = np.diff(energy, axis=1)
    flux = np.clip(flux, 0, None).sum(axis=0)
    flux = np.concatenate([[0.0], flux])

    if not flux.any():
        return np.zeros(0, dtype=int)

    # A fixed global threshold only finds attacks in the loudest passage, so
    # compare each frame against its own neighbourhood instead.
    span = max(3, int(round(0.4 / graph.frame_seconds)) | 1)
    local_mean = _moving_average(flux, span)
    threshold = local_mean + sensitivity * flux.std() * 0.5
    gap = max(1, int(round(min_separation / graph.frame_seconds)))

    peaks: list[int] = []
    for i in range(1, len(flux) - 1):
        if flux[i] < threshold[i] or flux[i] < flux[i - 1] or flux[i] < flux[i + 1]:
            continue
        if peaks and i - peaks[-1] < gap:
            if flux[i] > flux[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)

    # Energy starts rising as soon as the note edges into the analysis window,
    # half a window before it is actually struck; correct for that lead so the
    # onset lands on the attack.
    shifted = np.asarray(peaks, dtype=int) + graph.half_window_frames
    return shifted[shifted < energy.shape[1]]


def _moving_average(x: np.ndarray, size: int) -> np.ndarray:
    pad = size // 2
    padded = np.pad(x, pad, mode="edge")
    kernel = np.ones(size) / size
    return np.convolve(padded, kernel, mode="valid")[: x.size]


def _median_filter(x: np.ndarray, size: int) -> np.ndarray:
    if size < 3 or x.size < size:
        return x
    pad = size // 2
    padded = np.pad(x, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    return np.median(windows, axis=1)


def track(
    graph: SoundGraph,
    silence_db: float = -38.0,
    smoothing: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Per frame, pick the most salient pitch. Returns (midi, voiced)."""
    salience = harmonic_salience(graph)
    best = salience.argmax(axis=0) + graph.min_midi
    strength = salience.max(axis=0)

    rms = graph.rms
    ref = float(rms.max()) if rms.size else 0.0
    if ref > 0:
        db = 20.0 * np.log10(np.maximum(rms, 1e-10) / ref)
        voiced = db > silence_db
    else:
        voiced = np.zeros_like(rms, dtype=bool)

    # Frames with no clear winner are noise, not notes.
    positive = strength[strength > 0]
    if positive.size:
        voiced &= strength > 0.12 * float(np.median(positive))

    smoothed = _median_filter(best.astype(float), smoothing)
    return np.rint(smoothed).astype(int), voiced


def segment(
    graph: SoundGraph,
    min_duration: float = 0.07,
    silence_db: float = -38.0,
    smoothing: int = 5,
) -> list[NoteEvent]:
    """Collapse the frame-by-frame pitch track into discrete note events."""
    midi, voiced = track(graph, silence_db=silence_db, smoothing=smoothing)
    salience = harmonic_salience(graph)
    frame_seconds = graph.frame_seconds
    onsets = set(onset_frames(graph).tolist())

    events: list[NoteEvent] = []
    start = None
    for i in range(len(midi) + 1):
        active = i < len(midi) and voiced[i]
        same = active and start is not None and midi[i] == midi[start]
        if same and i not in onsets:
            continue
        restruck = same and i in onsets
        if start is not None:
            run = slice(start, i)
            duration = (i - start) * frame_seconds
            if duration >= min_duration:
                strengths = salience[midi[start] - graph.min_midi, run]
                peak = float(salience[:, run].max()) or 1.0
                events.append(
                    NoteEvent(
                        midi=int(midi[start]),
                        start=float(graph.times[start]),
                        duration=float(duration),
                        confidence=float(np.mean(strengths) / peak),
                    )
                )
            start = None
        if active:
            start = i
        if restruck:
            start = i

    events = _merge_repeats(events, onsets=onsets, graph=graph)
    return _ring_out(events, graph)


def _ring_out(events: list[NoteEvent], graph: SoundGraph, max_gap: float = 0.15) -> list[NoteEvent]:
    """Let a note ring until the next one is struck.

    An analysis window several notes wide clips the quiet tail off each event.
    On a ukulele a plucked string keeps sounding until the next note, so closing
    small gaps recovers the true rhythm instead of inventing rests.
    """
    for prev, nxt in zip(events, events[1:]):
        gap = nxt.start - prev.end
        if 0 < gap <= max_gap:
            prev.duration = nxt.start - prev.start

    # The last note has nothing to ring into, and the analysis window clips its
    # tail like any other; give it back that half window, up to the end of the
    # recording.
    if events and graph.times.size:
        audio_end = float(graph.times[-1]) + graph.n_fft / graph.sr
        last = events[-1]
        last.duration = min(last.duration + graph.half_window_frames * graph.frame_seconds,
                            audio_end - last.start)
    return events


def _merge_repeats(
    events: list[NoteEvent],
    gap: float = 0.05,
    onsets: set[int] | None = None,
    graph: SoundGraph | None = None,
) -> list[NoteEvent]:
    """Join same-pitch events split by a dropped frame or two - unless an attack
    sits in the gap, in which case they really are two notes."""
    onsets = onsets or set()
    merged: list[NoteEvent] = []
    for ev in events:
        struck_again = False
        if merged and graph is not None:
            lo = merged[-1].end - graph.frame_seconds
            hi = ev.start + graph.frame_seconds
            struck_again = any(lo <= graph.times[f] <= hi for f in onsets if f < graph.times.size)
        if merged and not struck_again and merged[-1].midi == ev.midi and ev.start - merged[-1].end <= gap:
            prev = merged[-1]
            prev.duration = ev.end - prev.start
            prev.confidence = max(prev.confidence, ev.confidence)
        else:
            merged.append(ev)
    return merged
