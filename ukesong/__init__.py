"""ukesong: turn a recording into ukulele tab.

Pipeline: audio -> sound graph (semitone x time energy) -> note events ->
fretboard fingerings -> tab.
"""

from .audio import load
from .pitch import NoteEvent, segment
from .spectral import SoundGraph, build, render_ascii
from .uke import Fingering, LOW_G_GCEA, STANDARD_GCEA, arrange
from .tab import render_notes, render_summary, render_tab

__all__ = [
    "load", "build", "render_ascii", "SoundGraph",
    "segment", "NoteEvent", "arrange", "Fingering",
    "STANDARD_GCEA", "LOW_G_GCEA",
    "render_tab", "render_notes", "render_summary",
]


def transcribe(path: str, tuning=None, max_fret: int = 12, min_duration: float = 0.07):
    """Convenience: file in, (sound graph, note events, fingerings) out."""
    samples, sr = load(path)
    graph = build(samples, sr)
    events = segment(graph, min_duration=min_duration)
    return graph, events, arrange(events, tuning=tuning, max_fret=max_fret)
