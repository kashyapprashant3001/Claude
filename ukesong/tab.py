"""Fingerings -> printable ukulele tab and note sheets."""

from __future__ import annotations

from .uke import Fingering, STRING_LABELS


def render_tab(fingerings: list[Fingering], columns: int = 64, labels: list[str] | None = None) -> str:
    """Four-line ASCII tab, wrapped into staves of `columns` characters."""
    labels = labels or STRING_LABELS
    if not fingerings:
        return "(nothing playable was detected)"

    cells: list[list[str]] = []
    for f in fingerings:
        column = ["-" * len(str(f.fret))] * 4
        column[f.string] = str(f.fret)
        cells.append(column)

    staves: list[str] = []
    line_width = 0
    buffers = [f"{label}|" for label in labels]
    for column in cells:
        width = len(column[0]) + 1  # note plus a separator dash
        if line_width + width > columns and line_width > 0:
            staves.append("\n".join(b + "-" for b in buffers))
            buffers = [f"{label}|" for label in labels]
            line_width = 0
        for i in range(4):
            buffers[i] += "-" + column[i]
        line_width += width
    staves.append("\n".join(b + "-" for b in buffers))
    return "\n\n".join(staves)


def render_notes(fingerings: list[Fingering]) -> str:
    """One line per note: when it happens, what it is, and where to put a finger."""
    lines = [f"{'time':>7}  {'dur':>5}  {'note':<5} {'string':<7} fret"]
    for f in fingerings:
        octave = ""
        if f.octave_shift:
            direction = "up" if f.octave_shift > 0 else "down"
            octave = f"  ({abs(f.octave_shift) // 12} oct {direction})"
        lines.append(
            f"{f.event.start:7.2f}  {f.event.duration:5.2f}  "
            f"{f.name:<5} {STRING_LABELS[f.string]:<7} {f.fret}{octave}"
        )
    return "\n".join(lines)


def render_summary(fingerings: list[Fingering]) -> str:
    if not fingerings:
        return "no notes detected"
    frets = [f.fret for f in fingerings if f.fret > 0]
    span = f"{min(frets)}-{max(frets)}" if frets else "open strings only"
    shifted = sum(1 for f in fingerings if f.octave_shift)
    end = max(f.event.end for f in fingerings)
    return (
        f"{len(fingerings)} notes over {end:.1f}s | fret range {span}"
        + (f" | {shifted} note(s) octave-shifted to fit the uke" if shifted else "")
    )
