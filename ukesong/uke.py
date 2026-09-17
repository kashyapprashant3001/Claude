"""Notes -> ukulele fingerings.

Standard (re-entrant) GCEA tuning. String 1 is the A closest to the floor when
you hold the uke; that is the order the tab is printed in.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pitch import NoteEvent
from .spectral import note_name

# index 0 == string 1 (A, printed on top), index 3 == string 4 (G)
STANDARD_GCEA = [69, 64, 60, 67]  # A4, E4, C4, G4
STRING_LABELS = ["A", "E", "C", "G"]
LOW_G_GCEA = [69, 64, 60, 55]     # same, with a wound low G3


@dataclass
class Fingering:
    string: int       # 0-based index into the tuning list
    fret: int
    event: NoteEvent
    octave_shift: int  # semitones the note was moved to fit the fretboard

    @property
    def midi(self) -> int:
        return self.event.midi + self.octave_shift

    @property
    def name(self) -> str:
        return note_name(self.midi)


def candidates(midi: int, tuning: list[int], max_fret: int) -> list[tuple[int, int]]:
    out = []
    for string, open_midi in enumerate(tuning):
        fret = midi - open_midi
        if 0 <= fret <= max_fret:
            out.append((string, fret))
    return out


def fit_to_range(midi: int, tuning: list[int], max_fret: int) -> tuple[int, int] | None:
    """Move `midi` by whole octaves until it is playable. Returns (midi, shift)."""
    lowest = min(tuning)
    highest = max(tuning) + max_fret
    shifted = midi
    shift = 0
    while shifted < lowest:
        shifted += 12
        shift += 12
    while shifted > highest:
        shifted -= 12
        shift -= 12
    if candidates(shifted, tuning, max_fret):
        return shifted, shift
    # Landed in a gap (only possible with odd tunings) - try one octave either way.
    for alt in (shifted + 12, shifted - 12):
        if candidates(alt, tuning, max_fret):
            return alt, shift + (alt - shifted)
    return None


def _cost(prev_fret: int | None, string: int, fret: int, prev_string: int | None) -> float:
    cost = 0.35 * fret  # open and low frets are easier
    if prev_fret is not None and fret > 0 and prev_fret > 0:
        cost += 1.2 * abs(fret - prev_fret)  # hand travel dominates
    if prev_string is not None:
        cost += 0.15 * abs(string - prev_string)
    return cost


def arrange(
    events: list[NoteEvent],
    tuning: list[int] | None = None,
    max_fret: int = 12,
) -> list[Fingering]:
    """Pick one fingering per note, minimising hand movement across the phrase.

    Each note has several places it could be played; choosing greedily sends you
    sliding up and down the neck. This is a Viterbi pass over the whole melody,
    so the fingering is chosen for the phrase rather than for one note.
    """
    tuning = tuning or STANDARD_GCEA
    playable: list[tuple[NoteEvent, int, int, list[tuple[int, int]]]] = []
    for ev in events:
        fitted = fit_to_range(ev.midi, tuning, max_fret)
        if fitted is None:
            continue
        midi, shift = fitted
        playable.append((ev, midi, shift, candidates(midi, tuning, max_fret)))

    if not playable:
        return []

    # Viterbi: best[i][c] = cheapest path ending on candidate c of note i.
    best: list[list[float]] = []
    back: list[list[int]] = []
    for i, (_, _, _, cands) in enumerate(playable):
        row_cost, row_back = [], []
        for string, fret in cands:
            if i == 0:
                row_cost.append(_cost(None, string, fret, None))
                row_back.append(-1)
                continue
            prev_cands = playable[i - 1][3]
            scored = [
                best[i - 1][j] + _cost(pf, string, fret, ps)
                for j, (ps, pf) in enumerate(prev_cands)
            ]
            j = min(range(len(scored)), key=scored.__getitem__)
            row_cost.append(scored[j])
            row_back.append(j)
        best.append(row_cost)
        back.append(row_back)

    idx = min(range(len(best[-1])), key=best[-1].__getitem__)
    chosen: list[int] = [idx]
    for i in range(len(playable) - 1, 0, -1):
        idx = back[i][idx]
        chosen.append(idx)
    chosen.reverse()

    return [
        Fingering(string=cands[c][0], fret=cands[c][1], event=ev, octave_shift=shift)
        for (ev, _, shift, cands), c in zip(playable, chosen)
    ]
