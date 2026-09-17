"""Command line entry point: `python -m ukesong song.wav`."""

from __future__ import annotations

import argparse
import json
import sys

from . import transcribe
from .spectral import render_ascii
from .tab import render_notes, render_summary, render_tab
from .uke import LOW_G_GCEA, STANDARD_GCEA, STRING_LABELS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ukesong",
        description="Convert a song into ukulele notes and tab.",
    )
    p.add_argument("audio", help="path to an audio file (WAV directly; others need ffmpeg)")
    p.add_argument("--tuning", choices=["gcea", "low-g"], default="gcea")
    p.add_argument("--max-fret", type=int, default=12)
    p.add_argument("--min-duration", type=float, default=0.07,
                   help="ignore notes shorter than this many seconds")
    p.add_argument("--columns", type=int, default=64, help="tab width in characters")
    p.add_argument("--graph", action="store_true", help="also print the sound graph")
    p.add_argument("--notes", action="store_true", help="also print the note-by-note sheet")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON instead")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tuning = LOW_G_GCEA if args.tuning == "low-g" else STANDARD_GCEA

    try:
        graph, events, fingerings = transcribe(
            args.audio, tuning=tuning, max_fret=args.max_fret, min_duration=args.min_duration
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ukesong: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({
            "tuning": args.tuning,
            "notes": [
                {
                    "note": f.name,
                    "midi": f.midi,
                    "start": round(f.event.start, 3),
                    "duration": round(f.event.duration, 3),
                    "string": STRING_LABELS[f.string],
                    "fret": f.fret,
                    "octave_shift": f.octave_shift,
                    "confidence": round(f.event.confidence, 3),
                }
                for f in fingerings
            ],
        }, indent=2))
        return 0

    if args.graph:
        print("sound graph (semitone rows x time):\n")
        print(render_ascii(graph))
        print()
    if args.notes:
        print(render_notes(fingerings))
        print()
    print(render_tab(fingerings, columns=args.columns))
    print()
    print(render_summary(fingerings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
