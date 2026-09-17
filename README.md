# ukesong

Turn a recording into notes you can play on a ukulele.

The idea it is built on: before anything can be transcribed, the sound has to
become *readable*. So the first stage turns audio into a **sound graph** — a
grid of semitone rows against time, where each cell is how much energy that
pitch had at that moment. Everything after that is reading the graph.

```
  A4 |                           -*#***********#*-
  G4 |            :*##############=             =#######-
  F4 |                                                 =#%%%%%%#%%%%%%%=.
  E4 |                                                 .:.    :-:    .+%%%%%%%#%%%%%%%.
  D4 |                                                                               #@@@@@@@%@@@@@@#.
  C4 |@@@@@@@@@@@@@*.                                                                              .%@@@@@
     +----------------------------------------------------------------------------------------------------
      0.0s                                                                                          5.5s
```

## Install and run

```bash
pip install numpy            # the only dependency
python -m ukesong song.wav                 # tab
python -m ukesong song.wav --notes --graph # note sheet and sound graph too
python -m ukesong song.wav --json          # machine-readable
```

WAV files decode with the standard library. Anything else (mp3, m4a, ...) is
handed to `ffmpeg` if it is installed.

```
A|---------0-0-----------------------------
E|---------------1-1-0-0-------------------
C|-0-0-------------------------2-2-0-------
G|-----0-0-----0---------------------------
```

## How it works

| Stage | Module | What it does |
| --- | --- | --- |
| Load | `audio.py` | Any input → mono float samples at 22.05 kHz |
| Sound graph | `spectral.py` | STFT → triangular semitone filterbank → energy per note per frame |
| Pitch | `pitch.py` | Harmonic-comb salience picks the fundamental, not its overtones |
| Onsets | `pitch.py` | Spectral flux against a local threshold, so a note struck twice stays two notes |
| Segment | `pitch.py` | Frame track → note events with start, duration, confidence |
| Fretboard | `uke.py` | Each note → the places it can be played on GCEA; Viterbi picks the path with the least hand movement |
| Render | `tab.py` | Four-line ASCII tab, note sheet, summary |

Three choices are worth calling out:

- **Semitone rows, not FFT bins.** A linear spectrogram wastes resolution up
  high and smears notes together down low. Music is logarithmic; the graph is
  too.
- **Harmonic comb.** A plucked string puts energy on its overtones as well as
  its fundamental, so the loudest bin is often an octave or a fifth above the
  note being played. Summing along the harmonic series scores the true
  fundamental highest.
- **Fingering is a path, not a choice per note.** Picking the lowest fret for
  each note in isolation sends your hand sliding up and down the neck. The
  arranger optimises across the whole phrase.

Notes below the uke's range are moved up by whole octaves (and flagged in the
note sheet) rather than dropped.

## What it does and does not do yet

Works: single-line melodies — whistling, humming, a solo instrument, a lead
line. Handles repeated notes, octave-shifts out-of-range notes, and keeps the
fingering in one hand position.

Not yet:

- **Polyphony.** One pitch per frame, so a full mix gives you whatever
  dominates. Feed it a melody, or separate the stems first.
- **Chords.** No chord detection or strumming charts — the graph already holds
  what is needed (chroma per frame), it just needs a template matcher.
- **Rhythm.** Durations are in seconds, not beats. Tempo estimation from the
  onset track, then quantising to a grid, is the next step.
- **Export.** Only ASCII and JSON; MusicXML or MIDI would open this up to
  notation software.

## Tests

```bash
python -m unittest discover -s tests
```

The suite synthesises its own audio (overtone-rich tones with attack and
release envelopes), so it checks the real pipeline end to end — pitch
accuracy, octave errors, repeated notes, timing, fretboard correctness and the
CLI — without shipping audio fixtures.
