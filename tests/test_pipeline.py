import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.synth import SR, melody, tone, write_wav  # noqa: E402
from ukesong import audio, pitch, spectral, tab, uke  # noqa: E402

# C4 D4 E4 F4 G4 A4 B4 C5
SCALE = [60, 62, 64, 65, 67, 69, 71, 72]


def graph_for(samples, sr=SR):
    return spectral.build(samples, sr)


class TestSoundGraph(unittest.TestCase):
    def test_single_note_peaks_on_its_own_row(self):
        graph = graph_for(tone(69, 1.0))  # A4
        row = graph.energy.mean(axis=1).argmax() + graph.min_midi
        self.assertEqual(int(row), 69)

    def test_graph_shape_and_times(self):
        graph = graph_for(tone(60, 0.5))
        self.assertEqual(graph.energy.shape[0], graph.max_midi - graph.min_midi + 1)
        self.assertEqual(graph.energy.shape[1], graph.times.size)
        self.assertLess(graph.times[-1], 0.6)

    def test_ascii_render_is_readable(self):
        rendered = spectral.render_ascii(graph_for(tone(69, 0.5)), width=40)
        self.assertIn("A4", rendered)
        self.assertTrue(any(ch in rendered for ch in "@%#"))


class TestPitchTracking(unittest.TestCase):
    def test_low_note_is_not_read_an_octave_high(self):
        """Overtone-rich tones are where naive peak-picking fails."""
        graph = graph_for(tone(48, 1.0, harmonics=6))  # C3
        midi, voiced = pitch.track(graph)
        self.assertEqual(int(np.median(midi[voiced])), 48)

    def test_scale_is_transcribed_in_order(self):
        samples = melody([(m, 0.35) for m in SCALE])
        events = pitch.segment(graph_for(samples), min_duration=0.1)
        self.assertEqual([e.midi for e in events], SCALE)

    def test_durations_are_about_right(self):
        events = pitch.segment(graph_for(melody([(64, 0.5), (67, 0.5)])), min_duration=0.1)
        self.assertEqual(len(events), 2)
        for ev in events:
            self.assertAlmostEqual(ev.duration, 0.5, delta=0.12)

    def test_repeated_notes_are_not_merged(self):
        """Same pitch struck twice must stay two notes - pitch alone can't tell."""
        samples = melody([(64, 0.4), (64, 0.4), (67, 0.4)])
        events = pitch.segment(graph_for(samples), min_duration=0.1)
        self.assertEqual([e.midi for e in events], [64, 64, 67])

    def test_onsets_land_on_the_attacks(self):
        samples = melody([(60, 0.5), (60, 0.5), (64, 0.5)])
        graph = graph_for(samples)
        times = [float(graph.times[f]) for f in pitch.onset_frames(graph)]
        # The first note-on is found by the voicing test; onsets exist to catch
        # the re-attacks that pitch alone would miss.
        self.assertEqual(len(times), 2)
        for expected, actual in zip([0.5, 1.0], times):
            self.assertAlmostEqual(actual, expected, delta=0.1)

    def test_note_boundaries_track_the_real_rhythm(self):
        samples = melody([(60, 0.4), (62, 0.4), (64, 0.4)])
        events = pitch.segment(graph_for(samples), min_duration=0.1)
        self.assertEqual(len(events), 3)
        for i, ev in enumerate(events):
            self.assertAlmostEqual(ev.start, i * 0.4, delta=0.1)

    def test_silence_produces_no_notes(self):
        quiet = np.zeros(SR, dtype=np.float32)
        self.assertEqual(pitch.segment(graph_for(quiet)), [])


class TestFretboard(unittest.TestCase):
    def test_open_strings_map_to_fret_zero(self):
        for string, open_midi in enumerate(uke.STANDARD_GCEA):
            self.assertIn((string, 0), uke.candidates(open_midi, uke.STANDARD_GCEA, 12))

    def test_out_of_range_note_is_octave_shifted(self):
        fitted = uke.fit_to_range(36, uke.STANDARD_GCEA, 12)  # C2, far below the uke
        self.assertIsNotNone(fitted)
        midi, shift = fitted
        self.assertEqual(shift % 12, 0)
        self.assertGreaterEqual(midi, min(uke.STANDARD_GCEA))

    def test_arrangement_keeps_the_hand_in_one_position(self):
        events = [pitch.NoteEvent(m, i * 0.5, 0.5, 1.0) for i, m in enumerate(SCALE)]
        fingerings = uke.arrange(events, max_fret=12)
        self.assertEqual(len(fingerings), len(SCALE))
        fretted = [f.fret for f in fingerings if f.fret > 0]
        self.assertLessEqual(max(fretted) - min(fretted), 4)

    def test_every_fingering_sounds_the_note_it_claims(self):
        events = [pitch.NoteEvent(m, i * 0.5, 0.5, 1.0) for i, m in enumerate(SCALE)]
        for f in uke.arrange(events, max_fret=12):
            self.assertEqual(uke.STANDARD_GCEA[f.string] + f.fret, f.midi)


class TestTab(unittest.TestCase):
    def test_tab_has_four_strings_and_marks_the_fret(self):
        events = [pitch.NoteEvent(69, 0.0, 0.5, 1.0)]  # open A
        rendered = tab.render_tab(uke.arrange(events))
        lines = rendered.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].startswith("A|"))
        self.assertIn("0", lines[0])

    def test_tab_wraps_into_staves(self):
        events = [pitch.NoteEvent(60 + (i % 12), i * 0.2, 0.2, 1.0) for i in range(60)]
        rendered = tab.render_tab(uke.arrange(events), columns=32)
        self.assertGreater(len(rendered.split("\n\n")), 1)

    def test_empty_input_is_reported_not_crashed(self):
        self.assertIn("nothing playable", tab.render_tab([]))


class TestEndToEnd(unittest.TestCase):
    def test_cli_transcribes_a_wav_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scale.wav")
            write_wav(path, melody([(m, 0.35) for m in SCALE]))
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            result = subprocess.run(
                [sys.executable, "-m", "ukesong", path, "--notes", "--graph"],
                capture_output=True, text=True, cwd=root,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("A|", result.stdout)
            self.assertIn("C4", result.stdout)
            self.assertIn("notes over", result.stdout)

    def test_stereo_and_resampling_survive_the_round_trip(self):
        import wave as wave_mod
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stereo.wav")
            mono = melody([(69, 0.6), (72, 0.6)], sr=44100)
            stereo = np.repeat(mono[:, None], 2, axis=1).reshape(-1)
            with wave_mod.open(path, "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes((stereo * 32767).astype("<i2").tobytes())

            samples, sr = audio.load(path)
            self.assertEqual(sr, audio.WORKING_SR)
            events = pitch.segment(spectral.build(samples, sr), min_duration=0.1)
            self.assertEqual([e.midi for e in events], [69, 72])

    def test_missing_file_exits_with_an_error(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            [sys.executable, "-m", "ukesong", "nope.wav"],
            capture_output=True, text=True, cwd=root,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("ukesong:", result.stderr)


if __name__ == "__main__":
    unittest.main()
