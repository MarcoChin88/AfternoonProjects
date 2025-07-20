from __future__ import annotations
import colorsys
import re
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from itertools import cycle

import sounddevice as sd
import librosa
import numpy as np

PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def pitch_to_rgb(pitch: str, _255: bool = False):
    pitch = pitch.replace('♯', '#')
    hue = PITCHES.index(pitch) / len(PITCHES)

    rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)

    if _255:
        tuple(int(round(c * 255)) for c in rgb)

    return rgb


@dataclass
class ChordType:
    shorthand: str
    offsets: tuple[int, ...]


class ChordTypes(ChordType, Enum):
    MAJOR = "", (4, 7)
    MINOR = "m", (3, 7)

    MAJOR7 = "maj7", (4, 7, 11)
    DOMINANT7 = "7", (4, 7, 10)
    MINOR7 = "m7", (3, 7, 10)
    MINOR_MAJOR7 = "m(maj7)", (3, 7, 11)

    DIMINISHED = "dim", (3, 6)
    DIMINISHED7 = "dim7", (3, 6, 9)
    HALF_DIMINISHED7 = "m7b5", (3, 6, 10)

    AUGMENTED = "aug", (4, 8)
    AUGMENTED7 = "+7", (4, 8, 10)

    SUSPENDED2 = "sus2", (2, 7)
    SUSPENDED4 = "sus4", (5, 7)

    MAJOR6 = "6", (4, 7, 9)
    MINOR6 = "m6", (3, 7, 9)


CHORD_SHORTHANDS = {_.value.shorthand: _ for _ in ChordTypes}


@dataclass
class Note:
    note_name: str

    @cached_property
    def pitch_and_octave(self) -> tuple[str, int]:
        match = re.match(r'^([A-Ga-g][#♯]?|[A-Ga-g]b?)(-?\d+)$', self.note_name)
        pitch, octave = match.groups()
        pitch = pitch.replace('♯', '#')
        return pitch.upper(), int(octave)

    def get_rgb(self, _255: bool = False):
        pitch, octave = self.pitch_and_octave
        hue = PITCHES.index(pitch) / len(PITCHES)

        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)

        if _255:
            tuple(int(round(c * 255)) for c in rgb)

        return rgb

    @cached_property
    def midi(self) -> int:
        return librosa.note_to_midi(self.note_name)

    @cached_property
    def freq(self) -> float:
        return float(librosa.note_to_hz(self.note_name))

    def get_tone(self, duration=1.0, sr=44100, taper_pct: float = 0.05):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        waveform = np.sin(2 * np.pi * self.freq * t)

        n = len(waveform)
        taper_len = int(taper_pct * n)

        if taper_len > 0:
            hanning_window = np.hanning(2 * taper_len)
            fade_in = hanning_window[:taper_len]
            fade_out = hanning_window[taper_len:]

            envelope = np.ones(n)
            envelope[:taper_len] = fade_in
            envelope[-taper_len:] = fade_out

            waveform *= envelope

        return waveform

    def play_tone(self, duration=1.0, sr=44100):
        sd.play(self.get_tone(duration, sr))
        sd.wait()

    def get_note_by_offset(self, offset: int) -> Note:
        if offset == 0:
            return self

        pitch, octave = self.pitch_and_octave
        pitch_idx: int = PITCHES.index(pitch)

        octave_offset, offest_pitch_idx = divmod(pitch_idx + offset, len(PITCHES))

        new_octave = octave + octave_offset
        new_pitch = PITCHES[offest_pitch_idx]

        return Note(note_name=f"{new_pitch}{new_octave}")

    def get_chord(self, chord_type: ChordTypes = ChordTypes.MAJOR):
        return Chord(root=self, chord_type=chord_type)


@dataclass
class Chord:
    root: Note
    chord_type: ChordTypes

    def get_chord_notes(self):
        notes = [self.root] + [self.root.get_note_by_offset(o) for o in self.chord_type.value.offsets]
        notes = [n.get_note_by_offset(-12) for n in notes] + notes
        return notes

    def get_chord_tone(self, duration: float = 1.0):
        return np.sum(np.array([_.get_tone(duration=duration) for _ in self.get_chord_notes()]), axis=0)

    def play_chord(self, duration: float = 1.0):
        sd.play(self.get_chord_tone(duration=duration))
        sd.wait()

    @classmethod
    def from_name(cls, name: str, octave: int = 3):
        match = re.compile(r'^([A-Ga-g][#b]?)(.*)$').match(name.strip())

        root, chord_type = match.groups()
        root_note = f"{root.upper()}{octave}"

        return Chord(root=Note(root_note), chord_type=CHORD_SHORTHANDS[chord_type.lower()])


def get_notes(min_note: str, max_note: str) -> list[Note]:
    min_note_midi: int = librosa.note_to_midi(min_note)
    max_note_midi: int = librosa.note_to_midi(max_note)

    return [
        Note(note_name=librosa.midi_to_note(midi))
        for midi in range(min_note_midi, max_note_midi + 1)
    ]


def play_chord(notes: list[Note], duration=1):
    tones = [n.get_tone(duration=duration) for n in notes]
    signal = np.sum(tones, axis=0)

    sd.play(signal)
    sd.wait()


if __name__ == "__main__":
    cp1 = cycle(Chord.from_name(_) for _ in [
        'E7', 'G7', 'A7', 'A#dim7', 'B7'
    ])
    cp2 = cycle(Chord.from_name(_) for _ in [
        'Fmaj7', 'G7', 'Em7', 'Am7'
    ])

    cp3 = cycle(
        Chord.from_name(_.strip(), octave=4)
        for _ in '| Am7 | D7 | Bm7 | E7 |'.split('|')
        if _.strip()
    )
    cp4 = cycle(
        Chord.from_name(_.strip(), octave=4)
        for _ in '|F |G|D|Am|'.split('|')
        if _.strip()
    )
    cp5 = cycle(
        Chord.from_name(_.strip(), octave=3)
        for _ in '|F |G|D|Am|'.split('|')
        if _.strip()
    )

    # for n in cycle(get_notes('C3', 'C4')):
    #     n.play_tone(duration=0.75)
    #
    note_dur = 2
    for c in cp4:
        c.play_chord(duration=note_dur)
