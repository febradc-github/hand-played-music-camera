"""Chord engine: maps root note + modifier to MIDI note numbers.

Provides the ``ChordEngine`` class which converts chord names (e.g., "Cmaj", "F#m7")
into lists of MIDI note numbers for playback or analysis.
"""

from __future__ import annotations


class ChordEngine:
    """Resolves chord names to MIDI note numbers.

    Given a root note (A-G, optionally with 'b' or '#') and a chord modifier
    (e.g., "maj", "m", "7", "dim"), produces a list of MIDI note numbers
    representing the notes in the chord.

    Args:
        octave: Base octave for note-to-MIDI conversion.  Defaults to 4
            (middle C = C4 = MIDI 60).

    Attributes:
        octave: The base octave used for MIDI number calculation.
    """

    # Semitone offsets from C for each natural note letter.
    _NOTE_BASE: dict[str, int] = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11,
    }

    def __init__(self, octave: int = 4) -> None:
        """Initialize the chord engine with a base octave.

        Args:
            octave: MIDI octave offset.  C4 (middle C) is MIDI 60.
        """
        self.octave = octave
        self._intervals: dict[str, list[int]] = self._build_intervals()

    @staticmethod
    def _build_intervals() -> dict[str, list[int]]:
        """Return the mapping of chord modifiers to semitone intervals from the root.

        Each interval list specifies the distance in semitones above the root
        note for every note in the chord.

        Returns:
            Dictionary keyed by modifier name with values as lists of semitone
            intervals.  The root is always interval 0.
        """
        return {
            "maj":  [0, 4, 7],
            "m":    [0, 3, 7],
            "aug":  [0, 4, 8],
            "dim":  [0, 3, 6],
            "sus2": [0, 2, 7],
            "sus4": [0, 5, 7],
            "add9": [0, 4, 7, 14],
            "7":    [0, 4, 7, 10],
            "maj7": [0, 4, 7, 11],
            "m7":   [0, 3, 7, 10],
        }

    def _note_to_midi(self, note_name: str) -> int:
        """Convert a note name to its MIDI note number.

        The note name consists of a letter A-G, optionally followed by a
        single accidental ('b' for flat or '#' for sharp).  The stored
        :attr:`octave` determines the octave offset.

        MIDI formula: ``semitone + (octave + 1) * 12``, where:
            - C = 0, D = 2, E = 4, F = 5, G = 7, A = 9, B = 11
            - Flat subtracts 1, sharp adds 1

        Examples:
            - C4 (octave=4) -> 0 + 5*12 = 60
            - A4 (octave=4) -> 9 + 5*12 = 69
            - F#4 (octave=4) -> 6 + 5*12 = 66

        Args:
            note_name: Note string such as "C", "F#", "Bb".

        Returns:
            Integer MIDI note number.

        Raises:
            KeyError: If the note letter is not in A-G.
        """
        note_letter = note_name[0].upper()
        accidental = 0

        if len(note_name) > 1:
            if note_name[1] == "#":
                accidental = 1
            elif note_name[1] == "b":
                accidental = -1

        semitone = self._NOTE_BASE[note_letter] + accidental
        return semitone + (self.octave + 1) * 12

    def resolve(self, root: str, modifier: str) -> list[int]:
        """Resolve a chord name to a list of MIDI note numbers.

        Args:
            root: Root note name (e.g., "C", "F#", "Eb").
            modifier: Chord modifier key (e.g., "maj", "m", "7", "dim").

        Returns:
            List of MIDI note numbers comprising the chord, sorted by
            interval order (root first).

        Raises:
            ValueError: If *modifier* is not a known chord type.
            KeyError: If *root* contains an invalid note letter.
        """
        if modifier not in self._intervals:
            raise ValueError(
                f"Unknown modifier: {modifier!r}. "
                f"Known modifiers: {list(self._intervals.keys())}"
            )

        root_midi = self._note_to_midi(root)
        intervals = self._intervals[modifier]
        return [root_midi + interval for interval in intervals]

    def get_root_notes(self) -> list[str]:
        """Return the list of natural root note letters.

        Returns:
            List of strings ``["A", "B", "C", "D", "E", "F", "G"]``.
        """
        return ["A", "B", "C", "D", "E", "F", "G"]

    def get_modifiers(self) -> list[str]:
        """Return all known chord modifier names.

        Returns:
            List of modifier strings (e.g., ``["maj", "m", "7", ...]``).
        """
        return list(self._intervals.keys())
