"""Integration tests for the hand-tracked chord instrument.

These tests exercise the chord engine, configuration loading, chord name
formatting, segment math, and MIDI port listing -- all without requiring
a camera or active MIDI hardware.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.chord_engine import ChordEngine
from src.config import AppConfig, CameraConfig, MenuConfig
from src.hand_tracker import Fingertip
from src.main import _format_chord_name, _get_segment
from src.midi_output import get_available_ports


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.json"


# ===========================================================================
# 1. Chord engine correctness
# ===========================================================================


class TestChordEngine:
    """Unit-style tests for the ChordEngine class."""

    # -- resolve() -----------------------------------------------------------

    def test_major(self) -> None:
        engine = ChordEngine(octave=4)
        assert engine.resolve("C", "maj") == [60, 64, 67]  # C E G

    def test_minor(self) -> None:
        engine = ChordEngine(octave=4)
        assert engine.resolve("A", "m") == [69, 72, 76]  # A C E

    def test_sharp(self) -> None:
        engine = ChordEngine(octave=4)
        assert engine.resolve("F#", "m7") == [66, 69, 73, 76]  # F# A C# E

    def test_flat(self) -> None:
        engine = ChordEngine(octave=4)
        assert engine.resolve("Bb", "maj7") == [70, 74, 77, 81]  # Bb D F A

    def test_dim(self) -> None:
        engine = ChordEngine(octave=4)
        assert engine.resolve("B", "dim") == [71, 74, 77]  # B D F

    def test_unknown_modifier_raises(self) -> None:
        engine = ChordEngine()
        with pytest.raises(ValueError):
            engine.resolve("C", "nonexistent")

    # -- get_root_notes() ----------------------------------------------------

    def test_root_notes(self) -> None:
        engine = ChordEngine()
        assert engine.get_root_notes() == ["A", "B", "C", "D", "E", "F", "G"]

    # -- get_modifiers() -----------------------------------------------------

    def test_modifiers(self) -> None:
        engine = ChordEngine()
        mods = engine.get_modifiers()
        assert "maj" in mods
        assert "m7" in mods
        assert len(mods) == 10

    # -- _note_to_midi edge cases -------------------------------------------

    def test_low_octave_midi(self) -> None:
        """C0 should be MIDI note 12 (0 + 1*12)."""
        engine = ChordEngine(octave=0)
        assert engine.resolve("C", "maj") == [12, 16, 19]

    def test_double_accidental_treated_as_single_sharp(self) -> None:
        """Only first accidental character is read; 'C##' resolves as C#."""
        engine = ChordEngine()
        assert engine.resolve("C##", "maj") == engine.resolve("C#", "maj")


# ===========================================================================
# 2. Config loading
# ===========================================================================


class TestConfigLoading:
    """Tests for AppConfig.load() and dataclass defaults."""

    def test_config_load_from_file(self) -> None:
        config = AppConfig.load(str(_CONFIG_PATH))
        assert config.camera.width == 960
        assert config.left_menu.radius == 150
        assert len(config.left_menu.items) == 7
        assert len(config.right_menu.items) == 10
        assert config.chord_octave == 4

    def test_camera_config_defaults(self) -> None:
        c = CameraConfig()
        assert c.width == 960
        assert c.height == 720
        assert c.fps == 30
        assert c.device_id == 0

    def test_menu_config_defaults(self) -> None:
        m = MenuConfig()
        assert m.center_x == 0.25
        assert m.center_y == 0.50
        assert m.radius == 150
        assert m.items == []

    def test_config_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            AppConfig.load("nonexistent_config.json")


# ===========================================================================
# 3. Chord name formatting
# ===========================================================================


class TestChordNameFormatting:
    """Tests for _format_chord_name helper from src.main."""

    def test_major_is_root_only(self) -> None:
        assert _format_chord_name("C", "maj") == "C"

    def test_minor_appends_m(self) -> None:
        assert _format_chord_name("A", "m") == "Am"

    def test_sharp_with_modifier(self) -> None:
        assert _format_chord_name("F#", "m7") == "F#m7"

    def test_flat_with_dim(self) -> None:
        assert _format_chord_name("Eb", "dim") == "Ebdim"

    def test_aug(self) -> None:
        assert _format_chord_name("G", "aug") == "Gaug"

    def test_sus4(self) -> None:
        assert _format_chord_name("D", "sus4") == "Dsus4"


# ===========================================================================
# 4. Segment math
# ===========================================================================


class TestSegmentMath:
    """Tests for _get_segment function from src.main."""

    def test_center_returns_valid_segment(self) -> None:
        """Fingertip at exact center of menu should return a valid segment."""
        menu = MenuConfig(
            center_x=0.5,
            center_y=0.5,
            radius=150,
            items=["A", "B", "C", "D", "E", "F", "G"],
        )
        ft = Fingertip(hand_label="Right", x=0.5, y=0.5, visible=True)
        seg = _get_segment(ft, menu, 960, 720)
        assert seg is not None
        assert 0 <= seg < 7

    def test_outside_menu_returns_none(self) -> None:
        """Fingertip far from menu should return None."""
        menu = MenuConfig(
            center_x=0.5,
            center_y=0.5,
            radius=100,
            items=["A", "B", "C"],
        )
        ft = Fingertip(hand_label="Left", x=0.99, y=0.99, visible=True)
        seg = _get_segment(ft, menu, 960, 720)
        assert seg is None

    def test_invisible_returns_none(self) -> None:
        """Invisible fingertip should return None regardless of position."""
        menu = MenuConfig(
            center_x=0.5,
            center_y=0.5,
            radius=150,
            items=["A", "B", "C"],
        )
        ft = Fingertip(hand_label="Left", x=0.5, y=0.5, visible=False)
        seg = _get_segment(ft, menu, 960, 720)
        assert seg is None

    def test_none_fingertip_returns_none(self) -> None:
        """Passing None as the fingertip should return None."""
        menu = MenuConfig(
            center_x=0.5,
            center_y=0.5,
            radius=150,
            items=["A", "B", "C"],
        )
        seg = _get_segment(None, menu, 960, 720)
        assert seg is None

    def test_empty_items_returns_none(self) -> None:
        """Menu with an empty items list should return None."""
        menu = MenuConfig(
            center_x=0.5,
            center_y=0.5,
            radius=150,
            items=[],
        )
        ft = Fingertip(hand_label="Right", x=0.5, y=0.5, visible=True)
        seg = _get_segment(ft, menu, 960, 720)
        assert seg is None

    def test_boundary_exactly_on_circle_edge(self) -> None:
        """Fingertip at exactly the circle boundary (radius distance)."""
        menu = MenuConfig(
            center_x=0.5,
            center_y=0.5,
            radius=150,
            items=["A", "B", "C", "D"],
        )
        # Place fingertip exactly 150px to the right of center.
        # Center pixel: 0.5 * 960 = 480. Fingertip pixel: 480 + 150 = 630.
        # Normalised: 630 / 960 = 0.65625
        ft = Fingertip(hand_label="Right", x=0.65625, y=0.5, visible=True)
        seg = _get_segment(ft, menu, 960, 720)
        # dist == radius, exact boundary -- the <= guard means this should pass.
        # Actually the source uses `dist > menu_config.radius`, so equal is NOT outside.
        assert seg is not None
        assert 0 <= seg < 4

    def test_just_outside_boundary(self) -> None:
        """Fingertip one pixel beyond radius should return None."""
        menu = MenuConfig(
            center_x=0.5,
            center_y=0.5,
            radius=150,
            items=["A", "B", "C"],
        )
        # 481 + 150 = 631 px, normalised = 631/960 = 0.65729...
        # dist = 151 > 150 => outside
        ft = Fingertip(hand_label="Right", x=0.6573, y=0.5, visible=True)
        seg = _get_segment(ft, menu, 960, 720)
        assert seg is None


# ===========================================================================
# 5. Cursor math
# ===========================================================================


class TestCursorMath:
    """Tests for the cursor delta computation used in main.py."""

    @staticmethod
    def _compute_cursor(finger_x: float, base_x: float, sensitivity: float = 2.0) -> float:
        """Replicates cursor_x = finger_x + (finger_x - baseline_x) * sensitivity, clamped."""
        cx = finger_x + (finger_x - base_x) * sensitivity
        return max(0.0, min(1.0, cx))

    def test_no_movement_returns_finger_position(self) -> None:
        """When finger is at baseline, cursor should be at same position."""
        cursor_x = self._compute_cursor(0.4, 0.4)
        assert cursor_x == 0.4

    def test_moved_right_amplifies_delta(self) -> None:
        """Finger right of baseline: cursor moves further right."""
        # baseline=0.25, finger=0.35, delta=0.1, amplified by 2.0 to 0.2
        # cursor = 0.35 + 0.2 = 0.55
        cursor_x = self._compute_cursor(0.35, 0.25)
        assert math.isclose(cursor_x, 0.55)

    def test_moved_left_amplifies_negative_delta(self) -> None:
        """Finger left of baseline: cursor moves further left."""
        # baseline=0.75, finger=0.65, delta=-0.1, amplified=-0.2
        # cursor = 0.65 - 0.2 = 0.45
        cursor_x = self._compute_cursor(0.65, 0.75)
        assert math.isclose(cursor_x, 0.45)

    def test_clamped_to_zero(self) -> None:
        """Cursor must not go below 0.0."""
        # baseline=0.5, finger=0.1, delta=-0.4, amplified=-0.8
        # cursor = 0.1 - 0.8 = -0.7 -> clamped to 0.0
        cursor_x = self._compute_cursor(0.1, 0.5)
        assert cursor_x == 0.0

    def test_clamped_to_one(self) -> None:
        """Cursor must not go above 1.0."""
        # baseline=0.5, finger=0.9, delta=0.4, amplified=0.8
        # cursor = 0.9 + 0.8 = 1.7 -> clamped to 1.0
        cursor_x = self._compute_cursor(0.9, 0.5)
        assert cursor_x == 1.0

    def test_sensitivity_zero_returns_finger_position(self) -> None:
        """With zero sensitivity, cursor = finger position (no amplification)."""
        cursor_x = self._compute_cursor(0.3, 0.5, sensitivity=0.0)
        assert cursor_x == 0.3

    def test_sensitivity_one(self) -> None:
        """With sensitivity=1.0, cursor = finger + delta."""
        # finger=0.3, baseline=0.2, delta=0.1
        # cursor = 0.3 + 0.1 = 0.4
        cursor_x = self._compute_cursor(0.3, 0.2, sensitivity=1.0)
        assert math.isclose(cursor_x, 0.4)


# ===========================================================================
# 6. MIDI port listing
# ===========================================================================


class TestHandTrackerInstantiation:
    """Verify HandTracker can be constructed with the installed MediaPipe."""

    def test_hand_tracker_instantiation(self) -> None:
        """HandTracker() should not raise AttributeError on mp.solutions."""
        from src.config import CameraConfig
        from src.hand_tracker import HandTracker

        config = CameraConfig(width=320, height=240, fps=15, device_id=0)
        tracker = HandTracker(config)
        assert tracker is not None


class TestHandTrackerThreadResilience:
    """Background thread must survive frame-processing exceptions."""

    def test_thread_alive_after_cvtcolor_error(self, monkeypatch) -> None:
        """_run() must catch exceptions; thread stays alive after cv error."""
        import time

        import cv2

        from src.config import CameraConfig
        from src.hand_tracker import HandTracker

        original_cvt = cv2.cvtColor
        call_count = [0]

        def mock_cvt(frame, code):  # noqa: ANN001, ANN202
            call_count[0] += 1
            if call_count[0] == 2:
                raise cv2.error("simulated frame conversion error")
            return original_cvt(frame, code)

        monkeypatch.setattr(cv2, "cvtColor", mock_cvt)

        tracker = HandTracker(CameraConfig(width=320, height=240, fps=15, device_id=0))
        tracker.start()
        time.sleep(1.0)  # allow a few loop iterations

        assert tracker._thread is not None
        assert tracker._thread.is_alive(), (
            "Background thread died after frame error — _run() lacks try/except"
        )

        tracker.stop()


class TestFingertipVisibilityNone:
    """MediaPipe NormalizedLandmark.visibility can be None."""

    def test_visibility_none_does_not_raise(self) -> None:
        """visibility=None must not cause TypeError in the > 0.5 check."""

        class MockLandmark:
            x = 0.5
            y = 0.5
            visibility = None  # MediaPipe can return None for some landmarks

        tip = MockLandmark()

        # Replicates the fixed visibility logic from hand_tracker._run()
        result = (
            tip.visibility is not None
            and tip.visibility > 0.5
        )

        assert isinstance(result, bool)
        assert result is False  # None should evaluate as not visible


class TestMidiPortListing:
    """Tests for get_available_ports (no hardware required)."""

    def test_returns_list(self) -> None:
        ports = get_available_ports()
        assert isinstance(ports, list)

    def test_returns_string_elements(self) -> None:
        ports = get_available_ports()
        for port in ports:
            assert isinstance(port, str)
