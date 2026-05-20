"""Main entry point for the hand-tracked chord instrument.

Wires together the HandTracker, MidiOutput, ChordEngine, and Renderer
modules into a real-time interactive application.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import pygame

from src.calibrator import run_calibration, CalibrationResult
from src.chord_engine import ChordEngine
from src.config import AppConfig, MenuConfig
from src.hand_tracker import Fingertip, HandTracker
from src.midi_output import MidiOutput, get_available_ports
from src.renderer import Renderer


_CURSOR_SENSITIVITY: float = 0.0

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_segment(
    fingertip: Fingertip | None,
    menu_config: MenuConfig,
    window_w: int,
    window_h: int,
) -> int | None:
    """Return the 0-based segment index the fingertip falls within, or ``None``.

    Args:
        fingertip: Detected fingertip with normalised 0-1 coordinates.
        menu_config: The radial menu to test against.
        window_w: Window pixel width (for coordinate conversion).
        window_h: Window pixel height (for coordinate conversion).

    Returns:
        Segment index (0-based), or ``None`` if the fingertip is outside
        the menu radius, invisible, or the menu has no items.
    """
    if fingertip is None or not fingertip.visible:
        return None

    items = menu_config.items
    if not items:
        return None

    cx = int(menu_config.center_x * window_w)
    cy = int(menu_config.center_y * window_h)
    fx = fingertip.x * window_w
    fy = fingertip.y * window_h

    dx, dy = fx - cx, fy - cy
    dist = math.hypot(dx, dy)
    if dist > menu_config.radius:
        return None

    angle = math.atan2(dy, dx) + math.pi / 2
    if angle < 0:
        angle += 2 * math.pi

    return int(angle // (2 * math.pi / len(items)))


# ---------------------------------------------------------------------------
# Chord name formatting
# ---------------------------------------------------------------------------


def _format_chord_name(root: str, modifier: str) -> str:
    """Format a chord name for display.

    The special ``"maj"`` modifier is rendered as just the root note
    (e.g. ``"C"`` instead of ``"Cmaj"``).  All other modifiers are
    concatenated directly after the root (e.g. ``"Am"``, ``"F#m7"``).
    """
    if modifier == "maj":
        return root
    return f"{root}{modifier}"


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, initialise subsystems, and run the main loop.

    Args:
        argv: Command-line argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 on success, 1 on configuration or runtime error).
    """
    # -- CLI ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Hand-Tracked Chord Instrument",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.json",
        help="Path to the JSON configuration file (default: config.json).",
    )
    parser.add_argument(
        "--list-midi",
        action="store_true",
        help="List available MIDI output ports and exit.",
    )
    args = parser.parse_args(argv)

    # -- List MIDI ports ------------------------------------------------------
    if args.list_midi:
        ports = get_available_ports()
        if not ports:
            print("No MIDI output ports available.")
        else:
            print("Available MIDI output ports:")
            for i, name in enumerate(ports):
                print(f"  [{i}] {name}")
        return 0

    # -- Load configuration ---------------------------------------------------
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Configuration file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        config = AppConfig.load(config_path)
    except Exception as exc:
        print(f"Failed to load configuration: {exc}", file=sys.stderr)
        return 1

    # -- Initialise modules ---------------------------------------------------
    tracker = HandTracker(config.camera)
    midi = MidiOutput(config.midi_port)
    engine = ChordEngine(config.chord_octave)
    renderer = Renderer(config.window, config.left_menu, config.right_menu)

    # -- Report MIDI port status ----------------------------------------------
    available = get_available_ports()
    print("Available MIDI output ports:")
    for i, name in enumerate(available):
        marker = " <-- ACTIVE" if name == midi.port_name else ""
        print(f"  [{i}] {name}{marker}")
    if not midi.is_active:
        print("WARNING: No MIDI port is active. Notes will not be heard.")

    # -- State ----------------------------------------------------------------
    current_notes: list[int] = []
    current_chord_name: str = ""

    window_w = config.window.width
    window_h = config.window.height

    # -- Main loop ------------------------------------------------------------
    try:
        tracker.start()

        # -- Calibration phase --------------------------------------------------
        calib: CalibrationResult | None = None
        while calib is None:
            calib = run_calibration(tracker, renderer, window_w, window_h)
            if calib is None:
                return 0
        print("Calibration complete.")

        while True:
            # --- Recalibration (KEYDOWN events only, before handle_events) ----
            should_recalibrate = False
            for event in pygame.event.get(pygame.KEYDOWN):
                if event.key == pygame.K_SPACE:
                    should_recalibrate = True

            if should_recalibrate:
                # Stop any playing chord
                if current_notes:
                    midi.stop_chord(current_notes)
                    current_notes = []
                    current_chord_name = ""
                # Re-run calibration
                calib = run_calibration(tracker, renderer, window_w, window_h)
                if calib is None:
                    break
                continue  # skip rest of this frame

            # --- Events (quit/escape) ----------------------------------------
            if not renderer.handle_events():
                break

            # --- Tracking -----------------------------------------------------
            frame = tracker.get_frame()
            all_fingertips = tracker.get_fingertips()
            hand_infos = tracker.get_hand_infos()

            # Separate by hand label
            left_fingertip: Fingertip | None = None
            right_fingertip: Fingertip | None = None
            for ft in all_fingertips:
                if ft.hand_label == "Left":
                    left_fingertip = ft
                elif ft.hand_label == "Right":
                    right_fingertip = ft

            # Compute cursor positions from deltas --------------------------------
            left_cursor: Fingertip | None = None
            right_cursor: Fingertip | None = None

            if left_fingertip is not None:
                cx = left_fingertip.x + (left_fingertip.x - calib.left_baseline_x) * _CURSOR_SENSITIVITY
                cy = left_fingertip.y + (left_fingertip.y - calib.left_baseline_y) * _CURSOR_SENSITIVITY
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                left_cursor = Fingertip(hand_label="Left", x=cx, y=cy, visible=True)

            if right_fingertip is not None:
                cx = right_fingertip.x + (right_fingertip.x - calib.right_baseline_x) * _CURSOR_SENSITIVITY
                cy = right_fingertip.y + (right_fingertip.y - calib.right_baseline_y) * _CURSOR_SENSITIVITY
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                right_cursor = Fingertip(hand_label="Right", x=cx, y=cy, visible=True)

            left_seg = _get_segment(left_cursor, config.left_menu, window_w, window_h)
            right_seg = _get_segment(right_cursor, config.right_menu, window_w, window_h)

            # --- Chord resolution & MIDI --------------------------------------
            if left_seg is not None and right_seg is not None:
                root_note = config.left_menu.items[left_seg]
                modifier = config.right_menu.items[right_seg]

                try:
                    new_notes = engine.resolve(root_note, modifier)
                except (ValueError, KeyError):
                    # Unknown combination — suppress playback
                    new_notes = []

                if new_notes != current_notes:
                    # Stop previous chord
                    if current_notes:
                        midi.stop_chord(current_notes)
                    # Start new chord
                    if new_notes:
                        midi.play_chord(new_notes)
                    current_notes = new_notes
                    current_chord_name = (
                        _format_chord_name(root_note, modifier)
                        if new_notes
                        else ""
                    )
            else:
                # At least one finger left the zone — stop playback
                if current_notes:
                    midi.stop_chord(current_notes)
                    current_notes = []
                    current_chord_name = ""

            # --- Render -------------------------------------------------------
            left_cursor_px = (
                (left_cursor.x * window_w, left_cursor.y * window_h)
                if left_cursor is not None else None
            )
            right_cursor_px = (
                (right_cursor.x * window_w, right_cursor.y * window_h)
                if right_cursor is not None else None
            )

            renderer.draw(
                frame,
                left_fingertip,
                right_fingertip,
                current_chord_name,
                left_cursor_px=left_cursor_px,
                right_cursor_px=right_cursor_px,
                hand_infos=hand_infos,
            )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # --- Cleanup ----------------------------------------------------------
        # Stop any lingering notes
        if current_notes:
            midi.stop_chord(current_notes)

        tracker.stop()
        midi.close()
        renderer.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
