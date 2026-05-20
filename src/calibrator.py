"""Calibration state machine for finger-tracking baseline setup.

Before the main loop, the user must place left and right index fingers
inside on-screen boxes for a continuous 3-second hold.  This module
captures the normalised fingertip positions at the moment calibration
completes so the main loop can track deltas from those baselines.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import pygame

from src.hand_tracker import HandTracker

if TYPE_CHECKING:
    from src.renderer import Renderer


class CalibrationState(Enum):
    """States of the calibration mini-state-machine."""

    IDLE = auto()
    ONE_PRESENT = auto()
    BOTH_PRESENT = auto()
    HOLDING = auto()
    DONE = auto()


@dataclass
class CalibrationResult:
    """Normalised (0-1) fingertip coordinates captured at calibration time.

    Attributes:
        left_baseline_x: Horizontal position of the left index fingertip.
        left_baseline_y: Vertical position of the left index fingertip.
        right_baseline_x: Horizontal position of the right index fingertip.
        right_baseline_y: Vertical position of the right index fingertip.
    """

    left_baseline_x: float
    left_baseline_y: float
    right_baseline_x: float
    right_baseline_y: float


# Box geometry (normalised 0-1 coordinates)
LEFT_BOX_CENTER_X = 0.25
LEFT_BOX_CENTER_Y = 0.50
RIGHT_BOX_CENTER_X = 0.75
RIGHT_BOX_CENTER_Y = 0.50
BOX_RADIUS_PX = 100
HOLD_DURATION = 3.0


def run_calibration(
    tracker: HandTracker,
    renderer: Renderer,
    window_w: int,
    window_h: int,
) -> CalibrationResult | None:
    """Run a blocking calibration mini-loop.

    The user must place their left index finger inside the on-screen left
    box **and** their right index finger inside the right box and hold
    steadily for ``HOLD_DURATION`` seconds.  The state machine advances
    through ``IDLE → ONE_PRESENT → BOTH_PRESENT → HOLDING → DONE``.

    Args:
        tracker: An already-started :class:`HandTracker` instance.
        renderer: The app's :class:`Renderer` (must have a
            ``draw_calibration(frame, left_present, right_present, progress)``
            method).
        window_w: Window width in pixels.
        window_h: Window height in pixels.

    Returns:
        A :class:`CalibrationResult` with the normalised (0-1) fingertip
        coordinates at the moment calibration completed, or ``None`` if the
        user quit (Escape key or window close).
    """
    # Pixel-space box centres -------------------------------------------------
    left_box_cx = LEFT_BOX_CENTER_X * window_w
    left_box_cy = LEFT_BOX_CENTER_Y * window_h
    right_box_cx = RIGHT_BOX_CENTER_X * window_w
    right_box_cy = RIGHT_BOX_CENTER_Y * window_h

    state: CalibrationState = CalibrationState.IDLE
    prev_state: CalibrationState | None = None
    hold_start: float = 0.0

    # Cached fingertip values captured at the moment DONE is reached
    left_baseline_x: float = 0.0
    left_baseline_y: float = 0.0
    right_baseline_x: float = 0.0
    right_baseline_y: float = 0.0

    running = True
    while running:
        # -- Event handling ---------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return None

        # -- Fingertip detection ----------------------------------------------
        frame = tracker.get_frame()
        fingertips = tracker.get_fingertips()
        hand_infos = tracker.get_hand_infos()

        # Any fingertip in either box — don't care about hand label
        left_ft = None
        right_ft = None
        left_present = False
        right_present = False
        for ft in fingertips:
            px = ft.x * window_w
            py = ft.y * window_h
            if math.hypot(px - left_box_cx, py - left_box_cy) < BOX_RADIUS_PX:
                left_present = True
                if left_ft is None:
                    left_ft = ft
            if math.hypot(px - right_box_cx, py - right_box_cy) < BOX_RADIUS_PX:
                right_present = True
                if right_ft is None:
                    right_ft = ft

        # -- State machine ----------------------------------------------------
        if state == CalibrationState.IDLE:
            if left_present and right_present:
                state = CalibrationState.BOTH_PRESENT
            elif left_present or right_present:
                state = CalibrationState.ONE_PRESENT

        elif state == CalibrationState.ONE_PRESENT:
            if left_present and right_present:
                state = CalibrationState.BOTH_PRESENT
            elif not left_present and not right_present:
                state = CalibrationState.IDLE

        elif state == CalibrationState.BOTH_PRESENT:
            if left_present and right_present:
                state = CalibrationState.HOLDING
                hold_start = time.monotonic()
            elif left_present or right_present:
                state = CalibrationState.ONE_PRESENT
            else:
                state = CalibrationState.IDLE

        elif state == CalibrationState.HOLDING:
            if left_present and right_present:
                elapsed = time.monotonic() - hold_start
                if elapsed >= HOLD_DURATION:
                    state = CalibrationState.DONE
                    # Capture baselines at this exact moment
                    left_baseline_x = left_ft.x  # type: ignore[union-attr]
                    left_baseline_y = left_ft.y  # type: ignore[union-attr]
                    right_baseline_x = right_ft.x  # type: ignore[union-attr]
                    right_baseline_y = right_ft.y  # type: ignore[union-attr]
            elif left_present or right_present:
                state = CalibrationState.ONE_PRESENT
            else:
                state = CalibrationState.IDLE

        elif state == CalibrationState.DONE:
            break

        # -- Render -----------------------------------------------------------
        if state == CalibrationState.HOLDING:
            progress = (time.monotonic() - hold_start) / HOLD_DURATION
            # Clamp to [0, 1] in case of minor overshoot
            if progress > 1.0:
                progress = 1.0
        else:
            progress = 0.0

        if state != prev_state:
            prev_state = state
            n_hands = len(hand_infos)
            print(f"[calibrator] state={state.name}  hands={n_hands}  "
                  f"left_present={left_present}  right_present={right_present}  "
                  f"left_label={left_ft.hand_label if left_ft else '-'}  "
                  f"right_label={right_ft.hand_label if right_ft else '-'}")

        renderer.draw_calibration(frame, left_present, right_present, progress, hand_infos=hand_infos)

    return CalibrationResult(
        left_baseline_x=left_baseline_x,
        left_baseline_y=left_baseline_y,
        right_baseline_x=right_baseline_x,
        right_baseline_y=right_baseline_y,
    )
