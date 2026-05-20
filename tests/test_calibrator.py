"""Tests for the calibration state machine and data types."""

from __future__ import annotations

import math

from src.calibrator import (
    CalibrationResult,
    CalibrationState,
    LEFT_BOX_CENTER_X,
    LEFT_BOX_CENTER_Y,
    RIGHT_BOX_CENTER_X,
    RIGHT_BOX_CENTER_Y,
    BOX_RADIUS_PX,
    HOLD_DURATION,
)
from src.hand_tracker import Fingertip


class TestCalibrationResult:
    def test_fields_exist(self) -> None:
        result = CalibrationResult(
            left_baseline_x=0.25,
            left_baseline_y=0.50,
            right_baseline_x=0.75,
            right_baseline_y=0.50,
        )
        assert result.left_baseline_x == 0.25
        assert result.left_baseline_y == 0.50
        assert result.right_baseline_x == 0.75
        assert result.right_baseline_y == 0.50

    def test_all_fields_are_floats(self) -> None:
        result = CalibrationResult(0.1, 0.2, 0.3, 0.4)
        assert isinstance(result.left_baseline_x, float)
        assert isinstance(result.left_baseline_y, float)
        assert isinstance(result.right_baseline_x, float)
        assert isinstance(result.right_baseline_y, float)


class TestCalibrationState:
    def test_has_all_states(self) -> None:
        states = set(CalibrationState)
        assert CalibrationState.IDLE in states
        assert CalibrationState.ONE_PRESENT in states
        assert CalibrationState.BOTH_PRESENT in states
        assert CalibrationState.HOLDING in states
        assert CalibrationState.DONE in states

    def test_states_are_unique(self) -> None:
        values = [s.value for s in CalibrationState]
        assert len(values) == len(set(values))


class TestBoxGeometry:
    def test_left_box_center_in_normalized_range(self) -> None:
        assert 0.0 <= LEFT_BOX_CENTER_X <= 1.0
        assert 0.0 <= LEFT_BOX_CENTER_Y <= 1.0

    def test_right_box_center_in_normalized_range(self) -> None:
        assert 0.0 <= RIGHT_BOX_CENTER_X <= 1.0
        assert 0.0 <= RIGHT_BOX_CENTER_Y <= 1.0

    def test_box_radius_positive(self) -> None:
        assert BOX_RADIUS_PX > 0

    def test_hold_duration_positive(self) -> None:
        assert HOLD_DURATION > 0


class TestBoxHitDetection:
    @staticmethod
    def _is_inside_box(
        finger_x: float, finger_y: float,
        box_cx: float, box_cy: float,
        window_w: int = 960, window_h: int = 720,
    ) -> bool:
        """Replicates the hit-test logic from run_calibration."""
        px = finger_x * window_w
        py = finger_y * window_h
        bx = box_cx * window_w
        by = box_cy * window_h
        return math.hypot(px - bx, py - by) < BOX_RADIUS_PX

    def test_finger_at_box_center_is_inside(self) -> None:
        assert self._is_inside_box(
            LEFT_BOX_CENTER_X, LEFT_BOX_CENTER_Y,
            LEFT_BOX_CENTER_X, LEFT_BOX_CENTER_Y,
        )

    def test_finger_at_box_edge_is_inside(self) -> None:
        # 99 pixels from center (radius is 100)
        edge_x = LEFT_BOX_CENTER_X + 99.0 / 960.0
        assert self._is_inside_box(
            edge_x, LEFT_BOX_CENTER_Y,
            LEFT_BOX_CENTER_X, LEFT_BOX_CENTER_Y,
        )

    def test_finger_far_outside_box_is_outside(self) -> None:
        far_x = LEFT_BOX_CENTER_X + 200.0 / 960.0
        assert not self._is_inside_box(
            far_x, LEFT_BOX_CENTER_Y,
            LEFT_BOX_CENTER_X, LEFT_BOX_CENTER_Y,
        )

    def test_right_box_hit_detection(self) -> None:
        assert self._is_inside_box(
            RIGHT_BOX_CENTER_X, RIGHT_BOX_CENTER_Y,
            RIGHT_BOX_CENTER_X, RIGHT_BOX_CENTER_Y,
        )
