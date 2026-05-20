"""Hand tracker using MediaPipe Hands running in a background thread.

Exposes thread-safe fingertip coordinates and camera frames for use by
the renderer and gesture-detection layers.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
import urllib.request
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mtp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import numpy as np

from src.config import CameraConfig


@dataclass
class Fingertip:
    """Coordinates of a single detected index fingertip (landmark 8).

    Attributes:
        hand_label: ``"Left"`` or ``"Right"`` as classified by MediaPipe.
        x: Horizontal position in normalised 0-1 space (0=left, 1=right).
        y: Vertical position in normalised 0-1 space (0=top, 1=bottom).
        visible: Whether the landmark is considered visible by MediaPipe.
    """

    hand_label: str
    x: float
    y: float
    visible: bool


@dataclass
class HandInfo:
    """All 21 hand landmarks for a single detected hand.

    Attributes:
        hand_label: ``"Left"`` or ``"Right"``.
        landmarks: 21 (x, y) tuples in normalised 0-1 space.
    """

    hand_label: str
    landmarks: list[tuple[float, float]]


# MediaPipe hand skeleton connections (landmark index pairs).
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (5, 9), (9, 10), (10, 11), (11, 12),  # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),                                # wrist to pinky base
]


class HandTracker:
    """Captures webcam frames in a background thread and runs MediaPipe Hands
    inference on every frame, exposing the latest fingertip positions and the
    raw camera image in a thread-safe manner.

    Typical usage::

        tracker = HandTracker(camera_config)
        tracker.start()
        # ... main loop ...
        fingertips = tracker.get_fingertips()
        frame = tracker.get_frame()
        tracker.stop()
    """

    # Landmark indices ------------------------
    INDEX_FINGER_TIP: int = 8

    # MediaPipe model / confidence thresholds
    _MODEL_FILENAME: str = "hand_landmarker.task"
    _MODEL_URL: str = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/"
        "hand_landmarker.task"
    )
    _MIN_DETECTION_CONFIDENCE: float = 0.7
    _MIN_TRACKING_CONFIDENCE: float = 0.7

    def __init__(self, camera_config: CameraConfig) -> None:
        """Initialise the hand tracker with camera settings.

        Args:
            camera_config: Width, height, fps and device ID for the webcam.
        """
        self._config: CameraConfig = camera_config

        # MediaPipe pipeline --------------------------------------------------
        self._download_model()
        base_options = mtp.BaseOptions(model_asset_path=self._MODEL_FILENAME)
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=self._MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=self._MIN_TRACKING_CONFIDENCE,
        )
        self._hands = HandLandmarker.create_from_options(options)

        # Thread / camera state ------------------------------------------------
        self._running: bool = False
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None

        # Shared data (protected by lock) -------------------------------------
        self._lock: threading.Lock = threading.Lock()
        self._fingertips: list[Fingertip] = []
        self._hand_infos: list[HandInfo] = []
        self._frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Model download helper
    # ------------------------------------------------------------------

    @classmethod
    def _download_model(cls) -> None:
        """Download the MediaPipe hand-landmarker model if not already present.

        Raises:
            RuntimeError: If the download fails (e.g. no network).
        """
        if os.path.exists(cls._MODEL_FILENAME):
            return
        try:
            urllib.request.urlretrieve(cls._MODEL_URL, cls._MODEL_FILENAME)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to download {cls._MODEL_FILENAME}. "
                f"Check your network connection. ({exc})"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the camera and begin the background tracking thread.

        The thread is spawned as a daemon so the process can exit cleanly
        even if ``stop()`` is never called explicitly.
        """
        if self._running:
            return

        cap = cv2.VideoCapture(self._config.device_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        cap.set(cv2.CAP_PROP_FPS, self._config.fps)
        self._capture = cap

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop, release the camera and close
        the MediaPipe Hands instance.
        """
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        if hasattr(self, "_hands"):
            self._hands.close()

        with self._lock:
            self._fingertips.clear()
            self._hand_infos.clear()
            self._frame = None

    def get_fingertips(self) -> list[Fingertip]:
        """Return a thread-safe copy of the current fingertip detections.

        Returns:
            A new list of :class:`Fingertip` objects representing the latest
            detected index-fingertip positions.  May be empty if no hands
            were detected.
        """
        with self._lock:
            return list(self._fingertips)

    def get_hand_infos(self) -> list[HandInfo]:
        """Return a thread-safe copy of the current hand landmark data.

        Returns:
            A new list of :class:`HandInfo` objects, each containing 21
            normalised (x, y) landmark coordinates for one detected hand.
        """
        with self._lock:
            return list(self._hand_infos)

    def get_frame(self) -> Optional[np.ndarray]:
        """Return a thread-safe copy of the latest raw camera frame.

        Returns:
            A BGR :class:`numpy.ndarray` (H x W x 3), or ``None`` if no
            frame has been captured yet or the tracker is stopped.
        """
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main loop executed in the background thread.

        Reads frames from ``cv2.VideoCapture``, runs MediaPipe inference,
        and populates the shared ``_fingertips`` / ``_frame`` fields under
        the protection of ``_lock``.
        """
        while self._running:
            try:
                if self._capture is None:
                    time.sleep(0.001)
                    continue

                grabbed, frame_bgr = self._capture.read()
                if not grabbed or frame_bgr is None:
                    time.sleep(0.001)
                    continue

                # Mirror the frame so it behaves like a mirror (left hand
                # appears on the left side of the screen).
                frame_bgr = cv2.flip(frame_bgr, 1)

                # Convert to RGB for MediaPipe ------------------------------------
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False  # performance hint for MediaPipe
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int(time.monotonic() * 1000)
                results = self._hands.detect_for_video(mp_image, timestamp_ms)

                # Build fingertip & hand-info lists ---------------------------------
                fingertips: list[Fingertip] = []
                hand_infos: list[HandInfo] = []
                if (
                    results.hand_landmarks
                    and results.handedness
                ):
                    for hand_landmarks, hand_classification in zip(
                        results.hand_landmarks,
                        results.handedness,
                    ):
                        # handedness is a list of Category; the first entry
                        # carries the classification for this hand.
                        label = hand_classification[0].category_name
                        # Map "Left" / "Right" from MediaPipe classification
                        # (which reports the *actual* hand side).
                        hand_label: str = label.title()  # "Left" / "Right"

                        index_tip = hand_landmarks[self.INDEX_FINGER_TIP]
                        fingertip = Fingertip(
                            hand_label=hand_label,
                            x=index_tip.x,  # already normalised 0-1
                            y=index_tip.y,  # already normalised 0-1
                            visible=(
                                index_tip.visibility is not None
                                and index_tip.visibility > 0.5
                            ),
                        )
                        fingertips.append(fingertip)

                        # Store all 21 landmarks for rendering
                        lm_list: list[tuple[float, float]] = [
                            (lm.x, lm.y) for lm in hand_landmarks
                        ]
                        hand_infos.append(HandInfo(hand_label, lm_list))

                # Publish results under lock --------------------------------------
                with self._lock:
                    self._frame = frame_bgr
                    self._fingertips = fingertips
                    self._hand_infos = hand_infos

                # Brief yield to cap loop rate (reduces CPU contention)
                time.sleep(0.001)
            except Exception:
                traceback.print_exc()
                time.sleep(0.005)
