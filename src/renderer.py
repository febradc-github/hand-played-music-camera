"""PyGame-based renderer for the hand-tracked chord instrument.

Renders the camera feed as a background with two radial pie menus overlaid.
Each menu supports per-segment highlight-on-select feedback driven by
fingertip position.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pygame

from src.config import WindowConfig, MenuConfig
from src.hand_tracker import HAND_CONNECTIONS

if TYPE_CHECKING:
    from src.hand_tracker import Fingertip, HandInfo


class Renderer:
    """Main renderer managing the PyGame window and radial menu drawing."""

    def __init__(
        self,
        window_config: WindowConfig,
        left_menu_config: MenuConfig,
        right_menu_config: MenuConfig,
    ) -> None:
        """Initialise PyGame, create the window, and store menu geometry.

        Args:
            window_config: Window dimensions and title.
            left_menu_config: Geometry and items for the left radial menu.
            right_menu_config: Geometry and items for the right radial menu.
        """
        pygame.init()

        self._window_config = window_config
        self._window_width = window_config.width
        self._window_height = window_config.height

        self._screen = pygame.display.set_mode(
            (self._window_width, self._window_height)
        )
        pygame.display.set_caption(window_config.title)

        self._left_menu_config = left_menu_config
        self._right_menu_config = right_menu_config

        self._left_center = (
            int(left_menu_config.center_x * self._window_width),
            int(left_menu_config.center_y * self._window_height),
        )
        self._right_center = (
            int(right_menu_config.center_x * self._window_width),
            int(right_menu_config.center_y * self._window_height),
        )

        self._font = pygame.font.Font(None, 24)
        self._chord_font = pygame.font.Font(None, 36)

        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_segment_index(
        menu_config: MenuConfig,
        pixel_center: tuple[int, int],
        fingertip_x: float,
        fingertip_y: float,
    ) -> int | None:
        """Return the 0-based segment index the fingertip is pointing at.

        Returns ``None`` when the fingertip falls outside the menu radius or
        the menu has no items.
        """
        items = menu_config.items
        if not items:
            return None

        cx, cy = pixel_center
        dx = fingertip_x - cx
        dy = fingertip_y - cy
        dist = math.hypot(dx, dy)

        if dist > menu_config.radius:
            return None

        # atan2 returns angle measured from positive x-axis (right).
        # Add pi/2 so that 0 rad corresponds to straight up (-90 deg).
        angle = math.atan2(dy, dx) + math.pi / 2
        if angle < 0:
            angle += 2 * math.pi

        segment_span = 2 * math.pi / len(items)
        return int(angle // segment_span)

    def _draw_hand_skeleton(
        self,
        landmarks: list[tuple[float, float]],
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> None:
        """Draw a hand skeleton overlay on ``self._screen``.

        Args:
            landmarks: 21 (x, y) tuples in normalised 0-1 space.
            color: RGB colour for the skeleton lines and dots.
        """
        w, h = self._window_width, self._window_height
        pts = [(int(lx * w), int(ly * h)) for lx, ly in landmarks]

        # Draw connections (lines)
        for a, b in HAND_CONNECTIONS:
            pygame.draw.line(self._screen, color, pts[a], pts[b], 2)

        # Draw landmark dots
        for px, py in pts:
            pygame.draw.circle(self._screen, color, (px, py), 4)

    def _draw_radial_menu(
        self,
        surface: pygame.Surface,
        menu_config: MenuConfig,
        pixel_center: tuple[int, int],
        highlight_index: int | None,
    ) -> None:
        """Draw one radial menu on *surface*.

        Args:
            surface: The destination ``pygame.Surface``.
            menu_config: Menu geometry and item labels.
            pixel_center: Pixel coordinates of the menu centre.
            highlight_index: Index of the segment to highlight, or ``None``.
        """
        items = menu_config.items
        if not items:
            return

        cx, cy = pixel_center
        radius = menu_config.radius
        num_items = len(items)
        segment_span = 2 * math.pi / num_items

        # Background circle (solid, opaque)
        pygame.draw.circle(surface, (50, 50, 50), pixel_center, radius)

        # Number of points per arc segment for smooth wedges
        arc_points = 32

        for i, label in enumerate(items):
            start_angle = -math.pi / 2 + i * segment_span
            end_angle = start_angle + segment_span

            # Choose fill colour
            if i == highlight_index:
                fill_color = (80, 180, 80)  # green highlight
            else:
                fill_color = (75, 75, 75)  # dark gray, fully opaque

            # Build wedge polygon: centre + arc points
            pts = [(cx, cy)]
            for j in range(arc_points):
                a = start_angle + (end_angle - start_angle) * j / (arc_points - 1)
                px = cx + radius * math.cos(a)
                py = cy + radius * math.sin(a)
                pts.append((px, py))

            pygame.draw.polygon(surface, fill_color, pts)

            # Segment divider line
            div_x = cx + radius * math.cos(start_angle)
            div_y = cy + radius * math.sin(start_angle)
            pygame.draw.line(surface, (40, 40, 40), (cx, cy), (div_x, div_y), 2)

            # Label at 60 % of radius along the segment midpoint
            mid_angle = start_angle + segment_span / 2
            label_x = cx + int(0.6 * radius * math.cos(mid_angle))
            label_y = cy + int(0.6 * radius * math.sin(mid_angle))

            text_surf = self._font.render(label, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=(label_x, label_y))
            surface.blit(text_surf, text_rect)

        # Outer border
        pygame.draw.circle(surface, (200, 200, 200), pixel_center, radius, 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray | None,
        left_fingertip: Fingertip | None,
        right_fingertip: Fingertip | None,
        current_chord_name: str | None = None,
        left_cursor_px: tuple[float, float] | None = None,
        right_cursor_px: tuple[float, float] | None = None,
        hand_infos: list[HandInfo] | None = None,
    ) -> None:
        """Compose and display one full frame.

        Args:
            frame: BGR camera image as a numpy array (height, width, 3),
                or ``None`` to render a black background.
            left_fingertip: Detected left-hand fingertip or ``None``.
            right_fingertip: Detected right-hand fingertip or ``None``.
            current_chord_name: Chord label to display, or ``None``/empty
                to suppress.
        """
        if frame is not None:
            # BGR (OpenCV) → RGB
            rgb = frame[:, :, ::-1]
            # surfarray.make_surface expects (width, height, 3)
            rgb_transposed = np.transpose(rgb, (1, 0, 2))
            cam_surf = pygame.surfarray.make_surface(rgb_transposed)
            # Scale to fill the window
            bg = pygame.transform.smoothscale(
                cam_surf, (self._window_width, self._window_height)
            )
        else:
            bg = pygame.Surface((self._window_width, self._window_height))
            bg.fill((0, 0, 0))

        self._screen.blit(bg, (0, 0))

        # Draw hand skeletons on top of the camera feed
        if hand_infos:
            for hi in hand_infos:
                self._draw_hand_skeleton(hi.landmarks)

        # Determine which segment (if any) each fingertip is pointing at
        left_highlight: int | None = None
        if left_fingertip is not None and left_fingertip.visible:
            fx = left_fingertip.x * self._window_width
            fy = left_fingertip.y * self._window_height
            left_highlight = self._compute_segment_index(
                self._left_menu_config, self._left_center, fx, fy
            )

        right_highlight: int | None = None
        if right_fingertip is not None and right_fingertip.visible:
            fx = right_fingertip.x * self._window_width
            fy = right_fingertip.y * self._window_height
            right_highlight = self._compute_segment_index(
                self._right_menu_config, self._right_center, fx, fy
            )

        # Draw both radial menus
        self._draw_radial_menu(
            self._screen,
            self._left_menu_config,
            self._left_center,
            left_highlight,
        )
        self._draw_radial_menu(
            self._screen,
            self._right_menu_config,
            self._right_center,
            right_highlight,
        )

        # Chord name label at bottom centre
        if current_chord_name:
            chord_surf = self._chord_font.render(
                current_chord_name, True, (255, 255, 255)
            )
            chord_rect = chord_surf.get_rect(
                center=(self._window_width // 2, self._window_height - 40)
            )
            self._screen.blit(chord_surf, chord_rect)

        if left_cursor_px is not None:
            self.draw_cursor(left_cursor_px[0], left_cursor_px[1])
        if right_cursor_px is not None:
            self.draw_cursor(right_cursor_px[0], right_cursor_px[1])

        pygame.display.flip()
        self.clock.tick(60)

    def draw_cursor(self, x: float, y: float) -> None:
        """Draw a filled green circle at the given pixel coordinates.

        Args:
            x: X pixel coordinate.
            y: Y pixel coordinate.
        """
        pygame.draw.circle(self._screen, (100, 255, 100), (int(x), int(y)), 8)

    def draw_calibration(
        self,
        frame: np.ndarray | None,
        left_present: bool,
        right_present: bool,
        progress: float,
        hand_infos: list[HandInfo] | None = None,
    ) -> None:
        """Render the calibration overlay with two target boxes.

        Each box shows whether the corresponding finger is inside it,
        a label, and (when active) a progress bar that fills over 3 s.

        Args:
            frame: BGR camera image background, or ``None`` for black.
            left_present: ``True`` when the left index finger is inside its box.
            right_present: ``True`` when the right index finger is inside its box.
            progress: Hold-timer progress in the range [0.0, 1.0].
        """
        # --- background (same logic as draw()) ---
        if frame is not None:
            rgb = frame[:, :, ::-1]
            rgb_transposed = np.transpose(rgb, (1, 0, 2))
            cam_surf = pygame.surfarray.make_surface(rgb_transposed)
            bg = pygame.transform.smoothscale(
                cam_surf, (self._window_width, self._window_height)
            )
        else:
            bg = pygame.Surface((self._window_width, self._window_height))
            bg.fill((0, 0, 0))

        self._screen.blit(bg, (0, 0))

        # Draw hand skeletons on top of the camera feed
        if hand_infos:
            for hi in hand_infos:
                self._draw_hand_skeleton(hi.landmarks)

        # --- box geometry ---
        half = 100
        box_size = half * 2  # 200 x 200

        left_cx = int(0.25 * self._window_width)
        left_cy = int(0.50 * self._window_height)
        right_cx = int(0.75 * self._window_width)
        right_cy = int(0.50 * self._window_height)

        for cx, cy, present, label in [
            (left_cx, left_cy, left_present, "Left Box"),
            (right_cx, right_cy, right_present, "Right Box"),
        ]:
            color = (0, 255, 0) if present else (255, 50, 50)

            # Solid fill when finger is present
            if present:
                fill_surf = pygame.Surface((box_size, box_size))
                fill_surf.fill((0, 100, 0))
                fill_surf.set_alpha(160)
                self._screen.blit(fill_surf, (cx - half, cy - half))

            # Border (thicker, more visible)
            rect = pygame.Rect(cx - half, cy - half, box_size, box_size)
            pygame.draw.rect(self._screen, color, rect, 5)

            # Label below the box
            label_surf = self._font.render(label, True, (255, 255, 255))
            label_rect = label_surf.get_rect(center=(cx, cy + half + 20))
            self._screen.blit(label_surf, label_rect)

            # Progress bar (only shown when timer is active)
            if progress > 0:
                bar_w = 200
                bar_h = 16
                bar_x = cx - bar_w // 2
                bar_y = cy + half + 20 + 20  # below the label

                # Background
                bar_bg = pygame.Rect(bar_x, bar_y, bar_w, bar_h)
                pygame.draw.rect(self._screen, (60, 60, 60), bar_bg)

                # Fill
                fill_w = int(progress * bar_w)
                if fill_w > 0:
                    bar_fill = pygame.Rect(bar_x, bar_y, fill_w, bar_h)
                    pygame.draw.rect(self._screen, (100, 255, 100), bar_fill)

        pygame.display.flip()
        self.clock.tick(30)

    def handle_events(self) -> bool:
        """Process the PyGame event queue.

        Returns:
            ``True`` if the application should continue running,
            ``False`` when a quit event (window close or Escape key) is
            received.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def cleanup(self) -> None:
        """Shut down PyGame and release display resources."""
        pygame.quit()
