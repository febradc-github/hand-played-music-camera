"""Application configuration dataclasses for the hand-tracked chord instrument."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CameraConfig:
    """Camera capture settings."""

    width: int = 960
    height: int = 720
    fps: int = 30
    device_id: int = 0

    def __repr__(self) -> str:
        return (
            f"CameraConfig(width={self.width}, height={self.height}, "
            f"fps={self.fps}, device_id={self.device_id})"
        )


@dataclass
class MenuConfig:
    """Configuration for a radial menu overlay."""

    center_x: float = 0.25
    center_y: float = 0.50
    radius: int = 150
    items: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"MenuConfig(center=({self.center_x}, {self.center_y}), "
            f"radius={self.radius}, items={self.items})"
        )


@dataclass
class WindowConfig:
    """Application window settings."""

    width: int = 960
    height: int = 720
    title: str = "Hand-Tracked Chord Instrument"

    def __repr__(self) -> str:
        return (
            f"WindowConfig(width={self.width}, height={self.height}, "
            f"title='{self.title}')"
        )


@dataclass
class AppConfig:
    """Top-level application configuration."""

    camera: CameraConfig
    midi_port: str
    left_menu: MenuConfig
    right_menu: MenuConfig
    chord_octave: int
    window: WindowConfig

    @classmethod
    def load(cls, path: str | Path) -> AppConfig:
        """Load configuration from a JSON file and return an AppConfig instance.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            A fully constructed AppConfig instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
            KeyError: If a required key is missing from the JSON.
        """
        config_path = Path(path)
        with open(config_path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = json.load(fh)

        camera = CameraConfig(**raw["camera"])
        midi_port: str = raw["midi_port"]
        left_menu = MenuConfig(**raw["left_menu"])
        right_menu = MenuConfig(**raw["right_menu"])
        chord_octave: int = raw["chord_octave"]
        window = WindowConfig(**raw["window"])

        return cls(
            camera=camera,
            midi_port=midi_port,
            left_menu=left_menu,
            right_menu=right_menu,
            chord_octave=chord_octave,
            window=window,
        )

    def __repr__(self) -> str:
        return (
            f"AppConfig(\n"
            f"  camera={self.camera!r},\n"
            f"  midi_port='{self.midi_port}',\n"
            f"  left_menu={self.left_menu!r},\n"
            f"  right_menu={self.right_menu!r},\n"
            f"  chord_octave={self.chord_octave},\n"
            f"  window={self.window!r},\n"
            f")"
        )
