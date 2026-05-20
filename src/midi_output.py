"""MIDI output wrapper around python-rtmidi.

Provides a MidiOutput class for sending note-on/note-off messages,
as well as a standalone helper for listing available ports.
"""

from __future__ import annotations

import time
import warnings
from typing import Optional, Union

import rtmidi


def get_available_ports() -> list[str]:
    """Return a list of available MIDI output port names.

    This is a standalone function that does not create a persistent
    MidiOutput instance.

    Returns:
        A list of port name strings, potentially empty if no ports
        are available.
    """
    midi_out = rtmidi.MidiOut()
    return midi_out.get_ports()


class MidiOutput:
    """Wraps a single ``rtmidi.MidiOut`` port for convenient note sending.

    Attributes:
        midiout: The underlying ``rtmidi.MidiOut`` instance, or *None*
            when no port could be opened (graceful degradation).
        port_name: The name of the port that was opened, or an empty
            string when no port is active.
    """

    def __init__(self, port_name: str = "") -> None:
        """Initialise and optionally open a MIDI output port.

        Args:
            port_name: Name of the port to open.  If an empty string
                (default), the first available port is opened.  If no
                ports are available or the named port cannot be found,
                ``midiout`` is set to *None* and a warning is printed.
        """
        self.midiout: Optional[rtmidi.MidiOut] = None
        self.port_name: str = ""

        available = get_available_ports()

        if not available:
            warnings.warn("No MIDI output ports available. MidiOutput will be inactive.")
            return

        self.open_port(port_name)

    # ------------------------------------------------------------------
    # Port discovery & management
    # ------------------------------------------------------------------

    @staticmethod
    def list_ports() -> list[str]:
        """Return a list of available MIDI output port names.

        Equivalent to calling ``get_available_ports()``.

        Returns:
            A list of port name strings.
        """
        return get_available_ports()

    def open_port(self, port_name: str = "") -> None:
        """Open the named MIDI output port, or the first available one.

        If *port_name* is a non-empty string the method searches for
        a port whose name matches exactly.  On failure (port not found
        or no ports available) ``self.midiout`` is set to *None* and
        a warning is emitted.

        Args:
            port_name: Name of the port to open.  An empty string uses
                the first available port.
        """
        # Close any previously opened port first
        if self.midiout is not None:
            try:
                self.midiout.close_port()
            except Exception:
                pass  # best-effort cleanup

        available = get_available_ports()

        if not available:
            self.midiout = None
            self.port_name = ""
            warnings.warn("No MIDI output ports available.")
            return

        if port_name:
            if port_name not in available:
                self.midiout = None
                self.port_name = ""
                warnings.warn(f"MIDI port '{port_name}' not found. Available: {available}")
                return
            target = port_name
            port_index = available.index(port_name)
        else:
            target = available[0]
            port_index = 0

        midi_out = rtmidi.MidiOut()
        midi_out.open_port(port_index)
        self.midiout = midi_out
        self.port_name = target

    # ------------------------------------------------------------------
    # Note sending
    # ------------------------------------------------------------------

    def note_on(self, note: int, velocity: int = 100, channel: int = 0) -> None:
        """Send a MIDI note-on message.

        Args:
            note: MIDI note number (0-127).
            velocity: Note-on velocity (0-127).  Defaults to 100.
            channel: MIDI channel (0-15).  Defaults to 0.
        """
        if self.midiout is None:
            warnings.warn("MidiOutput is inactive — no port available.")
            return

        status = 0x90 | (channel & 0x0F)
        self.midiout.send_message([status, note, velocity])

    def note_off(self, note: int, channel: int = 0) -> None:
        """Send a MIDI note-off message.

        Args:
            note: MIDI note number (0-127).
            channel: MIDI channel (0-15).  Defaults to 0.
        """
        if self.midiout is None:
            warnings.warn("MidiOutput is inactive — no port available.")
            return

        status = 0x80 | (channel & 0x0F)
        self.midiout.send_message([status, note, 0])

    def play_chord(self, notes: list[int], velocity: int = 100, channel: int = 0) -> None:
        """Send note-on messages for every note in *notes*.

        Args:
            notes: List of MIDI note numbers.
            velocity: Velocity for each note (0-127).  Defaults to 100.
            channel: MIDI channel (0-15).  Defaults to 0.
        """
        if self.midiout is None:
            warnings.warn("MidiOutput is inactive — no port available.")
            return

        for note in notes:
            self.note_on(note, velocity=velocity, channel=channel)

    def stop_chord(self, notes: list[int], channel: int = 0) -> None:
        """Send note-off messages for every note in *notes*.

        Args:
            notes: List of MIDI note numbers.
            channel: MIDI channel (0-15).  Defaults to 0.
        """
        if self.midiout is None:
            warnings.warn("MidiOutput is inactive — no port available.")
            return

        for note in notes:
            self.note_off(note, channel=channel)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying MIDI port.

        After calling this method the instance is still usable — you
        may call :meth:`open_port` again to re-open a port.
        """
        if self.midiout is not None:
            try:
                self.midiout.close_port()
            except Exception:
                pass
            finally:
                self.midiout = None
                self.port_name = ""

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """Return *True* if a MIDI port is currently opened."""
        return self.midiout is not None

    def __repr__(self) -> str:
        status = f"port='{self.port_name}'" if self.is_active else "inactive"
        return f"MidiOutput({status})"

    def __del__(self) -> None:
        """Attempt to close the port on garbage collection."""
        try:
            if self.midiout is not None:
                self.midiout.close_port()
        except Exception:
            pass  # interpreter may already have torn down the rtmidi module
