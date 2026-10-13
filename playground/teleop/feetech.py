"""Minimal Feetech STS3215 bus access for reading a leader arm.

Only what a passive leader needs: sync read of Present_Position and a plain read
of Homing_Offset. Torque is never touched. `scservo_sdk` (feetech-servo-sdk) is
imported on connect so this module loads without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

DEFAULT_BAUDRATE = 1_000_000
PROTOCOL_END = 0  # STS/SMS series use protocol 0 (little endian words)

# STS3215 control table (address, size in bytes).
ADDR_HOMING_OFFSET = 31
ADDR_PRESENT_POSITION = 56
WORD = 2

# Sign-magnitude sign bits used by the STS series.
PRESENT_POSITION_SIGN_BIT = 15
HOMING_OFFSET_SIGN_BIT = 11


def decode_sign_magnitude(value: int, sign_bit: int) -> int:
    """Decode a Feetech sign-magnitude register value."""
    magnitude = value & ((1 << sign_bit) - 1)
    return -magnitude if value & (1 << sign_bit) else magnitude


class BusError(IOError):
    """A read on the servo bus failed."""


@runtime_checkable
class LeaderBus(Protocol):
    """What the leader device needs from a servo bus. Tests inject a fake."""

    @property
    def is_connected(self) -> bool: ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def read_positions(self, ids: Sequence[int]) -> dict[int, int]:
        """Present_Position per motor id, decoded. Raises on any failure."""
        ...


class FeetechBus:
    """`LeaderBus` over a Feetech serial adapter (e.g. /dev/ttyACM0)."""

    def __init__(self, port: str, baudrate: int = DEFAULT_BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self._port_handler = None
        self._packet_handler = None
        self._sync_read = None
        self._sync_ids: tuple[int, ...] = ()

    @property
    def is_connected(self) -> bool:
        return self._port_handler is not None and self._port_handler.is_open

    def connect(self) -> None:
        import scservo_sdk as scs

        port_handler = scs.PortHandler(self.port)
        # The PyPI SDK computes a packet timeout too short for sync reads; use the
        # same fix as LeRobot (tx time + 3 bytes + 50 ms).
        port_handler.setPacketTimeout = _patched_set_packet_timeout.__get__(
            port_handler, scs.PortHandler
        )
        try:
            opened = port_handler.openPort()
        except Exception as exc:
            raise BusError(f"cannot open {self.port}: {exc}") from exc
        if not opened:
            raise BusError(f"cannot open {self.port}")
        if not port_handler.setBaudRate(self.baudrate):
            port_handler.closePort()
            raise BusError(f"cannot set baud rate {self.baudrate} on {self.port}")
        self._port_handler = port_handler
        self._packet_handler = scs.PacketHandler(PROTOCOL_END)
        self._sync_read = None
        self._sync_ids = ()

    def disconnect(self) -> None:
        if self._port_handler is not None:
            try:
                self._port_handler.closePort()
            except Exception:
                pass
        self._port_handler = None
        self._packet_handler = None
        self._sync_read = None
        self._sync_ids = ()

    def read_positions(self, ids: Sequence[int]) -> dict[int, int]:
        import scservo_sdk as scs

        if not self.is_connected:
            raise BusError("bus is not connected")
        ids = tuple(ids)
        if self._sync_read is None or self._sync_ids != ids:
            group = scs.GroupSyncRead(
                self._port_handler,
                self._packet_handler,
                ADDR_PRESENT_POSITION,
                WORD,
            )
            for motor_id in ids:
                group.addParam(motor_id)
            self._sync_read, self._sync_ids = group, ids

        try:
            result = self._sync_read.txRxPacket()
        except Exception as exc:  # serial errors after an unplug
            raise BusError(f"sync read failed: {exc}") from exc
        if result != scs.COMM_SUCCESS:
            msg = self._packet_handler.getTxRxResult(result)
            raise BusError(f"sync read failed: {msg}")

        positions = {}
        for motor_id in ids:
            if not self._sync_read.isAvailable(motor_id, ADDR_PRESENT_POSITION, WORD):
                raise BusError(f"no Present_Position from motor {motor_id}")
            raw = self._sync_read.getData(motor_id, ADDR_PRESENT_POSITION, WORD)
            positions[motor_id] = decode_sign_magnitude(raw, PRESENT_POSITION_SIGN_BIT)
        return positions

    def read_homing_offsets(self, ids: Sequence[int]) -> dict[int, int]:
        """Homing_Offset stored in each motor's EEPROM, decoded."""
        import scservo_sdk as scs

        if not self.is_connected:
            raise BusError("bus is not connected")
        offsets = {}
        for motor_id in ids:
            raw, result, _ = self._packet_handler.read2ByteTxRx(
                self._port_handler, motor_id, ADDR_HOMING_OFFSET
            )
            if result != scs.COMM_SUCCESS:
                msg = self._packet_handler.getTxRxResult(result)
                raise BusError(f"Homing_Offset read failed on motor {motor_id}: {msg}")
            offsets[motor_id] = decode_sign_magnitude(raw, HOMING_OFFSET_SIGN_BIT)
        return offsets


def _patched_set_packet_timeout(self, packet_length):
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (
        self.tx_time_per_byte * packet_length + self.tx_time_per_byte * 3.0 + 50
    )
