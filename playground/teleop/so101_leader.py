"""SO-101 leader arm as an Isaac Lab teleoperation device.

`advance()` returns six absolute joint targets in radians, in sim joint order
(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper), for a
joint-position action with ``use_default_offset=False``.

A failed read never raises out of `advance()`: the last good targets are held,
`status` reports the failure, and the bus is reopened after repeated failures.
"""

from __future__ import annotations

import logging
import time
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from isaaclab.devices.device_base import DeviceBase, DeviceCfg

from .feetech import DEFAULT_BAUDRATE, FeetechBus, LeaderBus
from .so101_calibration import (
    JOINT_NAMES,
    SIM_JOINT_LIMITS,
    LeaderCalibration,
    load_calibration,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = "calibration/so101_leader/default.json"

# Sim init pose of SO101_CFG; held until the first successful read.
SO101_INIT_JOINT_POS = (0.0, 0.0, 0.0, 1.57, 0.0, 0.0)


@dataclass
class LeaderStatus:
    connected: bool
    stale: bool  # True while the targets are a held value, not a fresh read
    consecutive_failures: int
    last_error: str | None
    last_read_time: float | None  # time.monotonic() of the last good read


class SO101Leader(DeviceBase):
    """Reads an SO-101 leader arm over the Feetech bus."""

    def __init__(self, cfg: SO101LeaderCfg, retargeters: list | None = None):
        super().__init__(retargeters)
        self.cfg = cfg
        self._sim_device = cfg.sim_device
        self._limits = dict(cfg.joint_limits or SIM_JOINT_LIMITS)

        self.calibration = cfg.calibration or load_calibration(
            resolve_repo_path(cfg.calibration_path)
        )
        self._ids = self.calibration.motor_ids

        self._targets = list(cfg.fallback_joint_pos)
        if len(self._targets) != len(JOINT_NAMES):
            raise ValueError("fallback_joint_pos needs one value per joint")
        self._stale = True
        self._failures = 0
        self._last_error: str | None = None
        self._last_read_time: float | None = None
        self._next_reconnect = 0.0
        self._callbacks: dict[str, Callable] = {}
        self._keyboard_sub = None

        factory = cfg.bus_factory or (lambda c: FeetechBus(c.port, c.baudrate))
        self._bus: LeaderBus = factory(cfg)
        # Fail loudly at startup: a missing arm is a setup error, not a glitch.
        self._bus.connect()
        self._verify_firmware_offsets()
        self._read()

        if cfg.bind_keyboard:
            self._subscribe_keyboard()

    def __del__(self):
        self.close()

    def __str__(self) -> str:
        s = self.status
        state = "connected" if s.connected else "disconnected"
        msg = f"SO-101 leader on {self.cfg.port} ({state}, {self.calibration.source}"
        msg += " calibration)"
        if s.stale:
            msg += f"\n\tholding last targets: {s.last_error}"
        if self._callbacks:
            msg += f"\n\tkeys: {', '.join(sorted(self._callbacks))}"
        return msg

    """
    Operations
    """

    @property
    def status(self) -> LeaderStatus:
        return LeaderStatus(
            connected=self._bus.is_connected,
            stale=self._stale,
            consecutive_failures=self._failures,
            last_error=self._last_error,
            last_read_time=self._last_read_time,
        )

    def reset(self):
        # Absolute device: nothing to integrate. Retry the bus right away.
        self._next_reconnect = 0.0

    def add_callback(self, key: Any, func: Callable):
        """Bind a keyboard key (carb name, e.g. "R") when running inside Kit."""
        self._callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Six joint targets in radians; the last good value if the read fails."""
        self._read()
        return torch.tensor(self._targets, dtype=torch.float32, device=self._sim_device)

    def close(self):
        if getattr(self, "_keyboard_sub", None) is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(
                    self._keyboard, self._keyboard_sub
                )
            except Exception:
                pass
            self._keyboard_sub = None
        bus = getattr(self, "_bus", None)
        if bus is not None:
            try:
                bus.disconnect()
            except Exception:
                pass

    """
    Internal helpers.
    """

    def _read(self):
        try:
            if not self._bus.is_connected:
                now = time.monotonic()
                if now < self._next_reconnect:
                    raise ConnectionError(self._last_error or "bus disconnected")
                self._next_reconnect = now + self.cfg.reconnect_interval_s
                self._bus.connect()
                logger.info("SO-101 leader reconnected on %s", self.cfg.port)
            present = self._bus.read_positions(self._ids)
            self._targets = self.calibration.to_joint_positions(present, self._limits)
        except Exception as exc:
            self._on_failure(exc)
            return
        if self._stale and self._failures:
            logger.info(
                "SO-101 leader reads recovered after %d failures", self._failures
            )
        self._stale = False
        self._failures = 0
        self._last_error = None
        self._last_read_time = time.monotonic()

    def _on_failure(self, exc: Exception):
        self._stale = True
        self._failures += 1
        self._last_error = f"{type(exc).__name__}: {exc}"
        if self._failures == 1 or self._failures % self.cfg.log_every_n_failures == 0:
            logger.warning(
                "SO-101 leader read failed (%d in a row), holding last targets: %s",
                self._failures,
                self._last_error,
            )
        if self._failures >= self.cfg.failures_before_reconnect:
            try:
                self._bus.disconnect()
            except Exception:
                pass

    def _verify_firmware_offsets(self):
        """Warn when a LeRobot calibration does not match the motors' EEPROM."""
        expected = {
            j.id: j.firmware_homing_offset
            for j in self.calibration.joints.values()
            if j.firmware_homing_offset is not None
        }
        read = getattr(self._bus, "read_homing_offsets", None)
        if not expected or read is None:
            return
        try:
            actual = read(sorted(expected))
        except Exception as exc:
            logger.warning(
                "could not read Homing_Offset to verify calibration: %s", exc
            )
            return
        wrong = {i: (expected[i], actual.get(i)) for i in expected}
        wrong = {i: v for i, v in wrong.items() if v[0] != v[1]}
        if wrong:
            logger.warning(
                "Homing_Offset in the motors differs from the LeRobot calibration "
                "(id: file, motor) %s; re-run calibration or the angles are shifted",
                wrong,
            )

    def _subscribe_keyboard(self):
        try:
            import carb
            import omni
        except ImportError:
            return
        self._carb = carb
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(
                event, *args
            ),
        )

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == self._carb.input.KeyboardEventType.KEY_PRESS:
            callback = self._callbacks.get(event.input.name)
            if callback is not None:
                callback()
        return True


def resolve_repo_path(path: str | Path) -> Path:
    """Expand ~ and resolve relative paths against the repo root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass
class SO101LeaderCfg(DeviceCfg):
    """Configuration for `SO101Leader`."""

    port: str = "/dev/ttyACM0"
    baudrate: int = DEFAULT_BAUDRATE
    # Playground or LeRobot JSON; relative paths are against the repo root.
    calibration_path: str = DEFAULT_CALIBRATION
    # Takes precedence over calibration_path when set.
    calibration: LeaderCalibration | None = None
    # Sim joint limits in radians; None uses so101_new_calib.urdf.
    joint_limits: Mapping[str, tuple[float, float]] | None = None
    fallback_joint_pos: tuple[float, ...] = SO101_INIT_JOINT_POS
    failures_before_reconnect: int = 10
    reconnect_interval_s: float = 1.0
    log_every_n_failures: int = 100
    # Dispatch add_callback keys from the Kit keyboard (no-op outside Kit).
    bind_keyboard: bool = True
    # Builds the bus from this cfg; None uses FeetechBus. Tests pass a fake.
    bus_factory: Callable[[SO101LeaderCfg], LeaderBus] | None = None
    class_type: type[DeviceBase] = SO101Leader
