"""SO-101 leader calibration: STS3215 ticks to sim joint targets in radians.

Two file formats load into the same `LeaderCalibration`:

* playground (written by `scripts/teleop/calibrate_leader.py`)::

      {"format": "playground-so101-leader", "version": 1,
       "joints": {"shoulder_pan": {"id": 1, "homing_offset": 12,
                  "range_min": 900, "range_max": 3200, "sign": 1}, ...}}

  The offset is applied here, in software (the motors are left untouched):
  ``ticks = present - homing_offset``, and ``ticks == 2047`` is 0 rad, i.e. the
  middle pose recorded at calibration.

* LeRobot (``~/.cache/huggingface/lerobot/calibration/teleoperators/<type>/<id>.json``)::

      {"shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -1470,
                        "range_min": 760, "range_max": 3400}, ...}

  LeRobot writes `homing_offset` into the motors' Homing_Offset register, so
  Present_Position already has it applied and it is not subtracted again. Zero is
  the middle of [range_min, range_max], as in LeRobot's DEGREES mode, and
  ``drive_mode == 1`` flips the sign.

Arm joints become ``sign * (ticks - center) * 2 pi / 4096`` clamped to the sim
limits. The gripper is mapped linearly from its calibrated range onto the sim
gripper range (range_min -> lower limit, range_max -> upper limit, reversed when
the sign is negative).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

TICKS_PER_TURN = 4096
HALF_TURN = 2047  # LeRobot's half-turn homing target, int(4095 / 2)
RAD_PER_TICK = 2.0 * math.pi / TICKS_PER_TURN

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_JOINT = "gripper"
JOINT_NAMES = ARM_JOINTS + (GRIPPER_JOINT,)
DEFAULT_MOTOR_IDS = {name: i + 1 for i, name in enumerate(JOINT_NAMES)}

# From so101_new_calib.urdf (TheRobotStudio SO-ARM100, external/isaac-envs).
SIM_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.174533, 1.74533),
}

PLAYGROUND_FORMAT = "playground-so101-leader"
PLAYGROUND_VERSION = 1
LEROBOT_KEYS = {"id", "drive_mode", "homing_offset", "range_min", "range_max"}


class CalibrationError(ValueError):
    """A calibration file is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class JointCalibration:
    """Per-motor calibration in the tick domain."""

    id: int
    range_min: float
    range_max: float
    homing_offset: int = 0  # subtracted in software from Present_Position
    sign: int = 1
    center: float = HALF_TURN  # tick value that maps to 0 rad
    firmware_homing_offset: int | None = None  # expected in EEPROM (LeRobot)

    def __post_init__(self):
        if self.sign not in (1, -1):
            raise CalibrationError(f"motor {self.id}: sign must be 1 or -1")
        if self.range_max <= self.range_min:
            raise CalibrationError(
                f"motor {self.id}: range_max ({self.range_max}) must exceed "
                f"range_min ({self.range_min})"
            )

    def unwrap(self, present: int) -> float:
        """Offset-corrected ticks, continuous within half a turn of `center`."""
        delta = (present - self.homing_offset - self.center + TICKS_PER_TURN // 2) % (
            TICKS_PER_TURN
        ) - TICKS_PER_TURN // 2
        return self.center + delta

    def to_radians(self, present: int) -> float:
        return self.sign * (self.unwrap(present) - self.center) * RAD_PER_TICK

    def to_fraction(self, present: int) -> float:
        """Position within [range_min, range_max] as 0..1, sign applied."""
        frac = (self.unwrap(present) - self.range_min) / (
            self.range_max - self.range_min
        )
        frac = min(1.0, max(0.0, frac))
        return frac if self.sign > 0 else 1.0 - frac


@dataclass(frozen=True)
class LeaderCalibration:
    joints: dict[str, JointCalibration]
    source: str = "playground"  # or "lerobot"
    path: Path | None = field(default=None, compare=False)

    def __post_init__(self):
        missing = [name for name in JOINT_NAMES if name not in self.joints]
        if missing:
            raise CalibrationError(f"calibration lacks joints: {', '.join(missing)}")

    @property
    def motor_ids(self) -> tuple[int, ...]:
        return tuple(self.joints[name].id for name in JOINT_NAMES)

    def to_joint_positions(
        self,
        present: Mapping[int, int],
        limits: Mapping[str, tuple[float, float]] = SIM_JOINT_LIMITS,
    ) -> list[float]:
        """Six sim joint targets (5 arm joints, then gripper) in radians."""
        out = []
        for name in ARM_JOINTS:
            joint = self.joints[name]
            lower, upper = limits[name]
            out.append(min(upper, max(lower, joint.to_radians(present[joint.id]))))
        grip = self.joints[GRIPPER_JOINT]
        lower, upper = limits[GRIPPER_JOINT]
        out.append(lower + grip.to_fraction(present[grip.id]) * (upper - lower))
        return out

    def to_playground_dict(self) -> dict:
        joints = {}
        for name in JOINT_NAMES:
            j = self.joints[name]
            joints[name] = {
                "id": j.id,
                "homing_offset": int(j.homing_offset),
                "range_min": int(j.range_min),
                "range_max": int(j.range_max),
                "sign": j.sign,
            }
        return {
            "format": PLAYGROUND_FORMAT,
            "version": PLAYGROUND_VERSION,
            "joints": joints,
        }


def load_calibration(path: str | Path) -> LeaderCalibration:
    """Load a playground or LeRobot calibration JSON (format is auto-detected)."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise CalibrationError(f"calibration file not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"{path}: invalid JSON ({exc})") from exc
    return parse_calibration(data, path=path)


def parse_calibration(data: Mapping, path: Path | None = None) -> LeaderCalibration:
    if not isinstance(data, Mapping):
        raise CalibrationError("calibration must be a JSON object")
    if data.get("format") == PLAYGROUND_FORMAT:
        return _parse_playground(data, path)
    if data and all(
        isinstance(v, Mapping) and LEROBOT_KEYS <= set(v) for v in data.values()
    ):
        return _parse_lerobot(data, path)
    raise CalibrationError(
        "unrecognised calibration format (expected playground or LeRobot JSON)"
    )


def save_calibration(calibration: LeaderCalibration, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration.to_playground_dict(), indent=4) + "\n")
    return path


def _parse_playground(data: Mapping, path: Path | None) -> LeaderCalibration:
    version = data.get("version")
    if version != PLAYGROUND_VERSION:
        raise CalibrationError(f"unsupported calibration version: {version}")
    joints = {}
    for name, entry in data.get("joints", {}).items():
        try:
            joints[name] = JointCalibration(
                id=int(entry["id"]),
                homing_offset=int(entry["homing_offset"]),
                range_min=int(entry["range_min"]),
                range_max=int(entry["range_max"]),
                sign=int(entry.get("sign", 1)),
            )
        except (KeyError, TypeError) as exc:
            raise CalibrationError(f"joint '{name}': {exc!r}") from exc
    return LeaderCalibration(joints=joints, source="playground", path=path)


def _parse_lerobot(data: Mapping, path: Path | None) -> LeaderCalibration:
    joints = {}
    for name, entry in data.items():
        range_min, range_max = int(entry["range_min"]), int(entry["range_max"])
        joints[name] = JointCalibration(
            id=int(entry["id"]),
            homing_offset=0,
            range_min=range_min,
            range_max=range_max,
            sign=-1 if int(entry["drive_mode"]) else 1,
            center=(range_min + range_max) / 2,
            firmware_homing_offset=int(entry["homing_offset"]),
        )
    return LeaderCalibration(joints=joints, source="lerobot", path=path)
