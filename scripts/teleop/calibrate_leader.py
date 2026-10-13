"""Calibrate an SO-101 leader arm without LeRobot, or stream its joint angles.

    .venv/bin/python scripts/teleop/calibrate_leader.py --port /dev/ttyACM0 --name default
    .venv/bin/python scripts/teleop/calibrate_leader.py --check --name default

Calibration only reads the motors; nothing is written to their EEPROM. It records
the middle pose (homing), each joint's range, and which way the gripper closes,
then writes `calibration/so101_leader/<name>.json` under the repo root.
"""

from __future__ import annotations

import argparse
import math
import select
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from playground.teleop.feetech import (  # noqa: E402
    DEFAULT_BAUDRATE,
    BusError,
    FeetechBus,
)
from playground.teleop.so101_calibration import (  # noqa: E402
    ARM_JOINTS,
    DEFAULT_MOTOR_IDS,
    GRIPPER_JOINT,
    HALF_TURN,
    JOINT_NAMES,
    SIM_JOINT_LIMITS,
    CalibrationError,
    JointCalibration,
    LeaderCalibration,
    load_calibration,
    save_calibration,
)

CALIBRATION_DIR = Path("calibration/so101_leader")
MIN_RANGE_TICKS = 20
TICKS_MAX = 4095


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = Path(args.calibration) if args.calibration else _default_path(args.name)
    if not path.is_absolute():
        path = REPO_ROOT / path

    try:
        calibration = load_calibration(path) if args.check else None
        bus = FeetechBus(args.port, args.baudrate)
        bus.connect()
    except (CalibrationError, BusError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        if args.check:
            check(bus, calibration, args.rate)
        else:
            calibration = calibrate(bus, invert=set(args.invert))
            save_calibration(calibration, path)
            print(f"\nSaved {path}")
    except KeyboardInterrupt:
        print()
    except BusError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    finally:
        bus.disconnect()
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--name", default="default", help="calibration file name")
    parser.add_argument(
        "--calibration",
        help="explicit calibration path (playground or LeRobot JSON for --check)",
    )
    parser.add_argument(
        "--check", action="store_true", help="stream converted joint angles"
    )
    parser.add_argument("--rate", type=float, default=20.0, help="--check rate in Hz")
    parser.add_argument(
        "--invert",
        action="append",
        default=[],
        choices=ARM_JOINTS,
        help="flip an arm joint's direction (repeatable)",
    )
    return parser.parse_args(argv)


def calibrate(bus: FeetechBus, invert: set[str]) -> LeaderCalibration:
    ids = [DEFAULT_MOTOR_IDS[name] for name in JOINT_NAMES]

    input(
        "1/3  Move the leader to the middle of every joint's range "
        "(the LeRobot SO-101 homing pose) and press ENTER..."
    )
    present = bus.read_positions(ids)
    offsets = {
        name: present[DEFAULT_MOTOR_IDS[name]] - HALF_TURN for name in JOINT_NAMES
    }

    print(
        "2/3  Move every joint, one by one, through its full range. "
        "Press ENTER when done."
    )
    homed = {
        name: JointCalibration(
            id=DEFAULT_MOTOR_IDS[name],
            homing_offset=offsets[name],
            range_min=0,
            range_max=TICKS_MAX,
        )
        for name in JOINT_NAMES
    }
    mins = {name: float(HALF_TURN) for name in JOINT_NAMES}
    maxes = dict(mins)
    while not _enter_pressed():
        present = bus.read_positions(ids)
        line = []
        for name, joint in homed.items():
            ticks = joint.unwrap(present[joint.id])
            mins[name] = min(mins[name], ticks)
            maxes[name] = max(maxes[name], ticks)
            line.append(f"{name[:8]} {int(mins[name]):4d}..{int(maxes[name]):4d}")
        print("\r" + " | ".join(line), end="", flush=True)
        time.sleep(0.02)
    print()

    narrow = [n for n in JOINT_NAMES if maxes[n] - mins[n] < MIN_RANGE_TICKS]
    if narrow:
        raise SystemExit(f"joints did not move enough: {', '.join(narrow)}")

    input("3/3  Close the gripper fully and press ENTER...")
    grip = homed[GRIPPER_JOINT]
    closed = grip.unwrap(bus.read_positions(ids)[grip.id])
    # Closed must map to the sim gripper's lower limit (range_min end).
    grip_sign = (
        1 if closed - mins[GRIPPER_JOINT] <= maxes[GRIPPER_JOINT] - closed else -1
    )

    joints = {}
    for name in JOINT_NAMES:
        sign = grip_sign if name == GRIPPER_JOINT else (-1 if name in invert else 1)
        joints[name] = JointCalibration(
            id=DEFAULT_MOTOR_IDS[name],
            homing_offset=offsets[name],
            range_min=int(math.floor(mins[name])),
            range_max=int(math.ceil(maxes[name])),
            sign=sign,
        )
    return LeaderCalibration(joints=joints, source="playground")


def check(bus: FeetechBus, calibration: LeaderCalibration, rate: float):
    ids = calibration.motor_ids
    print(f"{calibration.source} calibration; Ctrl+C to stop")
    print("  ".join(f"{name:>13}" for name in JOINT_NAMES))
    period = 1.0 / rate
    while True:
        try:
            present = bus.read_positions(ids)
        except OSError as exc:
            print(f"\rread failed: {exc}".ljust(100), end="", flush=True)
            time.sleep(period)
            continue
        q = calibration.to_joint_positions(present, SIM_JOINT_LIMITS)
        cells = [f"{v:+6.3f} ({math.degrees(v):+6.1f})" for v in q]
        print("\r" + "  ".join(f"{c:>13}" for c in cells), end="", flush=True)
        time.sleep(period)


def _default_path(name: str) -> Path:
    return CALIBRATION_DIR / f"{name}.json"


def _enter_pressed() -> bool:
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        sys.stdin.readline()
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
