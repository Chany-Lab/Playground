"""SO-101 leader device and calibration, against a fake Feetech bus."""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch

from isaaclab.devices.device_base import DeviceBase
from playground.teleop.feetech import BusError, decode_sign_magnitude
from playground.teleop.so101_calibration import (
    ARM_JOINTS,
    GRIPPER_JOINT,
    HALF_TURN,
    JOINT_NAMES,
    RAD_PER_TICK,
    SIM_JOINT_LIMITS,
    CalibrationError,
    JointCalibration,
    LeaderCalibration,
    load_calibration,
    parse_calibration,
    save_calibration,
)
from playground.teleop.so101_leader import SO101Leader, SO101LeaderCfg

REPO_ROOT = Path(__file__).resolve().parents[2]
URDF = REPO_ROOT / "external/isaac-envs/assets/robots/manipulation/so101/urdf/so101_new_calib.urdf"

IDS = {name: i + 1 for i, name in enumerate(JOINT_NAMES)}


class FakeBus:
    def __init__(self, positions=None, homing_offsets=None):
        self.positions = dict(positions or {i: HALF_TURN for i in IDS.values()})
        self.homing_offsets = homing_offsets
        self.connected = False
        self.fail = False
        self.refuse_connect = False
        self.connects = 0

    @property
    def is_connected(self):
        return self.connected

    def connect(self):
        if self.refuse_connect:
            raise BusError("no such port")
        self.connected = True
        self.connects += 1

    def disconnect(self):
        self.connected = False

    def read_positions(self, ids):
        if not self.connected:
            raise BusError("not connected")
        if self.fail:
            raise BusError("timeout")
        return {i: self.positions[i] for i in ids}


def playground_calibration(offset=0, sign=None, grip_range=(1500, 2500)):
    sign = sign or {}
    joints = {
        name: JointCalibration(
            id=IDS[name],
            homing_offset=offset,
            range_min=1000,
            range_max=3000,
            sign=sign.get(name, 1),
        )
        for name in ARM_JOINTS
    }
    joints[GRIPPER_JOINT] = JointCalibration(
        id=IDS[GRIPPER_JOINT],
        homing_offset=offset,
        range_min=grip_range[0],
        range_max=grip_range[1],
        sign=sign.get(GRIPPER_JOINT, 1),
    )
    return LeaderCalibration(joints=joints)


def make_leader(bus=None, calibration=None, **kw):
    bus = bus or FakeBus()
    cfg = SO101LeaderCfg(
        calibration=calibration or playground_calibration(),
        bus_factory=lambda cfg: bus,
        bind_keyboard=False,
        **kw,
    )
    return SO101Leader(cfg), bus


# ticks -> radians


def test_middle_pose_is_zero():
    joint = JointCalibration(id=1, range_min=1000, range_max=3000)
    assert joint.to_radians(HALF_TURN) == 0.0


def test_homing_offset_is_subtracted():
    joint = JointCalibration(id=1, homing_offset=500, range_min=1000, range_max=3000)
    assert joint.to_radians(HALF_TURN + 500) == 0.0
    assert joint.to_radians(HALF_TURN + 600) == pytest.approx(100 * RAD_PER_TICK)


def test_quarter_turn_and_sign():
    pos = JointCalibration(id=1, range_min=0, range_max=4095)
    neg = JointCalibration(id=1, range_min=0, range_max=4095, sign=-1)
    assert pos.to_radians(HALF_TURN + 1024) == pytest.approx(math.pi / 2)
    assert neg.to_radians(HALF_TURN + 1024) == pytest.approx(-math.pi / 2)


def test_wraps_across_encoder_zero():
    # Homed near the encoder seam: raw 4000 is the middle, raw 10 is 106 ticks past.
    joint = JointCalibration(
        id=1, homing_offset=4000 - HALF_TURN, range_min=1000, range_max=3000
    )
    assert joint.to_radians(4000) == 0.0
    assert joint.to_radians(10) == pytest.approx(106 * RAD_PER_TICK)
    assert joint.to_radians(3900) == pytest.approx(-100 * RAD_PER_TICK)


def test_sign_magnitude_decoding():
    assert decode_sign_magnitude(2047, 15) == 2047
    assert decode_sign_magnitude(0x8000 | 25, 15) == -25
    assert decode_sign_magnitude(0x800 | 300, 11) == -300


def test_negative_present_position_after_firmware_offset():
    # A firmware Homing_Offset can make Present_Position read slightly negative.
    joint = JointCalibration(id=1, range_min=0, range_max=2000, center=1000)
    assert joint.to_radians(-5) == pytest.approx(-1005 * RAD_PER_TICK)


def test_invalid_joint_calibration():
    with pytest.raises(CalibrationError):
        JointCalibration(id=1, range_min=3000, range_max=1000)
    with pytest.raises(CalibrationError):
        JointCalibration(id=1, range_min=0, range_max=10, sign=2)


# limits and gripper


def test_sim_limits_match_urdf():
    root = ET.parse(URDF).getroot()
    seen = set()
    for joint in root.findall("joint"):  # top level only, not <transmission>
        name = joint.get("name")
        if name in SIM_JOINT_LIMITS:
            seen.add(name)
            limit = joint.find("limit")
            lower, upper = float(limit.get("lower")), float(limit.get("upper"))
            assert SIM_JOINT_LIMITS[name] == pytest.approx((lower, upper))
    assert seen == set(SIM_JOINT_LIMITS)


def test_arm_targets_clamped_to_sim_limits():
    calib = playground_calibration()
    present = {i: HALF_TURN for i in IDS.values()}
    present[IDS["shoulder_pan"]] = HALF_TURN + 1800  # about +158 deg
    present[IDS["elbow_flex"]] = HALF_TURN - 1800
    q = calib.to_joint_positions(present)
    assert q[0] == pytest.approx(SIM_JOINT_LIMITS["shoulder_pan"][1])
    assert q[2] == pytest.approx(SIM_JOINT_LIMITS["elbow_flex"][0])
    assert q[1] == 0.0


def test_gripper_maps_range_onto_sim_range():
    calib = playground_calibration(grip_range=(1500, 2500))
    lower, upper = SIM_JOINT_LIMITS[GRIPPER_JOINT]
    grip = IDS[GRIPPER_JOINT]

    def gripper_at(ticks):
        present = {i: HALF_TURN for i in IDS.values()}
        present[grip] = ticks
        return calib.to_joint_positions(present)[5]

    assert gripper_at(1500) == pytest.approx(lower)
    assert gripper_at(2500) == pytest.approx(upper)
    assert gripper_at(2000) == pytest.approx((lower + upper) / 2)
    assert gripper_at(900) == pytest.approx(lower)  # clamped
    assert gripper_at(3500) == pytest.approx(upper)


def test_inverted_gripper():
    calib = playground_calibration(sign={GRIPPER_JOINT: -1})
    present = {i: HALF_TURN for i in IDS.values()}
    present[IDS[GRIPPER_JOINT]] = 1500
    lower, upper = SIM_JOINT_LIMITS[GRIPPER_JOINT]
    assert calib.to_joint_positions(present)[5] == pytest.approx(upper)


def test_custom_limits():
    calib = playground_calibration()
    limits = dict(SIM_JOINT_LIMITS, wrist_roll=(-0.1, 0.1))
    present = {i: HALF_TURN + 500 for i in IDS.values()}
    assert calib.to_joint_positions(present, limits)[4] == pytest.approx(0.1)


# calibration files


LEROBOT_JSON = {
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": -1470,
        "range_min": 800,
        "range_max": 3300,
    },
    "shoulder_lift": {
        "id": 2,
        "drive_mode": 0,
        "homing_offset": 1100,
        "range_min": 820,
        "range_max": 3200,
    },
    "elbow_flex": {
        "id": 3,
        "drive_mode": 1,
        "homing_offset": 900,
        "range_min": 870,
        "range_max": 3100,
    },
    "wrist_flex": {
        "id": 4,
        "drive_mode": 0,
        "homing_offset": -40,
        "range_min": 900,
        "range_max": 3150,
    },
    "wrist_roll": {
        "id": 5,
        "drive_mode": 0,
        "homing_offset": 15,
        "range_min": 0,
        "range_max": 4095,
    },
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": 1300,
        "range_min": 2000,
        "range_max": 3400,
    },
}


def test_load_lerobot_json(tmp_path):
    path = tmp_path / "leader.json"
    path.write_text(json.dumps(LEROBOT_JSON, indent=4))
    calib = load_calibration(path)

    assert calib.source == "lerobot"
    assert calib.motor_ids == (1, 2, 3, 4, 5, 6)
    pan = calib.joints["shoulder_pan"]
    # Firmware applies the offset; zero is the middle of the range.
    assert pan.homing_offset == 0
    assert pan.firmware_homing_offset == -1470
    assert pan.center == (800 + 3300) / 2
    assert pan.to_radians(2050) == 0.0
    assert pan.to_radians(2150) == pytest.approx(100 * RAD_PER_TICK)
    # drive_mode 1 flips the direction.
    elbow = calib.joints["elbow_flex"]
    assert elbow.to_radians(int(elbow.center) + 100) < 0


def test_load_playground_roundtrip(tmp_path):
    calib = playground_calibration(offset=123, sign={"wrist_roll": -1})
    path = save_calibration(calib, tmp_path / "sub" / "leader.json")
    loaded = load_calibration(path)
    assert loaded.source == "playground"
    assert loaded.joints == calib.joints


def test_unknown_or_incomplete_files_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"hello": 1}))
    with pytest.raises(CalibrationError):
        load_calibration(bad)

    partial = dict(LEROBOT_JSON)
    del partial["gripper"]
    bad.write_text(json.dumps(partial))
    with pytest.raises(CalibrationError, match="gripper"):
        load_calibration(bad)

    with pytest.raises(CalibrationError, match="not found"):
        load_calibration(tmp_path / "missing.json")


# device


def test_device_is_a_device_base_with_class_type():
    assert issubclass(SO101Leader, DeviceBase)
    assert SO101LeaderCfg().class_type is SO101Leader


def test_advance_shape_order_dtype():
    bus = FakeBus()
    bus.positions[IDS["shoulder_lift"]] = HALF_TURN + 200
    bus.positions[IDS["wrist_roll"]] = HALF_TURN - 300
    leader, _ = make_leader(bus)

    action = leader.advance()
    assert action.shape == (6,)
    assert action.dtype == torch.float32
    assert action.device.type == "cpu"
    expected = torch.zeros(6)
    expected[1] = 200 * RAD_PER_TICK
    expected[4] = -300 * RAD_PER_TICK
    expected[5] = SIM_JOINT_LIMITS[GRIPPER_JOINT][0] + 0.547 * (
        SIM_JOINT_LIMITS[GRIPPER_JOINT][1] - SIM_JOINT_LIMITS[GRIPPER_JOINT][0]
    )
    assert torch.allclose(action, expected, atol=1e-6)
    # Batched like record_demos does it.
    assert action.repeat(4, 1).shape == (4, 6)


def test_read_failure_holds_last_value():
    leader, bus = make_leader()
    bus.positions[IDS["shoulder_pan"]] = HALF_TURN + 400
    good = leader.advance()
    assert not leader.status.stale

    bus.fail = True
    bus.positions[IDS["shoulder_pan"]] = HALF_TURN - 400
    for _ in range(3):
        held = leader.advance()
        assert torch.equal(held, good)
    status = leader.status
    assert status.stale
    assert status.consecutive_failures == 3
    assert "timeout" in status.last_error

    bus.fail = False
    fresh = leader.advance()
    assert fresh[0] == pytest.approx(-400 * RAD_PER_TICK)
    assert not leader.status.stale
    assert leader.status.consecutive_failures == 0


def test_reconnects_after_repeated_failures():
    leader, bus = make_leader(failures_before_reconnect=2, reconnect_interval_s=0.0)
    bus.fail = True
    leader.advance()
    leader.advance()
    assert not bus.connected  # dropped after the second failure

    bus.fail = False
    leader.advance()
    assert bus.connected
    assert bus.connects == 2
    assert not leader.status.stale


def test_unplugged_bus_holds_value_without_raising():
    leader, bus = make_leader(reconnect_interval_s=0.0)
    good = leader.advance()
    bus.connected = False
    bus.refuse_connect = True
    for _ in range(5):
        assert torch.equal(leader.advance(), good)
    assert leader.status.stale and not leader.status.connected


def test_fallback_until_first_good_read():
    bus = FakeBus()
    bus.fail = True
    leader, _ = make_leader(bus)
    assert leader.status.stale
    assert leader.advance().tolist() == pytest.approx([0.0, 0.0, 0.0, 1.57, 0.0, 0.0])


def test_startup_without_arm_raises():
    bus = FakeBus()
    bus.refuse_connect = True
    with pytest.raises(BusError):
        make_leader(bus)


def test_firmware_offset_mismatch_warns(caplog):
    bus = FakeBus(positions={i: 2000 for i in IDS.values()})
    bus.read_homing_offsets = lambda ids: {i: 0 for i in ids}
    calib = parse_calibration(LEROBOT_JSON)
    with caplog.at_level("WARNING"):
        make_leader(bus, calibration=calib)
    assert "Homing_Offset" in caplog.text


def test_callbacks_and_reset():
    leader, _ = make_leader()
    called = []
    leader.add_callback("R", lambda: called.append(1))
    leader.reset()
    assert "R" in str(leader)


def test_calibration_path_from_cfg(tmp_path):
    path = tmp_path / "leader.json"
    path.write_text(json.dumps(LEROBOT_JSON))
    bus = FakeBus()
    cfg = SO101LeaderCfg(
        calibration_path=str(path), bus_factory=lambda cfg: bus, bind_keyboard=False
    )
    assert SO101Leader(cfg).calibration.source == "lerobot"


# calibration CLI


def load_cli():
    import importlib.util

    path = REPO_ROOT / "scripts/teleop/calibrate_leader.py"
    spec = importlib.util.spec_from_file_location("calibrate_leader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_calibrate_records_homing_ranges_and_gripper_sign(monkeypatch):
    cli = load_cli()
    bus = FakeBus(positions={i: 3000 for i in IDS.values()})  # middle pose
    bus.connect()
    sweep = iter([2400, 3600, 3000])  # each joint swings -600..+600 ticks
    prompts = []

    def fake_enter():
        try:
            ticks = next(sweep)
        except StopIteration:
            # Gripper closed at the high end of its sweep for step 3.
            bus.positions[IDS[GRIPPER_JOINT]] = 3590
            return True
        for i in IDS.values():
            bus.positions[i] = ticks
        return False

    monkeypatch.setattr(cli, "_enter_pressed", fake_enter)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    monkeypatch.setattr("builtins.input", lambda msg="": prompts.append(msg))

    calib = cli.calibrate(bus, invert={"wrist_flex"})
    assert len(prompts) == 2
    pan = calib.joints["shoulder_pan"]
    assert pan.homing_offset == 3000 - HALF_TURN
    assert (pan.range_min, pan.range_max) == (HALF_TURN - 600, HALF_TURN + 600)
    assert calib.joints["wrist_flex"].sign == -1
    assert calib.joints[GRIPPER_JOINT].sign == -1  # closes toward range_max

    present = {i: 3000 for i in IDS.values()}
    present[IDS[GRIPPER_JOINT]] = 3600
    q = calib.to_joint_positions(present)
    assert q[:5] == [0.0] * 5
    assert q[5] == pytest.approx(SIM_JOINT_LIMITS[GRIPPER_JOINT][0])
