"""Builds an environment from its Python script: `build(scene)` places props and the robot.

An environment script (Interface's data/sims/envs/<id>.py, or <id>/env.py) only defines

    def build(scene):
        scene.add("desk")
        scene.add("cube", pos=(0, 0.08, 0.75))
        scene.robot(pos=(0, -0.2, 0.75), yaw=90)

Props come from Isaac-Envs' assets/props/<name>/<name>.usd[a] on this machine (origin at their
bottom, centred). The robot is not named in the script: it is the environment's robot tag
(Interface's Environments page), sent with the scene in Isaac-Envs' layout and loaded from
<root>/sims/robots/<name>/<name>.usd[a]; scene.robot_name says which one. The robot is pinned to
the world with a fixed joint, so it stands when Play starts. scene.tool(<name>) places an end
effector from sims/tools/<name> the same way. The robot.yaml next to a robot's USD
(initial_pose) sets the joints it starts and holds in. Units are metres, Z up, yaw in degrees
about Z. pxr only, so the same code builds a stage with usd-core outside Isaac Sim.
"""

import importlib.util
from pathlib import Path

import yaml

from isaac_envs import ASSETS_ROOT
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

ASSETS_DIR = ASSETS_ROOT / "props"
USD_EXTS = (".usd", ".usda", ".usdc")
FLOOR_SIZE = 20.0


def _usd_file(folder: Path, name: str) -> Path:
    for ext in USD_EXTS:
        for p in (folder / name / f"{name}{ext}", folder / f"{name}{ext}"):
            if p.is_file():
                return p
    raise FileNotFoundError(f"no USD for '{name}' in {folder}")


def _place(prim: Usd.Prim, pos, yaw: float, scale: float = 1.0) -> None:
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddRotateZOp().Set(float(yaw))
    if scale != 1.0:
        xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))


class Scene:
    """What an environment script gets: a stage with /World, physics, lights and a floor."""

    def __init__(self, stage: Usd.Stage, root: Path, robot: str | None = None):
        self.stage = stage
        self.robots_dir = root / "sims" / "robots"
        self.tools_dir = root / "sims" / "tools"
        self.robot_name = robot
        self.robot_path: str | None = None
        self.tool_paths: list[str] = []
        self.leader_offsets: dict[str, float] = {}  # leader reading → joint value (robot.yaml)
        self.tcp: dict | None = None  # robot.yaml tcp (keyboard TCP jog)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        physics = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        physics.CreateGravityMagnitudeAttr(9.81)
        UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1000)
        key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        key.CreateIntensityAttr(2500)
        key.AddRotateXYZOp().Set(Gf.Vec3f(45, 0, 30))
        floor = UsdGeom.Cube.Define(stage, "/World/floor")
        floor.CreateSizeAttr(1.0)
        floor.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.8, 0.8)])
        floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.01))
        floor.AddScaleOp().Set(Gf.Vec3f(FLOOR_SIZE, FLOOR_SIZE, 0.02))
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    def _free_path(self, name: str) -> str:
        path, n = f"/World/{name}", 1
        while self.stage.GetPrimAtPath(path):
            path, n = f"/World/{name}_{n}", n + 1
        return path

    def add(
        self,
        asset: str,
        pos=(0, 0, 0),
        yaw: float = 0,
        name: str | None = None,
        scale: float = 1.0,
        color=None,
    ) -> Usd.Prim:
        """References the prop assets/props/<asset> at pos; color (r, g, b in 0–1) recolours its meshes."""
        prim = self.stage.DefinePrim(self._free_path(name or asset), "Xform")
        prim.GetReferences().AddReference(str(_usd_file(ASSETS_DIR, asset)))
        _place(prim, pos, yaw, scale)
        if color is not None:
            for p in Usd.PrimRange(prim):
                if p.IsA(UsdGeom.Gprim):
                    UsdGeom.Gprim(p).CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return prim

    def robot(self, pos=(0, 0, 0), yaw: float = 0, name: str | None = None) -> Usd.Prim:
        """References the robot (sims/robots/<name>, the tagged one by default) at /World/Robot, its
        base pinned where it stands."""
        if self.robot_path:
            raise ValueError("an environment places one robot")
        name = name or self.robot_name
        if not name:
            raise ValueError("no robot: tag the environment with one on the Environments page")
        prim = self._articulation("/World/Robot", self.robots_dir, name, pos, yaw)
        self.robot_path = str(prim.GetPath())
        usd = _usd_file(self.robots_dir, name)
        self.leader_offsets = leader_offsets(usd)
        self.tcp = tcp_config(usd)
        pose, percent = initial_pose(usd)
        if pose:
            Drives(self.stage, self.robot_path).start_at(pose, percent)
        return prim

    def tool(self, name: str, pos=(0, 0, 0), yaw: float = 0) -> Usd.Prim:
        """References an end effector (sims/tools/<name>, e.g. a robot hand) at /World/<name>, its base
        pinned where it stands. An arm with a hand mounted is a robot of its own in Isaac-Envs."""
        prim = self._articulation(self._free_path(name), self.tools_dir, name, pos, yaw)
        self.tool_paths.append(str(prim.GetPath()))
        return prim

    @property
    def teleop_path(self) -> str | None:
        """What teleoperation drives: the robot, else the first tool (a tool opened alone)."""
        return self.robot_path or (self.tool_paths[0] if self.tool_paths else None)

    def _articulation(self, path: str, folder: Path, name: str, pos, yaw: float) -> Usd.Prim:
        prim = self.stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(str(_usd_file(folder, name)))
        _place(prim, pos, yaw)
        # A robot file may carry its own PhysicsScene; the stage keeps one
        for p in Usd.PrimRange(prim):
            if p.IsA(UsdPhysics.Scene):
                p.SetActive(False)
        _pin(self.stage, prim)
        return prim


def _pin(stage: Usd.Stage, robot: Usd.Prim) -> None:
    """Fixed base: the articulation root moves from the base body to the robot Xform and a
    fixed joint ties the base to the world at its current pose."""
    # A robot file's own world joints (e.g. an active root_joint) pin it in world coordinates,
    # wherever the script places it
    for p in list(Usd.PrimRange(robot)):
        if p.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(p)
            if not j.GetBody0Rel().GetTargets() or not j.GetBody1Rel().GetTargets():
                p.SetActive(False)
    roots = [p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    bodies = [p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    base = next((p for p in roots if p.HasAPI(UsdPhysics.RigidBodyAPI)), None) or (
        bodies[0] if bodies else None
    )
    if base is None:
        return
    for p in roots:
        if p != robot:
            p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    UsdPhysics.ArticulationRootAPI.Apply(robot)
    m = UsdGeom.XformCache().GetLocalToWorldTransform(base).RemoveScaleShear()
    # Own name: robot files from URDF often carry an inactive "root_joint" that Define would reuse
    joint = UsdPhysics.FixedJoint.Define(stage, robot.GetPath().AppendChild("mimic_fixed_base"))
    joint.CreateBody1Rel().SetTargets([base.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(m.ExtractTranslation()))
    joint.CreateLocalRot0Attr(Gf.Quatf(m.ExtractRotationQuat()))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))


def build(stage: Usd.Stage, script: Path, root: Path, robot: str | None = None) -> Scene:
    """Runs the script's build(scene) on the stage; root is the folder the scene arrived in."""
    spec = importlib.util.spec_from_file_location(f"mimic_env_{script.stem}", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build", None)):
        raise ValueError(f"{script.name} defines no build(scene)")
    scene = Scene(stage, root, robot)
    module.build(scene)
    return scene


def robot_config(usd: Path) -> Path:
    """A robot's config next to its USD: <id>/robot.yaml for a folder robot, else <id>.yaml."""
    return usd.parent / "robot.yaml" if usd.parent.name == usd.stem else usd.with_suffix(".yaml")


def _floats(d) -> dict[str, float]:
    return {str(k): float(v) for k, v in (d or {}).items()}


def initial_pose(usd: Path) -> tuple[dict[str, float], tuple[str, ...]]:
    """initial_pose (joint name → degrees) and its percent joints (0–100 over the limits) from
    the robot's config; empty without one."""
    path = robot_config(usd)
    if not path.is_file():
        return {}, ()
    doc = yaml.safe_load(path.read_text()) or {}
    return _floats(doc.get("initial_pose")), tuple(doc.get("percent") or ())


def tcp_config(usd: Path) -> dict | None:
    """robot.yaml tcp: {link, offset (m, in the link frame), joints (the IK may move)}, or None."""
    path = robot_config(usd)
    doc = (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}
    tcp = doc.get("tcp")
    if not isinstance(tcp, dict) or not tcp.get("link"):
        return None
    return {
        "link": str(tcp["link"]),
        "offset": [float(v) for v in tcp.get("offset") or (0, 0, 0)],
        "joints": [str(j) for j in tcp.get("joints") or []] or None,
    }


def leader_offsets(usd: Path) -> dict[str, float]:
    """What to add to a leader reading to get the robot's joint value: the leader reads
    leader.rest while the robot is in initial_pose (a LeRobot leader reads 0° at the pose it was
    calibrated in, which need not be the USD's zero)."""
    path = robot_config(usd)
    if not path.is_file():
        return {}
    doc = yaml.safe_load(path.read_text()) or {}
    pose, rest = _floats(doc.get("initial_pose")), _floats((doc.get("leader") or {}).get("rest"))
    return {j: pose.get(j, 0.0) - v for j, v in rest.items()}


class Drives:
    """Joint drives of the placed robot by joint name, for teleoperation: targets in degrees
    (metres for a prismatic joint); a joint listed as percent takes 0–100 over its limits."""

    def __init__(self, stage: Usd.Stage, root: str = "/World/Robot"):
        self.drives: dict[str, tuple[UsdPhysics.DriveAPI, float | None, float | None]] = {}
        self.kinds: dict[str, str] = {}  # "angular" | "linear"
        self.offsets: dict[str, float] = {}  # added to a leader reading (scene.leader_offsets)
        prim = stage.GetPrimAtPath(root)
        if not prim:
            return
        for p in Usd.PrimRange(prim):
            for kind, joint_type in (
                ("angular", UsdPhysics.RevoluteJoint),
                ("linear", UsdPhysics.PrismaticJoint),
            ):
                if p.IsA(joint_type) and p.HasAPI(UsdPhysics.DriveAPI, kind):
                    # A drive without stiffness is passive (a hand's coupled finger segments)
                    if not UsdPhysics.DriveAPI(p, kind).GetStiffnessAttr().Get():
                        continue
                    j = joint_type(p)
                    lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
                    self.drives[p.GetName()] = (UsdPhysics.DriveAPI(p, kind), lo, hi)
                    self.kinds[p.GetName()] = kind

    def _value(self, name: str, value: float, percent: tuple[str, ...]) -> float:
        _, lo, hi = self.drives[name]
        if name in percent and lo is not None and hi is not None:
            value = lo + max(0.0, min(100.0, value)) / 100 * (hi - lo)
        if lo is not None and hi is not None:
            value = max(lo, min(hi, value))
        return float(value)

    def set(self, targets: dict[str, float], percent: tuple[str, ...] = ()) -> list[str]:
        """Sets the drive targets it knows (clamped to the limits); returns the names applied."""
        applied = []
        for name, value in targets.items():
            if name in self.drives:
                self.drives[name][0].GetTargetPositionAttr().Set(self._value(name, value, percent))
                applied.append(name)
        return applied

    def targets(self) -> dict[str, float]:
        """Present drive target per joint (degrees)."""
        return {n: float(d.GetTargetPositionAttr().Get() or 0.0) for n, (d, _, _) in self.drives.items()}

    def jog(self, velocities: dict[str, float], dt: float) -> None:
        """Moves drive targets by velocity × dt (degrees per second), stopping at the limits."""
        for name, v in velocities.items():
            if name in self.drives and v:
                drive, lo, hi = self.drives[name]
                value = float(drive.GetTargetPositionAttr().Get() or 0.0) + v * dt
                if lo is not None and hi is not None:
                    value = max(lo, min(hi, value))
                drive.GetTargetPositionAttr().Set(value)

    def start_at(self, pose: dict[str, float], percent: tuple[str, ...] = ()) -> list[str]:
        """Initial joint state (PhysicsJointStateAPI) and drive target, so the robot starts and
        holds there when Play starts."""
        applied = self.set(pose, percent)
        for name in applied:
            drive = self.drives[name][0]
            prim, kind = drive.GetPrim(), self.kinds[name]
            schema = f"PhysicsJointStateAPI:{kind}"
            if schema not in prim.GetAppliedSchemas():
                prim.AddAppliedSchema(schema)
            value = drive.GetTargetPositionAttr().Get()
            for attr, v in (("position", value), ("velocity", 0.0)):
                a = prim.GetAttribute(f"state:{kind}:physics:{attr}")
                if not a:
                    a = prim.CreateAttribute(f"state:{kind}:physics:{attr}", Sdf.ValueTypeNames.Float)
                a.Set(float(v))
        return applied
