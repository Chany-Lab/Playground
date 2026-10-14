"""Isaac Sim app started by server.py: builds environments on request (evaluation comes later).

Runs with the Python that has isaacsim installed (Playground's .venv). Commands arrive on a small
HTTP port that only server.py talks to; USD work runs on the main loop between app updates.

  GET  /state                 {"scene": path | null, "error": str | null}
  POST /open {"path", "root", "robot"}  build a new stage from the environment script at path;
                              root is the folder the scene arrived in (sims/robots/ is looked up there),
                              robot the one scene.robot() places (scene.py)
  GET  /state also lists "joints": {joint: drive target (degrees)} of the open robot
  POST /joints {"targets": {joint: value}, "percent": [joint, …], "play": bool}
         or    {"jog": {joint: degrees per second}, "play": bool}  keyboard teleoperation: the
                              targets move at those speeds (stopping at the limits) until JOG_HOLD_S
                              passes without a new command
         or    {"twist": [vx, vy, vz (m/s), wx, wy, wz (deg/s)], "play": bool}  the same for the TCP,
                              in its own frame (robot.yaml tcp, tcp.py); /state "tcp" is its pose
                              teleoperation: leader readings by joint name, shifted by the
                              robot's leader offsets (robot.yaml leader.rest → initial_pose), then
                              set as drive targets (degrees; percent joints 0–100 over their limits);
                              play starts the timeline. Only the newest command is applied

--device gpu (default) simulates on GPU 0: PhysX GPU dynamics and broadphase are turned on in
every PhysicsScene of a built stage, in the session layer; --device cpu keeps PhysX on the CPU.
Rendering is always on the NVIDIA GPU. A window app also enables the physics UI extensions
(Physics Inspector for moving joints, physics menus and properties).
"""

import argparse
import json
import queue
import time
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, required=True)
p.add_argument("--headless", action="store_true")
p.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
args = p.parse_args()
GPU = args.device == "gpu"
UI_EXTENSIONS = ("omni.physx.ui", "omni.physx.supportui", "isaacsim.gui.property")

# SimulationApp must start before any omni import
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless, "active_gpu": 0, "physics_gpu": 0})

# A window gets the physics authoring UI the lean default experience leaves out: Physics
# Inspector (joint sliders), physics menus and property widgets. The full experience kit
# crashes when started through SimulationApp, so only these extensions are added.
if not args.headless:
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    for ext in UI_EXTENSIONS:
        manager.set_extension_enabled_immediate(ext, True)

import carb  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene as env_scene  # noqa: E402
from tcp import TcpJog  # noqa: E402

carb.settings.get_settings().set_int("/physics/cudaDevice", 0 if GPU else -1)

state = {"scene": None, "error": None}
commands: queue.Queue = queue.Queue()
# Newest teleoperation command; older ones are dropped (the main loop applies one per update)
joints: dict = {"latest": None}
joints_lock = threading.Lock()
JOG_HOLD_S = 1.0  # a jog stops when no command came for this long (the backend sends a stop on release)
JOG_MAX_DT_S = 0.5  # longest frame a jog step integrates (a window app can render slowly)
STATE_EVERY_S = 0.2  # how often /state's joint targets are refreshed
jog = {"velocities": {}, "twist": None, "until": 0.0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200 if self.path == "/state" else 404, dict(state))

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/joints":
            with joints_lock:
                joints["latest"] = body
            return self._send(202, dict(state))
        if self.path != "/open":
            return self._send(404, {})
        commands.put(("open", body.get("path"), body.get("root"), body.get("robot") or None))
        self._send(202, dict(state))


def use_physics_device(stage) -> None:
    """GPU (or CPU) PhysX on every PhysicsScene of the stage, written to the session layer."""
    prev = stage.GetEditTarget()
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    try:
        for prim in [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]:
            api = PhysxSchema.PhysxSceneAPI.Apply(prim)
            api.CreateEnableGPUDynamicsAttr().Set(GPU)
            api.CreateBroadphaseTypeAttr().Set("GPU" if GPU else "MBP")
    finally:
        stage.SetEditTarget(prev)


server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()

def apply_joints(drives) -> None:
    with joints_lock:
        cmd, joints["latest"] = joints["latest"], None
    if not cmd or drives is None:
        return
    timeline = omni.timeline.get_timeline_interface()
    if cmd.get("play") and not timeline.is_playing():
        timeline.play()
    if "jog" in cmd or "twist" in cmd:
        twist = cmd.get("twist")
        jog.update(
            velocities=dict(cmd.get("jog") or {}),
            twist=[float(v) for v in twist] if twist and any(twist) else None,
            until=time.monotonic() + JOG_HOLD_S,
        )
        return
    targets = {j: v + drives.offsets.get(j, 0.0) for j, v in (cmd.get("targets") or {}).items()}
    drives.set(targets, tuple(cmd.get("percent") or ()))


def step_jog(drives, tcp, dt: float) -> None:
    if drives is None or time.monotonic() >= jog["until"]:
        return
    if jog["twist"] and tcp is not None:
        drives.jog(tcp.joint_speeds(jog["twist"]), dt)
    elif jog["velocities"]:
        drives.jog(jog["velocities"], dt)


drives = None  # scene.Drives of the open stage
tcp = None  # TcpJog of the open robot (robot.yaml tcp)
last, last_state = time.monotonic(), 0.0

# Runs until the window is closed or server.py terminates the process
while app.is_running():
    app.update()
    apply_joints(drives)
    now = time.monotonic()
    step_jog(drives, tcp, min(now - last, JOG_MAX_DT_S))
    last = now
    if now - last_state > STATE_EVERY_S:
        state["joints"] = {j: round(v, 2) for j, v in drives.targets().items()} if drives else None
        playing = omni.timeline.get_timeline_interface().is_playing()
        pose = tcp.pose() if tcp is not None and playing else None
        state["tcp"] = [round(v, 4) for v in pose] if pose else None
        last_state = now
    while not commands.empty():
        kind, path, root, robot = commands.get()
        if kind == "open":
            context = omni.usd.get_context()
            try:
                drives, tcp = None, None
                jog.update(velocities={}, twist=None, until=0.0)
                context.new_stage()
                built = env_scene.build(context.get_stage(), Path(path), Path(root), robot)
                use_physics_device(context.get_stage())
                drives = env_scene.Drives(context.get_stage(), built.teleop_path or "/World/Robot")
                drives.offsets = built.leader_offsets
                tcp = TcpJog(**built.tcp) if built.tcp else None
                state.update(scene=path, error=None)
            except Exception as e:  # the user's script: report it, keep the app up
                traceback.print_exc()
                state.update(scene=None, error=f"Could not build {Path(path).name}: {e}")

server.shutdown()
app.close()
