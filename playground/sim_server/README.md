# sim_server

The Isaac Sim server [Interface](https://github.com/Chany-Lab/Interface) talks to: it opens
Interface's environment scripts in Isaac Sim, with a window or headless, and takes keyboard and
leader-arm teleoperation. It moved here from Interface's `sim/runner/`.

| File | Role |
| :--- | :--- |
| `server.py` | HTTP front, stdlib only. Starts and stops `app.py`, unpacks the scenes Interface sends |
| `app.py` | Isaac Sim app (`SimulationApp`): builds the stage, drives the joints |
| `scene.py` | Runs an environment's `build(scene)`; pxr only |
| `tcp.py` | Keyboard TCP jog: damped least squares on the PhysX Jacobian |

A scene arrives as tar.gz: the environment under `envs/` and the robot as laid out in Isaac-Envs
(`sims/robots/<type>/`, plus the `assets/` folders it references). `scene.add(<name>)` places
`assets/props/<name>/` from this checkout's `external/isaac-envs`.

Interface starts the server itself in local mode. On a sim server:

```bash
uv sync
.venv/bin/python playground/sim_server/server.py --host 0.0.0.0 --port 8211
```

The API is described at the top of `server.py`; bump its `VERSION` when the protocol changes,
together with `JOINTS_VERSION` in Interface's backend.
