# sim_server

Isaac Sim server for [Interface](https://github.com/Chany-Lab/Interface): opens its environment scripts, with a window or headless, and takes teleoperation

<br>

### Files

| File | Role |
| :--- | :--- |
| `server.py` | HTTP front (stdlib only): starts `app.py`, unpacks scenes from Interface. API and `VERSION` at the top |
| `app.py` | Isaac Sim app: builds the stage, drives the joints |
| `scene.py` | Runs an environment's `build(scene)` (pxr only) |
| `tcp.py` | Keyboard TCP jog (damped least squares on the PhysX Jacobian) |

A scene arrives as tar.gz: `envs/…` and the robot in Isaac-Envs' layout (`sims/robots/<type>/`, `assets/…`).
`scene.add(<name>)` places `assets/props/<name>/` from this checkout's `external/isaac-envs`.

## Setup Guide

Interface starts the server itself in local mode. On a sim server:

```bash
uv sync
.venv/bin/python playground/sim_server/server.py --host 0.0.0.0 --port 8211
```

Bump `VERSION` with Interface's `JOINTS_VERSION` when the protocol changes.
