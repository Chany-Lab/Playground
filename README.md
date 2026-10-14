# Playground

Isaac Lab training, teleoperation and the Isaac Sim server on the [Isaac-Envs](https://github.com/Chany-Lab/Isaac-Envs) assets

<br>

### Modules

| Path | Content |
| :--- | :--- |
| `playground/teleop/` | SO-101 leader arm as an Isaac Lab device (Feetech bus, calibration) |
| `playground/sim_server/` | Isaac Sim server for [Interface](https://github.com/Chany-Lab/Interface) ([README](playground/sim_server/README.md)) |
| `playground/notify.py` | Discord / Slack notifications around a training run ([Notifier](https://github.com/Chany-Lab/Notifier)) |
| `external/IsaacLab` | Isaac Lab v2.3.2, installed from source |
| `external/isaac-envs` | Robots, props and scenes |

### Notifications

| Variable | Value |
| :--- | :--- |
| `DISCORD_WEBHOOK_URL`, `SLACK_WEBHOOK_URL` | Webhooks, in the environment or a git-ignored `.env` |
| `PLAYGROUND_NOTIFY_PROGRESS_MINUTES` | Progress interval for rsl_rl runs (default 30, `0` = off) |

A run sends started, finished / failed / stopped, and progress (iteration, mean reward).

## Setup Guide

Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.2, torch 2.7.0 (cu128), managed with uv.

```bash
git submodule update --init
git lfs install && git -C external/isaac-envs lfs pull
uv sync
export OMNI_KIT_ACCEPT_EULA=YES

uv run python scripts/teleop/calibrate_leader.py   # leader arm calibration
uv run pytest
```

ROS on the same machine sets `PYTHONPATH`; run with `env -u PYTHONPATH` if imports break.
