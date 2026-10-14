# Playground

Isaac Lab training, teleoperation and the Isaac Sim server on the [Isaac-Envs](https://github.com/Chany-Lab/Isaac-Envs) assets

<br>

### Modules

| Path | Content |
| :--- | :--- |
| `playground/teleop/` | SO-101 leader arm as an Isaac Lab device (Feetech bus, calibration) |
| `playground/sim_server/` | Isaac Sim server for [Interface](https://github.com/Chany-Lab/Interface) ([README](playground/sim_server/README.md)) |
| `playground/notify.py` | Discord / Slack notifications around a training run ([Notifier](https://github.com/Chany-Lab/Notifier)) |
| `external/IsaacLab` | Isaac Lab 3.0 (release/3.0.0), installed from source |
| `external/isaac-envs` | Robots, props and scenes |

### Notifications

| Variable | Value |
| :--- | :--- |
| `DISCORD_WEBHOOK_URL`, `SLACK_WEBHOOK_URL` | Webhooks, in the environment or a git-ignored `.env` |
| `PLAYGROUND_NOTIFY_PROGRESS_MINUTES` | Progress interval for rsl_rl runs (default 30, `0` = off) |

A run sends started, finished / failed / stopped, and progress (iteration, mean reward).

## Setup Guide

Python 3.12, Isaac Sim 6.2.0, Isaac Lab 3.0, torch 2.11.0 (cu130), managed with uv. Needs git-lfs and an NVIDIA driver 580 or newer.

```bash
./setup.sh            # submodules, LFS assets, uv sync (./setup.sh --check only checks)
export OMNI_KIT_ACCEPT_EULA=YES

uv run pytest -q tests
uv run python scripts/teleop/calibrate_leader.py   # leader arm calibration
```

Training uses Isaac Lab 3.0's unified entrypoint, headless unless `--viz kit`; logs go to `logs/<rl_library>/`:

```bash
uv run isaaclab train --rl_library rsl_rl --task <task> --num_envs 64
```

ROS on the same machine sets `PYTHONPATH`; run with `env -u PYTHONPATH` if imports break.
