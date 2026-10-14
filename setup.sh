#!/usr/bin/env bash
# Sets up Playground: checks the tools, fetches the submodules and their LFS assets, syncs the venv.
# Safe to run again. --check only reports what is missing and changes nothing.
#
#   ./setup.sh           set up (or update) this checkout
#   ./setup.sh --check   check only
set -euo pipefail

MIN_DRIVER=580
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK=0

usage() { sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; }

for arg in "$@"; do
    case "$arg" in
        --check) CHECK=1 ;;
        -h | --help) usage; exit 0 ;;
        *) echo "unknown option: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

cd "$ROOT"
# ROS and other setups export a PYTHONPATH that breaks Isaac Sim's imports.
unset PYTHONPATH

ok() { printf '  \033[32mok\033[0m    %s\n' "$*"; }
bad() { printf '  \033[31mmissing\033[0m %s\n' "$*"; FAILED=1; }
FAILED=0

echo "Tools"
command -v git > /dev/null && ok "git $(git --version | awk '{print $3}')" || bad "git"
if git lfs version > /dev/null 2>&1; then
    ok "git-lfs $(git lfs version | awk '{print $1}' | cut -d/ -f2)"
else
    bad "git-lfs (https://git-lfs.com)"
fi
if command -v uv > /dev/null; then
    ok "uv $(uv --version | awk '{print $2}')"
else
    bad "uv (curl -LsSf https://astral.sh/uv/install.sh | sh)"
fi
if command -v nvidia-smi > /dev/null; then
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
    if [ "${driver%%.*}" -ge "$MIN_DRIVER" ] 2> /dev/null; then
        ok "NVIDIA driver $driver"
    else
        bad "NVIDIA driver >= $MIN_DRIVER (found ${driver:-none})"
    fi
else
    bad "NVIDIA driver >= $MIN_DRIVER (no nvidia-smi)"
fi

echo "Checkout"
submodules_ready=1
for sub in external/IsaacLab external/isaac-envs; do
    state="$(git submodule status -- "$sub" 2> /dev/null | cut -c1)"
    if [ ! -e "$sub/.git" ] || [ "$state" = "-" ]; then
        submodules_ready=0
        bad "$sub (git submodule update --init)"
    elif [ "$state" = "+" ]; then
        ok "$sub (checked out at another commit than recorded)"
    else
        ok "$sub"
    fi
done
if [ "$submodules_ready" = 1 ]; then
    # LFS pointers left in place are ~130-byte text files starting with "version https://git-lfs".
    if git -C external/isaac-envs lfs ls-files 2> /dev/null | grep -q ' - '; then
        bad "external/isaac-envs LFS assets (git -C external/isaac-envs lfs pull)"
    else
        ok "external/isaac-envs LFS assets"
    fi
fi
if [ -x .venv/bin/python ] && uv sync --check --frozen > /dev/null 2>&1; then
    ok ".venv in sync with uv.lock"
else
    bad ".venv (uv sync)"
fi

if [ "$CHECK" = 1 ]; then
    [ "$FAILED" = 0 ] && echo "All set." || echo "Run ./setup.sh to fix what is missing."
    exit "$FAILED"
fi

# Tools cannot be installed from here; everything below can.
for tool in git uv; do
    command -v "$tool" > /dev/null || { echo "Install $tool first." >&2; exit 1; }
done
git lfs version > /dev/null 2>&1 || { echo "Install git-lfs first." >&2; exit 1; }

echo
echo "Submodules"
git submodule update --init
git -C external/isaac-envs lfs install --local > /dev/null
git -C external/isaac-envs lfs pull

echo
echo "Python environment (Python 3.12, Isaac Sim 6.2, Isaac Lab 3.0; the first sync downloads ~20 GB)"
uv sync

cat << 'EOF'

Done.

Isaac Sim needs its EULA accepted (https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html):

    export OMNI_KIT_ACCEPT_EULA=YES

Then, with PYTHONPATH unset (env -u PYTHONPATH) if ROS is sourced:

    uv run pytest -q tests                                       # tests
    uv run isaaclab train --rl_library rsl_rl --task <task id>   # training (headless; --viz kit for a window)
    .venv/bin/python playground/sim_server/server.py --port 8211 # Interface's sim server
EOF
