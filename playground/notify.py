"""Discord/Slack notifications around an Isaac Lab training run.

Webhooks come from DISCORD_WEBHOOK_URL / SLACK_WEBHOOK_URL, in the environment
or in a git-ignored `.env` at the repo root (the environment wins). Without
either, nothing is sent. PLAYGROUND_NOTIFY_PROGRESS_MINUTES (default 30, 0 =
off) sets how often progress is read from the run's tensorboard events.

Nothing here raises because of a notification: `Notifier.send` never raises,
and everything else that only serves notifications is wrapped.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEBHOOK_KEYS = ("DISCORD_WEBHOOK_URL", "SLACK_WEBHOOK_URL")
PROGRESS_KEY = "PLAYGROUND_NOTIFY_PROGRESS_MINUTES"
DEFAULT_PROGRESS_MINUTES = 30.0
DEFAULT_LIBRARY = "rsl_rl"
# rsl_rl's tensorboard tags; other frameworks just report the iteration.
REWARD_TAG = "Train/mean_reward"
LENGTH_TAG = "Train/mean_episode_length"


def read_dotenv(path: Path) -> dict[str, str]:
    """KEY=VALUE lines; blank lines, comments and `export ` prefixes allowed."""
    values: dict[str, str] = {}
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_config(dotenv: Path | None = REPO_ROOT / ".env") -> dict[str, str]:
    """`.env` values overridden by the environment."""
    config = read_dotenv(dotenv) if dotenv is not None else {}
    config.update({k: v for k, v in os.environ.items() if v})
    return config


def make_notifier(config: Mapping[str, str], source: str = "Playground"):
    """A `notifier.Notifier`, or None when no webhook is configured."""
    if not any(config.get(key) for key in WEBHOOK_KEYS):
        return None
    try:
        from notifier import Notifier

        notifier = Notifier(
            discord_url=config.get("DISCORD_WEBHOOK_URL") or None,
            slack_url=config.get("SLACK_WEBHOOK_URL") or None,
            source=source,
        )
    except Exception:
        return None
    return notifier if notifier.enabled else None


def task_from_argv(argv: list[str]) -> str:
    """The value of `--task` (or `--task=`), else "unknown task"."""
    for i, arg in enumerate(argv):
        if arg == "--task" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--task="):
            return arg.split("=", 1)[1]
    return "unknown task"


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def library_from_argv(argv: list[str]) -> str | None:
    """The value of `--rl_library` (or `--rl_library=`), else None."""
    for i, arg in enumerate(argv):
        if arg == "--rl_library" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--rl_library="):
            return arg.split("=", 1)[1]
    return None


def find_log_dir(library: str, since: float, cwd: Path | None = None) -> Path | None:
    """Newest run directory created by this run, e.g. logs/rsl_rl/<exp>/<time>.

    Isaac Lab's training entrypoint logs to logs/<library>/<experiment>/<run>
    relative to the working directory, <library> being `--rl_library`.
    """
    try:
        root = Path(cwd or Path.cwd()) / "logs" / library
        runs = [
            p for p in root.glob("*/*") if p.is_dir() and p.stat().st_mtime >= since - 2
        ]
        return max(runs, key=lambda p: p.stat().st_mtime) if runs else None
    except OSError:
        return None


def _display(path: Path) -> str:
    """Path relative to the working directory, so home paths are not sent."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


class EventProgress:
    """Latest iteration and rewards from a run's tensorboard event files.

    Reads incrementally, so each poll only parses what was appended. The
    writer flushes every couple of minutes, so values lag a little.
    """

    def __init__(self) -> None:
        self._loaders: dict[Path, object] = {}
        self.step: int | None = None
        self.values: dict[str, float] = {}

    def poll(self, log_dir: Path) -> None:
        from tensorboard.backend.event_processing.event_file_loader import (
            EventFileLoader,
        )

        for path in sorted(log_dir.glob("events.out.tfevents.*")):
            loader = self._loaders.setdefault(path, EventFileLoader(str(path)))
            for event in loader.Load():
                for value in event.summary.value:
                    number = _scalar(value)
                    if number is None:
                        continue
                    if self.step is None or event.step >= self.step:
                        self.step = event.step
                    self.values[value.tag] = number

    def fields(self) -> dict[str, object]:
        fields: dict[str, object] = {}
        if self.step is not None:
            fields["iteration"] = self.step
        for label, tag in (
            ("mean reward", REWARD_TAG),
            ("mean episode length", LENGTH_TAG),
        ):
            if tag in self.values:
                fields[label] = f"{self.values[tag]:.3f}"
        return fields


def _scalar(value) -> float | None:
    kind = value.WhichOneof("value")
    if kind == "simple_value":
        return float(value.simple_value)
    if kind == "tensor" and value.tensor.float_val:
        return float(value.tensor.float_val[0])
    return None


class RunNotifications:
    """The run's notifications; `finish` and friends send at most once."""

    def __init__(
        self,
        notifier,
        task: str,
        script: str,
        argv: list[str],
        progress_minutes: float,
        library: str = DEFAULT_LIBRARY,
    ):
        self.notifier = notifier
        self.task = task
        self.script = script
        self.library = library
        self.argv = argv
        self.progress_minutes = progress_minutes
        self.start = time.time()
        self.progress = EventProgress()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._sent_end = False
        self._thread: threading.Thread | None = None

    def _base_fields(self) -> dict[str, object]:
        return {"task": self.task, "script": self.script}

    def _send(
        self, title: str, level: str, fields: dict[str, object], text: str = ""
    ) -> None:
        try:
            self.notifier.send(title, text, level=level, fields=fields)
        except Exception:
            pass

    def _end_fields(self) -> dict[str, object]:
        fields = self._base_fields()
        fields["duration"] = format_duration(time.time() - self.start)
        try:
            log_dir = find_log_dir(self.library, self.start)
            if log_dir is not None:
                fields["log dir"] = _display(log_dir)
                self.progress.poll(log_dir)
        except Exception:
            pass
        fields.update(self.progress.fields())
        return fields

    def started(self) -> None:
        fields = self._base_fields()
        if self.argv:
            fields["args"] = " ".join(self.argv)[:900]
        self._send(f"Training started: {self.task}", "info", fields)
        if self.progress_minutes > 0:
            self._thread = threading.Thread(
                target=self._progress_loop, name="notify-progress", daemon=True
            )
            self._thread.start()

    def _end(self, title: str, level: str, text: str = "") -> None:
        with self._lock:
            if self._sent_end:
                return
            self._sent_end = True
        self._done.set()
        self._send(title, level, self._end_fields(), text)

    def finished(self) -> None:
        self._end(f"Training finished: {self.task}", "success")

    def failed(self, exc: BaseException) -> None:
        self._end(f"Training failed: {self.task}", "error", _describe(exc))

    def stopped(self) -> None:
        self._end(f"Training stopped: {self.task}", "warning", "KeyboardInterrupt")

    def _progress_loop(self) -> None:
        interval = self.progress_minutes * 60
        last_step = None
        while not self._done.wait(interval):
            try:
                log_dir = find_log_dir(self.library, self.start)
                if log_dir is None:
                    continue
                self.progress.poll(log_dir)
                if self.progress.step is None or self.progress.step == last_step:
                    continue
                last_step = self.progress.step
                fields = self._base_fields()
                fields["elapsed"] = format_duration(time.time() - self.start)
                fields.update(self.progress.fields())
                if not self._done.is_set():
                    self._send(f"Training progress: {self.task}", "info", fields)
            except Exception:
                pass


def _describe(exc: BaseException) -> str:
    # Hydra reports the real error and then calls sys.exit(1) inside its
    # except block, so the original exception is the SystemExit's context.
    if isinstance(exc, SystemExit) and exc.__context__ is not None:
        exc = exc.__context__
    message = f"{type(exc).__name__}: {exc}"
    return message[:1500]


def _hook_simulation_app_close(run: RunNotifications) -> None:
    """Send the end notification when the Kit app closes.

    Isaac Lab's launcher closes the app in a `finally` with the run's exit
    code (`SimulationApp.close(exit_code=...)`: 0, 130 for Ctrl+C, else an
    error), and Kit may end the process inside close(), before control
    returns here.
    """
    # The same import isaaclab.app does first thing; if it fails here (e.g. the
    # EULA prompt is declined, which raises SystemExit), leave it to the script.
    try:
        from isaacsim import SimulationApp
    except (Exception, SystemExit):
        return
    original = SimulationApp.close

    def close(self, *args, **kwargs):
        code = kwargs.get("exit_code", args[2] if len(args) > 2 else 0)
        if code in (None, 0):
            run.finished()
        elif code == 130:
            run.stopped()
        else:
            # close() runs in the launcher's `finally`, with the error in flight
            run.failed(sys.exception() or SystemExit(code))
        return original(self, *args, **kwargs)

    SimulationApp.close = close


@contextmanager
def run_notifications(
    task: str,
    script: str,
    argv: list[str],
    *,
    notifier=None,
    config: Mapping[str, str] | None = None,
    hook_app_close: bool = True,
    library: str | None = None,
) -> Iterator[RunNotifications | None]:
    """Notify started, then finished / failed / stopped, around a training run.

    `library` names the run's logs/<library> folder; by default `--rl_library`
    from argv, else rsl_rl.

    Yields None and sends nothing when no webhook is configured. Exceptions
    from the run are re-raised unchanged.
    """
    run = None
    try:
        config = load_config() if config is None else config
        notifier = notifier if notifier is not None else make_notifier(config)
        if notifier is not None:
            try:
                minutes = float(config.get(PROGRESS_KEY, DEFAULT_PROGRESS_MINUTES))
            except ValueError:
                minutes = DEFAULT_PROGRESS_MINUTES
            library = library or library_from_argv(argv) or DEFAULT_LIBRARY
            run = RunNotifications(notifier, task, script, argv, minutes, library)
            run.started()
            if hook_app_close:
                _hook_simulation_app_close(run)
    except Exception:
        run = None

    if run is None:
        yield None
        return
    try:
        yield run
    except KeyboardInterrupt:
        run.stopped()
        raise
    except SystemExit as exc:
        if exc.code in (None, 0):
            run.finished()
        else:
            run.failed(exc)
        raise
    except BaseException as exc:
        run.failed(exc)
        raise
    else:
        run.finished()
