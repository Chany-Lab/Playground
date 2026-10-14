"""playground.notify with a fake notifier: no Isaac Sim, no network."""

import os
import time

import pytest

from playground import notify
from playground.notify import run_notifications

URL = "https://example.invalid/webhook"


class FakeNotifier:
    enabled = True

    def __init__(self, fail=False):
        self.fail = fail
        self.sent = []

    def send(self, title, text="", level="info", fields=None):
        self.sent.append((title, level, text, dict(fields or {})))
        if self.fail:
            raise RuntimeError("webhook down")
        return []


def run(notifier, body, config=None):
    with run_notifications(
        "Task-v0",
        "reinforcement_learning/train.py",
        ["--task", "Task-v0"],
        notifier=notifier,
        config=config or {notify.PROGRESS_KEY: "0"},
        hook_app_close=False,
    ):
        body()


def levels(notifier):
    return [(title.split(":")[0], level) for title, level, _, _ in notifier.sent]


def test_started_then_finished(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = FakeNotifier()
    run(fake, lambda: None)
    assert levels(fake) == [
        ("Training started", "info"),
        ("Training finished", "success"),
    ]
    fields = fake.sent[1][3]
    assert fields["task"] == "Task-v0" and "duration" in fields


def test_finished_reports_log_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = FakeNotifier()
    run(
        fake,
        lambda: (tmp_path / "logs" / "rsl_rl" / "exp" / "2026_run").mkdir(parents=True),
    )
    assert fake.sent[1][3]["log dir"] == os.path.join(
        "logs", "rsl_rl", "exp", "2026_run"
    )


def test_failed_reraises_with_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = FakeNotifier()

    def boom():
        raise ValueError("bad config")

    with pytest.raises(ValueError):
        run(fake, boom)
    assert levels(fake) == [("Training started", "info"), ("Training failed", "error")]
    assert fake.sent[1][2] == "ValueError: bad config"


def test_hydra_style_exit_reports_original_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = FakeNotifier()

    def hydra_exit():
        try:
            raise KeyError("missing")
        except KeyError:
            raise SystemExit(1)

    with pytest.raises(SystemExit):
        run(fake, hydra_exit)
    assert fake.sent[1][1] == "error"
    assert fake.sent[1][2].startswith("KeyError")


def test_keyboard_interrupt_is_stopped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = FakeNotifier()

    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(fake, interrupt)
    assert levels(fake) == [
        ("Training started", "info"),
        ("Training stopped", "warning"),
    ]


def test_end_is_sent_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = FakeNotifier()
    with run_notifications(
        "T",
        "rsl_rl/train.py",
        [],
        notifier=fake,
        config={notify.PROGRESS_KEY: "0"},
        hook_app_close=False,
    ) as handle:
        handle.finished()  # what the SimulationApp.close hook does
    assert [level for _, level, _, _ in fake.sent] == ["info", "success"]


def test_send_failures_never_raise(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = FakeNotifier(fail=True)
    run(fake, lambda: None)
    assert len(fake.sent) == 2

    def boom():
        raise ValueError("x")

    with pytest.raises(ValueError):  # the run's own error, not the webhook's
        run(fake, boom)


def test_nothing_without_webhooks(monkeypatch):
    monkeypatch.setattr(notify, "RunNotifications", None)  # would fail if used
    with run_notifications("T", "rsl_rl/train.py", [], config={}) as handle:
        assert handle is None
    assert notify.make_notifier({}) is None
    assert notify.make_notifier({"DISCORD_WEBHOOK_URL": ""}) is None


def test_make_notifier_with_webhook():
    notifier = notify.make_notifier({"SLACK_WEBHOOK_URL": URL})
    assert notifier is not None and notifier.enabled


def test_read_dotenv(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# webhooks\n"
        "\n"
        f"DISCORD_WEBHOOK_URL={URL}\n"
        f"export SLACK_WEBHOOK_URL='{URL}/slack'\n"
        'PLAYGROUND_NOTIFY_PROGRESS_MINUTES = "15"\n'
        "not a pair\n"
    )
    assert notify.read_dotenv(path) == {
        "DISCORD_WEBHOOK_URL": URL,
        "SLACK_WEBHOOK_URL": f"{URL}/slack",
        "PLAYGROUND_NOTIFY_PROGRESS_MINUTES": "15",
    }
    assert notify.read_dotenv(tmp_path / "missing") == {}


def test_environment_overrides_dotenv(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(f"DISCORD_WEBHOOK_URL={URL}/file\n")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", f"{URL}/env")
    assert notify.load_config(path)["DISCORD_WEBHOOK_URL"] == f"{URL}/env"
    monkeypatch.delenv("DISCORD_WEBHOOK_URL")
    assert notify.load_config(path)["DISCORD_WEBHOOK_URL"] == f"{URL}/file"


def test_library_from_argv():
    assert notify.library_from_argv(["--rl_library", "skrl", "--task", "A"]) == "skrl"
    assert notify.library_from_argv(["--rl_library=sb3"]) == "sb3"
    assert notify.library_from_argv(["--task", "A"]) is None


def test_log_dir_follows_rl_library(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = FakeNotifier()
    with run_notifications(
        "Task-v0",
        "reinforcement_learning/train.py",
        ["--rl_library", "skrl", "--task", "Task-v0"],
        notifier=fake,
        config={notify.PROGRESS_KEY: "0"},
        hook_app_close=False,
    ):
        (tmp_path / "logs" / "skrl" / "exp" / "run").mkdir(parents=True)
    assert fake.sent[1][3]["log dir"] == os.path.join("logs", "skrl", "exp", "run")


def test_task_from_argv():
    assert notify.task_from_argv(["--task", "A-v0", "--headless"]) == "A-v0"
    assert notify.task_from_argv(["--task=B-v0"]) == "B-v0"
    assert notify.task_from_argv(["--headless"]) == "unknown task"


def test_progress_from_tensorboard_events(tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    log_dir = tmp_path / "logs" / "rsl_rl" / "exp" / "run"
    writer = SummaryWriter(log_dir=str(log_dir))
    for it in range(3):
        writer.add_scalar(notify.REWARD_TAG, 1.5 * it, it)
    writer.flush()
    progress = notify.EventProgress()
    progress.poll(log_dir)
    assert progress.fields()["iteration"] == 2
    assert progress.fields()["mean reward"] == "3.000"
    writer.add_scalar(notify.REWARD_TAG, 9.0, 10)  # incremental read
    writer.close()
    progress.poll(log_dir)
    assert progress.fields() == {"iteration": 10, "mean reward": "9.000"}
    assert notify.find_log_dir("rsl_rl", time.time() - 60, cwd=tmp_path) == log_dir


@pytest.mark.parametrize(
    "exit_code, expected",
    [
        (0, ("Training finished", "success")),
        (130, ("Training stopped", "warning")),
        (1, ("Training failed", "error")),
    ],
)
def test_app_close_reports_launcher_exit_code(monkeypatch, exit_code, expected):
    import sys
    import types

    class FakeApp:
        def close(self, wait_for_replicator=True, skip_cleanup=False, exit_code=0):
            self.closed_with = exit_code

    monkeypatch.setitem(
        sys.modules, "isaacsim", types.SimpleNamespace(SimulationApp=FakeApp)
    )
    fake = FakeNotifier()
    run = notify.RunNotifications(fake, "Task-v0", "train.py", [], 0)
    notify._hook_simulation_app_close(run)
    app = FakeApp()
    app.close(exit_code=exit_code)
    assert app.closed_with == exit_code
    assert levels(fake) == [expected]
