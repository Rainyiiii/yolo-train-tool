from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parent
PANEL_SCRIPT = ROOT / "train_panel.py"
PID_FILE = ROOT / ".train_panel.pid.json"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "panel.log"
PANEL_URL = "http://127.0.0.1:8989/"
PANEL_MARKER = "MyAutoTrain 团队训练平台"


def read_pid_record() -> dict[str, object] | None:
    try:
        value = json.loads(PID_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    return value


def remove_pid_file() -> None:
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


def managed_process(record: dict[str, object] | None = None) -> psutil.Process | None:
    record = record or read_pid_record()
    if not record:
        return None
    try:
        pid = int(record["pid"])
        expected_created = float(record["create_time"])
        process = psutil.Process(pid)
        if abs(process.create_time() - expected_created) > 1.0:
            return None
        command = " ".join(process.cmdline()).lower()
        if "train_panel.py" not in command:
            return None
        return process
    except (KeyError, TypeError, ValueError, psutil.Error):
        return None


def discover_panel_process() -> psutil.Process | None:
    try:
        connections = psutil.net_connections(kind="tcp")
    except psutil.Error:
        return None
    for connection in connections:
        if not connection.laddr or connection.status != psutil.CONN_LISTEN:
            continue
        if connection.laddr.port != 8989 or not connection.pid:
            continue
        try:
            process = psutil.Process(connection.pid)
            command = " ".join(process.cmdline()).lower()
            working_dir = Path(process.cwd()).resolve()
            if "train_panel.py" in command and working_dir == ROOT:
                return process
        except (OSError, psutil.Error):
            continue
    return None


def write_pid_record(process: psutil.Process) -> None:
    PID_FILE.write_text(
        json.dumps(
            {"pid": process.pid, "create_time": process.create_time()},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def panel_ready(timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(PANEL_URL, timeout=timeout) as response:
            content = response.read(200_000).decode("utf-8", errors="ignore")
        return response.status == 200 and PANEL_MARKER in content
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def log_tail(max_lines: int = 20) -> str:
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


def start_panel(no_browser: bool = False) -> int:
    process = managed_process()
    if process and panel_ready():
        print(f"MyAutoTrain is already running (PID {process.pid}).")
        print(PANEL_URL)
        if not no_browser:
            webbrowser.open(PANEL_URL)
        return 0

    remove_pid_file()
    if panel_ready():
        process = discover_panel_process()
        if process:
            write_pid_record(process)
            print(f"Attached to the running MyAutoTrain panel (PID {process.pid}).")
            print(PANEL_URL)
            if not no_browser:
                webbrowser.open(PANEL_URL)
            return 0
        print("Port 8989 already hosts MyAutoTrain, but its process could not be identified.")
        print("Close the old instance in Task Manager, then run the start script again.")
        return 2

    if not PANEL_SCRIPT.exists():
        print(f"Panel script not found: {PANEL_SCRIPT}", file=sys.stderr)
        return 1

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    creation_flags = 0
    if os.name == "nt":
        creation_flags = (
            subprocess.CREATE_NO_WINDOW
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
        )

    with LOG_FILE.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting MyAutoTrain\n")
        log.flush()
        process_handle = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(PANEL_SCRIPT),
                "--host",
                "127.0.0.1",
                "--port",
                "8989",
                "--no-browser",
            ],
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            creationflags=creation_flags,
        )

    process = psutil.Process(process_handle.pid)
    write_pid_record(process)

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process_handle.poll() is not None:
            remove_pid_file()
            print("MyAutoTrain exited during startup.", file=sys.stderr)
            tail = log_tail()
            if tail:
                print(tail, file=sys.stderr)
            return 1
        if panel_ready():
            print(f"MyAutoTrain started successfully (PID {process.pid}).")
            print(PANEL_URL)
            if not no_browser:
                webbrowser.open(PANEL_URL)
            return 0
        time.sleep(0.4)

    try:
        process.terminate()
        process.wait(timeout=5)
    except psutil.Error:
        pass
    remove_pid_file()
    print("MyAutoTrain startup timed out after 20 seconds.", file=sys.stderr)
    tail = log_tail()
    if tail:
        print(tail, file=sys.stderr)
    return 1


def stop_panel() -> int:
    record = read_pid_record()
    process = managed_process(record)
    if process is None:
        remove_pid_file()
        if panel_ready():
            process = discover_panel_process()
            if process:
                write_pid_record(process)
            else:
                print("MyAutoTrain is reachable, but its process could not be identified.", file=sys.stderr)
                return 2
        else:
            print("MyAutoTrain is not running.")
            return 0

    if process is None:
        return 0

    pid = process.pid
    try:
        process.terminate()
        process.wait(timeout=10)
    except psutil.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    except psutil.NoSuchProcess:
        pass
    finally:
        remove_pid_file()

    print(f"MyAutoTrain stopped (PID {pid}).")
    return 0


def show_status() -> int:
    process = managed_process()
    if process and panel_ready():
        print(f"MyAutoTrain is running (PID {process.pid}).")
        print(PANEL_URL)
        return 0
    if panel_ready():
        print("MyAutoTrain is running, but it is not managed by the launcher.")
        return 2
    print("MyAutoTrain is stopped.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage the local MyAutoTrain panel.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--no-browser", action="store_true")
    subparsers.add_parser("stop")
    subparsers.add_parser("status")
    args = parser.parse_args()

    if args.command == "start":
        return start_panel(no_browser=args.no_browser)
    if args.command == "stop":
        return stop_panel()
    return show_status()


if __name__ == "__main__":
    raise SystemExit(main())
