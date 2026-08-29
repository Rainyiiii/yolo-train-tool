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

from platform_paths import LOG_DIR, PRODUCT_NAME, STATE_DIR, ensure_workspace
from platform_subprocess import hidden_creationflags


ROOT = Path(__file__).resolve().parent
ensure_workspace()
PANEL_SCRIPT = ROOT / "train_panel.py"
PID_FILE = STATE_DIR / "panel-service.json"
LOG_FILE = LOG_DIR / "panel.log"
DEFAULT_PANEL_PORT = 8989
PANEL_MARKER = PRODUCT_NAME


def panel_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/"


def record_port(record: dict[str, object] | None, default: int = DEFAULT_PANEL_PORT) -> int:
    try:
        port = int((record or {}).get("port", default))
    except (TypeError, ValueError):
        return default
    return port if 1 <= port <= 65535 else default


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


def discover_panel_process(port: int) -> psutil.Process | None:
    try:
        connections = psutil.net_connections(kind="tcp")
    except psutil.Error:
        return None
    for connection in connections:
        if not connection.laddr or connection.status != psutil.CONN_LISTEN:
            continue
        if connection.laddr.port != port or not connection.pid:
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


def write_pid_record(process: psutil.Process, port: int) -> None:
    PID_FILE.write_text(
        json.dumps(
            {"pid": process.pid, "create_time": process.create_time(), "port": port},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def panel_ready(port: int, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(panel_url(port), timeout=timeout) as response:
            content = response.read(200_000).decode("utf-8", errors="ignore")
        return response.status == 200 and PANEL_MARKER in content
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def discover_running_panels() -> list[tuple[psutil.Process, int, str]]:
    """Find verified YOLO panel processes started from any installation root."""
    panels: list[tuple[psutil.Process, int, str]] = []
    seen_pids: set[int] = set()
    try:
        connections = psutil.net_connections(kind="tcp")
    except psutil.Error:
        return panels
    for connection in connections:
        if (
            not connection.laddr
            or connection.status != psutil.CONN_LISTEN
            or not connection.pid
            or connection.pid in seen_pids
        ):
            continue
        seen_pids.add(connection.pid)
        try:
            process = psutil.Process(connection.pid)
            arguments = [str(value) for value in process.cmdline()]
            script = next(
                (value for value in arguments if Path(value).name.lower() == "train_panel.py"),
                "",
            )
            if not script:
                continue
            port = int(connection.laddr.port)
            if not 1 <= port <= 65535 or not panel_ready(port, timeout=0.4):
                continue
            panels.append((process, port, str(Path(script).resolve())))
        except (OSError, psutil.Error):
            continue
    return panels


def list_running_panels() -> int:
    records: list[dict[str, object]] = []
    for process, port, script in discover_running_panels():
        try:
            records.append(
                {
                    "pid": process.pid,
                    "create_time": process.create_time(),
                    "port": port,
                    "script": script,
                }
            )
        except psutil.Error:
            continue
    print(json.dumps(records, ensure_ascii=False))
    return 0


def stop_running_panels() -> int:
    panels = discover_running_panels()
    if not panels:
        print(f"没有发现正在运行的 {PRODUCT_NAME} 服务。")
        return 0

    processes = [process for process, _, _ in panels]
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=10)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    if alive:
        psutil.wait_procs(alive, timeout=5)
    remove_pid_file()
    ports = ", ".join(str(port) for _, port, _ in panels)
    print(f"已关闭 {len(panels)} 个 {PRODUCT_NAME} 服务（端口：{ports}）。")
    return 0


def log_tail(max_lines: int = 20) -> str:
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


def start_panel(no_browser: bool = False, port: int = DEFAULT_PANEL_PORT) -> int:
    record = read_pid_record()
    process = managed_process(record)
    active_port = record_port(record, port)
    if process and panel_ready(active_port):
        print(f"{PRODUCT_NAME} 已在运行（PID {process.pid}）。")
        print(panel_url(active_port))
        if not no_browser:
            webbrowser.open(panel_url(active_port))
        return 0

    remove_pid_file()
    if panel_ready(port):
        process = discover_panel_process(port)
        if process:
            write_pid_record(process, port)
            print(f"已接管正在运行的 {PRODUCT_NAME}（PID {process.pid}）。")
            print(panel_url(port))
            if not no_browser:
                webbrowser.open(panel_url(port))
            return 0
        print(f"{port} 端口已有 {PRODUCT_NAME} 页面，但无法识别对应进程。")
        print("请在任务管理器中关闭旧实例后重试。")
        return 2

    if not PANEL_SCRIPT.exists():
        print(f"Panel script not found: {PANEL_SCRIPT}", file=sys.stderr)
        return 1

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    creation_flags = hidden_creationflags(new_process_group=True, detached=True)

    with LOG_FILE.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting {PRODUCT_NAME}\n")
        log.flush()
        process_handle = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(PANEL_SCRIPT),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
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
    write_pid_record(process, port)

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process_handle.poll() is not None:
            remove_pid_file()
            print(f"{PRODUCT_NAME} 在启动过程中退出。", file=sys.stderr)
            tail = log_tail()
            if tail:
                print(tail, file=sys.stderr)
            return 1
        if panel_ready(port):
            listening_process = discover_panel_process(port) or process
            write_pid_record(listening_process, port)
            print(f"{PRODUCT_NAME} 启动成功（PID {listening_process.pid}）。")
            print(panel_url(port))
            if not no_browser:
                webbrowser.open(panel_url(port))
            return 0
        time.sleep(0.4)

    try:
        process.terminate()
        process.wait(timeout=5)
    except psutil.Error:
        pass
    remove_pid_file()
    print(f"{PRODUCT_NAME} 启动等待超过 20 秒。", file=sys.stderr)
    tail = log_tail()
    if tail:
        print(tail, file=sys.stderr)
    return 1


def stop_panel() -> int:
    record = read_pid_record()
    port = record_port(record)
    process = managed_process(record)
    if process is None:
        remove_pid_file()
        if panel_ready(port):
            process = discover_panel_process(port)
            if process:
                write_pid_record(process, port)
            else:
                print(f"{PRODUCT_NAME} 可以访问，但无法识别对应进程。", file=sys.stderr)
                return 2
        else:
            print(f"{PRODUCT_NAME} 没有运行。")
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

    print(f"{PRODUCT_NAME} 已停止（PID {pid}）。")
    return 0


def show_status(port: int = DEFAULT_PANEL_PORT) -> int:
    record = read_pid_record()
    active_port = record_port(record, port)
    process = managed_process(record)
    if process and panel_ready(active_port):
        print(f"{PRODUCT_NAME} 正在运行（PID {process.pid}）。")
        print(panel_url(active_port))
        return 0
    if panel_ready(port):
        print(f"{PRODUCT_NAME} 正在运行，但不是由当前启动器管理。")
        return 2
    print(f"{PRODUCT_NAME} 已停止。")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Manage the local {PRODUCT_NAME} panel.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--no-browser", action="store_true")
    start_parser.add_argument("--port", type=int, default=DEFAULT_PANEL_PORT)
    subparsers.add_parser("stop")
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--port", type=int, default=DEFAULT_PANEL_PORT)
    subparsers.add_parser("list")
    subparsers.add_parser("stop-all")
    args = parser.parse_args()

    if getattr(args, "port", DEFAULT_PANEL_PORT) not in range(1, 65536):
        parser.error("--port 必须在 1 到 65535 之间")

    if args.command == "start":
        return start_panel(no_browser=args.no_browser, port=args.port)
    if args.command == "stop":
        return stop_panel()
    if args.command == "list":
        return list_running_panels()
    if args.command == "stop-all":
        return stop_running_panels()
    return show_status(port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
