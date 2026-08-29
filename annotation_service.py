# -*- coding: utf-8 -*-
"""Background service manager for the YOLO Team annotation server."""

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

from annotation_server import DEFAULT_WORKSPACE, discover_lan_addresses
from platform_paths import LOG_DIR, PRODUCT_NAME, STATE_DIR, ensure_workspace
from platform_subprocess import hidden_creationflags


ROOT = Path(__file__).resolve().parent
ensure_workspace()
SERVER_SCRIPT = ROOT / "annotation_server.py"
PID_FILE = STATE_DIR / "annotation-service.json"
LOG_FILE = LOG_DIR / "annotation-server.log"
PORT = 9000
LOCAL_URL = f"http://127.0.0.1:{PORT}/"


def read_record() -> dict:
    try:
        value = json.loads(PID_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def remove_record() -> None:
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


def managed_process(record: dict | None = None) -> psutil.Process | None:
    record = record or read_record()
    try:
        process = psutil.Process(int(record["pid"]))
        if abs(process.create_time() - float(record["create_time"])) > 1:
            return None
        if "annotation_server.py" not in " ".join(process.cmdline()).casefold():
            return None
        return process
    except (KeyError, TypeError, ValueError, psutil.Error):
        return None


def discover_process() -> psutil.Process | None:
    try:
        connections = psutil.net_connections(kind="tcp")
    except psutil.Error:
        return None
    for connection in connections:
        if not connection.laddr or connection.status != psutil.CONN_LISTEN or connection.laddr.port != PORT or not connection.pid:
            continue
        try:
            process = psutil.Process(connection.pid)
            if "annotation_server.py" in " ".join(process.cmdline()).casefold():
                return process
        except psutil.Error:
            continue
    return None


def health() -> bool:
    try:
        with urllib.request.urlopen(f"{LOCAL_URL}api/health", timeout=1.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return response.status == 200 and payload.get("ok") is True
    except (OSError, ValueError, TimeoutError, urllib.error.URLError):
        return False


def write_record(process: psutil.Process, shared: bool, workspace: Path) -> None:
    PID_FILE.write_text(json.dumps({
        "pid": process.pid,
        "create_time": process.create_time(),
        "shared": shared,
        "workspace": str(workspace),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stop() -> int:
    process = managed_process() or discover_process()
    if process is None:
        remove_record()
        print("协作标注服务没有运行。")
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
    remove_record()
    print(f"协作标注服务已停止（PID {pid}）。")
    return 0


def start(shared: bool, workspace: Path, no_browser: bool) -> int:
    record = read_record()
    process = managed_process(record)
    if process and health():
        current_shared = bool(record.get("shared"))
        current_workspace = Path(str(record.get("workspace") or DEFAULT_WORKSPACE)).resolve()
        if current_shared == shared and current_workspace == workspace:
            print(f"协作标注服务已经运行（PID {process.pid}）。")
            print(LOCAL_URL)
            if not no_browser:
                webbrowser.open(LOCAL_URL)
            return 0
        stop()
    elif health():
        process = discover_process()
        if process:
            stop()
        else:
            print(f"端口 {PORT} 已被其他程序占用。", file=sys.stderr)
            return 2
    remove_record()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    creation_flags = hidden_creationflags(new_process_group=True)
    command = [sys.executable, "-u", str(SERVER_SCRIPT), "--port", str(PORT), "--workspace", str(workspace), "--no-browser"]
    if shared:
        command.append("--share")
    with LOG_FILE.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting annotation server; shared={shared}; workspace={workspace}\n")
        log.flush()
        handle = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            creationflags=creation_flags,
            start_new_session=(os.name != "nt"),
        )
    process = psutil.Process(handle.pid)
    write_record(process, shared, workspace)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if handle.poll() is not None:
            remove_record()
            print("协作标注服务启动失败，请查看 logs/annotation_server.log。", file=sys.stderr)
            return 1
        if health():
            listener = discover_process()
            if listener is not None:
                process = listener
                write_record(process, shared, workspace)
            print(f"协作标注服务已启动（PID {process.pid}）。")
            print(f"本机地址：{LOCAL_URL}")
            if shared:
                for address in discover_lan_addresses():
                    print(f"局域网地址：http://{address}:{PORT}/")
            if not no_browser:
                webbrowser.open(LOCAL_URL)
            return 0
        time.sleep(0.35)
    stop()
    print("协作标注服务启动超时，请查看日志。", file=sys.stderr)
    return 1


def status_payload() -> dict:
    record = read_record()
    process = managed_process(record) or discover_process()
    running = bool(process and health())
    shared = bool(record.get("shared")) if running else False
    return {
        "running": running,
        "shared": shared,
        "pid": process.pid if process else None,
        "workspace": str(Path(str(record.get("workspace") or DEFAULT_WORKSPACE)).resolve()),
        "local_url": LOCAL_URL,
        "lan_urls": [f"http://{address}:{PORT}/" for address in discover_lan_addresses()] if shared else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Manage {PRODUCT_NAME} annotation server")
    subparsers = parser.add_subparsers(dest="action", required=True)
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--share", action="store_true")
    start_parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    start_parser.add_argument("--no-browser", action="store_true")
    subparsers.add_parser("stop")
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.action == "start":
        return start(args.share, Path(args.workspace).expanduser().resolve(), args.no_browser)
    if args.action == "stop":
        return stop()
    payload = status_payload()
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else payload)
    return 0 if payload["running"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
