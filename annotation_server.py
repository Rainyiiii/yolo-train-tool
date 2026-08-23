# -*- coding: utf-8 -*-
"""Local-first LAN collaboration server for YOLO Team dataset annotation."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import socket
import sys
import time
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import psutil

from annotation_exports import export_dataset, import_project_package
from annotation_store import AnnotationError, AnnotationStore
from annotation_ui import ANNOTATION_HTML
from platform_paths import ANNOTATION_HUB_DIR, PRODUCT_NAME


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ANNOTATION_HUB_DIR
SESSION_COOKIE = "yolo_team_annotation_session"
MAX_JSON_BODY = 4 * 1024 * 1024
MAX_IMAGE_BODY = 64 * 1024 * 1024
LOGIN_WINDOW_SECONDS = 5 * 60
LOGIN_ATTEMPTS = 10
LOGIN_FAILURES: dict[str, list[float]] = {}


def configure_stdio() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def discover_lan_addresses() -> list[str]:
    addresses: list[tuple[int, str]] = []
    skipped = ("loopback", "docker", "wsl", "vethernet", "vmware", "virtualbox", "tailscale", "zerotier", "meta", "vpn")
    stats = psutil.net_if_stats()
    for interface, entries in psutil.net_if_addrs().items():
        if interface in stats and not stats[interface].isup:
            continue
        penalty = 10 if any(word in interface.casefold() for word in skipped) else 0
        for entry in entries:
            if entry.family != socket.AF_INET:
                continue
            ip = entry.address
            if ip.startswith(("127.", "169.254.")):
                continue
            private = ip.startswith(("10.", "192.168.")) or ip.startswith("172.")
            addresses.append((penalty + (0 if private else 5), ip))
    return [ip for _, ip in sorted(dict.fromkeys(addresses))]


class AnnotationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], store: AnnotationStore, shared: bool):
        super().__init__(address, AnnotationHandler)
        self.store = store
        self.shared = shared

    def handle_error(self, request: Any, client_address: tuple[str, int]) -> None:
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


class AnnotationHandler(BaseHTTPRequestHandler):
    server: AnnotationHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format_string: str, *args: Any) -> None:
        sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] {self.client_address[0]} {format_string % args}\n")

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'")

    def send_json(self, payload: Any, status: int = 200, cookie: str | None = None) -> None:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        if cookie is not None:
            self.send_header("Set-Cookie", cookie)
        self._security_headers()
        self.end_headers()
        self.wfile.write(raw)

    def send_html(self) -> None:
        raw = ANNOTATION_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(raw)

    def send_file(self, path: Path, download_name: str | None = None) -> None:
        if not path.is_file():
            raise AnnotationError("文件不存在。", 404)
        size = path.stat().st_size
        self.send_response(200)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        if download_name:
            encoded = urllib.parse.quote(download_name)
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{encoded}")
        self._security_headers()
        self.end_headers()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(256 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _cookie_token(self) -> str:
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except Exception:
            return ""
        morsel = cookie.get(SESSION_COOKIE)
        return morsel.value if morsel else ""

    def current_user(self, required: bool = True) -> dict[str, Any] | None:
        user = self.server.store.session_user(self._cookie_token())
        if required and user is None:
            raise AnnotationError("请先登录。", 401)
        return user

    def read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise AnnotationError("请求长度无效。") from exc
        if length < 0 or length > MAX_JSON_BODY:
            raise AnnotationError("请求内容过大。", 413)
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeError, ValueError) as exc:
            raise AnnotationError("请求不是有效的 JSON。") from exc
        if not isinstance(payload, dict):
            raise AnnotationError("请求内容必须是 JSON 对象。")
        return payload

    def _validate_origin(self) -> None:
        origin = self.headers.get("Origin", "")
        host = self.headers.get("Host", "")
        if origin and urllib.parse.urlparse(origin).netloc.casefold() != host.casefold():
            raise AnnotationError("拒绝跨站请求。", 403)

    def _urls(self) -> list[str]:
        port = self.server.server_address[1]
        if not self.server.shared:
            return [f"http://127.0.0.1:{port}/"]
        addresses = discover_lan_addresses() or ["127.0.0.1"]
        return [f"http://{address}:{port}/" for address in addresses[:4]]

    def _login_allowed(self) -> bool:
        ip = self.client_address[0]
        cutoff = time.time() - LOGIN_WINDOW_SECONDS
        recent = [stamp for stamp in LOGIN_FAILURES.get(ip, []) if stamp > cutoff]
        LOGIN_FAILURES[ip] = recent
        return len(recent) < LOGIN_ATTEMPTS

    def _login_failed(self) -> None:
        LOGIN_FAILURES.setdefault(self.client_address[0], []).append(time.time())

    def do_GET(self) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            if parsed.path == "/":
                self.send_html()
                return
            if parsed.path == "/api/health":
                self.send_json({"ok": True, "service": f"{PRODUCT_NAME} Annotation Server"})
                return
            if parsed.path == "/api/bootstrap":
                self.send_json({
                    "needs_setup": self.server.store.needs_setup(),
                    "user": self.current_user(False),
                    "shared": self.server.shared,
                    "urls": self._urls(),
                })
                return
            user = self.current_user()
            assert user is not None
            if parsed.path == "/api/projects":
                self.send_json({"items": self.server.store.list_projects(user)})
                return
            if parsed.path == "/api/users":
                self.send_json({"items": self.server.store.list_users(user)})
                return
            if parsed.path == "/api/items":
                project_id = int(params.get("project_id", ["0"])[0])
                status = params.get("status", [""])[0]
                self.send_json({"items": self.server.store.list_items(project_id, user, status)})
                return
            if parsed.path == "/api/item":
                item_id = int(params.get("id", ["0"])[0])
                self.send_json({"item": self.server.store.item_detail(item_id, user)})
                return
            if parsed.path == "/api/image":
                item_id = int(params.get("id", ["0"])[0])
                path, _ = self.server.store.image_path(item_id, user)
                self.send_file(path)
                return
            if parsed.path == "/api/export":
                project_id = int(params.get("project_id", ["0"])[0])
                export_format = params.get("format", ["yolo"])[0]
                path = export_dataset(self.server.store, user, project_id, export_format)
                self.send_file(path, path.name)
                return
            raise AnnotationError("接口不存在。", 404)
        except AnnotationError as exc:
            self.send_json({"error": str(exc)}, exc.status)
        except (ValueError, TypeError):
            self.send_json({"error": "请求参数无效。"}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            print(f"GET {self.path} failed: {exc}", file=sys.stderr, flush=True)
            self.send_json({"error": "服务器处理请求时发生错误，请查看服务日志。"}, 500)

    def do_POST(self) -> None:
        try:
            self._validate_origin()
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/projects/upload-image":
                user = self.current_user()
                assert user is not None
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise AnnotationError("请求长度无效。") from exc
                if length <= 0 or length > MAX_IMAGE_BODY:
                    raise AnnotationError("单张图片必须小于 64 MB。", 413)
                params = urllib.parse.parse_qs(parsed.query)
                item = self.server.store.import_uploaded_image(
                    user,
                    int(params.get("project_id", ["0"])[0]),
                    params.get("filename", [""])[0],
                    self.rfile.read(length),
                )
                self.send_json({"item": item})
                return
            body = self.read_json()
            if self.path == "/api/setup":
                user = self.server.store.create_user(body.get("username", ""), body.get("password", ""), "admin")
                token = self.server.store.create_session(user["id"])
                cookie = f"{SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=604800"
                self.send_json({"user": user}, cookie=cookie)
                return
            if self.path == "/api/login":
                if not self._login_allowed():
                    raise AnnotationError("登录失败次数过多，请五分钟后再试。", 429)
                try:
                    user = self.server.store.authenticate(body.get("username", ""), body.get("password", ""))
                except AnnotationError:
                    self._login_failed()
                    raise
                token = self.server.store.create_session(user["id"])
                LOGIN_FAILURES.pop(self.client_address[0], None)
                cookie = f"{SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=604800"
                self.send_json({"user": user}, cookie=cookie)
                return
            if self.path == "/api/logout":
                self.server.store.delete_session(self._cookie_token())
                cookie = f"{SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0"
                self.send_json({"ok": True}, cookie=cookie)
                return
            user = self.current_user()
            assert user is not None
            if self.path == "/api/users":
                created = self.server.store.create_user(body.get("username", ""), body.get("password", ""), body.get("role", "annotator"), user)
                self.send_json({"user": created})
                return
            if self.path == "/api/projects":
                project = self.server.store.create_project(user, body.get("name", ""), body.get("labels", []), body.get("source_dir") or None)
                self.send_json({"project": project})
                return
            if self.path == "/api/projects/import-images":
                count = self.server.store.import_images(user, int(body.get("project_id", 0)), body.get("source_dir", ""))
                self.send_json({"imported": count})
                return
            if self.path == "/api/tasks/assign":
                assigned = self.server.store.assign_items(
                    user,
                    int(body.get("project_id", 0)),
                    int(body.get("assignee_id", 0)),
                    int(body.get("count", 0)),
                    body.get("item_ids", []),
                )
                self.send_json({"assigned": assigned})
                return
            if self.path == "/api/item/acquire":
                item = self.server.store.acquire_item(int(body.get("item_id", 0)), user)
                self.send_json({"item": item})
                return
            if self.path == "/api/item/save":
                item = self.server.store.save_item(
                    int(body.get("item_id", 0)), user, body.get("boxes", []),
                    int(body.get("revision", -1)), bool(body.get("submit")),
                )
                self.send_json({"item": item})
                return
            if self.path == "/api/item/review":
                item = self.server.store.review_item(
                    int(body.get("item_id", 0)), user, bool(body.get("approve")), body.get("comment", ""),
                )
                self.send_json({"item": item})
                return
            if self.path == "/api/item/release":
                self.server.store.release_item(int(body.get("item_id", 0)), user)
                self.send_json({"ok": True})
                return
            if self.path == "/api/project-package/import":
                project = import_project_package(self.server.store, user, body.get("path", ""))
                self.send_json({"project": project})
                return
            raise AnnotationError("接口不存在。", 404)
        except AnnotationError as exc:
            self.send_json({"error": str(exc)}, exc.status)
        except (ValueError, TypeError):
            self.send_json({"error": "请求参数无效。"}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            print(f"POST {self.path} failed: {exc}", file=sys.stderr, flush=True)
            self.send_json({"error": "服务器处理请求时发生错误，请查看服务日志。"}, 500)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{PRODUCT_NAME} local-first collaborative annotation server")
    parser.add_argument("--host", default="127.0.0.1", help="127.0.0.1 for personal mode; 0.0.0.0 for LAN sharing")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--share", action="store_true", help="listen on all network interfaces")
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main() -> int:
    configure_stdio()
    args = build_parser().parse_args()
    host = "0.0.0.0" if args.share else args.host
    shared = host not in {"127.0.0.1", "localhost", "::1"}
    store = AnnotationStore(args.workspace)
    try:
        server = AnnotationHTTPServer((host, args.port), store, shared)
    except OSError as exc:
        print(f"无法监听 {host}:{args.port}：{exc}", file=sys.stderr)
        return 1
    local_url = f"http://127.0.0.1:{args.port}/"
    print(f"{PRODUCT_NAME}协作标注中心：{local_url}")
    print(f"Workspace: {store.workspace}")
    print("Mode: " + ("LAN shared" if shared else "personal"))
    if shared:
        for address in discover_lan_addresses():
            print(f"LAN_URL=http://{address}:{args.port}/")
    if not args.no_browser:
        import threading
        threading.Timer(0.8, lambda: webbrowser.open(local_url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
