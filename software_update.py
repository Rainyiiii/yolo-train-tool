"""GitHub Release update discovery for YOLO Team Training Platform."""

from __future__ import annotations

import json
import os
import re
import ssl
import subprocess
import threading
import time
import base64
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

import certifi

from platform_subprocess import hidden_creationflags


REPOSITORY = "Rainyiiii/yolo-train-tool"
RELEASES_API = f"https://api.github.com/repos/{REPOSITORY}/releases?per_page=20"
RELEASES_PAGE = f"https://github.com/{REPOSITORY}/releases"
INSTALLER_PATTERN = re.compile(r"^YOLO-Team-Training-Platform-Setup-v(.+)\.exe$", re.IGNORECASE)
SEMVER_PATTERN = re.compile(
    r"^v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:-(?P<pre>[0-9A-Za-z.-]+))?$"
)
_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"checked_at": 0.0, "payload": None}
_CACHE_SECONDS = 15 * 60


def current_version(root: Path | None = None) -> str:
    version_file = (root or Path(__file__).resolve().parent) / "VERSION.txt"
    return version_file.read_text(encoding="utf-8-sig").strip()


def semantic_version_key(value: str) -> tuple[int, int, int, int, tuple[tuple[int, Any], ...]]:
    match = SEMVER_PATTERN.fullmatch(value.strip())
    if not match:
        raise ValueError(f"无效版本号：{value}")
    prerelease = match.group("pre")
    pre_parts: tuple[tuple[int, Any], ...] = ()
    if prerelease:
        normalized: list[tuple[int, Any]] = []
        for part in prerelease.split("."):
            normalized.append((0, int(part)) if part.isdigit() else (1, part.casefold()))
        pre_parts = tuple(normalized)
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
        1 if prerelease is None else 0,
        pre_parts,
    )


def _release_payload(release: dict[str, Any], installed_version: str) -> dict[str, Any]:
    tag = str(release.get("tag_name") or "").strip()
    version = tag[1:] if tag.lower().startswith("v") else tag
    assets = release.get("assets") if isinstance(release.get("assets"), list) else []
    installer = next(
        (
            asset
            for asset in assets
            if isinstance(asset, dict)
            and INSTALLER_PATTERN.fullmatch(str(asset.get("name") or ""))
            and str(asset.get("state") or "uploaded") == "uploaded"
        ),
        None,
    )
    return {
        "current_version": installed_version,
        "latest_version": version,
        "tag": tag,
        "update_available": semantic_version_key(version) > semantic_version_key(installed_version),
        "prerelease": bool(release.get("prerelease")),
        "published_at": str(release.get("published_at") or release.get("created_at") or ""),
        "notes": str(release.get("body") or "暂无更新说明。").strip(),
        "release_url": str(release.get("html_url") or RELEASES_PAGE),
        "download_url": str(installer.get("browser_download_url") or "") if installer else "",
        "asset_name": str(installer.get("name") or "") if installer else "",
        "asset_size": int(installer.get("size") or 0) if installer else 0,
        "asset_digest": str(installer.get("digest") or "") if installer else "",
    }


def select_latest_release(releases: list[dict[str, Any]], installed_version: str) -> dict[str, Any]:
    candidates: list[tuple[tuple[int, int, int, int, tuple[tuple[int, Any], ...]], dict[str, Any]]] = []
    for release in releases:
        if not isinstance(release, dict) or release.get("draft"):
            continue
        tag = str(release.get("tag_name") or "").strip()
        try:
            key = semantic_version_key(tag)
        except ValueError:
            continue
        candidates.append((key, release))
    if not candidates:
        raise RuntimeError("GitHub Releases 中没有可识别的平台版本。")
    _, latest = max(candidates, key=lambda item: item[0])
    return _release_payload(latest, installed_version)


def _fetch_json_with_windows_https(timeout: float) -> list[dict[str, Any]]:
    powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    if not powershell.is_file():
        raise RuntimeError("未找到 Windows PowerShell HTTPS 回退组件。")
    script = (
        "$ErrorActionPreference='Stop';"
        "$ProgressPreference='SilentlyContinue';"
        "[Console]::OutputEncoding=New-Object Text.UTF8Encoding($false);"
        f"$headers=@{{Accept='application/vnd.github+json';'User-Agent'='YOLO-Team-Training-Platform-Updater';'X-GitHub-Api-Version'='2022-11-28'}};"
        f"@(Invoke-RestMethod -UseBasicParsing -Headers $headers -Uri '{RELEASES_API}' -TimeoutSec {max(3, int(timeout))}) | ConvertTo-Json -Depth 20 -Compress"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    completed = subprocess.run(
        [str(powershell), "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout + 5,
        creationflags=hidden_creationflags(),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(detail or f"Windows HTTPS 回退失败（退出码 {completed.returncode}）。")
    payload = json.loads(completed.stdout.lstrip("\ufeff").strip())
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise RuntimeError("Windows HTTPS 回退返回了无法识别的数据。")
    return payload


def _fetch_json(timeout: float = 8.0) -> list[dict[str, Any]]:
    request = Request(
        RELEASES_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "YOLO-Team-Training-Platform-Updater",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        tls_context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(request, timeout=timeout, context=tls_context) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        if os.name != "nt":
            raise
        return _fetch_json_with_windows_https(timeout)
    if not isinstance(payload, list):
        raise RuntimeError("GitHub 更新接口返回了无法识别的数据。")
    return payload


def check_for_updates(
    *,
    force: bool = False,
    fetcher: Callable[[], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    installed_version = current_version()
    now = time.time()
    with _CACHE_LOCK:
        cached = _CACHE.get("payload")
        if not force and cached and now - float(_CACHE.get("checked_at") or 0) < _CACHE_SECONDS:
            return dict(cached)
    try:
        payload = select_latest_release((fetcher or _fetch_json)(), installed_version)
        payload.update({"ok": True, "error": "", "checked_at": int(now)})
    except Exception as exc:
        payload = {
            "ok": False,
            "error": f"检查更新失败：{exc}",
            "current_version": installed_version,
            "latest_version": "",
            "update_available": False,
            "release_url": RELEASES_PAGE,
            "download_url": "",
            "notes": "",
            "checked_at": int(now),
        }
    with _CACHE_LOCK:
        _CACHE.update({"checked_at": now, "payload": dict(payload)})
    return payload
