# -*- coding: utf-8 -*-
"""Compile RDK X5 bundles in WSL and validate them on a board over SSH."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_stdio()


DEFAULT_DISTRO = "Ubuntu-22.04"
DEFAULT_VENV = "~/.local/share/yolo-team-training-platform/rdk-x5-venv"
DEFAULT_REMOTE_ROOT = "~/yolo-team-training-platform/rdk-x5-deployments"
MODEL_ZOO_URL = "https://github.com/D-Robotics/rdk_model_zoo.git"
MODEL_ZOO_BRANCH = "rdk_x5"
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
SAFE_HOST = re.compile(r"^[A-Za-z0-9._-]+$")
SAFE_LINUX_PATH = re.compile(r"^(?:~(?:/[A-Za-z0-9._-]+)+|/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*)$")
TOOLCHAIN_MARKER = ".yolo-team-rdk-toolchain"


def _require_program(name: str) -> str:
    program = shutil.which(name)
    if not program:
        raise RuntimeError(f"未找到 {name}，请在 Windows 可选功能中安装 OpenSSH/WSL 后重试。")
    return program


def _safe_name(value: str, label: str) -> str:
    value = str(value or "").strip()
    if not value or not SAFE_NAME.fullmatch(value):
        raise ValueError(f"{label}只能包含字母、数字、点、下划线和短横线。")
    return value


def _safe_host(value: str) -> str:
    value = str(value or "").strip()
    if not value or not SAFE_HOST.fullmatch(value):
        raise ValueError("RDK X5 主机名/IP 无效；当前支持 IPv4 或普通局域网主机名。")
    return value


def _safe_port(value: str | int) -> str:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("SSH 端口必须是 1–65535 的整数。") from exc
    if not 1 <= port <= 65535:
        raise ValueError("SSH 端口必须是 1–65535 的整数。")
    return str(port)


def _safe_linux_path(value: str, label: str) -> str:
    value = str(value or "").strip().rstrip("/")
    if not SAFE_LINUX_PATH.fullmatch(value) or "/../" in f"{value}/" or value.endswith("/.."):
        raise ValueError(f"{label}必须是 ~/目录 或绝对 Linux 目录，且不能包含空格或上级路径。")
    return value


def _linux_assignment(name: str, value: str, label: str) -> str:
    path = _safe_linux_path(value, label)
    if path.startswith("~/"):
        return f'{name}="$HOME/{path[2:]}"'
    return f"{name}={shlex.quote(path)}"


def _run_stream(command: list[str], input_text: str = "") -> None:
    print("$ " + subprocess.list2cmdline(command), flush=True)
    child_env = os.environ.copy()
    if Path(command[0]).name.lower() in {"wsl", "wsl.exe"}:
        child_env["WSL_UTF8"] = "1"
    process = subprocess.Popen(
        command,
        env=child_env,
        stdin=subprocess.PIPE if input_text else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if input_text and process.stdin is not None:
        process.stdin.write(input_text.encode("utf-8"))
        process.stdin.close()
    if process.stdout is not None:
        for raw_line in iter(process.stdout.readline, b""):
            # Redirected wsl.exe diagnostics can be UTF-16LE while Linux output is UTF-8.
            encoding = "utf-16-le" if b"\x00" in raw_line else "utf-8"
            print(raw_line.decode(encoding, errors="replace").rstrip("\r\n\x00"), flush=True)
        process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"命令执行失败，退出码 {return_code}。")


def _capture(command: list[str]) -> str:
    child_env = os.environ.copy()
    if Path(command[0]).name.lower() in {"wsl", "wsl.exe"}:
        child_env["WSL_UTF8"] = "1"
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_env,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("命令检查超时，请确认 WSL/SSH 没有等待首次初始化或交互输入。") from exc
    stdout_encoding = "utf-16-le" if b"\x00" in completed.stdout else "utf-8"
    stderr_encoding = "utf-16-le" if b"\x00" in completed.stderr else "utf-8"
    stdout = completed.stdout.decode(stdout_encoding, errors="replace").replace("\x00", "")
    stderr = completed.stderr.decode(stderr_encoding, errors="replace").replace("\x00", "")
    if completed.returncode:
        detail = (stderr or stdout).strip()
        raise RuntimeError(detail or f"命令执行失败，退出码 {completed.returncode}。")
    return stdout.strip()


def build_wsl_command(distro: str, user: str = "") -> list[str]:
    command = ["wsl.exe", "-d", _safe_name(distro, "WSL 发行版")]
    if user:
        command += ["-u", _safe_name(user, "WSL 用户名")]
    return [*command, "--", "bash", "-s"]


def _run_wsl_stream(distro: str, shell_code: str, user: str = "") -> None:
    _run_stream(build_wsl_command(distro, user), shell_code)


def _capture_wsl(distro: str, shell_code: str, user: str = "") -> str:
    _require_program("wsl.exe")
    command = build_wsl_command(distro, user)
    child_env = os.environ.copy()
    child_env["WSL_UTF8"] = "1"
    try:
        completed = subprocess.run(
            command,
            input=shell_code.encode("utf-8"),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_env,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("WSL 状态检查超时；首次安装后请先重启 Windows，再返回平台继续配置。") from exc
    stdout = completed.stdout.decode("utf-8", errors="replace").replace("\x00", "").strip()
    stderr_encoding = "utf-16-le" if b"\x00" in completed.stderr else "utf-8"
    stderr = completed.stderr.decode(stderr_encoding, errors="replace").replace("\x00", "").strip()
    if completed.returncode != 0:
        detail = stderr or stdout or "没有输出"
        raise RuntimeError(f"命令失败（退出码 {completed.returncode}）：{detail}")
    return stdout


def installed_wsl_distros() -> list[str]:
    program = shutil.which("wsl.exe")
    if not program:
        return []
    try:
        output = _capture([program, "--list", "--quiet"])
    except RuntimeError:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def inspect_wsl_environment(distro: str = DEFAULT_DISTRO, venv: str = DEFAULT_VENV) -> dict[str, Any]:
    distro = _safe_name(distro, "WSL 发行版")
    _safe_linux_path(venv, "WSL 工具链目录")
    program = shutil.which("wsl.exe")
    components = [
        {"id": "wsl", "label": "Windows WSL", "status": "missing", "detail": "尚未安装"},
        {"id": "distro", "label": distro, "status": "missing", "detail": "尚未安装"},
        {"id": "python", "label": "Python 3.10", "status": "missing", "detail": "等待 Ubuntu"},
        {"id": "toolchain", "label": "RDK X5 编译工具链", "status": "missing", "detail": "尚未配置"},
    ]
    result: dict[str, Any] = {
        "overall": "not_installed",
        "summary": "尚未安装 WSL；仅在需要编译 RDK X5 模型时安装。",
        "components": components,
        "distro": distro,
        "venv": venv,
        "can_install": True,
        "can_setup": False,
        "can_remove": False,
        "restart_required": False,
    }
    if not program:
        return result

    try:
        try:
            version_output = _capture([program, "--version"])
        except RuntimeError:
            version_output = _capture([program, "--status"])
        version = next((line.strip() for line in version_output.splitlines() if line.strip()), "WSL 命令可用")
    except RuntimeError as exc:
        components[0].update(status="error", detail=str(exc))
        result.update(overall="error", summary="WSL 命令存在，但 Windows 组件尚未就绪。")
        return result

    components[0].update(status="ready", detail=version)
    distros = installed_wsl_distros()
    result["installed_distros"] = distros
    if distro.lower() not in {item.lower() for item in distros}:
        result.update(
            overall="needs_distro",
            summary=f"WSL 已可用；安装 {distro} 后即可配置 RDK 工具链。",
            can_install=True,
        )
        return result

    components[1].update(status="ready", detail="已安装，可由平台按需启动")
    result["can_install"] = False
    result["can_setup"] = True
    venv_line = _linux_assignment("VENV", venv, "WSL 工具链目录")
    probe = f'''set -eu
{venv_line}
echo "ARCH=$(uname -m)"
echo "SYSTEM=$(grep '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '\"')"
echo "PYTHON=$(python3 --version 2>&1 || true)"
if python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)' 2>/dev/null; then echo "PYTHON_OK=1"; else echo "PYTHON_OK=0"; fi
if [ -x "$VENV/bin/hb_mapper" ] && [ -f "$VENV/{TOOLCHAIN_MARKER}" ]; then
  echo "TOOLCHAIN_OK=1"
  echo "MAPPER=$($VENV/bin/hb_mapper --version 2>&1 | head -n 1)"
  echo "SIZE=$(du -sh "$VENV" 2>/dev/null | awk '{{print $1}}')"
else
  echo "TOOLCHAIN_OK=0"
fi
echo "FREE=$(df -h / | tail -n 1 | awk '{{print $4}}')"'''
    try:
        output = _capture_wsl(distro, probe)
    except RuntimeError as exc:
        components[1].update(status="error", detail=str(exc))
        result.update(overall="error", summary=f"{distro} 已安装，但当前无法启动。")
        return result
    facts = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    result["facts"] = facts
    architecture = facts.get("ARCH", "未知架构")
    system_name = facts.get("SYSTEM", distro)
    components[1]["detail"] = f"{system_name} · {architecture} · 根目录剩余 {facts.get('FREE', '?')}"
    if architecture != "x86_64":
        components[1].update(status="error")
        result.update(overall="error", summary="RDK X5 编译环境要求 x86_64 WSL。")
        return result
    python_ok = facts.get("PYTHON_OK") == "1"
    components[2].update(
        status="ready" if python_ok else "warning",
        detail=facts.get("PYTHON", "未找到 Python 3.10"),
    )
    toolchain_ok = facts.get("TOOLCHAIN_OK") == "1"
    if toolchain_ok:
        mapper = facts.get("MAPPER", "hb_mapper 已安装")
        size = facts.get("SIZE", "未知大小")
        components[3].update(status="ready", detail=f"{mapper} · 占用 {size}")
        result.update(
            overall="ready",
            summary="RDK X5 编译环境已就绪，可以直接编译 Bayes-e .bin。",
            can_remove=True,
        )
    else:
        components[3]["detail"] = "未配置或缺少平台标记；点击“配置编译环境”安装"
        result.update(overall="needs_setup", summary="Ubuntu 已就绪；还需要配置平台专用 RDK 工具链。")
    return result


def _wsl_path(distro: str, windows_path: Path) -> str:
    resolved = windows_path.expanduser().resolve()
    return _capture([_require_program("wsl.exe"), "-d", _safe_name(distro, "WSL 发行版"), "--", "wslpath", "-a", str(resolved)])


def wsl_probe_script(venv: str) -> str:
    venv_line = _linux_assignment("VENV", venv, "WSL 工具链目录")
    shell = f'''set -eu
{venv_line}
ARCH="$(uname -m)"
. /etc/os-release
echo "WSL 系统：$PRETTY_NAME"
echo "WSL 架构：$ARCH"
python3 --version
echo "WSL 根目录空间：$(df -h / | tail -n 1 | awk '{{print $4}}') 可用"
test "$ARCH" = "x86_64" || {{ echo "RDK X5 编译节点必须是 x86_64" >&2; exit 1; }}
python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)' || {{ echo "官方工具链推荐 Python 3.10；当前版本不匹配" >&2; exit 1; }}
if [ -x "$VENV/bin/hb_mapper" ]; then
  "$VENV/bin/hb_mapper" --version
  echo "RDK_WSL_TOOLCHAIN=ready"
else
  echo "RDK_WSL_TOOLCHAIN=missing"
fi
echo "RDK_WSL_READY=1"'''
    return shell


def install_wsl_distro(distro: str, venv: str = DEFAULT_VENV) -> dict[str, Any]:
    distro = _safe_name(distro, "WSL 发行版")
    program = _require_program("wsl.exe")
    if distro.lower() in {item.lower() for item in installed_wsl_distros()}:
        print(f"{distro} 已安装，跳过重复安装。", flush=True)
        print("RDK_WSL_INSTALL=ready", flush=True)
        return inspect_wsl_environment(distro, venv)

    print(f"准备安装可选组件：WSL2 + {distro}。Windows 可能显示管理员授权窗口。", flush=True)
    arguments = ["--install", "--distribution", distro, "--no-launch"]
    if os.name == "nt":
        powershell = _require_program("powershell.exe")
        quoted_program = program.replace("'", "''")
        quoted_args = ",".join("'" + item.replace("'", "''") + "'" for item in arguments)
        script = (
            f"$arguments=@({quoted_args});"
            f"$process=Start-Process -FilePath '{quoted_program}' -ArgumentList $arguments "
            "-Verb RunAs -Wait -PassThru -WindowStyle Hidden; exit $process.ExitCode"
        )
        _run_stream([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script])
    else:
        _run_stream([program, *arguments])

    status = inspect_wsl_environment(distro, venv)
    if status.get("overall") in {"needs_setup", "ready"}:
        print("RDK_WSL_INSTALL=installed", flush=True)
    else:
        status["restart_required"] = True
        status["summary"] = "Windows 已接受 WSL 安装；请重启电脑后返回此页面继续配置。"
        print("RDK_WSL_INSTALL=reboot_required", flush=True)
    return status


def wsl_prerequisite_script() -> str:
    return '''set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if python3 -c 'import venv' >/dev/null 2>&1 && command -v git >/dev/null 2>&1; then
  echo "WSL 基础组件已就绪"
  exit 0
fi
command -v apt-get >/dev/null 2>&1 || { echo "当前发行版不支持 apt-get；请使用 Ubuntu 22.04" >&2; exit 1; }
apt-get update
apt-get install -y python3-venv git ca-certificates
echo "WSL 基础组件安装完成"'''


def wsl_setup_script(venv: str) -> str:
    venv_line = _linux_assignment("VENV", venv, "WSL 工具链目录")
    shell = f'''set -euo pipefail
{venv_line}
test "$(uname -m)" = "x86_64" || {{ echo "RDK X5 编译节点必须是 x86_64" >&2; exit 1; }}
python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)' || {{ echo "需要 Python 3.10" >&2; exit 1; }}
python3 -c 'import venv' || {{ echo "缺少 python3-venv；请先执行 sudo apt install python3.10-venv" >&2; exit 1; }}
if [ ! -x "$VENV/bin/python" ]; then
  mkdir -p "$(dirname "$VENV")"
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install "pip==26.2.1" "setuptools==80.9.0" "wheel==0.48.0"
"$VENV/bin/python" -m pip install "torch==2.13.0+cpu" "torchvision==0.28.0+cpu" --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/python" -m pip install rdkx5-yolo-mapper==1.0.0
"$VENV/bin/python" -m pip install "requests>=2.23,<3" "polars>=0.20,<2" "nvidia-ml-py>=12,<14" "ultralytics-thop>=2,<3"
"$VENV/bin/python" -m pip install --no-deps "ultralytics==8.4.120"
"$VENV/bin/hb_mapper" --version
"$VENV/bin/python" -c 'import cv2, numpy, onnx, onnxruntime, pkg_resources, torch, ultralytics; from ultralytics.nn.modules.block import AAttn; print(f"RDK 编译环境：torch={{torch.__version__}} ultralytics={{ultralytics.__version__}} numpy={{numpy.__version__}} opencv={{cv2.__version__}} onnx={{onnx.__version__}} onnxruntime={{onnxruntime.__version__}}")'
printf 'managed_by=yolo-team-training-platform\ncomponent=rdk-x5-toolchain\n' > "$VENV/{TOOLCHAIN_MARKER}"
echo "RDK_WSL_TOOLCHAIN=ready"
echo "RDK_WSL_READY=1"'''
    return shell


def wsl_remove_script(venv: str) -> str:
    venv_line = _linux_assignment("VENV", venv, "WSL 工具链目录")
    return f'''set -euo pipefail
{venv_line}
HOME_REAL="$(realpath -m "$HOME")"
VENV_REAL="$(realpath -m "$VENV")"
case "$VENV_REAL" in "$HOME_REAL"/*) ;; *) echo "拒绝移除用户目录之外的工具链：$VENV_REAL" >&2; exit 1;; esac
test -f "$VENV_REAL/{TOOLCHAIN_MARKER}" || {{ echo "目录缺少平台管理标记，拒绝删除：$VENV_REAL" >&2; exit 1; }}
CACHE_ROOT="$HOME_REAL/.cache/yolo-team-training-platform"
rm -rf -- "$VENV_REAL"
rm -rf -- "$CACHE_ROOT/rdk-model-zoo-x5" "$CACHE_ROOT/rdk-jobs"
rmdir "$CACHE_ROOT" 2>/dev/null || true
echo "已移除平台专用 RDK X5 工具链与编译缓存。"
echo "RDK_WSL_TOOLCHAIN=removed"'''


def remove_wsl_toolchain(distro: str, venv: str) -> None:
    _require_program("wsl.exe")
    if _safe_name(distro, "WSL 发行版").lower() not in {item.lower() for item in installed_wsl_distros()}:
        raise RuntimeError(f"未安装 WSL 发行版 {distro}，没有可移除的平台工具链。")
    _run_wsl_stream(distro, wsl_remove_script(venv))


def _find_manifest(bundle: Path) -> Optional[Path]:
    candidates = sorted(bundle.parent.glob("*.manifest.json"))
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if Path(str(data.get("artifact") or "")).resolve() == bundle.resolve():
                return candidate
        except (OSError, ValueError, TypeError):
            continue
    return candidates[0] if len(candidates) == 1 else None


def _update_manifest(bundle: Path, bin_path: Path, board: Optional[dict[str, Any]] = None) -> Optional[Path]:
    manifest_path = _find_manifest(bundle)
    if manifest_path is None:
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    vendor = data.get("vendor_conversion") if isinstance(data.get("vendor_conversion"), dict) else {}
    vendor.update({"status": "board_validated" if board else "compiled", "compiled_artifact": str(bin_path)})
    data["vendor_conversion"] = vendor
    data["final_artifact"] = str(bin_path)
    if board:
        data["board_validation"] = board
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def compile_bundle(bundle_path: str, distro: str, venv: str) -> Path:
    _require_program("wsl.exe")
    bundle = Path(bundle_path).expanduser().resolve()
    if not (bundle / "conversion-plan.json").is_file() or not (bundle / "convert_rdk_x5.sh").is_file():
        raise ValueError("请选择平台生成的 rdk-x5-npu-bundle 目录。")
    bundle_wsl = _wsl_path(distro, bundle)
    venv_line = _linux_assignment("VENV", venv, "WSL 工具链目录")
    job_name = re.sub(r"[^A-Za-z0-9._-]+", "-", bundle.parent.name).strip("-.") or "rdk-x5"
    shell = f'''set -euo pipefail
{venv_line}
BUNDLE={shlex.quote(bundle_wsl)}
MODEL_ZOO="$HOME/.cache/yolo-team-training-platform/rdk-model-zoo-x5"
WORK_DIR="$HOME/.cache/yolo-team-training-platform/rdk-jobs/{job_name}"
test -x "$VENV/bin/hb_mapper" || {{ echo "编译工具链尚未配置；请先点击配置 WSL 编译环境" >&2; exit 1; }}
"$VENV/bin/python" -c 'import cv2, numpy, onnx, pkg_resources, torch, ultralytics; from ultralytics.nn.modules.block import AAttn; assert ultralytics.__version__ == "8.4.120"; assert numpy.__version__ == "1.23.0"; assert cv2.__version__ == "4.6.0"; assert onnx.__version__ == "1.15.0"'
export PATH="$VENV/bin:$PATH"
export RDK_MODEL_ZOO_DIR="$MODEL_ZOO"
export RDK_WORK_DIR="$WORK_DIR"
cd "$BUNDLE"
bash convert_rdk_x5.sh'''
    _run_wsl_stream(distro, shell)
    bins = sorted((bundle / "output").glob("*_bayese_*_nv12.bin"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not bins:
        bins = sorted((bundle / "output").glob("*.bin"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not bins:
        raise RuntimeError("WSL 编译结束，但 Windows 转换包中没有找到 .bin。")
    bin_path = bins[0].resolve()
    manifest = _update_manifest(bundle, bin_path)
    print(f"RDK_X5_BIN={bin_path}", flush=True)
    if manifest:
        print(f"DEPLOY_MANIFEST={manifest}", flush=True)
    return bin_path


def _ssh_options(port: str, identity_file: str = "") -> list[str]:
    options = [
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=8",
        "-o", "ServerAliveInterval=15",
        "-o", "StrictHostKeyChecking=accept-new",
        "-p", _safe_port(port),
    ]
    if str(identity_file or "").strip():
        identity = Path(identity_file).expanduser().resolve()
        if not identity.is_file():
            raise ValueError("SSH 私钥文件不存在。")
        options += ["-i", str(identity)]
    return options


def build_board_probe_command(host: str, user: str, port: str, identity_file: str = "") -> list[str]:
    target = f"{_safe_name(user, 'SSH 用户名')}@{_safe_host(host)}"
    shell = '''set -eu
echo "板卡主机：$(hostname)"
echo "板卡架构：$(uname -m)"
test "$(uname -m)" = "aarch64" || { echo "目标不是 aarch64 RDK X5" >&2; exit 1; }
if command -v rdkos_info >/dev/null 2>&1; then rdkos_info; elif [ -f /etc/version ]; then echo "RDK OS：$(cat /etc/version)"; else cat /etc/os-release; fi
python3 -c 'import hbm_runtime; print("hbm_runtime：可用")'
if command -v hrt_model_exec >/dev/null 2>&1; then echo "hrt_model_exec：可用"; else echo "hrt_model_exec：未安装，将使用 hbm_runtime 推理验证"; fi
if command -v git >/dev/null 2>&1; then echo "Git：可用"; else echo "Git：未安装；首次图片推理需要预先准备 Model Zoo"; fi
echo "RDK_BOARD_READY=1"'''
    return [_require_program("ssh"), *_ssh_options(port, identity_file), target, shell]


def _scp_options(port: str, identity_file: str = "") -> list[str]:
    ssh_options = _ssh_options(port, identity_file)
    converted: list[str] = []
    index = 0
    while index < len(ssh_options):
        item = ssh_options[index]
        if item == "-p":
            converted += ["-P", ssh_options[index + 1]]
            index += 2
            continue
        converted.append(item)
        index += 1
    return converted


def _remote_job_dir(remote_root: str, bundle: Path, bin_path: Path) -> str:
    root = _safe_linux_path(remote_root, "板端部署目录")
    name_source = bundle.parent.name if bundle.is_dir() else bin_path.stem
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name_source).strip("-.") or "rdk-x5-model"
    return f"{root}/{name}"


def deploy_and_test(
    bundle_path: str,
    bin_file: str,
    host: str,
    user: str,
    port: str,
    identity_file: str,
    remote_root: str,
    test_image: str = "",
    task: str = "detect",
) -> Path | None:
    ssh = _require_program("ssh")
    scp = _require_program("scp")
    bundle = Path(bundle_path).expanduser().resolve()
    model = Path(bin_file).expanduser().resolve()
    if not bundle.is_dir() or not (bundle / "conversion-plan.json").is_file():
        raise ValueError("请选择有效的 RDK X5 转换包。")
    if not model.is_file() or model.suffix.lower() != ".bin":
        raise ValueError("请选择 WSL 编译生成的 RDK X5 .bin 模型。")
    classes = bundle / "classes.txt"
    if not classes.is_file():
        raise ValueError("转换包缺少 classes.txt。")
    image = Path(test_image).expanduser().resolve() if str(test_image or "").strip() else None
    if image is not None and (not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}):
        raise ValueError("板端测试图片不存在或格式不支持。")
    task = str(task or "detect").strip().lower()
    if task not in {"detect", "classify"}:
        raise ValueError("当前一键板端推理支持目标检测和图像分类。")

    safe_user = _safe_name(user, "SSH 用户名")
    safe_host = _safe_host(host)
    target = f"{safe_user}@{safe_host}"
    ssh_options = _ssh_options(port, identity_file)
    scp_options = _scp_options(port, identity_file)
    remote_dir = _remote_job_dir(remote_root, bundle, model)
    remote_dir_line = _linux_assignment("DEPLOY_DIR", remote_dir, "板端部署目录")

    prepare = f'''set -eu
{remote_dir_line}
mkdir -p "$DEPLOY_DIR"
chmod 700 "$DEPLOY_DIR"
echo "板端部署目录：$DEPLOY_DIR"'''
    _run_stream([ssh, *ssh_options, target, prepare])
    _run_stream([scp, *scp_options, str(model), f"{target}:{remote_dir}/model.bin"])
    _run_stream([scp, *scp_options, str(classes), f"{target}:{remote_dir}/classes.txt"])
    if image is not None:
        _run_stream([scp, *scp_options, str(image), f"{target}:{remote_dir}/test-image{image.suffix.lower()}"])

    test_suffix = image.suffix.lower() if image is not None else ""
    remote_test = f'''set -euo pipefail
{remote_dir_line}
MODEL="$DEPLOY_DIR/model.bin"
test -s "$MODEL" || {{ echo "上传后的 model.bin 不存在" >&2; exit 1; }}
MODEL_PATH="$MODEL" python3 -c 'import os, hbm_runtime; hbm_runtime.HB_HBMRuntime(os.environ["MODEL_PATH"]); print("BPU 模型加载通过")'
if command -v hrt_model_exec >/dev/null 2>&1; then
  hrt_model_exec model_info --model_file "$MODEL"
  hrt_model_exec perf --model_file "$MODEL" --thread_num 1
fi'''
    if image is not None:
        remote_test += f'''
MODEL_ZOO="$HOME/.cache/yolo-team-training-platform/rdk-model-zoo-x5"
RUNTIME="$MODEL_ZOO/samples/vision/ultralytics_yolo/runtime/python/main.py"
if [ ! -f "$RUNTIME" ]; then
  command -v git >/dev/null 2>&1 || {{ echo "板端缺少 Git，无法获取官方推理示例" >&2; exit 1; }}
  mkdir -p "$(dirname "$MODEL_ZOO")"
  git clone --depth 1 --branch {MODEL_ZOO_BRANCH} {MODEL_ZOO_URL} "$MODEL_ZOO"
fi
python3 "$RUNTIME" --task {shlex.quote(task)} --model-path "$MODEL" --test-img "$DEPLOY_DIR/test-image{test_suffix}" --label-file "$DEPLOY_DIR/classes.txt" --img-save-path "$DEPLOY_DIR/result.jpg"
test -s "$DEPLOY_DIR/result.jpg" || {{ echo "推理结束但未生成结果图片" >&2; exit 1; }}
echo "RDK_REMOTE_RESULT=$DEPLOY_DIR/result.jpg"'''
    remote_test += '\necho "RDK_BOARD_VALIDATED=1"'
    _run_stream([ssh, *ssh_options, target, remote_test])

    local_result: Optional[Path] = None
    if image is not None:
        result_dir = bundle / "board-results"
        result_dir.mkdir(parents=True, exist_ok=True)
        host_name = re.sub(r"[^A-Za-z0-9._-]+", "-", safe_host)
        local_result = (result_dir / f"{host_name}-result.jpg").resolve()
        _run_stream([scp, *scp_options, f"{target}:{remote_dir}/result.jpg", str(local_result)])

    board_record = {
        "status": "passed",
        "host": safe_host,
        "user": safe_user,
        "remote_dir": remote_dir,
        "validated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "result_image": str(local_result) if local_result else None,
    }
    manifest = _update_manifest(bundle, model, board_record)
    print(f"RDK_X5_REMOTE_DIR={remote_dir}", flush=True)
    if local_result:
        print(f"RDK_X5_RESULT={local_result}", flush=True)
    if manifest:
        print(f"DEPLOY_MANIFEST={manifest}", flush=True)
    return local_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO团队训练平台 RDK X5 WSL/SSH deployment helper")
    sub = parser.add_subparsers(dest="action", required=True)

    for name in ("wsl-status", "wsl-install", "wsl-check", "wsl-setup", "wsl-remove"):
        item = sub.add_parser(name)
        item.add_argument("--distro", default=DEFAULT_DISTRO)
        item.add_argument("--venv", default=DEFAULT_VENV)

    compile_parser = sub.add_parser("wsl-compile")
    compile_parser.add_argument("--distro", default=DEFAULT_DISTRO)
    compile_parser.add_argument("--venv", default=DEFAULT_VENV)
    compile_parser.add_argument("--bundle", required=True)

    board_check = sub.add_parser("board-check")
    board_check.add_argument("--host", required=True)
    board_check.add_argument("--user", required=True)
    board_check.add_argument("--port", default="22")
    board_check.add_argument("--identity-file", default="")

    board_deploy = sub.add_parser("board-deploy")
    board_deploy.add_argument("--bundle", required=True)
    board_deploy.add_argument("--bin", required=True)
    board_deploy.add_argument("--host", required=True)
    board_deploy.add_argument("--user", required=True)
    board_deploy.add_argument("--port", default="22")
    board_deploy.add_argument("--identity-file", default="")
    board_deploy.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    board_deploy.add_argument("--test-image", default="")
    board_deploy.add_argument("--task", choices=("detect", "classify"), default="detect")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.action == "wsl-status":
        print(json.dumps(inspect_wsl_environment(args.distro, args.venv), ensure_ascii=False, indent=2))
    elif args.action == "wsl-install":
        install_wsl_distro(args.distro, args.venv)
    elif args.action == "wsl-check":
        _require_program("wsl.exe")
        _run_wsl_stream(args.distro, wsl_probe_script(args.venv))
    elif args.action == "wsl-setup":
        _require_program("wsl.exe")
        _run_wsl_stream(args.distro, wsl_prerequisite_script(), user="root")
        _run_wsl_stream(args.distro, wsl_setup_script(args.venv))
    elif args.action == "wsl-remove":
        remove_wsl_toolchain(args.distro, args.venv)
    elif args.action == "wsl-compile":
        compile_bundle(args.bundle, args.distro, args.venv)
    elif args.action == "board-check":
        _run_stream(build_board_probe_command(args.host, args.user, args.port, args.identity_file))
    elif args.action == "board-deploy":
        deploy_and_test(
            args.bundle, args.bin, args.host, args.user, args.port, args.identity_file,
            args.remote_root, args.test_image, args.task,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"RDK X5 操作失败：{exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
