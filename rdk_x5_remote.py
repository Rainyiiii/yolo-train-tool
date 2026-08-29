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
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(detail or f"命令执行失败，退出码 {completed.returncode}。")
    return completed.stdout.strip().replace("\x00", "")


def build_wsl_command(distro: str) -> list[str]:
    return ["wsl.exe", "-d", _safe_name(distro, "WSL 发行版"), "--", "bash", "-s"]


def _run_wsl_stream(distro: str, shell_code: str) -> None:
    _run_stream(build_wsl_command(distro), shell_code)


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
echo "RDK_WSL_TOOLCHAIN=ready"
echo "RDK_WSL_READY=1"'''
    return shell


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

    for name in ("wsl-check", "wsl-setup"):
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
    if args.action == "wsl-check":
        _require_program("wsl.exe")
        _run_wsl_stream(args.distro, wsl_probe_script(args.venv))
    elif args.action == "wsl-setup":
        _require_program("wsl.exe")
        _run_wsl_stream(args.distro, wsl_setup_script(args.venv))
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
