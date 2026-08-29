#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在算力机器上创建完全隔离的 YOLO 训练环境。

不会安装 Python、CUDA Toolkit、Conda 或任何系统级依赖；所有 Python 包、缓存、模型和日志
均存放在 --root 指定目录。NVIDIA 驱动仍需由算力机器管理员预先安装。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterable

from platform_subprocess import hidden_creationflags

APP_NAME = "remote-train-env"
CONFIG_NAME = "environment.json"
SUPPORTED_TORCH = {"cu118", "cu121", "cu124", "cu126", "cu128", "cpu", "none"}


def is_windows() -> bool:
    return os.name == "nt"


def bin_dir(venv: Path) -> Path:
    return venv / ("Scripts" if is_windows() else "bin")


def venv_python(venv: Path) -> Path:
    return bin_dir(venv) / ("python.exe" if is_windows() else "python")


def venv_yolo(venv: Path) -> Path:
    return bin_dir(venv) / ("yolo.exe" if is_windows() else "yolo")


def print_command(command: Iterable[object]) -> None:
    print("$ " + subprocess.list2cmdline([str(item) for item in command]), flush=True)


def run(command: Iterable[object], *, env: dict[str, str] | None = None) -> None:
    command = [str(item) for item in command]
    print_command(command)
    subprocess.run(command, check=True, env=env, creationflags=hidden_creationflags())


def env_vars(root: Path) -> dict[str, str]:
    """返回只影响本次子进程的缓存配置，避免写入用户主目录。"""
    data = os.environ.copy()
    cache = root / "cache"
    data.update(
        {
            "PIP_CACHE_DIR": str(cache / "pip"),
            "TORCH_HOME": str(cache / "torch"),
            "HF_HOME": str(cache / "huggingface"),
            "YOLO_CONFIG_DIR": str(root / "config"),
            "MPLCONFIGDIR": str(cache / "matplotlib"),
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    return data


def ensure_directories(root: Path) -> None:
    for relative in ("cache/pip", "cache/torch", "cache/huggingface", "cache/matplotlib", "config", "jobs", "logs"):
        (root / relative).mkdir(parents=True, exist_ok=True)


def locate_host_python(requested: str) -> list[str]:
    """定位宿主 Python；打包成 EXE 后不能使用 sys.executable 创建 venv。"""
    if requested:
        candidate = Path(requested).expanduser()
        if candidate.is_file():
            return [str(candidate)]
        found = shutil.which(requested)
        if found:
            return [found]
        raise SystemExit(f"找不到 --python 指定的解释器：{requested}")

    candidates: list[list[str]] = []
    if getattr(sys, "frozen", False):
        if is_windows() and shutil.which("py"):
            candidates.extend([
                ["py", "-3.14"],
                ["py", "-3.13"],
                ["py", "-3.12"],
                ["py", "-3.11"],
                ["py", "-3.10"],
                ["py", "-3"],
            ])
        candidates.extend([["python3"], ["python"]])
    else:
        candidates.append([sys.executable])
        if is_windows() and shutil.which("py"):
            candidates.extend([
                ["py", "-3.14"],
                ["py", "-3.13"],
                ["py", "-3.12"],
                ["py", "-3.11"],
                ["py", "-3.10"],
            ])
        candidates.extend([["python3"], ["python"]])

    for command in candidates:
        try:
            probe = subprocess.run(
                command + ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
                creationflags=hidden_creationflags(),
            )
            if probe.returncode != 0:
                continue
            major, minor = (int(value) for value in probe.stdout.strip().split(".", maxsplit=1))
            if major == 3 and 10 <= minor <= 14:
                return command
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
    raise SystemExit("未找到 Python 3.10–3.14。请在不改系统 PATH 的前提下，使用 --python 指定 python.exe 的完整路径。")


def torch_index_url(channel: str) -> str:
    return "" if channel in {"none", ""} else f"https://download.pytorch.org/whl/{channel}"


def write_config(root: Path, python_command: list[str], torch_cuda: str) -> None:
    config = {
        "app": APP_NAME,
        "root": str(root),
        "venv": str(root / ".venv"),
        "host_python": python_command,
        "torch_cuda": torch_cuda,
        "cache_policy": "all caches are kept below root/cache",
    }
    (root / CONFIG_NAME).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_launcher(root: Path) -> Path:
    venv = root / ".venv"
    if is_windows():
        path = root / "train_env.cmd"
        content = f'''@echo off
setlocal
set "TRAIN_ENV_ROOT=%~dp0"
set "PIP_CACHE_DIR=%TRAIN_ENV_ROOT%cache\\pip"
set "TORCH_HOME=%TRAIN_ENV_ROOT%cache\\torch"
set "HF_HOME=%TRAIN_ENV_ROOT%cache\\huggingface"
set "YOLO_CONFIG_DIR=%TRAIN_ENV_ROOT%config"
set "MPLCONFIGDIR=%TRAIN_ENV_ROOT%cache\\matplotlib"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONNOUSERSITE=1"
"%TRAIN_ENV_ROOT%.venv\\Scripts\\yolo.exe" %*
'''
    else:
        path = root / "train_env.sh"
        content = f'''#!/usr/bin/env bash
set -euo pipefail
TRAIN_ENV_ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
export PIP_CACHE_DIR="$TRAIN_ENV_ROOT/cache/pip"
export TORCH_HOME="$TRAIN_ENV_ROOT/cache/torch"
export HF_HOME="$TRAIN_ENV_ROOT/cache/huggingface"
export YOLO_CONFIG_DIR="$TRAIN_ENV_ROOT/config"
export MPLCONFIGDIR="$TRAIN_ENV_ROOT/cache/matplotlib"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONNOUSERSITE=1
exec "$TRAIN_ENV_ROOT/.venv/bin/yolo" "$@"
'''
    path.write_text(content, encoding="utf-8", newline="\r\n" if is_windows() else "\n")
    if not is_windows():
        path.chmod(path.stat().st_mode | 0o111)
    return path


def setup(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    venv = root / ".venv"
    ensure_directories(root)
    host_python = locate_host_python(args.python)

    if not venv_python(venv).exists():
        run(host_python + ["-m", "venv", str(venv)])
    python = venv_python(venv)
    local_env = env_vars(root)
    run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], env=local_env)

    if args.torch_cuda != "none":
        torch_command: list[object] = [python, "-m", "pip", "install", "--upgrade", "torch", "torchvision", "torchaudio"]
        index = torch_index_url(args.torch_cuda)
        if index:
            torch_command += ["--index-url", index]
        run(torch_command, env=local_env)
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "ultralytics>=8.3,<9",
            "onnx>=1.16",
            "onnxsim>=0.4",
            "onnxruntime-gpu>=1.18" if args.torch_cuda != "cpu" else "onnxruntime>=1.18",
            "pyyaml>=6.0",
        ],
        env=local_env,
    )
    write_config(root, host_python, args.torch_cuda)
    launcher = write_launcher(root)
    print(f"\n环境已就绪：{root}")
    print(f"YOLO 启动器：{launcher}")
    print("所有 pip、模型与训练缓存均位于该目录；不会安装 Conda、CUDA Toolkit 或全局 Python 包。")
    doctor(argparse.Namespace(root=str(root), skip_import=False))


def doctor(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    venv = root / ".venv"
    python = venv_python(venv)
    print(f"环境目录：{root}")
    print(f"虚拟环境：{venv}")
    print(f"Python：{python if python.exists() else '未创建'}")
    if not python.exists():
        raise SystemExit("环境未创建，请先运行 setup。")
    if not args.skip_import:
        code = textwrap.dedent(
            """
            import torch, ultralytics
            print('torch=', torch.__version__)
            print('ultralytics=', ultralytics.__version__)
            print('cuda_available=', torch.cuda.is_available())
            if torch.cuda.is_available():
                print('gpu=', torch.cuda.get_device_name(0))
            """
        )
        run([python, "-c", code], env=env_vars(root))
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            creationflags=hidden_creationflags(),
        )
    else:
        print("提示：未找到 nvidia-smi；若需 GPU 训练，请确认 NVIDIA 驱动已由管理员安装。")


def run_yolo(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    executable = venv_yolo(root / ".venv")
    if not executable.exists():
        raise SystemExit("环境未创建或 ultralytics 未安装，请先运行 setup。")
    run([executable, *args.yolo_args], env=env_vars(root))


def clean_cache(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    cache = root / "cache"
    if cache.exists():
        shutil.rmtree(cache)
    ensure_directories(root)
    print(f"已清理本环境缓存：{cache}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在指定目录创建隔离的 YOLO 训练环境，不写入系统 Python/Conda/CUDA。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：remote_train_env.exe setup --root D:\\yolo_train_env --torch-cuda cu128",
    )
    parser.add_argument("--root", default="./remote_train_env", help="隔离环境根目录；建议放在数据盘")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_root_argument(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--root", default=argparse.SUPPRESS, help="隔离环境根目录；建议放在数据盘")

    setup_parser = commands.add_parser("setup", help="创建虚拟环境并安装训练依赖")
    add_root_argument(setup_parser)
    setup_parser.add_argument("--python", default="", help="宿主 Python 3.10–3.14 的路径或命令")

    setup_parser.add_argument("--torch-cuda", choices=sorted(SUPPORTED_TORCH), default="cu128", help="匹配驱动的 PyTorch CUDA 通道")
    setup_parser.set_defaults(func=setup)

    doctor_parser = commands.add_parser("doctor", help="检查隔离环境和 GPU")
    add_root_argument(doctor_parser)
    doctor_parser.add_argument("--skip-import", action="store_true", help="仅检查目录，不导入 torch/ultralytics")
    doctor_parser.set_defaults(func=doctor)

    run_parser = commands.add_parser("run", help="在隔离环境内执行 yolo 参数")
    add_root_argument(run_parser)
    run_parser.add_argument("yolo_args", nargs=argparse.REMAINDER, help="例如 detect train model=yolo11n.pt data=data.yaml epochs=100 device=0")
    run_parser.set_defaults(func=run_yolo)

    clean_parser = commands.add_parser("clean-cache", help="仅删除该环境的下载/模型缓存，不删除虚拟环境")
    add_root_argument(clean_parser)
    clean_parser.set_defaults(func=clean_cache)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
