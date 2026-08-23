#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE_ROOT="$SCRIPT_ROOT/workspace"
LOG_DIR="$WORKSPACE_ROOT/logs"
LOG_FILE="$LOG_DIR/installation-ubuntu.log"
VENV_DIR="$SCRIPT_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
REQUIREMENTS="$SCRIPT_ROOT/requirements.txt"
DEFAULTS_EXAMPLE="$SCRIPT_ROOT/train_panel_defaults.example.json"
DEFAULTS_FILE="$WORKSPACE_ROOT/config/settings.json"
NO_START=0

if [[ "${1:-}" == "--no-start" ]]; then
    NO_START=1
elif [[ $# -gt 0 ]]; then
    echo "用法：bash ubuntu_install_and_start.sh [--no-start]" >&2
    exit 2
fi

mkdir -p "$LOG_DIR" "$WORKSPACE_ROOT/config" "$WORKSPACE_ROOT/state" \
    "$WORKSPACE_ROOT/datasets" "$WORKSPACE_ROOT/annotation-hub" \
    "$WORKSPACE_ROOT/training-runs" "$WORKSPACE_ROOT/model-assets/base-models" \
    "$WORKSPACE_ROOT/exports/datasets" "$WORKSPACE_ROOT/exports/deployments" \
    "$WORKSPACE_ROOT/test-results" "$WORKSPACE_ROOT/cache" "$WORKSPACE_ROOT/temp" "$WORKSPACE_ROOT/backups"
export YOLO_TEAM_PLATFORM_HOME="$SCRIPT_ROOT"
export YOLO_TEAM_PLATFORM_DATA="$WORKSPACE_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "YOLO团队训练平台 Ubuntu 安装器"
echo "安装目录：$SCRIPT_ROOT"
echo "日志文件：$LOG_FILE"

if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    if [[ "${ID:-}" != "ubuntu" ]]; then
        echo "警告：当前系统标识为 ${PRETTY_NAME:-未知系统}，脚本按 Ubuntu/Debian 方式继续。"
    fi
fi

run_privileged() {
    if [[ "$(id -u)" -eq 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        echo "需要 root 权限或 sudo：$*" >&2
        return 1
    fi
}

find_supported_python() {
    local candidate path version major minor
    local candidates=(python3.14 python3.13 python3.12 python3.11 python3.10 python3 python)
    for candidate in "${candidates[@]}"; do
        path="$(command -v "$candidate" 2>/dev/null || true)"
        [[ -n "$path" ]] || continue
        version="$("$path" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
        [[ "$version" =~ ^([0-9]+)\.([0-9]+)$ ]] || continue
        major="${BASH_REMATCH[1]}"
        minor="${BASH_REMATCH[2]}"
        if (( major == 3 && minor >= 10 && minor <= 14 )); then
            PYTHON_BIN="$path"
            PYTHON_VERSION="$version"
            return 0
        fi
    done
    return 1
}

if ! find_supported_python; then
    echo "未找到 Python 3.10–3.14，尝试通过 apt 安装 Ubuntu 的 Python、venv 和 pip 组件。"
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "当前系统没有 apt-get。请手动安装 Python 3.10–3.14 后重试。" >&2
        exit 1
    fi
    run_privileged apt-get update
    run_privileged apt-get install -y python3 python3-venv python3-pip
    if ! find_supported_python; then
        echo "apt 安装后仍未找到 Python 3.10–3.14。请安装受支持版本后重试。" >&2
        exit 1
    fi
fi

echo "使用 Python $PYTHON_VERSION：$PYTHON_BIN"

if [[ -x "$VENV_PYTHON" ]]; then
    VENV_VERSION="$("$VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [[ ! "$VENV_VERSION" =~ ^3\.(10|11|12|13|14)$ ]]; then
        echo "已有 .venv 不是 Python 3.10–3.14：${VENV_VERSION:-无法读取}" >&2
        echo "请备份后移走 $VENV_DIR，再重新运行本脚本。" >&2
        exit 1
    fi
else
    echo "创建项目专用虚拟环境"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "更新 pip 安装工具"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

HAS_NVIDIA=0
VRAM_MB=0
TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    HAS_NVIDIA=1
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    VRAM_TEXT="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
    VRAM_DIGITS="$(printf '%s' "$VRAM_TEXT" | tr -cd '0-9')"
    if [[ "$VRAM_DIGITS" =~ ^[0-9]+$ ]]; then
        VRAM_MB="$VRAM_DIGITS"
    fi
    echo "检测到 NVIDIA GPU，显存：${VRAM_MB} MiB"
    echo "安装 CUDA 12.8 PyTorch 组件"
else
    echo "未检测到 NVIDIA GPU，安装 CPU PyTorch 组件"
fi

"$VENV_PYTHON" -m pip install --upgrade torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"

echo "安装 YOLO团队训练平台功能组件（包含 onnxruntime）"
"$VENV_PYTHON" -m pip install --upgrade -r "$REQUIREMENTS"

if [[ ! -f "$DEFAULTS_FILE" && -f "$DEFAULTS_EXAMPLE" ]]; then
    cp "$DEFAULTS_EXAMPLE" "$DEFAULTS_FILE"
    DEFAULTS_FILE="$DEFAULTS_FILE" WORKSPACE_ROOT="$WORKSPACE_ROOT" HAS_NVIDIA="$HAS_NVIDIA" VRAM_MB="$VRAM_MB" \
        "$VENV_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["DEFAULTS_FILE"])
data = json.loads(path.read_text(encoding="utf-8"))
has_nvidia = os.environ.get("HAS_NVIDIA") == "1"
vram_mb = int(os.environ.get("VRAM_MB", "0") or 0)
workspace = Path(os.environ["WORKSPACE_ROOT"])
data.update({
    "dataset_root": str(workspace / "datasets"),
    "train_images_dir": str(workspace / "datasets" / "default" / "images"),
    "train_annotations_dir": str(workspace / "datasets" / "default" / "annotations"),
    "base_model": str(workspace / "model-assets" / "base-models" / "yolo11n.pt"),
    "export_output_dir": str(workspace / "exports" / "deployments"),
    "test_output_dir": str(workspace / "test-results"),
    "label_images_dir": str(workspace / "annotation-hub" / "quick-label" / "images"),
    "label_annotations_dir": str(workspace / "annotation-hub" / "quick-label" / "annotations"),
})
if has_nvidia:
    data["train_device"] = "cuda"
    data["torch_cuda"] = "cu128"
    if vram_mb >= 7500:
        data["batch"], data["train_workers"] = "16", "4"
    elif vram_mb >= 5500:
        data["batch"], data["train_workers"] = "8", "4"
    else:
        data["batch"], data["train_workers"] = "4", "2"
else:
    data["train_device"] = "cpu"
    data["torch_cuda"] = "cpu"
    data["train_cache"] = "False"
    data["batch"], data["train_workers"] = "4", "2"
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
fi

BASE_MODEL="$WORKSPACE_ROOT/model-assets/base-models/yolo11n.pt"
if [[ ! -f "$BASE_MODEL" ]]; then
    echo "下载默认 YOLO11n 基础模型"
    YOLO_BASE_MODEL_DESTINATION="$BASE_MODEL" "$VENV_PYTHON" -c "import os, shutil; from pathlib import Path; from ultralytics import YOLO; model=YOLO('yolo11n.pt'); src=Path(getattr(model, 'ckpt_path', 'yolo11n.pt')).resolve(); dst=Path(os.environ['YOLO_BASE_MODEL_DESTINATION']); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); print('YOLO11n ready')"
else
    echo "默认 YOLO11n 基础模型已经就绪"
fi

echo "执行系统自检"
"$VENV_PYTHON" "$SCRIPT_ROOT/system_check.py" --write-report

if (( NO_START == 0 )); then
    echo "启动 YOLO团队训练平台"
    "$VENV_PYTHON" "$SCRIPT_ROOT/panel_service.py" start --no-browser
else
    echo "已按 --no-start 跳过面板启动。"
fi

echo "安装完成。"
echo "网页地址：http://127.0.0.1:8989/"
echo "启动：bash ubuntu_start_train_panel.sh"
echo "停止：bash ubuntu_stop_train_panel.sh"
