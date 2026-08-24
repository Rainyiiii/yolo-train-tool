# -*- coding: utf-8 -*-
import argparse
import importlib.util
import json
import locale
import math
import mimetypes
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import uuid
import webbrowser


import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass
from http import HTTPStatus

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import cv2
import yaml

from base_models import base_model_catalog, validate_model_name
from device_profiles import public_device_profiles
from model_assets import collect_model_assets, register_asset_manifest, register_asset_root, register_external_model
from annotation_service import status_payload as annotation_service_status
from project_manager import activate_project, create_project, inspect_dataset, project_catalog, update_project
from platform_paths import (
    ANNOTATION_HUB_DIR,
    CONFIG_DIR,
    DATASETS_DIR,
    DEPLOYMENT_EXPORTS_DIR,
    LOG_DIR,
    MODEL_ASSETS_DIR,
    STATE_DIR,
    TEST_RESULTS_DIR,
    TRAINING_RUNS_DIR,
    ensure_workspace,
)



def configure_stdio() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass



configure_stdio()


SCRIPT_ROOT = Path(__file__).resolve().parent
ensure_workspace()

WORKFLOW_SCRIPT = SCRIPT_ROOT / "host_train_export.py"
EXPORT_SCRIPT = SCRIPT_ROOT / "export_model.py"
TEST_SCRIPT = SCRIPT_ROOT / "model_test.py"
LABEL_SCRIPT = SCRIPT_ROOT / "video_track_label.py"
ANNOTATION_SERVICE_SCRIPT = SCRIPT_ROOT / "annotation_service.py"
DEFAULT_VM_WORK_DIR = "~/yolo-team-training-platform/maixcam-jobs"
USER_DEFAULTS_FILE = CONFIG_DIR / "settings.json"
MODEL_REGISTRY_FILE = CONFIG_DIR / "model-registry.json"
BASE_MODEL_DOWNLOAD_SCRIPT = SCRIPT_ROOT / "base_models.py"
STOP_EXPORT_SIGNAL_FILE = STATE_DIR / "training-stop-export.signal"
LATEST_JOB_LOG_FILE = LOG_DIR / "latest-job.log"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg"}

TRAIN_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}






DEFAULT_VALUES: dict[str, Any] = {
    "active_project_id": "",
    "dataset_root": str(DATASETS_DIR),
    "train_task": "detect",
    "train_images_dir": str(DATASETS_DIR / "default" / "images"),
    "train_annotations_dir": str(DATASETS_DIR / "default" / "annotations"),
    "prepared_dataset_yaml": "",
    "train_ratio_percent": "80",
    "val_ratio_percent": "10",
    "img_width": "640",
    "img_height": "480",
    "image_resize_mode": "letterbox",

    "epochs": "10000",
    "batch": "16",
    "train_workers": "4",
    "patience": "0",
    "lr0": "0.005",
    "conda_env": "",
    "base_model": str(MODEL_ASSETS_DIR / "base-models" / "yolo11n.pt"),
    "torch_cuda": "cu128",
    "train_device": "cuda",
    "train_cache": "disk",
    "raw_dataset_root": "",
    "raw_images_dir": "",
    "raw_labels_dir": "",
    "raw_output_dir": "",
    "raw_class_names": "",
    "raw_overwrite": False,
    "asset_scan_root": "",

    "project_name": "default-project",
    "model_name": "yolo-model",
    "operator_mode": "recommended",
    "train_mode": "local",
    "remote_train_user": "",
    "remote_train_host": "127.0.0.1",
    "remote_train_port": "22",
    "remote_train_work_dir": "C:/YOLOTeamTrainingPlatform/workspace/training-jobs",
    "model_path": "",
    "classes_path": "",
    "calib_dir": "",
    "test_image": "",
    "vm_user": "",
    "vm_host": "",
    "vm_work_dir": DEFAULT_VM_WORK_DIR,
    "skip_vm_convert": False,
    "deploy_model": "",
    "deployment_target": "generic_onnx",
    "export_format": "auto",
    "export_chip": "",
    "export_output_dir": str(DEPLOYMENT_EXPORTS_DIR),
    "export_data": "",
    "export_int8": False,
    "test_model": "",
    "test_source": "camera",
    "test_image_file": "",
    "test_image_folder": "",
    "test_output_dir": str(TEST_RESULTS_DIR),
    "camera_index": "0",
    "conf": "0.25",
    "label_video_dir": "",
    "label_video": "",
    "label_camera_index": "0",
    "label_source_type": "video",
    "label_images_input_dir": "",
    "label_name": "object",

    "label_interval": "5",
    "label_images_dir": str(ANNOTATION_HUB_DIR / "quick-label" / "images"),
    "label_annotations_dir": str(ANNOTATION_HUB_DIR / "quick-label" / "annotations"),
    "label_prefix": "track",
    "label_tracker": "csrt",
    "label_start_frame": "0",
    "label_max_frames": "0",
    "label_display_scale": "1.0",
    "label_jpeg_quality": "95",
}





STATE_LOCK = threading.RLock()
STATE: dict[str, Any] = {
    "values": DEFAULT_VALUES.copy(),
    "logs": [],
    "markers": {},
    "train_progress": {
        "phase": "idle",
        "task": "",
        "epoch": 0,
        "total_epochs": 0,
        "batch": 0,
        "total_batches": 0,
        "percent": 0.0,
        "gpu_mem": "",
        "loss": None,
        "box_loss": None,
        "cls_loss": None,
        "dfl_loss": None,
        "instances": None,
        "size": None,
        "speed": "",
        "elapsed": "",
        "eta": "",
        "val_batch": 0,
        "val_total": 0,
        "val_percent": 0.0,
        "metrics": {},
        "history": [],
        "updated_at": "",
    },
    "running": False,
    "job": None,
    "exit_code": None,
    "started_at": None,
    "finished_at": None,
    "last_error": "",
}
@dataclass
class LabelTrackObject:
    obj_id: int
    label: str
    bbox: tuple[int, int, int, int]
    tracker: object
    ok: bool = True
    sample_count: int = 1
    quality: float = 1.0
    warning: str = ""


class TemplateTracker:
    def __init__(self, search_scale: float = 2.5, min_score: float = 0.45):
        self.search_scale = search_scale
        self.min_score = min_score
        self.template = None
        self.bbox: Optional[tuple[int, int, int, int]] = None
        self.last_score: Optional[float] = None

    def init(self, frame, bbox: tuple[int, int, int, int]) -> bool:
        x, y, w, h = sanitize_label_bbox(bbox, frame.shape[1], frame.shape[0])
        if w <= 2 or h <= 2:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.template = gray[y:y + h, x:x + w].copy()
        self.bbox = (x, y, w, h)
        return True

    def update(self, frame):
        if self.template is None or self.bbox is None:
            return False, (0, 0, 0, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self.bbox
        pad_x, pad_y = max(int(w * self.search_scale), 20), max(int(h * self.search_scale), 20)
        sx1, sy1 = max(0, x - pad_x), max(0, y - pad_y)
        sx2, sy2 = min(frame.shape[1], x + w + pad_x), min(frame.shape[0], y + h + pad_y)
        search = gray[sy1:sy2, sx1:sx2]
        if search.shape[0] < h or search.shape[1] < w:
            return False, self.bbox
        result = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        self.last_score = float(score)
        self.bbox = sanitize_label_bbox((sx1 + loc[0], sy1 + loc[1], w, h), frame.shape[1], frame.shape[0])
        return score >= self.min_score, self.bbox


class MultiTemplateTracker:
    """以 CSRT 为主跟踪，多个参考模板仅用于 CSRT 丢失后的恢复。"""
    def __init__(self, search_scale: float = 2.5, min_score: float = 0.55):
        self.search_scale = search_scale
        self.min_score = min_score
        self.samples: list[tuple[Any, int, int]] = []
        self.bbox: Optional[tuple[int, int, int, int]] = None
        self.primary = None
        self.last_score: Optional[float] = None

    def _reset_primary(self, frame, bbox: tuple[int, int, int, int]) -> None:
        self.primary = make_label_cv_tracker("csrt")
        if self.primary is not None and not init_label_tracker(self.primary, frame, bbox):
            self.primary = None

    def init(self, frame, bbox: tuple[int, int, int, int]) -> bool:
        if make_label_cv_tracker("csrt") is None:
            raise RuntimeError("多视角实验模式需要 OpenCV CSRT；请安装 opencv-contrib-python 后重启标注工具。")
        self.samples = []
        self.bbox = None
        self.primary = None
        return self.add_sample(frame, bbox)

    def add_sample(self, frame, bbox: tuple[int, int, int, int]) -> bool:
        x, y, w, h = sanitize_label_bbox(bbox, frame.shape[1], frame.shape[0])
        if w <= 2 or h <= 2:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.samples.append((gray[y:y + h, x:x + w].copy(), w, h))
        self.bbox = (x, y, w, h)
        self._reset_primary(frame, self.bbox)
        return True

    def _recover_from_templates(self, frame):
        if self.bbox is None:
            return False, (0, 0, 0, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self.bbox
        pad_x, pad_y = max(int(w * self.search_scale), 20), max(int(h * self.search_scale), 20)
        sx1, sy1 = max(0, x - pad_x), max(0, y - pad_y)
        sx2, sy2 = min(frame.shape[1], x + w + pad_x), min(frame.shape[0], y + h + pad_y)
        search = gray[sy1:sy2, sx1:sx2]
        best_score, best_bbox = -1.0, self.bbox
        for template, sample_w, sample_h in self.samples:
            if search.shape[0] < sample_h or search.shape[1] < sample_w:
                continue
            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(result)
            if score > best_score:
                best_score = score
                best_bbox = sanitize_label_bbox((sx1 + loc[0], sy1 + loc[1], sample_w, sample_h), frame.shape[1], frame.shape[0])
        if best_score < self.min_score:
            self.last_score = max(0.0, float(best_score))
            return False, self.bbox
        self.last_score = float(best_score)
        self.bbox = best_bbox
        self._reset_primary(frame, self.bbox)
        return True, self.bbox

    def update(self, frame):
        if not self.samples or self.bbox is None:
            return False, (0, 0, 0, 0)
        if self.primary is not None:
            ok, bbox = self.primary.update(frame)
            if ok:
                self.bbox = sanitize_label_bbox(bbox, frame.shape[1], frame.shape[0])
                self.last_score = None
                return True, self.bbox
        return self._recover_from_templates(frame)


def sanitize_label_bbox(bbox, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = [int(round(value)) for value in bbox]
    x, y = max(0, min(x, width - 1)), max(0, min(y, height - 1))
    return x, y, max(1, min(w, width - x)), max(1, min(h, height - y))


def label_tracking_quality(
    previous: tuple[int, int, int, int],
    current: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    tracker_score: Optional[float] = None,
) -> tuple[float, str]:
    """Conservative drift check used before an automatically tracked box is saved."""
    px, py, pw, ph = sanitize_label_bbox(previous, frame_width, frame_height)
    cx, cy, cw, ch = sanitize_label_bbox(current, frame_width, frame_height)
    previous_area = max(1.0, float(pw * ph))
    current_area = max(1.0, float(cw * ch))
    area_ratio = current_area / previous_area
    previous_center = (px + pw / 2.0, py + ph / 2.0)
    current_center = (cx + cw / 2.0, cy + ch / 2.0)
    center_distance = math.hypot(current_center[0] - previous_center[0], current_center[1] - previous_center[1])
    reference_diagonal = max(20.0, math.hypot(pw, ph))
    motion_ratio = center_distance / reference_diagonal
    previous_aspect = pw / max(1.0, float(ph))
    current_aspect = cw / max(1.0, float(ch))
    aspect_ratio = current_aspect / max(0.01, previous_aspect)

    quality = 1.0
    warnings: list[str] = []
    if motion_ratio > 2.5:
        quality = 0.0
        warnings.append("目标位置突变")
    elif motion_ratio > 1.25:
        quality *= 0.45
        warnings.append("目标移动幅度较大")
    elif motion_ratio > 0.75:
        quality *= 0.75

    if not 0.35 <= area_ratio <= 2.8:
        quality *= 0.2
        warnings.append("目标尺寸突变")
    elif not 0.6 <= area_ratio <= 1.7:
        quality *= 0.7
    if not 0.45 <= aspect_ratio <= 2.2:
        quality *= 0.35
        warnings.append("目标形状突变")
    if tracker_score is not None:
        quality = min(quality, max(0.0, min(1.0, float(tracker_score))))
        if tracker_score < 0.45:
            warnings.append("图像匹配置信度低")
    return round(max(0.0, min(1.0, quality)), 3), "；".join(dict.fromkeys(warnings))


def make_label_cv_tracker(name: str):
    upper = name.upper()
    candidates = []
    if hasattr(cv2, "legacy"):
        candidates.append((cv2.legacy, f"Tracker{upper}_create"))
    candidates.append((cv2, f"Tracker{upper}_create"))
    for module, factory in candidates:
        if hasattr(module, factory):
            return getattr(module, factory)()
    return None


def make_label_tracker(name: str):
    normalized = name.lower()
    if normalized == "template":
        return TemplateTracker()
    if normalized == "multi_template":
        return MultiTemplateTracker()
    tracker = make_label_cv_tracker(name)
    if tracker is not None:
        return tracker
    append_log(f"[网页标注] 跟踪器 {name} 不可用，已回退到 Template。\n")
    return TemplateTracker()


def init_label_tracker(tracker, frame, bbox: tuple[int, int, int, int]) -> bool:
    result = tracker.init(frame, bbox)
    return True if result is None else bool(result)


def label_object_data(obj: LabelTrackObject) -> dict[str, Any]:
    x, y, w, h = obj.bbox
    return {
        "id": obj.obj_id, "label": obj.label, "x": x, "y": y, "w": w, "h": h,
        "ok": obj.ok, "sample_count": obj.sample_count,
        "quality": obj.quality, "warning": obj.warning,
    }


LABEL_SESSIONS: dict[str, dict[str, Any]] = {}
LABEL_SESSIONS_LOCK = threading.RLock()
MODEL_REGISTRY_LOCK = threading.RLock()


MAX_LOG_LINES = 3000
MAX_LABEL_VIDEOS = 2000
MAX_VIDEO_PREVIEW_WIDTH = 560
VIDEO_PREVIEW_CACHE_LIMIT = 80
VIDEO_PREVIEW_CACHE: OrderedDict[tuple[str, int, int], bytes] = OrderedDict()
VIDEO_MIME_TYPES = {
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv",
    ".mpg": "video/mpeg",
    ".mpeg": "video/mpeg",
}





def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def clean_values(values: Optional[dict[str, Any]]) -> dict[str, Any]:

    merged = DEFAULT_VALUES.copy()
    if values:
        source = values.copy()
        legacy_img_size = source.get("img_size")
        if legacy_img_size is not None:
            source.setdefault("img_width", legacy_img_size)
            source.setdefault("img_height", legacy_img_size)
        for key in merged:
            if key in source:
                merged[key] = as_bool(source[key]) if isinstance(DEFAULT_VALUES[key], bool) else str(source[key])
        if not str(source.get("deploy_model", "")).strip():
            merged["deploy_model"] = str(source.get("test_model") or source.get("model_path") or "")
    try:
        train_ratio = max(1, min(99, round(float(merged["train_ratio_percent"]))))
    except (TypeError, ValueError):
        train_ratio = 80
    try:
        val_ratio = max(1, min(99, round(float(merged["val_ratio_percent"]))))
    except (TypeError, ValueError):
        val_ratio = 10
    if train_ratio + val_ratio > 100:
        val_ratio = max(1, 100 - train_ratio)
    merged["train_ratio_percent"] = str(train_ratio)
    merged["val_ratio_percent"] = str(val_ratio)
    return merged


def load_user_defaults() -> dict[str, Any]:
    if not USER_DEFAULTS_FILE.is_file():
        return DEFAULT_VALUES.copy()
    try:
        data = json.loads(USER_DEFAULTS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"无法读取默认配置 {USER_DEFAULTS_FILE}: {exc}", file=sys.stderr)
        return DEFAULT_VALUES.copy()
    return clean_values(data if isinstance(data, dict) else None)


def save_user_defaults(values: dict[str, Any]) -> dict[str, Any]:
    defaults = clean_values(values)
    tmp_path = USER_DEFAULTS_FILE.with_suffix(USER_DEFAULTS_FILE.suffix + ".tmp")
    tmp_path.write_text(json.dumps(defaults, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(USER_DEFAULTS_FILE)
    return defaults


def quote_cmd(cmd: list[Any]) -> str:
    return subprocess.list2cmdline([str(x) for x in cmd])


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["NO_COLOR"] = "1"
    python_bin = str(Path(sys.executable).resolve().parent)
    current_path = env.get("PATH") or env.get("Path") or ""
    path_parts = [part for part in current_path.split(os.pathsep) if part]
    if python_bin.lower() not in {part.lower() for part in path_parts}:
        env["PATH"] = os.pathsep.join([python_bin, *path_parts])
    return env


def subprocess_creationflags() -> int:
    if os.name == "nt":
        return subprocess.CREATE_NEW_PROCESS_GROUP
    return 0


def terminate_process_tree(proc: subprocess.Popen[Any], timeout: float = 3.0) -> None:
    if proc.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            proc.terminate()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            proc.kill()
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()


def ps_single_quote(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def remote_train_stop_cmd(values: dict[str, Any], markers: dict[str, Any], export_only: bool = False) -> Optional[list[str]]:
    job_name = str(markers.get("remote_job_name") or "").strip()
    if not job_name:
        return None
    user = str(values.get("remote_train_user") or "").strip()
    host = str(values.get("remote_train_host") or "").strip()
    if not user or not host:
        return None

    port = str(values.get("remote_train_port") or "22").strip() or "22"
    ssh_cmd = ["ssh"]
    if port != "22":
        ssh_cmd += ["-p", port]

    filters = "$_.CommandLine -and $_.CommandLine.Contains($needle) -and $_.ProcessId -ne $current"
    if export_only:
        filters += " -and ($_.CommandLine.Contains(' detect train') -or $_.CommandLine.Contains('detect train') -or $_.CommandLine.Contains('ultralytics'))"
    ps_command = (
        f"$needle={ps_single_quote(job_name)}; "
        "$current=$PID; "
        "Get-CimInstance Win32_Process | "
        f"Where-Object {{ {filters} }} | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    return ssh_cmd + [f"{user}@{host}", f"powershell -NoProfile -ExecutionPolicy Bypass -Command \"{ps_command}\""]



def stop_remote_training(values: dict[str, Any], markers: dict[str, Any], export_only: bool = False) -> None:
    cmd = remote_train_stop_cmd(values, markers, export_only=export_only)
    if not cmd:
        return
    append_log("[remote stop and export requested]\n" if export_only else "[remote stop requested]\n")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
            check=False,
            env=subprocess_env(),
            creationflags=subprocess_creationflags(),
        )
        output = decode_process_output(result.stdout or b"")
        if output.strip():
            append_log(output if output.endswith("\n") else output + "\n")
        append_log(f"[remote stop exit code {result.returncode}]\n")
    except Exception as exc:
        append_log(f"[remote stop error] {exc}\n")



def decode_process_output(raw: bytes) -> str:
    encodings = ["utf-8", locale.getpreferredencoding(False), "gb18030"]
    tried: set[str] = set()

    for encoding in encodings:
        normalized = (encoding or "").lower()
        if not normalized or normalized in tried:
            continue
        tried.add(normalized)
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def append_log(text: str) -> None:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).replace("\r", "\n")
    with STATE_LOCK:
        STATE["logs"].append(text)
        if len(STATE["logs"]) > MAX_LOG_LINES:
            STATE["logs"] = STATE["logs"][-MAX_LOG_LINES:]
    try:
        LATEST_JOB_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LATEST_JOB_LOG_FILE.open("a", encoding="utf-8") as log_file:
            log_file.write(text)
    except OSError:
        pass


def strip_ansi(text: str) -> str:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return text.replace("\r", "").strip()


def parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: str) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def format_gib(value: float) -> str:
    if value <= 0:
        return "0 GB"
    return f"{value:.1f} GB" if value < 100 else f"{value:.0f} GB"


MODEL_RESOURCE_PROFILES: dict[str, dict[str, float]] = {
    "n": {"base_vram": 0.80, "per_image_vram": 0.025, "base_ram": 1.70},
    "s": {"base_vram": 1.10, "per_image_vram": 0.045, "base_ram": 1.95},
    "m": {"base_vram": 1.60, "per_image_vram": 0.080, "base_ram": 2.35},
    "l": {"base_vram": 2.20, "per_image_vram": 0.130, "base_ram": 2.90},
    "x": {"base_vram": 3.00, "per_image_vram": 0.190, "base_ram": 3.60},
}
MAX_IMAGE_SIZE_SAMPLES = 300
LOCAL_RESOURCE_CACHE: dict[str, Any] = {"updated_at": 0.0, "data": None}


def infer_model_size(base_model: str) -> str:

    name = Path(str(base_model or "")).name.lower()
    match = re.search(r"yolo(?:v?\d+)?([nslmx])(?:[._-]|$)", name)
    if match:
        return match.group(1)
    match = re.search(r"(?:^|[._-])([nslmx])(?:[._-]|$)", name)
    return match.group(1) if match else "n"


def read_image_size(path: Path) -> Optional[tuple[int, int]]:
    try:
        with path.open("rb") as fh:
            head = fh.read(32)
            if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24:
                return struct.unpack(">II", head[16:24])
            if head.startswith(b"BM") and len(head) >= 26:
                return struct.unpack("<II", head[18:26])
            if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
                if head[12:16] == b"VP8X" and len(head) >= 30:
                    return (int.from_bytes(head[24:27], "little") + 1, int.from_bytes(head[27:30], "little") + 1)
                if head[12:16] == b"VP8L" and len(head) >= 25:
                    b0, b1, b2, b3 = head[21:25]
                    width = 1 + (((b1 & 0x3F) << 8) | b0)
                    height = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
                    return width, height
            if head.startswith(b"\xff\xd8"):
                fh.seek(2)
                while True:
                    marker = fh.read(2)
                    if len(marker) < 2:
                        return None
                    while marker[0] != 0xFF:
                        marker = marker[1:] + fh.read(1)
                        if len(marker) < 2:
                            return None
                    code = marker[1]
                    if code in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                        data = fh.read(7)
                        if len(data) < 7:
                            return None
                        return struct.unpack(">HH", data[3:7])[::-1]
                    if code in {0xD8, 0xD9, 0x01} or 0xD0 <= code <= 0xD7:
                        continue
                    size_data = fh.read(2)
                    if len(size_data) < 2:
                        return None
                    segment_size = struct.unpack(">H", size_data)[0]
                    if segment_size < 2:
                        return None
                    fh.seek(segment_size - 2, 1)
    except OSError:
        return None
    return None


def split_class_names(raw: Any) -> list[str]:
    return [item.strip() for item in re.split(r"[\r\n,;，；]+", str(raw or "")) if item.strip()]


def direct_files_with_extensions(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions),
        key=lambda path: path.name.lower(),
    )


def choose_dataset_directory(root: Path, explicit: str, candidates: tuple[str, ...], extensions: set[str]) -> Path:
    explicit = str(explicit or "").strip()
    if explicit:
        directory = Path(explicit).expanduser().resolve()
        if directory.is_dir():
            return directory
    for name in candidates:
        directory = root / name if name else root
        if direct_files_with_extensions(directory, extensions):
            return directory.resolve()
    return (root / candidates[0]).resolve()


def _dataset_names(data: dict[str, Any]) -> list[str]:
    raw_names = data.get("names", [])
    if isinstance(raw_names, dict):
        def sort_key(item: tuple[Any, Any]) -> tuple[int, str]:
            key = str(item[0])
            return (int(key), "") if key.isdigit() else (10**9, key)
        return [str(value) for _, value in sorted(raw_names.items(), key=sort_key)]
    if isinstance(raw_names, (list, tuple)):
        return [str(value) for value in raw_names]
    return []


def _resolve_yolo_split(root: Path, yaml_path: Path, split: str, raw_value: Any) -> Optional[Path]:
    if isinstance(raw_value, (list, tuple)):
        raw_value = raw_value[0] if raw_value else ""
    text = str(raw_value or "").strip().replace("\\", "/")
    candidates: list[Path] = []
    if text:
        raw_path = Path(text).expanduser()
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.extend((yaml_path.parent / raw_path, root / raw_path))
            trimmed = re.sub(r"^(\.\./)+", "", text)
            if trimmed != text:
                candidates.append(root / trimmed)
    aliases = ("valid", "val") if split == "val" else (split,)
    for alias in aliases:
        candidates.extend((root / alias / "images", root / "images" / alias))
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = os.path.normcase(str(resolved))
        if key not in seen and resolved.is_dir():
            return resolved
        seen.add(key)
    return None


def inspect_prepared_yolo_dataset(source: Any) -> dict[str, Any]:
    source_text = str(source or "").strip()
    if not source_text:
        raise ValueError("请先选择下载的数据集最外层目录。")
    source_path = Path(source_text).expanduser().resolve()
    if source_path.is_file():
        yaml_path = source_path
        root = yaml_path.parent
    else:
        root = source_path
        if not root.is_dir():
            raise ValueError(f"数据集目录不存在：{root}")
        yaml_path = next((path for path in (root / "data.yaml", root / "dataset.yaml", root / "data.yml") if path.is_file()), None)
        if yaml_path is None:
            raise ValueError("没有在最外层找到 data.yaml 或 dataset.yaml。")
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig")) or {}
    except Exception as exc:
        raise ValueError(f"无法读取数据集 YAML：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("data.yaml 内容不是有效的 YOLO 数据集配置。")

    names = _dataset_names(data)
    split_info: dict[str, dict[str, Any]] = {}
    class_ids: set[int] = set()
    total_boxes = 0
    total_invalid = 0
    for split in ("train", "val", "test"):
        images_dir = _resolve_yolo_split(root, yaml_path, split, data.get(split))
        if images_dir is None:
            if split == "test":
                continue
            raise ValueError(f"无法找到 {split} 图片目录，请检查 data.yaml 和目录布局。")
        labels_dir = images_dir.parent / "labels" if images_dir.name.lower() == "images" else root / split / "labels"
        if not labels_dir.is_dir() and split == "val":
            labels_dir = root / "valid" / "labels"
        if not labels_dir.is_dir():
            raise ValueError(f"无法找到 {split} 标签目录：{labels_dir}")
        images = sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in TRAIN_IMAGE_EXTENSIONS)
        labels = sorted(path for path in labels_dir.rglob("*.txt") if path.is_file())
        image_stems = {path.relative_to(images_dir).with_suffix("").as_posix().casefold() for path in images}
        label_stems = {path.relative_to(labels_dir).with_suffix("").as_posix().casefold() for path in labels}
        boxes = 0
        invalid = 0
        for label_path in labels:
            for line in label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    class_id = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                    if len(coords) not in {4, 8} or class_id < 0 or not all(0.0 <= value <= 1.0 for value in coords):
                        raise ValueError
                except ValueError:
                    invalid += 1
                    continue
                class_ids.add(class_id)
                boxes += 1
        split_info[split] = {
            "images_dir": str(images_dir), "labels_dir": str(labels_dir),
            "image_count": len(images), "label_count": len(labels),
            "matched_count": len(image_stems & label_stems),
            "images_without_labels": len(image_stems - label_stems),
            "labels_without_images": len(label_stems - image_stems),
            "box_count": boxes, "invalid_lines": invalid,
        }
        total_boxes += boxes
        total_invalid += invalid
    if not names and class_ids:
        names = [f"class_{index}" for index in range(max(class_ids) + 1)]
    if not names:
        raise ValueError("data.yaml 中没有类别名称，标签中也没有可识别类别。")
    if max(class_ids, default=-1) >= len(names):
        raise ValueError("TXT 标签中的类别编号超出了 data.yaml 的 names 范围。")
    for required in ("train", "val"):
        current = split_info[required]
        if current["image_count"] == 0 or current["matched_count"] == 0:
            raise ValueError(f"{required} 集没有找到可用的图片/TXT 标签对。")
    return {
        "format": "yolo-split", "root": str(root), "yaml_path": str(yaml_path),
        "class_names": names, "splits": split_info,
        "image_count": sum(item["image_count"] for item in split_info.values()),
        "label_count": sum(item["label_count"] for item in split_info.values()),
        "matched_count": sum(item["matched_count"] for item in split_info.values()),
        "box_count": total_boxes, "invalid_lines": total_invalid,
        "images_without_labels": sum(item["images_without_labels"] for item in split_info.values()),
        "labels_without_images": sum(item["labels_without_images"] for item in split_info.values()),
        "images_dir": split_info["train"]["images_dir"],
        "labels_dir": split_info["train"]["labels_dir"],
        "output_dir": str(root),
    }


def inspect_yolo_txt_dataset(values: dict[str, Any]) -> dict[str, Any]:
    root_text = str(values.get("raw_dataset_root", "")).strip()
    if not root_text:
        raise ValueError("请先选择原始数据集根目录。")
    root = Path(root_text).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"原始数据集目录不存在：{root}")
    if any((root / name).is_file() for name in ("data.yaml", "dataset.yaml", "data.yml")):
        return inspect_prepared_yolo_dataset(root)

    images_dir = choose_dataset_directory(
        root,
        values.get("raw_images_dir", ""),
        ("Main", "images", "Images", "JPEGImages", ""),
        TRAIN_IMAGE_EXTENSIONS,
    )
    labels_dir = choose_dataset_directory(
        root,
        values.get("raw_labels_dir", ""),
        ("Main_labels", "labels", "Labels", "annotations", ""),
        {".txt"},
    )
    images = direct_files_with_extensions(images_dir, TRAIN_IMAGE_EXTENSIONS)
    labels = direct_files_with_extensions(labels_dir, {".txt"})
    if not images:
        raise ValueError(f"没有在图片目录找到支持的图片：{images_dir}")
    if not labels:
        raise ValueError(f"没有在标签目录找到 YOLO TXT 标注：{labels_dir}")

    image_by_stem: dict[str, Path] = {}
    label_by_stem: dict[str, Path] = {}
    duplicate_stems: set[str] = set()
    for path in images:
        key = path.stem.casefold()
        if key in image_by_stem:
            duplicate_stems.add(path.stem)
        image_by_stem[key] = path
    for path in labels:
        key = path.stem.casefold()
        if key in label_by_stem:
            duplicate_stems.add(path.stem)
        label_by_stem[key] = path

    matched_keys = sorted(image_by_stem.keys() & label_by_stem.keys())
    class_ids: set[int] = set()
    box_count = 0
    invalid_lines = 0
    empty_labels = 0
    for key in matched_keys:
        valid_in_file = 0
        for line in label_by_stem[key].read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                if len(parts) != 5:
                    raise ValueError
                class_id = int(parts[0])
                coords = [float(value) for value in parts[1:]]
                if class_id < 0 or not all(0.0 <= value <= 1.0 for value in coords):
                    raise ValueError
            except ValueError:
                invalid_lines += 1
                continue
            class_ids.add(class_id)
            box_count += 1
            valid_in_file += 1
        if valid_in_file == 0:
            empty_labels += 1

    max_class_id = max(class_ids, default=-1)
    configured_names = split_class_names(values.get("raw_class_names", ""))
    class_names = [
        configured_names[index] if index < len(configured_names) else f"class_{index}"
        for index in range(max_class_id + 1)
    ]
    output_text = str(values.get("raw_output_dir", "")).strip()
    output_dir = Path(output_text).expanduser().resolve() if output_text else (root / "converted_voc").resolve()
    return {
        "root": str(root),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "output_dir": str(output_dir),
        "image_count": len(images),
        "label_count": len(labels),
        "matched_count": len(matched_keys),
        "images_without_labels": len(image_by_stem.keys() - label_by_stem.keys()),
        "labels_without_images": len(label_by_stem.keys() - image_by_stem.keys()),
        "empty_labels": empty_labels,
        "box_count": box_count,
        "invalid_lines": invalid_lines,
        "class_ids": sorted(class_ids),
        "class_names": class_names,
        "duplicate_stems": sorted(duplicate_stems),
    }


def convert_yolo_txt_to_voc(values: dict[str, Any]) -> dict[str, Any]:
    info = inspect_yolo_txt_dataset(values)
    if info.get("format") == "yolo-split":
        raise ValueError("这个数据集已经是可训练的 YOLO 格式，请点击“直接导入”，无需转换 XML。")
    if info["duplicate_stems"]:
        raise ValueError("存在重名图片或标注，无法安全转换：" + "、".join(info["duplicate_stems"][:8]))
    if info["matched_count"] == 0:
        raise ValueError("没有找到同名的图片和 TXT 标注。")
    if not info["class_ids"]:
        raise ValueError("没有找到有效的 YOLO 检测框。")

    images_dir = Path(info["images_dir"])
    labels_dir = Path(info["labels_dir"])
    output_root = Path(info["output_dir"])
    output_images = output_root / "images"
    output_annotations = output_root / "annotations"
    source_resolved = {images_dir.resolve(), labels_dir.resolve()}
    if output_root.resolve() in source_resolved or output_images.resolve() in source_resolved or output_annotations.resolve() in source_resolved:
        raise ValueError("输出目录不能覆盖原始图片或标签目录。")

    configured_names = split_class_names(values.get("raw_class_names", ""))
    max_class_id = max(info["class_ids"])
    class_names = [
        configured_names[index] if index < len(configured_names) else f"class_{index}"
        for index in range(max_class_id + 1)
    ]
    overwrite = as_bool(values.get("raw_overwrite", False))
    output_images.mkdir(parents=True, exist_ok=True)
    output_annotations.mkdir(parents=True, exist_ok=True)

    images = direct_files_with_extensions(images_dir, TRAIN_IMAGE_EXTENSIONS)
    labels_by_stem = {
        path.stem.casefold(): path
        for path in direct_files_with_extensions(labels_dir, {".txt"})
    }
    converted = 0
    boxes_written = 0
    skipped_existing = 0
    invalid_lines = 0
    unreadable_images: list[str] = []
    for image_path in images:
        label_path = labels_by_stem.get(image_path.stem.casefold())
        if label_path is None:
            continue
        size = read_image_size(image_path)
        if not size:
            unreadable_images.append(image_path.name)
            continue
        width, height = size
        destination_image = output_images / image_path.name
        xml_path = output_annotations / f"{image_path.stem}.xml"
        if xml_path.exists() and destination_image.exists() and not overwrite:
            skipped_existing += 1
            continue

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = output_images.name
        ET.SubElement(root, "filename").text = image_path.name
        ET.SubElement(root, "path").text = str(destination_image)
        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "YOLO TXT conversion"
        size_node = ET.SubElement(root, "size")
        ET.SubElement(size_node, "width").text = str(width)
        ET.SubElement(size_node, "height").text = str(height)
        ET.SubElement(size_node, "depth").text = "3"
        ET.SubElement(root, "segmented").text = "0"

        file_boxes = 0
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                if len(parts) != 5:
                    raise ValueError
                class_id = int(parts[0])
                xc, yc, bw, bh = (float(value) for value in parts[1:])
                if class_id < 0 or class_id >= len(class_names):
                    raise ValueError
                if not all(0.0 <= value <= 1.0 for value in (xc, yc, bw, bh)) or bw <= 0 or bh <= 0:
                    raise ValueError
            except ValueError:
                invalid_lines += 1
                continue
            xmin = max(0, min(width - 1, int(round((xc - bw / 2) * width))))
            ymin = max(0, min(height - 1, int(round((yc - bh / 2) * height))))
            xmax = max(xmin + 1, min(width, int(round((xc + bw / 2) * width))))
            ymax = max(ymin + 1, min(height, int(round((yc + bh / 2) * height))))
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = class_names[class_id]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            box = ET.SubElement(obj, "bndbox")
            ET.SubElement(box, "xmin").text = str(xmin)
            ET.SubElement(box, "ymin").text = str(ymin)
            ET.SubElement(box, "xmax").text = str(xmax)
            ET.SubElement(box, "ymax").text = str(ymax)
            file_boxes += 1

        if file_boxes == 0:
            continue
        if overwrite or not destination_image.exists():
            shutil.copy2(image_path, destination_image)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
        converted += 1
        boxes_written += file_boxes

    (output_root / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    return {
        **info,
        "output_dir": str(output_root),
        "output_images_dir": str(output_images),
        "output_annotations_dir": str(output_annotations),
        "class_names": class_names,
        "converted_count": converted,
        "boxes_written": boxes_written,
        "skipped_existing": skipped_existing,
        "invalid_lines": invalid_lines,
        "unreadable_images": unreadable_images,
    }


def empty_local_resources() -> dict[str, Any]:
    return {"ram_total_gib": None, "ram_available_gib": None, "gpu_name": "", "gpu_total_gib": None, "gpu_free_gib": None}


def query_local_resources(cache_seconds: float = 5.0) -> dict[str, Any]:
    now = time.time()
    cached = LOCAL_RESOURCE_CACHE.get("data")
    if isinstance(cached, dict) and now - float(LOCAL_RESOURCE_CACHE.get("updated_at") or 0.0) < cache_seconds:
        return cached.copy()

    resources = empty_local_resources()
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        resources["ram_total_gib"] = mem.total / (1024 ** 3)
        resources["ram_available_gib"] = mem.available / (1024 ** 3)
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        lines = (result.stdout or "").strip().splitlines()
        if lines:
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) >= 3:
                resources["gpu_name"] = parts[0]
                resources["gpu_total_gib"] = float(parts[1]) / 1024
                resources["gpu_free_gib"] = float(parts[2]) / 1024
    except Exception:
        pass
    LOCAL_RESOURCE_CACHE["updated_at"] = now
    LOCAL_RESOURCE_CACHE["data"] = resources.copy()
    return resources



def image_dimensions(values: dict[str, Any]) -> tuple[int, int]:
    width = max(32, parse_int(str(values.get("img_width", ""))) or 448)
    height = max(32, parse_int(str(values.get("img_height", ""))) or 448)
    return width, height


def estimate_train_resources(values: dict[str, Any]) -> dict[str, Any]:
    img_width, img_height = image_dimensions(values)
    img_pixels = img_width * img_height
    batch = max(1, parse_int(str(values.get("batch", ""))) or 1)
    cache_mode = str(values.get("train_cache") or "False")
    train_device = str(values.get("train_device") or "cuda")
    train_mode = str(values.get("train_mode") or "local")
    base_model = str(values.get("base_model") or "")

    train_task = str(values.get("train_task") or "detect")
    model_size = infer_model_size(base_model)
    profile = MODEL_RESOURCE_PROFILES[model_size]
    if train_task == "classify":
        profile = {
            "base_vram": profile["base_vram"] * 0.65,
            "per_image_vram": profile["per_image_vram"] * 0.55,
            "base_ram": profile["base_ram"] * 0.8,
        }
    image_count = 0
    image_bytes = 0
    sampled_images = 0
    sampled_pixels = 0
    images_dir = Path(str(values.get("train_images_dir") or "")).expanduser()
    if images_dir.is_dir():
        try:
            for path in images_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in TRAIN_IMAGE_EXTENSIONS:
                    image_count += 1
                    try:
                        image_bytes += path.stat().st_size
                    except OSError:
                        pass
                    if sampled_images < MAX_IMAGE_SIZE_SAMPLES:
                        size = read_image_size(path)
                        if size:
                            width, height = size
                            if width > 0 and height > 0:
                                sampled_images += 1
                                sampled_pixels += width * height
        except OSError:
            pass

    img_scale = img_pixels / (640 * 640)
    avg_pixels = sampled_pixels / sampled_images if sampled_images else img_pixels
    raw_cache_gib = image_count * avg_pixels * 3 / (1024 ** 3) if image_count else 0.0
    disk_cache_gib = raw_cache_gib * 1.20
    cache_ram_gib = raw_cache_gib * 1.15 if cache_mode == "True" else 0.0
    cache_disk_gib = disk_cache_gib if cache_mode == "disk" else 0.0

    batch_tensor_gib = batch * img_pixels * 3 * 4 / (1024 ** 3)
    loader_ram_gib = min(4.0, batch_tensor_gib * 2.2 + 0.35)
    augment_ram_gib = min(3.0, batch_tensor_gib * 1.4 + 0.25)
    safety_ram_gib = max(0.6, (profile["base_ram"] + loader_ram_gib + augment_ram_gib + cache_ram_gib) * 0.15)
    train_ram_gib = profile["base_ram"] + loader_ram_gib + augment_ram_gib + safety_ram_gib
    total_ram_gib = train_ram_gib + cache_ram_gib

    raw_vram_gib = profile["base_vram"] + batch * profile["per_image_vram"] * img_scale
    train_vram_gib = raw_vram_gib * 1.18 + 0.25 if train_device == "cuda" else 0.0

    local = query_local_resources() if train_mode == "local" else empty_local_resources()
    risk = "safe"
    risk_text = "资源预估"
    if train_device == "cuda" and train_mode == "local" and local.get("gpu_free_gib") is not None:

        free_vram = float(local["gpu_free_gib"])
        if train_vram_gib > free_vram * 0.95:
            risk, risk_text = "danger", "显存可能不足"
        elif train_vram_gib > free_vram * 0.75:
            risk, risk_text = "warning", "显存接近上限"
    if train_mode == "local" and local.get("ram_available_gib") is not None:

        free_ram = float(local["ram_available_gib"])
        if total_ram_gib > free_ram * 0.95:
            risk, risk_text = "danger", "内存可能不足"
        elif risk == "safe" and total_ram_gib > free_ram * 0.75:
            risk, risk_text = "warning", "内存接近上限"

    notes = [
        f"模型档位={model_size}，显存按 base {profile['base_vram']:.1f}GB + batch*{profile['per_image_vram']:.3f}GB*(imgsz/640)^2 估算。",
        f"RAM=训练 {format_gib(train_ram_gib)} + cache {format_gib(cache_ram_gib)}。",
    ]
    if sampled_images:
        notes.append(f"cache 按真实图片尺寸采样 {sampled_images}/{image_count} 张估算。")
    elif not image_count:
        notes.append("未读取到图片数量，cache 部分按 0 估算。")
    else:
        notes.append("未能读取图片尺寸，cache 暂按 ImgSize 估算。")
    if cache_mode == "True":
        notes.append("内存 cache 会额外占用 RAM。")
    elif cache_mode == "disk":
        notes.append("disk cache 主要额外占用磁盘。")
    if train_device != "cuda":
        notes.append("当前选择 CPU 训练，显存按 0 估算。")
    elif train_mode != "local":
        notes.append("远程训练模式下只估算训练需求，不检测本机 GPU 余量。")
    elif local.get("gpu_free_gib") is not None:
        notes.append(f"当前 GPU 可用约 {format_gib(float(local['gpu_free_gib']))}。")


    return {
        "image_count": image_count,
        "image_bytes": image_bytes,
        "sampled_images": sampled_images,
        "avg_image_mp": round(avg_pixels / 1_000_000, 2) if image_count else 0,
        "img_size": f"{img_width} × {img_height}",
        "batch": batch,
        "base_model": base_model,
        "model_size": model_size,
        "train_mode": train_mode,
        "cache_mode": cache_mode,

        "risk": risk,
        "risk_text": risk_text,
        "ram_gib": round(total_ram_gib, 2),
        "vram_gib": round(train_vram_gib, 2),
        "train_ram_gib": round(train_ram_gib, 2),
        "cache_ram_gib": round(cache_ram_gib, 2),
        "cache_disk_gib": round(cache_disk_gib, 2),
        "ram_text": format_gib(total_ram_gib),
        "vram_text": "0 GB" if train_device != "cuda" else format_gib(train_vram_gib),
        "cache_text": format_gib(cache_ram_gib) if cache_mode == "True" else (format_gib(cache_disk_gib) if cache_mode == "disk" else "0 GB"),
        "local_resources": local,
        "note": " ".join(notes),
    }



def reset_train_progress_locked() -> None:

    STATE["train_progress"] = {
        "phase": "idle",
        "task": "",
        "epoch": 0,
        "total_epochs": 0,
        "batch": 0,
        "total_batches": 0,
        "percent": 0.0,
        "gpu_mem": "",
        "loss": None,
        "box_loss": None,
        "cls_loss": None,
        "dfl_loss": None,
        "instances": None,
        "size": None,
        "speed": "",
        "elapsed": "",
        "eta": "",
        "val_batch": 0,
        "val_total": 0,
        "val_percent": 0.0,
        "metrics": {},
        "history": [],
        "updated_at": "",
    }


def append_epoch_history(progress: dict[str, Any]) -> None:
    epoch = progress.get("epoch")
    if not epoch:
        return
    history = progress.setdefault("history", [])
    item = {
        "epoch": epoch,
        "loss": progress.get("loss"),
        "box_loss": progress.get("box_loss"),
        "cls_loss": progress.get("cls_loss"),
        "dfl_loss": progress.get("dfl_loss"),
        "precision": progress.get("metrics", {}).get("precision"),
        "recall": progress.get("metrics", {}).get("recall"),
        "map50": progress.get("metrics", {}).get("map50"),
        "map50_95": progress.get("metrics", {}).get("map50_95"),
        "top1_acc": progress.get("metrics", {}).get("top1_acc"),
        "top5_acc": progress.get("metrics", {}).get("top5_acc"),
    }
    if history and history[-1].get("epoch") == epoch:
        history[-1] = item
    else:
        history.append(item)
    if len(history) > 500:
        del history[:-500]


def parse_train_output(line: str) -> None:
    clean = strip_ansi(line)
    if not clean:
        return
    now = time.strftime("%H:%M:%S")
    detect_match = re.search(
        r"(?P<epoch>\d+)\s*/\s*(?P<total_epochs>\d+)\s+(?P<gpu_mem>\S+)\s+"
        r"(?P<box_loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"(?P<cls_loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"(?P<dfl_loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"(?P<instances>\d+)\s+(?P<size>\d+)\s*:\s*(?P<percent>\d+(?:\.\d+)?)%.*?"
        r"(?P<batch>\d+)\s*/\s*(?P<total_batches>\d+)"
        r"(?:\s+(?P<speed>[\d.]+it/s))?(?:\s+(?P<elapsed>[^<\s]+))?(?:<(?P<eta>\S+))?",
        clean,
    )
    if detect_match:
        data = detect_match.groupdict()
        with STATE_LOCK:
            progress = STATE["train_progress"]
            progress.update({
                "phase": "train", "task": "detect", "epoch": parse_int(data["epoch"]) or 0,
                "total_epochs": parse_int(data["total_epochs"]) or 0, "batch": parse_int(data["batch"]) or 0,
                "total_batches": parse_int(data["total_batches"]) or 0, "percent": parse_float(data["percent"]) or 0.0,
                "gpu_mem": data.get("gpu_mem") or "", "loss": None, "box_loss": parse_float(data["box_loss"]),
                "cls_loss": parse_float(data["cls_loss"]), "dfl_loss": parse_float(data["dfl_loss"]),
                "instances": parse_int(data["instances"]), "size": parse_int(data["size"]),
                "speed": data.get("speed") or "", "elapsed": data.get("elapsed") or "", "eta": data.get("eta") or "",
                "updated_at": now,
            })
            if progress["percent"] >= 100:
                append_epoch_history(progress)
        return

    classify_match = re.search(
        r"(?P<epoch>\d+)\s*/\s*(?P<total_epochs>\d+)\s+(?P<gpu_mem>\S+)\s+"
        r"(?P<loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+(?P<instances>\d+)\s+(?P<size>\d+)\s*:\s*"
        r"(?P<percent>\d+(?:\.\d+)?)%.*?(?P<batch>\d+)\s*/\s*(?P<total_batches>\d+)"
        r"(?:\s+(?P<speed>[\d.]+it/s))?(?:\s+(?P<elapsed>[^<\s]+))?(?:<(?P<eta>\S+))?",
        clean,
    )
    if classify_match:
        data = classify_match.groupdict()
        with STATE_LOCK:
            progress = STATE["train_progress"]
            progress.update({
                "phase": "train", "task": "classify", "epoch": parse_int(data["epoch"]) or 0,
                "total_epochs": parse_int(data["total_epochs"]) or 0, "batch": parse_int(data["batch"]) or 0,
                "total_batches": parse_int(data["total_batches"]) or 0, "percent": parse_float(data["percent"]) or 0.0,
                "gpu_mem": data.get("gpu_mem") or "", "loss": parse_float(data["loss"]), "box_loss": None,
                "cls_loss": None, "dfl_loss": None, "instances": parse_int(data["instances"]),
                "size": parse_int(data["size"]), "speed": data.get("speed") or "", "elapsed": data.get("elapsed") or "",
                "eta": data.get("eta") or "", "updated_at": now,
            })
            if progress["percent"] >= 100:
                append_epoch_history(progress)
        return

    val_match = re.search(r"Class\s+Images\s+Instances.*?:\s*(?P<percent>\d+(?:\.\d+)?)%.*?(?P<batch>\d+)\s*/\s*(?P<total>\d+)", clean)
    if val_match:
        data = val_match.groupdict()
        with STATE_LOCK:
            STATE["train_progress"].update({"phase": "val", "task": "detect", "val_batch": parse_int(data["batch"]) or 0, "val_total": parse_int(data["total"]) or 0, "val_percent": parse_float(data["percent"]) or 0.0, "updated_at": now})
        return

    classify_val_match = re.search(r"classes\s+top1_acc\s+top5_acc\s*:\s*(?P<percent>\d+(?:\.\d+)?)%.*?(?P<batch>\d+)\s*/\s*(?P<total>\d+)", clean, re.IGNORECASE)
    if classify_val_match:
        data = classify_val_match.groupdict()
        with STATE_LOCK:
            STATE["train_progress"].update({"phase": "val", "task": "classify", "val_batch": parse_int(data["batch"]) or 0, "val_total": parse_int(data["total"]) or 0, "val_percent": parse_float(data["percent"]) or 0.0, "updated_at": now})
        return

    detect_metric_match = re.match(r"^all\s+(?P<images>\d+)\s+(?P<instances>\d+)\s+(?P<precision>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+(?P<recall>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+(?P<map50>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+(?P<map50_95>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$", clean)
    if detect_metric_match:
        data = detect_metric_match.groupdict()
        metrics = {"images": parse_int(data["images"]), "instances": parse_int(data["instances"]), "precision": parse_float(data["precision"]), "recall": parse_float(data["recall"]), "map50": parse_float(data["map50"]), "map50_95": parse_float(data["map50_95"])}
        with STATE_LOCK:
            progress = STATE["train_progress"]
            progress.update({"phase": "metrics", "task": "detect", "metrics": metrics, "val_percent": 100.0, "updated_at": now})
            append_epoch_history(progress)
        return

    classify_metric_match = re.match(r"^all\s+(?P<top1_acc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+(?P<top5_acc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$", clean, re.IGNORECASE)
    if classify_metric_match:
        data = classify_metric_match.groupdict()
        metrics = {"top1_acc": parse_float(data["top1_acc"]), "top5_acc": parse_float(data["top5_acc"])}
        with STATE_LOCK:
            progress = STATE["train_progress"]
            progress.update({"phase": "metrics", "task": "classify", "metrics": metrics, "val_percent": 100.0, "updated_at": now})
            append_epoch_history(progress)



def parse_marker(line: str) -> None:
    markers = {
        "REMOTE_JOB_NAME=": "remote_job_name",
        "REMOTE_WORK_DIR=": "remote_work_dir",
        "TRAIN_MODEL_ONNX=": "model_path",
        "TRAIN_OUTPUT_DIR=": "train_output_dir",
        "TRAIN_MANIFEST=": "training_manifest",
        "TRAIN_PLOT_DIR=": "train_plot_dir",
        "TRAIN_CLASSES=": "classes_path",
        "TRAIN_CALIB_DIR=": "calib_dir",
        "TRAIN_TEST_IMAGE=": "test_image",
        "TRAIN_MODEL_PT=": "test_model",
        "DEPLOY_ARTIFACT=": "deploy_artifact",
        "DEPLOY_MANIFEST=": "deploy_manifest",
        "TRAIN_STOP_EXPORT_REQUESTED=": "stop_export",
        "TEST_OUTPUT_IMAGE=": "test_output_image",
        "BASE_MODEL=": "base_model",
        "MODEL_SHA256=": "model_sha256",

        "CONVERT_FINAL_TAR=": "convert_final_tar",
        "CONVERT_PACKAGE=": "convert_package",
        "CONVERT_WORK_DIR=": "convert_work_dir",
    }
    with STATE_LOCK:
        for prefix, key in markers.items():
            if line.startswith(prefix):
                value = line[len(prefix):].strip()
                STATE["markers"][key] = value
                if key in STATE["values"]:
                    STATE["values"][key] = value
                if key == "test_model" and not STATE["values"].get("model_path"):
                    STATE["values"]["model_path"] = value
                if key == "test_model":
                    STATE["values"]["deploy_model"] = value
                if key in {"training_manifest", "deploy_manifest"}:
                    try:
                        with MODEL_REGISTRY_LOCK:
                            register_asset_manifest(MODEL_REGISTRY_FILE, value)
                    except OSError as exc:
                        append_log(f"[模型资产] 无法更新本机索引：{exc}\n")
                if key == "base_model":
                    try:
                        save_user_defaults(STATE["values"])
                    except OSError as exc:
                        append_log(f"[基础模型] 无法保存默认模型：{exc}\n")
                return


def build_common_args(values: dict[str, Any], stage: str) -> list[Any]:
    return [
        sys.executable,
        str(WORKFLOW_SCRIPT),
        "--stage", stage,
        "--dataset-root", values["dataset_root"],
        "--output-root", str(TRAINING_RUNS_DIR),
        "--train-task", values["train_task"],
        "--images-dir", values["train_images_dir"],
        "--annotations-dir", values["train_annotations_dir"],
        "--dataset-yaml", values.get("prepared_dataset_yaml", ""),
        "--train-ratio-percent", values["train_ratio_percent"],
        "--val-ratio-percent", values["val_ratio_percent"],
        "--img-width", values["img_width"],
        "--img-height", values["img_height"],
        "--image-resize-mode", values["image_resize_mode"],

        "--epochs", values["epochs"],
        "--batch", values["batch"],
        "--workers", values["train_workers"],
        "--patience", values["patience"],
        "--lr0", values["lr0"],
        "--conda-env", values["conda_env"],
        "--base-model", values["base_model"],
        "--torch-cuda", values["torch_cuda"],
        "--train-device", values["train_device"],
        "--train-cache", values["train_cache"],
        "--stop-export-signal", str(STOP_EXPORT_SIGNAL_FILE),
        "--project-name", values["project_name"],


        "--model-name", values["model_name"],
        "--operator-mode", values["operator_mode"],
        "--train-mode", values["train_mode"],
        "--remote-train-user", values["remote_train_user"],
        "--remote-train-host", values["remote_train_host"],
        "--remote-train-port", values["remote_train_port"],
        "--remote-train-work-dir", values["remote_train_work_dir"],
        "--vm-user", values["vm_user"],
        "--vm-host", values["vm_host"],
        "--vm-work-dir", values["vm_work_dir"],
    ]


def build_train_cmd(values: dict[str, Any]) -> list[Any]:
    return build_common_args(values, "train")


def build_convert_cmd(values: dict[str, Any]) -> list[Any]:
    cmd = build_common_args(values, "convert") + [
        "--model-path", values["model_path"],
        "--classes-path", values["classes_path"],
        "--calib-dir", values["calib_dir"],
        "--test-image", values["test_image"],
    ]
    if as_bool(values["skip_vm_convert"]):
        cmd.append("--skip-vm-convert")
    return cmd


def build_export_cmd(values: dict[str, Any]) -> list[Any]:
    cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--model", values["deploy_model"],
        "--target", values["deployment_target"],
        "--format", values["export_format"],
        "--imgsz", f"{values['img_height']},{values['img_width']}",
    ]
    for option, key in (
        ("--chip", "export_chip"),
        ("--data", "export_data"),
        ("--classes", "classes_path"),
        ("--output-dir", "export_output_dir"),
    ):
        if values.get(key, "").strip():
            cmd += [option, values[key]]
    if as_bool(values.get("export_int8")):
        cmd.append("--int8")
    return cmd


def build_test_cmd(values: dict[str, Any]) -> list[Any]:
    cmd = [
        sys.executable,
        str(TEST_SCRIPT),
        "--model", values["test_model"],
        "--source", values["test_source"],
        "--camera-index", values["camera_index"],
        "--conf", values["conf"],
    ]
    if values["test_source"] == "image":
        cmd += ["--image", values["test_image_file"]]
    elif values["test_source"] == "folder":
        cmd += ["--folder", values["test_image_folder"]]
        if values["test_output_dir"].strip():
            cmd += ["--output-dir", values["test_output_dir"]]
    return cmd


def build_label_cmd(values: dict[str, Any]) -> list[Any]:
    source_type = values.get("label_source_type", "video")
    cmd = [
        sys.executable,
        str(LABEL_SCRIPT),
        "--labels", values["label_name"],
        "--interval", values["label_interval"],
        "--images-dir", values["label_images_dir"],
        "--annotations-dir", values["label_annotations_dir"],
        "--prefix", values["label_prefix"],
        "--tracker", values["label_tracker"],
        "--start-frame", values["label_start_frame"],
        "--max-frames", values["label_max_frames"],
        "--display-scale", values["label_display_scale"],
        "--jpeg-quality", values["label_jpeg_quality"],
    ]
    if source_type == "images":
        cmd += ["--images-input-dir", values["label_images_input_dir"]]
    elif source_type == "camera":
        cmd += ["--video", values["label_camera_index"]]
    else:
        cmd += ["--video", values["label_video"]]
    return cmd



def command_for(action: str, values: dict[str, Any]) -> list[Any]:

    if action == "train":
        return build_train_cmd(values)
    if action == "convert":
        return build_convert_cmd(values)
    if action == "export":
        return build_export_cmd(values)
    if action == "test":
        return build_test_cmd(values)
    if action == "label":
        return build_label_cmd(values)
    if action == "model_download":
        cmd: list[Any] = [
            sys.executable,
            str(BASE_MODEL_DOWNLOAD_SCRIPT),
            "--model", values["model_download_name"],
            "--output-dir", str(MODEL_ASSETS_DIR / "base-models"),
        ]
        if as_bool(values.get("model_download_force")):
            cmd.append("--force")
        return cmd
    if action == "annotation_personal":
        return [sys.executable, str(ANNOTATION_SERVICE_SCRIPT), "start", "--no-browser"]
    if action == "annotation_share":
        return [sys.executable, str(ANNOTATION_SERVICE_SCRIPT), "start", "--share", "--no-browser"]
    if action == "annotation_stop":
        return [sys.executable, str(ANNOTATION_SERVICE_SCRIPT), "stop"]
    if action == "train_ssh":

        return ["ssh", "-p", values["remote_train_port"], f"{values['remote_train_user']}@{values['remote_train_host']}", "hostname"]
    if action == "vm_ssh":
        return ["ssh", f"{values['vm_user']}@{values['vm_host']}", "hostname"]
    raise ValueError(f"unknown action: {action}")


def validate(action: str, values: dict[str, Any]) -> None:
    if action == "train":
        if not str(values.get("dataset_root", "")).strip():
            raise ValueError("请先选择训练输出目录。")
        if not str(values.get("train_images_dir", "")).strip():
            raise ValueError("请先选择训练图片目录。")
        dataset = Path(values["dataset_root"])
        images_dir = Path(values["train_images_dir"])
        annotations_dir = Path(values["train_annotations_dir"]) if str(values.get("train_annotations_dir", "")).strip() else None
        prepared_yaml = str(values.get("prepared_dataset_yaml", "")).strip()
        if not dataset.is_dir():
            raise ValueError("Dataset Root 必须是有效文件夹，用于保存训练输出。")
        if not images_dir.is_dir():
            raise ValueError("Images Dir 必须是有效图片文件夹。")
        train_task = values.get("train_task", "detect")
        if train_task not in {"detect", "classify"}:
            raise ValueError("训练任务必须为目标检测或图像分类。")
        if train_task == "detect":
            if prepared_yaml:
                inspect_prepared_yolo_dataset(prepared_yaml)
            else:
                if annotations_dir is None:
                    raise ValueError("请选择 XML 标注目录，或直接导入带 data.yaml 的 YOLO 数据集。")
                if not annotations_dir.is_dir():
                    raise ValueError("目标检测需要有效的 XML 标注文件夹。")
        if train_task == "classify":
            class_dirs = [path for path in images_dir.iterdir() if path.is_dir()]
            if len(class_dirs) < 2:
                raise ValueError("图像分类的 Images Dir 下至少需要两个类别子文件夹。")
        train_ratio = parse_float(str(values.get("train_ratio_percent", "")))
        val_ratio = parse_float(str(values.get("val_ratio_percent", "")))
        if train_ratio is None or not 1 <= train_ratio < 100:
            raise ValueError("训练集比例必须在 1% 到 99% 之间。")
        if val_ratio is None or not 1 <= val_ratio < 100:
            raise ValueError("验证集比例必须在 1% 到 99% 之间。")
        if train_ratio + val_ratio > 100:
            raise ValueError("训练集与验证集比例之和不能超过 100%。")
        epochs = parse_int(str(values.get("epochs", "")))
        if epochs is None or epochs < 1:
            raise ValueError("训练轮数 Epochs 必须是大于等于 1 的整数。")
        batch = parse_int(str(values.get("batch", "")))
        if batch is None or batch == 0 or batch < -1:
            raise ValueError("Batch 必须是正整数，或使用 -1 自动选择。")
        workers = parse_int(str(values.get("train_workers", "")))
        if workers is None or not 0 <= workers <= 16:
            raise ValueError("数据加载进程必须是 0 到 16 的整数；多数电脑建议使用 2 到 4。")
        patience = parse_int(str(values.get("patience", "")))
        if patience is None or patience < 0:
            raise ValueError("早停耐心值必须是大于等于 0 的整数。")
        lr0 = parse_float(str(values.get("lr0", "")))
        if lr0 is None or lr0 <= 0:
            raise ValueError("初始学习率 Lr0 必须是大于 0 的数字。")
        for key, label in (("img_width", "图片宽度"), ("img_height", "图片高度")):
            value = parse_int(str(values.get(key, "")))
            if value is None or value < 32 or value % 32:
                raise ValueError(f"{label}必须是大于等于 32 的 32 倍数，以兼容 YOLO 和常见边缘设备。")
        if values.get("image_resize_mode") not in {"crop", "letterbox", "stretch"}:
            raise ValueError("图片适配方式必须为裁剪、等比缩放或拉伸。")
        if not WORKFLOW_SCRIPT.exists():
            raise ValueError(f"未找到 host_train_export.py: {WORKFLOW_SCRIPT}")
        if values["train_mode"] == "remote-windows" and not values["remote_train_user"].strip():
            raise ValueError("远程 Windows 训练需要填写 Remote User。")
    elif action == "convert":
        if values.get("train_task") == "classify":
            raise ValueError("当前 MaixCAM 转换流程仅支持目标检测 ONNX，分类模型训练完成后可直接使用 .pt 或 .onnx。")
        for key, label in (("model_path", "ONNX 模型"), ("classes_path", "classes.txt"), ("calib_dir", "校准图片目录")):
            if not values[key].strip():
                raise ValueError(f"{label} 不能为空。")
    elif action == "export":
        if not EXPORT_SCRIPT.is_file():
            raise ValueError(f"未找到多平台导出脚本：{EXPORT_SCRIPT}")
        source = Path(values.get("deploy_model", "").strip()).expanduser()
        if not source.is_file():
            raise ValueError("请选择训练完成的 .pt 或已有 .onnx 模型。")
        if source.suffix.lower() not in {".pt", ".onnx"}:
            raise ValueError("部署模型目前支持 .pt 或 .onnx。")
        if values.get("deployment_target") not in {
            "generic_onnx", "raspberry_pi", "rockchip_rknn", "drobotics_rdk",
            "maixcam", "nvidia_jetson", "intel_openvino",
        }:
            raise ValueError("请选择有效的部署平台。")
        if as_bool(values.get("export_int8")) and not values.get("export_data", "").strip():
            raise ValueError("INT8 导出需要填写 data.yaml 作为代表性校准数据。")
    elif action == "test":
        if not TEST_SCRIPT.exists():
            raise ValueError(f"未找到 model_test.py: {TEST_SCRIPT}")
        if not values["test_model"].strip():
            raise ValueError("测试模型不能为空。")
        if values["test_source"] == "image" and not values["test_image_file"].strip():
            raise ValueError("选择单张图片测试时必须填写图片路径。")
        if values["test_source"] == "folder" and not values["test_image_folder"].strip():
            raise ValueError("选择图片文件夹测试时必须填写文件夹路径。")
    elif action == "label":
        if not LABEL_SCRIPT.exists():
            raise ValueError(f"未找到 video_track_label.py: {LABEL_SCRIPT}")
        source_type = values.get("label_source_type", "video")
        if source_type == "images":
            image_dir = Path(values.get("label_images_input_dir", "").strip())
            if not image_dir.is_dir():
                raise ValueError("图片集文件夹必须是有效文件夹。")
        elif source_type == "camera":
            camera_index = values.get("label_camera_index", "").strip()
            if not camera_index.isdigit():
                raise ValueError("摄像头索引必须是非负整数，例如 0 或 1。")
        elif not values["label_video"].strip():
            raise ValueError("请先从视频队列选择或填写视频路径。")
        if not values["label_name"].strip():
            raise ValueError("请先填写至少一个标签名称。")
    elif action == "model_download":
        if not BASE_MODEL_DOWNLOAD_SCRIPT.is_file():
            raise ValueError("基础模型下载组件不存在，请重新安装平台。")
        validate_model_name(str(values.get("model_download_name") or ""))
    elif action in {"annotation_personal", "annotation_share", "annotation_stop"}:
        if not ANNOTATION_SERVICE_SCRIPT.is_file():
            raise ValueError(f"未找到协作标注服务管理器：{ANNOTATION_SERVICE_SCRIPT}")
    elif action == "train_ssh":
        if not values["remote_train_user"].strip():
            raise ValueError("请先填写 Remote User。")

    elif action == "vm_ssh":
        if not values["vm_user"].strip() or not values["vm_host"].strip():
            raise ValueError("请先填写 VM User 和 VM Host/IP。")


def train_preflight(values: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, str]] = []

    def add(check_id: str, label: str, status: str, detail: str) -> None:
        checks.append({"id": check_id, "label": label, "status": status, "detail": detail})

    try:
        validate("train", values)
        add("config", "训练参数", "ok", "必填项和数值格式正确")
    except Exception as exc:
        add("config", "训练参数", "error", str(exc))

    train_ratio = parse_float(str(values.get("train_ratio_percent", "")))
    val_ratio = parse_float(str(values.get("val_ratio_percent", "")))
    split_valid = train_ratio is not None and val_ratio is not None and 1 <= train_ratio < 100 and 1 <= val_ratio < 100 and train_ratio + val_ratio <= 100
    test_ratio = max(0.0, 100.0 - (train_ratio or 0.0) - (val_ratio or 0.0))
    if split_valid:
        add("split", "数据集划分", "ok", f"训练 {train_ratio:g}% / 验证 {val_ratio:g}% / 测试 {test_ratio:g}%")
    else:
        add("split", "数据集划分", "error", "训练集与验证集都至少为 1%，且两者之和不能超过 100%")

    img_width, img_height = image_dimensions(values)
    workers = parse_int(str(values.get("train_workers", ""))) or 0
    if values.get("train_task") == "detect" and img_width != img_height:
        add("shape", "实际训练张量", "ok", f"{img_width}×{img_height}，自动启用矩形训练；不会再补成正方形")
        add("amp", "稳定性保护", "ok", "矩形训练自动关闭当前环境中不稳定的半精度；实测吞吐仍高于原配置")
    else:
        add("shape", "实际训练张量", "ok", f"{img_width}×{img_height}")
    if workers == 0:
        add("loader", "数据加载", "warn", "workers=0 会让 GPU 等待 CPU；RTX 4060 建议设为 4")
    else:
        add("loader", "数据加载", "ok", f"{workers} 个数据加载进程")
    patience = parse_int(str(values.get("patience", ""))) or 0
    if patience == 0:
        add("stopping", "停止方式", "ok", "自动早停已关闭，只在你手动点击停止时结束")
    else:
        add("stopping", "停止方式", "warn", f"连续 {patience} 轮没有提升会自动停止")

    dataset_root_text = str(values.get("dataset_root", "")).strip()
    dataset_root = Path(dataset_root_text).resolve() if dataset_root_text else None
    if dataset_root and dataset_root.is_dir() and os.access(dataset_root, os.W_OK):
        add("output", "输出目录", "ok", str(dataset_root))
    elif dataset_root and dataset_root.is_dir():
        add("output", "输出目录", "error", f"没有写入权限：{dataset_root}")
    elif dataset_root_text:
        add("output", "输出目录", "error", f"文件夹不存在：{dataset_root_text}")
    else:
        add("output", "输出目录", "error", "尚未选择训练输出目录")

    prepared_yaml = str(values.get("prepared_dataset_yaml", "")).strip()
    if prepared_yaml and values.get("train_task", "detect") == "detect":
        try:
            info = inspect_prepared_yolo_dataset(prepared_yaml)
            split_parts = []
            for name, label in (("train", "训练"), ("val", "验证"), ("test", "测试")):
                if name in info["splits"]:
                    split_parts.append(f"{label} {info['splits'][name]['matched_count']}")
            add("dataset", "YOLO 数据集", "ok", f"沿用原划分：{' / '.join(split_parts)}；{len(info['class_names'])} 个类别，无需 XML")
        except Exception as exc:
            add("dataset", "YOLO 数据集", "error", str(exc))

    images_dir_text = str(values.get("train_images_dir", "")).strip()
    images_dir = Path(images_dir_text).resolve() if images_dir_text else None
    train_task = values.get("train_task", "detect")
    if images_dir is None or not images_dir.is_dir():
        detail = f"文件夹不存在：{images_dir_text}" if images_dir_text else "尚未选择训练图片目录"
        add("dataset", "训练数据", "error", detail)
    elif train_task == "detect" and not prepared_yaml:
        annotations_dir_text = str(values.get("train_annotations_dir", "")).strip()
        annotations_dir = Path(annotations_dir_text).resolve() if annotations_dir_text else None
        if annotations_dir is None or not annotations_dir.is_dir():
            detail = f"文件夹不存在：{annotations_dir_text}" if annotations_dir_text else "尚未选择 XML 标注目录"
            add("dataset", "训练数据", "error", detail)
            annotations_dir = None
        image_paths = [
            path for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in TRAIN_IMAGE_EXTENSIONS
        ]
        xml_paths = [path for path in annotations_dir.rglob("*.xml") if path.is_file()] if annotations_dir else []
        image_stems = {path.stem.lower() for path in image_paths}
        xml_stems = {path.stem.lower() for path in xml_paths}
        matched = len(image_stems & xml_stems)
        if annotations_dir is None:
            pass
        elif matched == 0:
            add("dataset", "训练数据", "error", "没有找到同名的图片和 XML 标注")
        elif matched < (3 if test_ratio > 0 else 2):
            add("dataset", "训练数据", "error", f"当前比例启用了 {'3 个' if test_ratio > 0 else '2 个'}数据子集，但只有 {matched} 对有效图片/XML")
        elif matched < 10:
            add("dataset", "训练数据", "warn", f"找到 {matched} 对图片/XML；可以训练，但建议至少准备几十张")
        else:
            unmatched = len(image_stems - xml_stems) + len(xml_stems - image_stems)
            detail = f"找到 {matched} 对图片/XML"
            if unmatched:
                detail += f"，另有 {unmatched} 个未匹配文件会被忽略"
            add("dataset", "训练数据", "ok", detail)
    elif train_task == "classify" and images_dir is not None:
        class_counts = {
            path.name: sum(
                1 for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in TRAIN_IMAGE_EXTENSIONS
            )
            for path in images_dir.iterdir()
            if path.is_dir()
        }
        minimum_per_class = 3 if test_ratio > 0 else 2
        empty_classes = [name for name, count in class_counts.items() if count < minimum_per_class]
        total_images = sum(class_counts.values())
        if empty_classes:
            add("dataset", "训练数据", "error", f"这些类别少于 {minimum_per_class} 张图片，无法覆盖全部数据子集：{', '.join(empty_classes)}")
        elif total_images < 20:
            add("dataset", "训练数据", "warn", f"{len(class_counts)} 个类别，共 {total_images} 张图片；可以训练，但数据较少")
        else:
            add("dataset", "训练数据", "ok", f"{len(class_counts)} 个类别，共 {total_images} 张图片")

    base_model = str(values.get("base_model", "")).strip()
    model_path = Path(base_model)
    if not model_path.is_absolute():
        model_path = SCRIPT_ROOT / model_path
    if model_path.is_file():
        add("model", "基础模型", "ok", str(model_path.resolve()))
    elif re.fullmatch(r"yolo[\w.-]+\.pt", Path(base_model).name, re.IGNORECASE):
        add("model", "基础模型", "warn", f"{base_model} 尚未下载，首次训练时需要联网")
    else:
        add("model", "基础模型", "error", f"找不到模型：{base_model}")

    if values.get("train_mode") == "remote-windows":
        add("runtime", "训练环境", "warn", "远程模式已选择；请先点击“测试训练 SSH”确认连接")
    else:
        missing = [
            module for module in ("torch", "ultralytics", "onnx", "onnxsim", "onnxslim", "onnxruntime")
            if importlib.util.find_spec(module) is None
        ]
        yolo_exe = Path(sys.executable).parent / ("yolo.exe" if os.name == "nt" else "yolo")
        if missing:
            add("runtime", "Python 环境", "error", f"缺少依赖：{', '.join(missing)}")
        elif not yolo_exe.is_file():
            add("runtime", "Python 环境", "error", f"找不到 YOLO 命令：{yolo_exe}")
        else:
            add("runtime", "Python 环境", "ok", f"Python {sys.version_info.major}.{sys.version_info.minor}，YOLO 命令可用")

        if values.get("train_device") == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    add("device", "训练设备", "ok", f"{gpu_name}，{memory_gib:.1f} GB 显存")
                else:
                    add("device", "训练设备", "error", "当前 PyTorch 无法使用 CUDA；请改用 CPU 或修复 CUDA 环境")
            except Exception as exc:
                add("device", "训练设备", "error", f"CUDA 检查失败：{exc}")
        else:
            add("device", "训练设备", "warn", "当前选择 CPU，训练速度会明显较慢")

    ready = not any(item["status"] == "error" for item in checks)
    if ready:
        warnings = sum(item["status"] == "warn" for item in checks)
        summary = "训练前检查通过" + (f"，有 {warnings} 项提示" if warnings else "")
    else:
        first_error = next(item["detail"] for item in checks if item["status"] == "error")
        summary = f"暂时不能开始训练：{first_error}"
    return {"ready": ready, "checks": checks, "summary": summary}


def start_job(action: str, values: dict[str, Any]) -> None:
    global current_proc, stop_requested

    validate(action, values)
    cmd = command_for(action, values)
    with STATE_LOCK:
        if STATE["running"]:
            raise RuntimeError("已有任务正在运行，请等待完成或先停止。")
        current_proc = None
        stop_requested = False
        STATE["values"] = values.copy()
        STATE["logs"] = []
        STATE["running"] = True
        STATE["job"] = action
        STATE["exit_code"] = None
        STATE["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        STATE["finished_at"] = None
        STATE["last_error"] = ""
        if action == "train":
            reset_train_progress_locked()
            STATE["markers"].pop("remote_job_name", None)
            STATE["markers"].pop("remote_work_dir", None)
            STATE["markers"].pop("stop_export", None)
            try:
                STOP_EXPORT_SIGNAL_FILE.unlink(missing_ok=True)
            except OSError:
                pass
            STATE["train_progress"]["phase"] = "pending"
            STATE["train_progress"]["updated_at"] = time.strftime("%H:%M:%S")

    try:
        LATEST_JOB_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        LATEST_JOB_LOG_FILE.write_text(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] action={action}\n",
            encoding="utf-8",
        )
    except OSError:
        pass


    def worker() -> None:
        global current_proc, stop_requested

        proc: Optional[subprocess.Popen[Any]] = None
        append_log("$ " + quote_cmd(cmd) + "\n")
        try:
            proc = subprocess.Popen(
                [str(x) for x in cmd],
                cwd=str(SCRIPT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=subprocess_env(),
                creationflags=subprocess_creationflags(),
                start_new_session=(os.name != "nt"),
            )
            with STATE_LOCK:
                current_proc = proc
                should_stop = stop_requested
            if should_stop:
                terminate_process_tree(proc)

            if proc.stdout is not None:
                buffer = bytearray()
                while True:
                    chunk = proc.stdout.read(1)
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    if chunk in (b"\n", b"\r"):
                        line = decode_process_output(bytes(buffer))
                        buffer.clear()
                        append_log(line)
                        parse_marker(line.strip())
                        if action == "train":
                            parse_train_output(line)
                if buffer:
                    line = decode_process_output(bytes(buffer))
                    append_log(line)
                    parse_marker(line.strip())
                    if action == "train":
                        parse_train_output(line)

            code = proc.wait()

            with STATE_LOCK:
                stopped = stop_requested
                STATE["exit_code"] = code
            if stopped:
                append_log(f"\n[stopped, exit code {code}]\n")
            else:
                append_log(f"\n[exit code {code}]\n")
        except Exception as exc:
            with STATE_LOCK:
                stopped = stop_requested
                STATE["exit_code"] = -15 if stopped else -1
                STATE["last_error"] = "" if stopped else str(exc)
            if stopped:
                append_log("\n[stopped]\n")
            else:
                append_log(f"\n[error] {exc}\n")
        finally:
            with STATE_LOCK:
                STATE["running"] = False
                STATE["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                current_proc = None
                stop_requested = False


    threading.Thread(target=worker, daemon=True).start()


def stop_train_and_export() -> bool:
    with STATE_LOCK:
        if not STATE["running"] or STATE.get("job") != "train":
            return False
        values = STATE["values"].copy()
        markers = STATE["markers"].copy()
        STATE["markers"]["stop_export"] = "已请求停止训练并导出当前 best 模型"
        STATE["train_progress"]["phase"] = "export"
        STATE["train_progress"]["updated_at"] = time.strftime("%H:%M:%S")

    append_log("\n[stop training and export requested]\n")
    if values.get("train_mode") == "remote-windows":
        threading.Thread(target=stop_remote_training, args=(values, markers, True), daemon=True).start()
    else:
        try:
            STOP_EXPORT_SIGNAL_FILE.write_text(str(time.time()), encoding="utf-8")
        except OSError as exc:
            append_log(f"[stop export signal error] {exc}\n")
    return True


def stop_job() -> bool:
    global stop_requested
    with STATE_LOCK:
        if not STATE["running"]:
            return False
        proc = current_proc
        values = STATE["values"].copy()
        markers = STATE["markers"].copy()
        job = STATE["job"]
        stop_requested = True

    append_log("\n[stop requested]\n")
    if job == "train" and values.get("train_mode") == "remote-windows":
        threading.Thread(target=stop_remote_training, args=(values, markers), daemon=True).start()
    if proc and proc.poll() is None:
        terminate_process_tree(proc)
    return True


def resolve_under(path: str, base: Path) -> Path:

    target = Path(path).resolve()
    base = base.resolve()
    if target != base and base not in target.parents:
        raise ValueError("路径不在当前打标输出目录内。")
    return target


def parse_label_names(raw: str) -> list[str]:
    labels = [part.strip() for part in re.split(r"[,;\n]+", raw) if part.strip()]
    return labels or ["object"]


def label_image_files(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in TRAIN_IMAGE_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


def write_label_voc_xml(xml_path: Path, image_name: str, frame, objects: list[LabelTrackObject]) -> None:
    height, width = frame.shape[:2]
    depth = frame.shape[2] if len(frame.shape) == 3 else 1
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = image_name
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(root, "segmented").text = "0"
    for obj in objects:
        if not obj.ok:
            continue
        x, y, w, h = obj.bbox
        item = ET.SubElement(root, "object")
        ET.SubElement(item, "name").text = obj.label
        ET.SubElement(item, "truncated").text = "0"
        ET.SubElement(item, "difficult").text = "0"
        ET.SubElement(item, "occluded").text = "0"
        box = ET.SubElement(item, "bndbox")
        ET.SubElement(box, "xmin").text = str(x)
        ET.SubElement(box, "ymin").text = str(y)
        ET.SubElement(box, "xmax").text = str(x + w)
        ET.SubElement(box, "ymax").text = str(y + h)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(xml_path, encoding="UTF-8", xml_declaration=True)


def label_session_state(session: dict[str, Any]) -> dict[str, Any]:
    frame = session["frame"]
    height, width = frame.shape[:2]
    return {
        "session_id": session["id"],
        "source_type": session["source_type"],
        "frame_index": session["frame_index"],
        "frame_count": session["frame_count"],
        "width": width,
        "height": height,
        "objects": [label_object_data(obj) for obj in session["objects"]],
        "labels": session["labels"],
        "tracker": session["tracker"],
        "saved": session["saved"],
        "interval": session["interval"],
        "processed": session["processed"],
        "max_frames": session["max_frames"],
        "ended": session["ended"],
        "lost": any(not obj.ok for obj in session["objects"]),
        "review_skipped": session.get("review_skipped", 0),
        "last_warning": session.get("last_warning", ""),
        "last_auto_saved": session.get("last_auto_saved", False),
    }


def get_label_session(session_id: str) -> dict[str, Any]:
    with LABEL_SESSIONS_LOCK:
        session = LABEL_SESSIONS.get(session_id)
    if session is None:
        raise ValueError("标注会话不存在或已结束。")
    return session


def open_label_camera(camera_index: int):
    backends = [(cv2.CAP_ANY, "default")]
    if sys.platform.startswith("win"):
        backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MSMF"), *backends]
    for backend, name in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, frame, name
        cap.release()
    raise ValueError("摄像头无法打开。请关闭占用程序、检查系统摄像头权限，或尝试其他索引。")


def start_label_session(values: dict[str, Any]) -> dict[str, Any]:
    validate("label", values)
    source_type = values.get("label_source_type", "video")
    cap = None
    files: Optional[list[Path]] = None
    source_image_path: Optional[Path] = None
    frame_count = 0
    start_frame = max(0, int(values.get("label_start_frame", "0") or 0))
    if source_type == "images":
        files = label_image_files(Path(values["label_images_input_dir"]).resolve())
        if not files:
            raise ValueError("图片集文件夹内没有可标注图片。")
        frame_index = min(start_frame, len(files) - 1)
        source_image_path = files[frame_index]
        frame = cv2.imread(str(source_image_path))
        if frame is None:
            raise ValueError("无法读取起始图片。")
        frame_count = len(files)
    elif source_type == "camera":
        cap, frame, backend = open_label_camera(int(values["label_camera_index"]))
        frame_index = 0
        append_log(f"\n[网页标注] 摄像头 {values['label_camera_index']} 已通过 {backend} 打开。\n")
    else:
        video_path = Path(values["label_video"]).expanduser().resolve()
        if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("视频文件不存在或格式不支持。")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise ValueError("视频无法打开，可能是当前 OpenCV 不支持该编码。")
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise ValueError("无法读取视频起始帧。")
        frame_index = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    images_dir = Path(values["label_images_dir"]).resolve()
    annotations_dir = Path(values["label_annotations_dir"]).resolve()
    if source_type != "images":
        images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex
    session = {
        "id": session_id, "lock": threading.RLock(), "source_type": source_type, "cap": cap, "files": files,
        "frame": frame, "frame_index": frame_index, "frame_count": frame_count, "source_image_path": source_image_path,
        "objects": [], "next_object_id": 1, "saved": 0, "processed": 0, "ended": False,
        "review_skipped": 0, "last_warning": "", "last_auto_saved": False,
        "labels": parse_label_names(values["label_name"]), "tracker": values["label_tracker"],
        "interval": max(1, int(values["label_interval"] or 1)), "max_frames": max(0, int(values["label_max_frames"] or 0)),
        "images_dir": images_dir, "annotations_dir": annotations_dir, "prefix": values["label_prefix"].strip() or "track",
        "jpeg_quality": max(1, min(100, int(values["label_jpeg_quality"] or 95))),
    }
    with LABEL_SESSIONS_LOCK:
        LABEL_SESSIONS[session_id] = session
    append_log(f"[网页标注] 会话已创建：{source_type}，请在页面框选目标。\n")
    return label_session_state(session)


def save_label_session_sample(session: dict[str, Any], automatic: bool = False) -> bool:
    objects = session["objects"]
    if not objects:
        session["last_warning"] = "当前帧没有目标框。"
        return False
    invalid = [obj for obj in objects if not obj.ok]
    if invalid:
        names = "、".join(f"#{obj.obj_id} {obj.label}" for obj in invalid[:4])
        session["last_warning"] = f"{names} 需要复核；为避免生成缺框标签，本帧未保存。"
        if automatic:
            session["review_skipped"] = session.get("review_skipped", 0) + 1
        return False
    frame = session["frame"]
    source_image_path = session["source_image_path"]
    if source_image_path is not None:
        image_name = source_image_path.name
        xml_path = session["annotations_dir"] / f"{source_image_path.stem}.xml"
    else:
        stem = f"{session['prefix']}_{session['frame_index']:06d}"
        image_name = stem + ".jpg"
        image_path = session["images_dir"] / image_name
        if not cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), session["jpeg_quality"]]):
            raise ValueError("标注图片保存失败。")
        xml_path = session["annotations_dir"] / f"{stem}.xml"
    write_label_voc_xml(xml_path, image_name, frame, objects)
    session["saved"] += 1
    session["last_warning"] = ""
    session["last_auto_saved"] = automatic
    return True


def advance_label_session(session: dict[str, Any]) -> dict[str, Any]:
    if session["ended"]:
        return label_session_state(session)
    if session["max_frames"] and session["processed"] >= session["max_frames"]:
        session["ended"] = True
        return label_session_state(session)
    if session["source_type"] == "images":
        next_index = session["frame_index"] + 1
        if next_index >= len(session["files"]):
            session["ended"] = True
            return label_session_state(session)
        frame = cv2.imread(str(session["files"][next_index]))
        if frame is None:
            raise ValueError("下一张图片无法读取。")
        session["frame_index"] = next_index
        session["source_image_path"] = session["files"][next_index]
    else:
        ok, frame = session["cap"].read()
        if not ok or frame is None:
            session["ended"] = True
            return label_session_state(session)
        if session["source_type"] == "camera":
            session["frame_index"] += 1
        else:
            session["frame_index"] = max(0, int(session["cap"].get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        session["source_image_path"] = None
    session["frame"] = frame
    session["processed"] += 1
    session["last_auto_saved"] = False
    session["last_warning"] = ""
    for obj in session["objects"]:
        if not obj.ok:
            continue
        previous_bbox = obj.bbox
        ok, bbox = obj.tracker.update(frame)
        obj.bbox = sanitize_label_bbox(bbox, frame.shape[1], frame.shape[0])
        tracker_score = getattr(obj.tracker, "last_score", None)
        obj.quality, drift_warning = label_tracking_quality(
            previous_bbox, obj.bbox, frame.shape[1], frame.shape[0], tracker_score
        )
        obj.ok = bool(ok) and obj.quality >= 0.35
        if not ok:
            obj.warning = "跟踪器未找到目标"
        elif not obj.ok:
            obj.warning = drift_warning or "跟踪质量过低"
        else:
            obj.warning = drift_warning
    if session["objects"] and session["frame_index"] % session["interval"] == 0:
        save_label_session_sample(session, automatic=True)
    elif any(not obj.ok for obj in session["objects"]):
        names = "、".join(f"#{obj.obj_id} {obj.label}" for obj in session["objects"] if not obj.ok)
        session["last_warning"] = f"{names} 跟踪异常，请在继续前复核。"
    return label_session_state(session)


def end_label_session(session_id: str) -> None:
    with LABEL_SESSIONS_LOCK:
        session = LABEL_SESSIONS.pop(session_id, None)
    if session is None:
        return
    with session["lock"]:
        cap = session.get("cap")
        if cap is not None:
            cap.release()
        session["ended"] = True
    append_log("[网页标注] 会话已结束。\n")





def parse_voc_xml(xml_path: Path) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename", default=xml_path.with_suffix(".jpg").name)
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="")
        box = obj.find("bndbox")
        if box is None:
            continue
        boxes.append({
            "name": name,
            "xmin": int(float(box.findtext("xmin", default="0"))),
            "ymin": int(float(box.findtext("ymin", default="0"))),
            "xmax": int(float(box.findtext("xmax", default="0"))),
            "ymax": int(float(box.findtext("ymax", default="0"))),
        })
    return {"filename": filename, "boxes": boxes}


def label_result_images_dir(values: dict[str, Any]) -> Path:
    if values.get("label_source_type") == "images":
        return Path(values.get("label_images_input_dir", "")).resolve()
    return Path(values["label_images_dir"]).resolve()


def list_label_results(values: dict[str, Any]) -> list[dict[str, Any]]:
    images_dir = label_result_images_dir(values)
    annotations_dir = Path(values["label_annotations_dir"]).resolve()
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        return []
    results = []
    xml_files = sorted(annotations_dir.glob("*.xml"), key=lambda p: p.stat().st_mtime, reverse=True)
    for xml_path in xml_files[:300]:
        try:
            meta = parse_voc_xml(xml_path)
        except Exception:
            continue
        image_path = images_dir / meta["filename"]
        if not image_path.exists():
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                candidate = images_dir / (xml_path.stem + ext)
                if candidate.exists():
                    image_path = candidate
                    break
        if not image_path.exists():
            continue
        results.append({
            "stem": xml_path.stem,
            "image": str(image_path),
            "xml": str(xml_path),
            "boxes": meta["boxes"],
            "mtime": int(xml_path.stat().st_mtime),
        })
    return results


def list_label_videos(values: dict[str, Any]) -> list[dict[str, Any]]:
    video_dir_raw = values.get("label_video_dir", "").strip()
    if not video_dir_raw:
        return []
    video_dir = Path(video_dir_raw).expanduser().resolve()
    if not video_dir.is_dir():
        return []
    videos = []
    for path in video_dir.rglob("*"):
        if len(videos) >= MAX_LABEL_VIDEOS:
            break
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        videos.append({
            "name": path.name,
            "stem": path.stem,
            "path": str(path),
            "rel": str(path.relative_to(video_dir)),
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
        })
    videos.sort(key=lambda item: item["rel"].lower())
    return videos





def list_train_plots() -> tuple[Path, list[dict[str, Any]]]:
    with STATE_LOCK:
        plot_dir_raw = str(STATE["markers"].get("train_plot_dir", "")).strip()
    if not plot_dir_raw:
        return Path(), []
    plot_dir = Path(plot_dir_raw).resolve()
    if not plot_dir.is_dir():
        return plot_dir, []
    items = [
        {"name": path.name, "mtime": int(path.stat().st_mtime)}
        for path in sorted(plot_dir.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.suffix.lower() in TRAIN_IMAGE_EXTENSIONS
    ]
    return plot_dir, items


def model_asset_catalog(values: dict[str, Any]) -> dict[str, Any]:
    roots = [TRAINING_RUNS_DIR, values.get("asset_scan_root", "")]
    deployment_roots = [DEPLOYMENT_EXPORTS_DIR, values.get("export_output_dir", "")]
    with MODEL_REGISTRY_LOCK:
        return collect_model_assets(MODEL_REGISTRY_FILE, roots, deployment_roots)


def send_train_plot(handler: BaseHTTPRequestHandler, plot_dir: Path, name: str) -> None:
    image_path = resolve_under(name, plot_dir)
    if not image_path.is_file() or image_path.suffix.lower() not in TRAIN_IMAGE_EXTENSIONS:
        raise ValueError("训练图片不存在或格式不支持。")
    raw = image_path.read_bytes()
    content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def pick_image_file(initial_path: str = "") -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("当前环境不支持系统文件选择器，请手动粘贴图片路径。") from exc
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    initial_dir = str(Path(initial_path).expanduser().parent) if initial_path else str(SCRIPT_ROOT)
    if not Path(initial_dir).is_dir():
        initial_dir = str(SCRIPT_ROOT)
    selected = filedialog.askopenfilename(
        parent=root,
        initialdir=initial_dir,
        title="选择测试图片",
        filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")],
    )
    root.destroy()
    return selected


def pick_model_file(initial_path: str = "", include_onnx: bool = False) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("当前环境不支持系统文件选择器，请手动粘贴模型路径。") from exc
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    initial_dir = str(Path(initial_path).expanduser().parent) if initial_path else str(SCRIPT_ROOT)
    if not Path(initial_dir).is_dir():
        initial_dir = str(SCRIPT_ROOT)
    selected = filedialog.askopenfilename(
        parent=root,
        initialdir=initial_dir,
        title="选择训练或部署模型" if include_onnx else "选择 YOLO 基础模型",
        filetypes=[("YOLO 模型", "*.pt *.onnx"), ("所有文件", "*.*")] if include_onnx else [("PyTorch 模型", "*.pt"), ("所有文件", "*.*")],
    )
    root.destroy()
    return selected


def pick_directory(initial_dir: str = "", title: str = "选择文件夹") -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("当前环境不支持系统文件夹选择器，请手动粘贴文件夹路径。") from exc
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    start = initial_dir if initial_dir and Path(initial_dir).expanduser().exists() else str(SCRIPT_ROOT)
    selected = filedialog.askdirectory(parent=root, initialdir=start, title=title)
    root.destroy()
    return selected


def apply_project_to_values(project: dict[str, Any], current: dict[str, str]) -> dict[str, str]:
    """Make the selected project the source of truth for training-related fields."""
    values = clean_values(current)
    dataset_root = Path(project.get("dataset_root") or project.get("root") or "").expanduser().resolve()
    task = str(project.get("task") or "detect")
    values.update({
        "active_project_id": str(project.get("id") or ""),
        "project_name": str(project.get("id") or project.get("name") or "project"),
        "dataset_root": str(dataset_root),
        "raw_dataset_root": str(dataset_root),
        "train_task": task,
        "raw_class_names": ", ".join(project.get("labels") or []),
    })
    if task == "classify":
        image_root = dataset_root / "images"
        values.update({
            "train_images_dir": str(image_root if image_root.is_dir() else dataset_root),
            "train_annotations_dir": "",
            "prepared_dataset_yaml": "",
        })
        return values

    yaml_candidates = [dataset_root / "data.yaml", dataset_root / "dataset.yaml"]
    yaml_path = next((candidate for candidate in yaml_candidates if candidate.is_file()), None)
    if yaml_path is not None:
        try:
            prepared = inspect_prepared_yolo_dataset(str(dataset_root))
            train_split = prepared["splits"]["train"]
            values.update({
                "train_images_dir": train_split["images_dir"],
                "train_annotations_dir": "",
                "prepared_dataset_yaml": prepared["yaml_path"],
                "raw_class_names": ", ".join(prepared.get("class_names") or project.get("labels") or []),
            })
            return values
        except Exception:
            pass

    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"
    values.update({
        "train_images_dir": str(images_dir if images_dir.is_dir() else dataset_root),
        "train_annotations_dir": str(annotations_dir) if annotations_dir.is_dir() else "",
        "prepared_dataset_yaml": "",
    })
    return values


def read_video_preview_frame(video_path: Path):
    backends = [cv2.CAP_ANY]
    if hasattr(cv2, "CAP_FFMPEG"):
        backends.append(cv2.CAP_FFMPEG)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    tried = set()
    for backend in backends:
        if backend in tried:
            continue
        tried.add(backend)
        cap = cv2.VideoCapture(str(video_path), backend)
        try:
            if not cap.isOpened():
                continue
            for frame_no in (0, 1, 3, 10, 30):
                if frame_no:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ok, frame = cap.read()
                if ok and frame is not None:
                    return frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(30):
                ok = cap.grab()
                if not ok:
                    break
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    return frame
        finally:
            cap.release()
    raise ValueError("无法读取视频预览帧，可能是编码器不受当前 OpenCV 支持。")


def render_video_preview(video_path: Path) -> bytes:
    video_path = video_path.resolve()
    if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("视频文件不存在或格式不支持。")
    stat = video_path.stat()
    cache_key = (str(video_path), int(stat.st_mtime), stat.st_size)
    cached = VIDEO_PREVIEW_CACHE.get(cache_key)
    if cached is not None:
        VIDEO_PREVIEW_CACHE.move_to_end(cache_key)
        return cached
    frame = read_video_preview_frame(video_path)
    h, w = frame.shape[:2]
    if w > MAX_VIDEO_PREVIEW_WIDTH:
        scale = MAX_VIDEO_PREVIEW_WIDTH / w
        frame = cv2.resize(frame, (MAX_VIDEO_PREVIEW_WIDTH, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
    if not ok:
        raise ValueError("视频预览图编码失败。")
    raw = encoded.tobytes()
    VIDEO_PREVIEW_CACHE[cache_key] = raw
    VIDEO_PREVIEW_CACHE.move_to_end(cache_key)
    while len(VIDEO_PREVIEW_CACHE) > VIDEO_PREVIEW_CACHE_LIMIT:
        VIDEO_PREVIEW_CACHE.popitem(last=False)
    return raw






def send_video_file(handler: BaseHTTPRequestHandler, video_path: Path) -> None:
    video_path = video_path.resolve()
    if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("视频文件不存在或格式不支持。")
    file_size = video_path.stat().st_size
    start = 0
    end = min(file_size - 1, 8 * 1024 * 1024 - 1)
    status = HTTPStatus.PARTIAL_CONTENT
    range_header = handler.headers.get("Range", "")
    if range_header.startswith("bytes="):
        raw_range = range_header.removeprefix("bytes=").split(",", 1)[0].strip()
        left, _, right = raw_range.partition("-")
        if left:
            start = int(left)
            end = int(right) if right else file_size - 1
        elif right:
            suffix_len = int(right)
            start = max(file_size - suffix_len, 0)
            end = file_size - 1
        if start < 0 or end < start or start >= file_size:
            handler.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            handler.send_header("Content-Range", f"bytes */{file_size}")
            handler.end_headers()
            return
        end = min(end, file_size - 1)
    length = end - start + 1
    content_type = VIDEO_MIME_TYPES.get(video_path.suffix.lower()) or mimetypes.guess_type(str(video_path))[0] or "application/octet-stream"

    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
    handler.send_header("Content-Length", str(length))
    handler.end_headers()
    with video_path.open("rb") as fh:
        fh.seek(start)
        remaining = length
        while remaining > 0:
            chunk = fh.read(min(256 * 1024, remaining))
            if not chunk:
                break
            try:
                handler.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                break
            remaining -= len(chunk)








def render_label_preview(image_path: Path, xml_path: Path) -> bytes:


    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError("图片不可读取。")
    meta = parse_voc_xml(xml_path)
    line_thickness = max(5, round(min(frame.shape[:2]) / 180))
    text_scale = max(0.7, min(frame.shape[:2]) / 1400)
    for box in meta["boxes"]:
        p1 = (box["xmin"], box["ymin"])
        p2 = (box["xmax"], box["ymax"])
        cv2.rectangle(frame, p1, p2, (40, 220, 120), line_thickness, cv2.LINE_AA)
        label_y = max(24, p1[1] - 8)
        cv2.putText(frame, box["name"], (p1[0], label_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), line_thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, box["name"], (p1[0], label_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (240, 255, 245), max(2, line_thickness // 2), cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise ValueError("预览图编码失败。")
    return encoded.tobytes()


HTML_PAGE = r'''<!doctype html>

<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>YOLO团队训练平台</title>
<style>
:root{--bg:#09111f;--panel:rgba(255,255,255,.08);--panel2:rgba(255,255,255,.13);--text:#eef6ff;--muted:#9fb0c7;--line:rgba(255,255,255,.14);--blue:#56a8ff;--green:#30d287;--purple:#a78bfa;--orange:#ffbd5a;--red:#ff6678;--shadow:0 24px 70px rgba(0,0,0,.35);font-family:"Microsoft YaHei UI","Segoe UI",system-ui,sans-serif}*{box-sizing:border-box}body{margin:0;color:var(--text);background:radial-gradient(circle at 15% 10%,#173c73 0,#09111f 34%),radial-gradient(circle at 86% 0,#41226d 0,transparent 30%),linear-gradient(135deg,#09111f,#0c1528);min-height:100vh}.wrap{width:min(1380px,calc(100% - 36px));margin:0 auto;padding:28px 0 36px}.hero{display:grid;grid-template-columns:1.2fr .8fr;gap:18px;align-items:stretch;margin-bottom:18px}.card{background:var(--panel);border:1px solid var(--line);box-shadow:var(--shadow);border-radius:26px;backdrop-filter:blur(18px)}.title{padding:30px}.eyebrow{display:inline-flex;gap:8px;align-items:center;color:#cde4ff;background:rgba(86,168,255,.13);border:1px solid rgba(86,168,255,.25);padding:7px 12px;border-radius:999px;font-size:13px}.dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 16px var(--green)}h1{font-size:42px;letter-spacing:-.04em;margin:18px 0 10px}.subtitle{color:var(--muted);line-height:1.8;margin:0;max-width:780px}.guide{padding:24px}.steps{display:grid;gap:12px}.step{display:flex;gap:12px;align-items:flex-start;padding:13px;border:1px solid var(--line);background:rgba(255,255,255,.06);border-radius:18px}.num{flex:0 0 30px;width:30px;height:30px;border-radius:12px;background:linear-gradient(135deg,var(--blue),var(--purple));display:grid;place-items:center;font-weight:800}.step b{display:block;margin-bottom:3px}.step span{color:var(--muted);font-size:13px;line-height:1.5}.layout{display:grid;grid-template-columns:290px 1fr;gap:18px}.side{position:sticky;top:18px;height:fit-content;padding:16px}.nav{display:grid;gap:10px}.nav button{all:unset;cursor:pointer;padding:15px 16px;border-radius:18px;color:var(--muted);border:1px solid transparent;display:flex;justify-content:space-between;align-items:center}.nav button.active{background:linear-gradient(135deg,rgba(86,168,255,.22),rgba(167,139,250,.18));border-color:rgba(255,255,255,.18);color:var(--text)}.status{margin-top:14px;padding:14px;border-radius:18px;background:rgba(0,0,0,.23);border:1px solid var(--line);overflow:hidden}.pill{display:inline-flex;align-items:center;gap:8px;padding:7px 11px;border-radius:999px;font-size:13px;border:1px solid var(--line);color:var(--muted);max-width:100%}.pill span:last-child{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.pill.run{color:#bff8dc;border-color:rgba(48,210,135,.35);background:rgba(48,210,135,.1)}.pill.idle{color:#d5e5ff;background:rgba(86,168,255,.08)}.main{display:grid;gap:18px}.tab{display:none}.tab.active{display:block}.section{padding:22px;margin-bottom:18px}.section h2{margin:0 0 6px;font-size:24px}.hint{margin:0 0 18px;color:var(--muted);line-height:1.65}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.field{grid-column:span 6}.field.sm{grid-column:span 3}.field.full{grid-column:1/-1}label{display:block;color:#d9e9ff;font-size:13px;margin:0 0 7px}input,select{width:100%;background:rgba(5,12,24,.72);border:1px solid var(--line);border-radius:14px;color:var(--text);padding:12px 13px;outline:none;transition:.2s}input:focus,select:focus{border-color:rgba(86,168,255,.72);box-shadow:0 0 0 4px rgba(86,168,255,.13)}.choice{display:flex;gap:10px;flex-wrap:wrap}.choice label{margin:0;cursor:pointer}.choice input{display:none}.choice span{display:inline-flex;padding:10px 13px;border-radius:14px;border:1px solid var(--line);background:rgba(255,255,255,.06);color:var(--muted)}.choice input:checked+span{color:var(--text);border-color:rgba(86,168,255,.58);background:rgba(86,168,255,.18)}.actions{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin-top:18px}.btns{display:flex;gap:10px;flex-wrap:wrap}.btn{all:unset;cursor:pointer;border-radius:15px;padding:12px 17px;font-weight:700;border:1px solid var(--line);background:var(--panel2)}.btn.primary{background:linear-gradient(135deg,#238bff,#8b5cf6);border:0}.btn.green{background:linear-gradient(135deg,#18aa69,#22c98a);border:0}.btn.blue{background:linear-gradient(135deg,#1877f2,#56a8ff);border:0}.btn.red{background:rgba(255,102,120,.14);border-color:rgba(255,102,120,.35);color:#ffd7dd}.cmd{margin-top:14px;background:#050b15;border:1px solid var(--line);border-radius:18px;padding:14px;color:#bad4f6;white-space:pre-wrap;word-break:break-all;font-family:"Cascadia Mono",Consolas,monospace;font-size:12px;line-height:1.55}.log{height:360px;overflow:auto;background:#030813;border:1px solid rgba(255,255,255,.12);border-radius:20px;padding:16px;white-space:pre-wrap;color:#c6f6d5;font-family:"Cascadia Mono",Consolas,monospace;font-size:12px;line-height:1.55}.toast{position:fixed;right:22px;bottom:22px;max-width:420px;padding:14px 16px;border-radius:16px;background:#101d33;border:1px solid var(--line);box-shadow:var(--shadow);display:none}.toast.show{display:block}.mini{color:var(--muted);font-size:12px;margin-top:7px}.markers{display:grid;gap:8px;margin-top:10px;min-width:0}.marker{display:grid;grid-template-columns:1fr;gap:5px;align-items:start;padding:10px;border-radius:14px;background:rgba(255,255,255,.05);border:1px solid var(--line);font-size:12px;color:var(--muted);min-width:0;overflow:hidden}.marker b{color:#dcecff;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.marker span{min-width:0;overflow-wrap:anywhere;word-break:break-all;white-space:normal;line-height:1.45}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px;margin-top:16px}.sample{overflow:hidden;border-radius:18px;border:1px solid var(--line);background:rgba(255,255,255,.06)}.sample img{width:100%;display:block;aspect-ratio:16/10;object-fit:cover;background:#050b15}.sample .meta{padding:11px;font-size:12px;color:var(--muted);display:grid;gap:8px}.sample .delete{all:unset;cursor:pointer;text-align:center;padding:9px 10px;border-radius:12px;background:rgba(255,102,120,.14);border:1px solid rgba(255,102,120,.35);color:#ffd7dd;font-weight:700}.empty{padding:16px;border:1px dashed var(--line);border-radius:18px;color:var(--muted);margin-top:14px}.input-action{display:flex;gap:10px;align-items:stretch}.input-action input{min-width:0;flex:1}.input-action .btn{white-space:nowrap}@media(max-width:520px){.input-action{flex-direction:column}}.label-workspace{display:grid;grid-template-columns:minmax(280px,360px) 1fr;gap:16px;margin-top:18px}.train-board{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:18px}.progress-card{border:1px solid var(--line);background:rgba(255,255,255,.055);border-radius:22px;padding:16px;overflow:hidden}.progress-card.wide{grid-column:1/-1}.progress-head{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:12px}.progress-head b{font-size:16px}.progress-head span{color:var(--muted);font-size:12px}.bar{height:16px;border-radius:999px;background:rgba(5,12,24,.72);border:1px solid var(--line);overflow:hidden}.bar div{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--green),var(--blue),var(--purple));transition:width .25s ease}.metrics-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-top:12px}.metrics-grid div{padding:10px;border-radius:14px;background:rgba(0,0,0,.18);border:1px solid var(--line);min-width:0}.metrics-grid span{display:block;color:var(--muted);font-size:12px;margin-bottom:4px}.metrics-grid b{display:block;color:#eff8ff;font-size:18px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.loss-grid{grid-template-columns:repeat(3,1fr)}canvas{width:100%;height:180px;margin-top:10px;border-radius:14px;background:rgba(3,8,19,.72);border:1px solid rgba(255,255,255,.1)}.panel{border:1px solid var(--line);background:rgba(255,255,255,.055);border-radius:22px;padding:16px}.panel h3{margin:0 0 10px;font-size:16px}.queue-head{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:10px}.count{color:var(--muted);font-size:12px}.video-list{display:grid;gap:8px;max-height:300px;overflow:auto;padding-right:4px}.video-preview{margin-top:14px}.video-preview-box{min-height:220px;border:1px dashed var(--line);border-radius:18px;background:rgba(5,12,24,.42);display:grid;place-items:center;overflow:hidden;color:var(--muted);font-size:12px;text-align:center;position:relative}.video-preview-box.clickable{cursor:pointer;border-style:solid;border-color:rgba(86,168,255,.42)}.video-preview-box img,.video-preview-box video{width:100%;height:100%;display:block;object-fit:contain;background:#050b15}.video-preview-box video{min-height:220px}.play-overlay{position:absolute;inset:0;display:grid;place-items:center;background:linear-gradient(180deg,rgba(5,12,24,.08),rgba(5,12,24,.48));pointer-events:none}.play-button{width:66px;height:66px;border-radius:50%;display:grid;place-items:center;background:rgba(86,168,255,.86);box-shadow:0 14px 36px rgba(0,0,0,.42);color:white;font-size:30px;line-height:1;transform:translateY(-2px)}.video-item{all:unset;cursor:pointer;display:grid;gap:5px;padding:11px 12px;border-radius:15px;border:1px solid var(--line);background:rgba(5,12,24,.42)}.video-item.active{border-color:rgba(86,168,255,.75);background:rgba(86,168,255,.16)}.video-item.done{border-color:rgba(48,210,135,.38)}.video-item b{font-size:13px;color:#edf7ff;word-break:break-all}.video-item span{font-size:12px;color:var(--muted);word-break:break-all}.current-video{display:grid;gap:8px;padding:12px 14px;border-radius:18px;background:rgba(86,168,255,.12);border:1px solid rgba(86,168,255,.24);margin-bottom:14px}.current-video b{word-break:break-all}.label-config{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.label-config .field{grid-column:span 6}.label-config .field.sm{grid-column:span 3}.label-config .field.full{grid-column:1/-1}.quick-help{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}.quick-help div{padding:10px;border-radius:14px;background:rgba(0,0,0,.18);border:1px solid var(--line);font-size:12px;color:var(--muted)}.quick-help b{display:block;color:#e6f2ff;margin-bottom:3px}@media(max-width:1100px){.label-workspace{grid-template-columns:1fr}.quick-help{grid-template-columns:repeat(2,1fr)}}@media(max-width:980px){.hero,.layout{grid-template-columns:1fr}.side{position:static}.field,.field.sm,.label-config .field,.label-config .field.sm{grid-column:1/-1}h1{font-size:32px}}.label-studio{margin-top:18px}.label-studio[hidden]{display:none}.label-stage{position:relative;background:#030813;border:1px solid var(--line);border-radius:20px;overflow:hidden;min-height:300px;display:grid;place-items:center}.label-stage img{display:block;width:100%;max-height:650px;object-fit:contain;user-select:none}.label-stage canvas{position:absolute;inset:0;width:100%;height:100%;margin:0;border:0;background:transparent;touch-action:none;cursor:crosshair}.label-stage.empty-stage{color:var(--muted);padding:28px;text-align:center}.label-studio-grid{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px}.label-status{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.label-status span{padding:7px 10px;border-radius:999px;background:rgba(86,168,255,.12);border:1px solid rgba(86,168,255,.24);font-size:12px;color:#dcecff}.label-object-list{display:grid;gap:8px;max-height:360px;overflow:auto}.label-object{all:unset;cursor:pointer;display:flex;justify-content:space-between;gap:8px;padding:11px;border:1px solid var(--line);border-radius:14px;background:rgba(5,12,24,.42);font-size:12px}.label-object.active{border-color:rgba(86,168,255,.8);background:rgba(86,168,255,.17)}.label-object.lost{border-color:rgba(255,102,120,.55);color:#ffd7dd}.label-object span{color:var(--muted)}.label-help{font-size:12px;color:var(--muted);line-height:1.65;margin-top:12px}@media(max-width:980px){.label-studio-grid{grid-template-columns:1fr}}
.onboarding{display:grid;gap:14px;margin:16px 0 18px;padding:17px;border-radius:20px;background:linear-gradient(135deg,rgba(86,168,255,.13),rgba(48,210,135,.07));border:1px solid rgba(86,168,255,.28)}.onboarding-head{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;flex-wrap:wrap}.onboarding-head b{font-size:17px}.onboarding-head span{display:block;color:var(--muted);font-size:12px;margin-top:5px}.preset-row{display:flex;gap:9px;flex-wrap:wrap}.preset{all:unset;cursor:pointer;padding:9px 13px;border-radius:13px;border:1px solid var(--line);background:rgba(5,12,24,.45);font-size:13px}.preset:hover{border-color:rgba(86,168,255,.65);background:rgba(86,168,255,.12)}.readiness{display:grid;gap:9px;margin:14px 0 18px}.readiness[hidden]{display:none}.check-item{display:grid;grid-template-columns:24px minmax(110px,170px) 1fr;gap:10px;align-items:start;padding:11px 13px;border-radius:15px;border:1px solid var(--line);background:rgba(5,12,24,.42);font-size:13px}.check-icon{width:22px;height:22px;border-radius:50%;display:grid;place-items:center;font-weight:800}.check-item.ok .check-icon{background:rgba(48,210,135,.18);color:#8ff0bd}.check-item.warn .check-icon{background:rgba(255,189,90,.18);color:#ffd38f}.check-item.error .check-icon{background:rgba(255,102,120,.18);color:#ff9cab}.check-detail{color:var(--muted);overflow-wrap:anywhere}.advanced-toggle{color:#cde4ff}.advanced-setting{transition:.2s}body:not(.show-advanced) .advanced-setting{display:none}.last-error{margin-top:10px;padding:10px;border-radius:12px;background:rgba(255,102,120,.12);border:1px solid rgba(255,102,120,.28);color:#ffd7dd;font-size:12px;line-height:1.5;overflow-wrap:anywhere}.eyebrow.offline{color:#ffd7dd;background:rgba(255,102,120,.12);border-color:rgba(255,102,120,.3)}.btn[disabled],.preset[disabled]{opacity:.5;cursor:not-allowed}.field-note{display:flex;justify-content:space-between;gap:8px;align-items:center}.field-note .mini{margin-top:0}@media(max-width:680px){.check-item{grid-template-columns:24px 1fr}.check-detail{grid-column:2}.onboarding-head{display:grid}.preset-row{display:grid;grid-template-columns:1fr 1fr}.preset{display:block;text-align:center}}
/* YOLO Team Training Platform light theme */
:root{--bg:#fff8f8;--panel:#ffffff;--panel2:#fff2f2;--text:#302525;--muted:#806e6e;--line:#eedddd;--blue:#e53935;--green:#27a66b;--purple:#ef6b68;--orange:#e88935;--red:#dc2f2f;--shadow:0 16px 42px rgba(126,43,43,.10)}
body{color:var(--text);background:radial-gradient(circle at 88% 0,#ffe4e4 0,transparent 30%),linear-gradient(180deg,#fffafa,#fff4f4);font-size:14px}.wrap{width:min(1240px,calc(100% - 32px));padding:22px 0 32px}.hero{grid-template-columns:1fr;margin-bottom:16px}.card{background:var(--panel);border-color:var(--line);box-shadow:var(--shadow);border-radius:20px;backdrop-filter:none}.title{padding:24px 26px}.guide{padding:14px}.steps{grid-template-columns:repeat(3,1fr)}.step{padding:12px;background:#fffafa;border-color:var(--line);border-radius:14px}.num{border-radius:10px;color:#fff;background:linear-gradient(135deg,#e53935,#f06a66)}h1{font-size:36px;margin:14px 0 8px;color:#291f1f}.subtitle,.hint,.mini,.step span{color:var(--muted)}.eyebrow{color:#b32424;background:#fff0f0;border-color:#f3cccc}.eyebrow .dot{background:#e53935;box-shadow:0 0 0 4px #ffe2e2}.layout{grid-template-columns:220px 1fr;gap:16px}.side{padding:12px}.nav{gap:6px}.nav button{padding:13px 14px;border-radius:13px}.nav button:hover{background:#fff5f5;color:#5d4141}.nav button.active{color:#b82121;background:#fff0f0;border-color:#f0cccc}.nav button span{font-size:11px;color:#bca4a4}.status{background:#fffafa;border-color:var(--line);border-radius:14px}.pill.idle{color:#825f5f;background:#fff}.pill.run{color:#137a4c;background:#effaf4;border-color:#bce7cf}.pill.run .dot{background:#27a66b}.section{padding:22px}.section h2{color:#2d2222}.onboarding{background:linear-gradient(135deg,#fff0f0,#fffafa);border-color:#f0cccc;border-radius:16px}.preset{color:#674b4b;background:#fff;border-color:#e9d4d4}.preset:hover{color:#b82121;border-color:#e9a9a9;background:#fff3f3}.advanced-toggle{color:#b82121}label{color:#5f4848;font-weight:650}input,select{color:#352828;background:#fff;border-color:#e7d4d4;border-radius:12px}input:focus,select:focus{border-color:#e45757;box-shadow:0 0 0 4px rgba(229,57,53,.10)}.choice span{color:#745e5e;background:#fff;border-color:#e8d7d7;border-radius:12px}.choice input:checked+span{color:#b82121;border-color:#e79a9a;background:#fff0f0}.btn{color:#5a4242;background:#fff;border-color:#e6d0d0;border-radius:12px}.btn:hover{background:#fff5f5}.btn.primary,.btn.green,.btn.blue{color:#fff;background:linear-gradient(135deg,#d92f2f,#ed625e);border:0}.btn.red{color:#bd2525;background:#fff;border-color:#e7a6a6}.progress-card,.panel{background:#fffafa;border-color:#eedddd;border-radius:16px}.metrics-grid div,.quick-help div{background:#fff;border-color:#eadada}.metrics-grid b,.quick-help b,.marker b,.video-item b{color:#3a2b2b}.bar{background:#f3e7e7;border-color:#ead4d4}.bar div{background:linear-gradient(90deg,#e53935,#f27a74)}.readiness .check-item{background:#fff;border-color:#eadada}.check-item.ok .check-icon{background:#eaf8f0;color:#168354}.check-item.warn .check-icon{background:#fff3e6;color:#b76a1e}.check-item.error .check-icon{background:#fff0f0;color:#c82d2d}.last-error{color:#b62929;background:#fff1f1;border-color:#efc5c5}.marker,.sample,.video-item,.label-object{background:#fff;border-color:#eadada}.sample .delete{color:#b62929;background:#fff1f1;border-color:#efc5c5}.empty{border-color:#e4cfcf}.cmd,.log{color:#5d3c3c;background:#fffafa;border-color:#e7d3d3}.toast{color:#fff;background:#b82d2d;border-color:#a52424}.current-video,.label-status span{color:#9f2727;background:#fff0f0;border-color:#efcaca}.video-preview-box{color:var(--muted);background:#fffafa;border-color:#e8d5d5}.play-button{background:#e53935;box-shadow:0 12px 28px rgba(152,32,32,.25)}.progress-head b,.panel h3{color:#392a2a}.label-stage{border-color:#e4d0d0}.field.full>.onboarding{margin-bottom:0}@media(max-width:980px){.steps{grid-template-columns:1fr}.layout{grid-template-columns:1fr}.side{position:static}.nav{grid-template-columns:repeat(3,1fr)}.nav button{justify-content:center;gap:7px}.status{margin-top:10px}}@media(max-width:620px){.nav{grid-template-columns:1fr 1fr}.wrap{width:min(100% - 20px,1240px)}h1{font-size:29px}.metrics-grid{grid-template-columns:repeat(2,1fr)}}
.asset-summary{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.asset-summary div{padding:15px;border:1px solid var(--line);border-radius:15px;background:#fffafa}.asset-summary span{display:block;color:var(--muted);font-size:12px}.asset-summary b{display:block;margin-top:4px;font-size:25px}.asset-library{display:grid;gap:16px;margin-top:18px}.asset-dataset{border:1px solid var(--line);border-radius:18px;background:#fffafa;padding:17px}.asset-dataset-head{display:flex;justify-content:space-between;gap:14px;align-items:flex-start;flex-wrap:wrap}.asset-dataset-head h3{margin:0 0 5px;font-size:19px}.asset-path{color:var(--muted);font-size:12px;overflow-wrap:anywhere}.asset-badges{display:flex;gap:7px;flex-wrap:wrap}.asset-badge{display:inline-flex;padding:5px 8px;border-radius:999px;background:#fff;border:1px solid var(--line);font-size:11px;color:#765f5f}.asset-badge.ok{color:#167a4d;background:#eef9f3;border-color:#c4e8d3}.asset-badge.warn{color:#a85c18;background:#fff6e9;border-color:#f2d7b6}.asset-run-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:12px;margin-top:14px}.asset-run{border:1px solid var(--line);border-radius:15px;background:#fff;padding:14px;min-width:0}.asset-run h4{margin:0;font-size:16px}.asset-run-meta{display:grid;gap:5px;margin:10px 0;color:var(--muted);font-size:12px}.artifact-row{display:grid;gap:7px;padding:9px 0;border-top:1px solid #f0e2e2}.artifact-name{display:flex;justify-content:space-between;gap:8px;font-size:12px}.artifact-name b{overflow-wrap:anywhere}.asset-actions{display:flex;gap:7px;flex-wrap:wrap}.asset-actions .btn{padding:7px 10px;font-size:11px}.asset-deployments{margin-top:8px;padding-top:8px;border-top:1px dashed var(--line);font-size:12px;color:var(--muted)}@media(max-width:760px){.asset-summary{grid-template-columns:repeat(2,1fr)}.asset-run-grid{grid-template-columns:1fr}}
.model-toolbar{display:grid;grid-template-columns:repeat(2,minmax(0,240px));gap:12px;margin:16px 0}.model-family-list{display:grid;gap:16px}.model-family-card{padding:17px;border:1px solid var(--line);border-radius:16px;background:#fafbfd}.model-family-copy{max-width:760px}.model-family-copy h3{margin:0 0 5px}.model-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(215px,1fr));gap:11px;margin-top:14px}.base-model-card{display:grid;gap:10px;padding:14px;border:1px solid #e0e6ed;border-radius:13px;background:#fff;min-width:0}.base-model-card.downloaded{border-color:#b9dfca;background:#fbfffc}.base-model-title{display:flex;justify-content:space-between;gap:8px;align-items:flex-start}.base-model-title h4{margin:0;font-size:15px;overflow-wrap:anywhere}.resource-dots{display:flex;gap:3px}.resource-dots i{width:7px;height:7px;border-radius:50%;background:#dce3ea}.resource-dots i.on{background:#e04a4f}.model-license{margin:16px 0;padding:13px 15px;border:1px solid #f0d2a9;border-radius:13px;background:#fff8ed;color:#79562f;font-size:12px;line-height:1.65}@media(max-width:620px){.model-toolbar{grid-template-columns:1fr}.model-grid{grid-template-columns:1fr}}
.hero{display:block}.hero .title{display:flex;align-items:center;gap:22px;padding:18px 24px}.hero h1{font-size:30px;margin:0;white-space:nowrap}.hero .subtitle{line-height:1.55}.guide{display:none}.project-summary{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:16px 0}.project-summary div{padding:14px;border:1px solid var(--line);border-radius:14px;background:#fffafa}.project-summary span{display:block;color:var(--muted);font-size:12px}.project-summary b{display:block;font-size:24px;margin-top:4px}.project-list{display:grid;gap:14px;margin-top:18px}.project-card{border:1px solid var(--line);border-radius:17px;padding:16px;background:#fffafa}.project-card.active{border-color:#e88d8d;box-shadow:0 0 0 3px #fff0f0}.project-head{display:flex;justify-content:space-between;gap:14px;align-items:flex-start;flex-wrap:wrap}.project-head h3{margin:0 0 5px}.project-health{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}.health{padding:5px 9px;border-radius:999px;border:1px solid var(--line);font-size:11px}.health.ok{color:#167a4d;background:#eef9f3;border-color:#c4e8d3}.health.warning{color:#a85c18;background:#fff6e9;border-color:#f2d7b6}.health.error{color:#bd2525;background:#fff0f0;border-color:#efc5c5}.project-issues{display:grid;gap:5px;color:#9b4c36;font-size:12px;margin:10px 0}.project-preview{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:9px}.project-preview figure{margin:0;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#fff}.project-preview img{width:100%;aspect-ratio:4/3;object-fit:cover;display:block}.project-preview figcaption{padding:7px;font-size:10px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.project-preview figure.missing{border-color:#e7a6a6}.project-create{padding:16px;border-radius:16px;background:#fffafa;border:1px solid var(--line)}button:focus-visible,input:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid rgba(229,57,53,.28);outline-offset:2px}@media(max-width:980px){.hero .title{display:block}.hero h1{margin:10px 0}.project-summary{grid-template-columns:1fr 1fr 1fr}}@media(max-width:620px){.project-summary{grid-template-columns:1fr}.hero .title{padding:16px}.hero h1{font-size:25px}.section{padding:16px}}
/* v3.2.2 calm workspace theme */
:root{color-scheme:light;--bg:#f3f6fa;--panel:#fff;--panel2:#f7f9fc;--text:#182230;--muted:#667386;--line:#dfe5ec;--line-strong:#cbd4df;--blue:#2f6fed;--green:#18845c;--purple:#7b61d1;--orange:#b86b1f;--red:#d83b42;--brand:#df3f45;--brand-dark:#bd2930;--brand-soft:#fff0f1;--shadow:0 10px 30px rgba(23,34,48,.065),0 1px 3px rgba(23,34,48,.05);font-family:"Microsoft YaHei UI","Segoe UI Variable","Segoe UI",system-ui,sans-serif}
html{scroll-behavior:smooth}body{color:var(--text);background:linear-gradient(180deg,#f9fafc 0,#f3f6fa 260px);font-size:14px;line-height:1.5}body:before{content:"";position:fixed;inset:0 0 auto;height:260px;pointer-events:none;background:radial-gradient(circle at 12% -25%,rgba(223,63,69,.10),transparent 42%),radial-gradient(circle at 82% -30%,rgba(47,111,237,.08),transparent 38%);z-index:-1}
.wrap{width:min(1440px,calc(100% - 40px));padding:18px 0 34px}.card{background:rgba(255,255,255,.97);border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow)}
.hero{margin-bottom:14px}.hero .title{display:grid;grid-template-columns:auto minmax(280px,1fr) minmax(240px,480px) auto;gap:18px;align-items:center;padding:16px 20px;min-height:82px}.brand-mark{width:46px;height:46px;display:grid;place-items:center;border-radius:14px;background:linear-gradient(145deg,#ef555b,#c92d34);box-shadow:0 8px 18px rgba(216,59,66,.22);color:#fff;font-size:19px;font-weight:850;letter-spacing:-.04em}.brand-copy{min-width:0}.brand-kicker{display:block;margin-bottom:1px;color:#8a5660;font-size:10px;font-weight:800;letter-spacing:.13em}.hero h1{margin:0;color:#182230;font-size:26px;font-weight:800;letter-spacing:-.035em;white-space:nowrap}.hero .subtitle{max-width:none;color:#6b7686;font-size:13px;line-height:1.65}.eyebrow{justify-self:end;color:#526173;background:#f7f9fc;border-color:#dce3eb;padding:8px 11px;white-space:nowrap}.eyebrow .dot{background:#24a36f;box-shadow:0 0 0 4px #dff5eb}
.layout{grid-template-columns:228px minmax(0,1fr);gap:14px}.side{top:14px;padding:12px;max-height:calc(100vh - 28px);overflow:auto;scrollbar-width:thin}.nav{gap:4px}.nav button{position:relative;justify-content:flex-start;gap:11px;min-height:43px;padding:8px 10px;color:#59677a;border-radius:11px;font-size:13px;font-weight:600;transition:background .16s,color .16s,transform .16s}.nav button:hover{color:#252f3d;background:#f4f6f9;transform:translateX(1px)}.nav button.active{color:#b5262d;background:var(--brand-soft);border-color:#f4c9cc;box-shadow:inset 3px 0 0 var(--brand)}.nav button span{order:-1;display:grid;place-items:center;flex:0 0 28px;width:28px;height:28px;color:#7a8797;background:#f1f4f7;border:1px solid #e2e7ed;border-radius:9px;font-size:10px;font-variant-numeric:tabular-nums}.nav button.active span{color:#fff;background:var(--brand);border-color:var(--brand)}
.status{margin-top:10px;padding:12px;background:#f7f9fc;border-color:#e1e6ec;border-radius:12px}.pill{padding:6px 9px;font-size:12px}.pill.idle{color:#526173;background:#fff}.pill.run{color:#126b49;background:#eaf8f1;border-color:#bfe7d2}.status .mini{margin:8px 2px 0}
.main{gap:14px;min-width:0}.section{padding:24px;margin-bottom:0}.section>h2{position:relative;margin:0 0 6px;padding-left:14px;color:#192432;font-size:22px;font-weight:800;letter-spacing:-.02em}.section>h2:before{content:"";position:absolute;left:0;top:.22em;width:4px;height:1.05em;border-radius:99px;background:var(--brand)}.hint{margin-bottom:20px;color:#6d7888;line-height:1.7}.section>h2+.hint{padding-bottom:17px;border-bottom:1px solid #edf0f4}
.grid{gap:16px 14px}.field{min-width:0}label{margin-bottom:7px;color:#435064;font-size:12px;font-weight:700}input,select{min-height:42px;padding:10px 12px;color:#1e2938;background:#fff;border-color:#d7dee7;border-radius:10px;box-shadow:0 1px 2px rgba(20,31,46,.025);transition:border-color .16s,box-shadow .16s,background .16s}input::placeholder{color:#9ba6b4}input:hover,select:hover{border-color:#bbc6d3}input:focus,select:focus{border-color:#4e7fe7;box-shadow:0 0 0 3px rgba(47,111,237,.12)}input[readonly]{color:#586679;background:#f7f9fb}
.choice{gap:8px}.choice span{min-height:39px;align-items:center;padding:8px 12px;color:#5f6d7f;background:#fff;border-color:#d8dfe7;border-radius:10px}.choice span:hover{border-color:#b7c2d0;background:#fafbfc}.choice input:checked+span{color:#b1262d;border-color:#e9aeb2;background:#fff1f2;box-shadow:inset 0 0 0 1px rgba(216,59,66,.08)}
.actions{margin-top:20px;padding-top:16px;border-top:1px solid #edf0f4}.btns{gap:8px}.btn{display:inline-flex;align-items:center;justify-content:center;min-height:38px;padding:8px 13px;color:#3d4a5d;background:#fff;border-color:#d5dde6;border-radius:9px;font-size:12px;font-weight:750;line-height:1.3;box-shadow:0 1px 2px rgba(23,34,48,.04);transition:transform .15s,box-shadow .15s,background .15s,border-color .15s}.btn:hover{background:#f7f9fb;border-color:#b9c4d1;box-shadow:0 4px 12px rgba(23,34,48,.08);transform:translateY(-1px)}.btn:active{transform:translateY(0);box-shadow:none}.btn.primary{color:#fff;background:linear-gradient(135deg,#e3494f,#cf3037);border-color:#cf3037;box-shadow:0 5px 13px rgba(216,59,66,.18)}.btn.primary:hover{background:linear-gradient(135deg,#d93a41,#bd252d)}.btn.green{color:#fff;background:linear-gradient(135deg,#24966c,#147553);border-color:#147553;box-shadow:0 5px 13px rgba(24,132,92,.18)}.btn.green:hover{background:linear-gradient(135deg,#1d865f,#0f6848)}.btn.blue{color:#fff;background:linear-gradient(135deg,#3d78ed,#265fcf);border-color:#265fcf;box-shadow:0 5px 13px rgba(47,111,237,.18)}.btn.blue:hover{background:linear-gradient(135deg,#316be0,#1f53bd)}.btn.red{color:#b4232a;background:#fff5f5;border-color:#efc3c6}.btn.red:hover{color:#9f1d24;background:#ffebec;border-color:#e8a8ad}.btn[disabled],.preset[disabled]{opacity:.48;transform:none;box-shadow:none}
.onboarding{margin:16px 0 20px;padding:16px;background:linear-gradient(135deg,#f8faff,#fff);border-color:#dbe4f1;border-radius:14px}.onboarding-head b{color:#253247;font-size:15px}.onboarding-head span{color:#748094}.preset-row{gap:8px}.preset{padding:8px 12px;color:#475569;background:#fff;border-color:#d7e0ea;border-radius:9px;font-size:12px;font-weight:650}.preset:hover{color:#285ec5;border-color:#9bb7ea;background:#f4f7ff}.advanced-toggle{color:#285ec5;background:#f4f7ff;border-color:#bfd0f1}
.project-summary,.asset-summary{gap:10px}.project-summary div,.asset-summary div{position:relative;padding:14px 16px;background:#f8fafc;border-color:#e2e7ed;border-radius:12px;overflow:hidden}.project-summary div:after,.asset-summary div:after{content:"";position:absolute;right:-12px;bottom:-20px;width:58px;height:58px;border-radius:50%;background:rgba(47,111,237,.055)}.project-summary span,.asset-summary span{color:#748094;font-size:11px;font-weight:650}.project-summary b,.asset-summary b{color:#202b3a;font-size:23px;font-variant-numeric:tabular-nums}.project-create,.project-card,.asset-dataset{background:#fafbfd;border-color:#e1e6ed;border-radius:14px}.project-create{padding:18px}.project-create h3{color:#263246}.project-card.active{border-color:#e5a3a7;box-shadow:0 0 0 3px #fff0f1}.asset-run{border-color:#e0e6ed;border-radius:12px;box-shadow:0 2px 7px rgba(23,34,48,.035)}.asset-badge{color:#59677a;background:#f7f9fb;border-color:#dce3ea}.health,.asset-badge{font-weight:650}
.progress-card,.panel{background:#fafbfd;border-color:#e1e6ed;border-radius:14px}.metrics-grid div,.quick-help div{background:#fff;border-color:#e1e6ed;border-radius:10px}.metrics-grid b,.quick-help b,.marker b,.video-item b,.progress-head b,.panel h3{color:#263246}.bar{height:12px;background:#e8edf3;border-color:#dce3eb}.bar div{background:linear-gradient(90deg,#22a171,#3f78e5)}canvas{background:#fff;border-color:#e0e6ed}.readiness .check-item,.marker,.sample,.video-item,.label-object{background:#fff;border-color:#dfe5ec;border-radius:11px}.empty{margin-top:14px;padding:18px;color:#7b8797;background:#fafbfd;border-color:#ccd5e0;border-radius:12px;text-align:center}.cmd{color:#42526a;background:#f5f7fa;border-color:#dce3eb;border-radius:11px}.log{color:#c8f7dd;background:#111827;border-color:#273449;border-radius:12px}.toast{right:20px;bottom:20px;color:#fff;background:#253247;border-color:#33435a;border-radius:11px;box-shadow:0 14px 34px rgba(18,28,43,.22)}.last-error{color:#ad242b;background:#fff1f2;border-color:#f0c5c8}.current-video,.label-status span{color:#2b5dbd;background:#f1f5ff;border-color:#cbd9f5}.video-preview-box{background:#f5f7fa;border-color:#d8e0e8}.label-stage{border-color:#d9e1e9;border-radius:13px}.play-button{background:#2f6fed;box-shadow:0 12px 28px rgba(47,111,237,.25)}
.input-action{gap:8px}.field-note{color:#526173}.split-controls{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr) 170px;gap:10px;align-items:stretch;margin-top:9px}.split-controls>div{padding:11px 13px;border:1px solid #dfe5ec;border-radius:11px;background:#fafbfd}.split-controls span{display:flex;justify-content:space-between;gap:10px;color:#657286;font-size:11px;font-weight:650}.split-controls span b{color:#263246;font-size:12px}.split-controls input[type=range]{min-height:28px;padding:0;box-shadow:none;border:0;background:transparent;accent-color:var(--brand)}.split-controls .test-share{display:flex;align-items:center;justify-content:space-between;gap:12px;background:#f3f7ff;border-color:#d7e2f5}.test-share span{display:block}.test-share>b{color:#275fc8;font-size:22px;font-variant-numeric:tabular-nums}button:focus-visible,input:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid rgba(47,111,237,.22);outline-offset:2px}.side::-webkit-scrollbar,.video-list::-webkit-scrollbar,.log::-webkit-scrollbar{width:8px;height:8px}.side::-webkit-scrollbar-thumb,.video-list::-webkit-scrollbar-thumb,.log::-webkit-scrollbar-thumb{background:#cbd4df;border-radius:99px;border:2px solid transparent;background-clip:padding-box}
@media(max-width:1120px){.hero .title{grid-template-columns:auto minmax(250px,1fr) auto}.hero .subtitle{display:none}.layout{grid-template-columns:214px minmax(0,1fr)}.section{padding:21px}.metrics-grid{grid-template-columns:repeat(3,1fr)}}
@media(max-width:980px){.wrap{width:min(100% - 24px,1440px);padding-top:12px}.hero .title{display:grid;grid-template-columns:auto 1fr auto}.hero h1{margin:0}.layout{grid-template-columns:1fr}.side{position:sticky;top:0;z-index:30;max-height:none;padding:7px;border-radius:13px}.nav{display:flex;grid-template-columns:none;gap:3px;overflow-x:auto;scrollbar-width:none}.nav::-webkit-scrollbar{display:none}.nav button{flex:1 0 auto;justify-content:center;min-height:40px;padding:7px 10px;white-space:nowrap}.nav button:hover{transform:none}.nav button.active{box-shadow:inset 0 -3px 0 var(--brand)}.nav button span{display:none}.status{display:none}.field,.field.sm,.label-config .field,.label-config .field.sm{grid-column:span 6}.section{padding:20px}}
@media(max-width:720px){.wrap{width:min(100% - 16px,1440px)}.hero .title{grid-template-columns:auto 1fr;padding:13px 14px;min-height:0}.brand-mark{width:40px;height:40px;border-radius:12px}.brand-kicker{display:none}.hero h1{font-size:21px}.hero .subtitle{display:none}.hero .eyebrow{grid-column:1/-1;justify-self:stretch;justify-content:center}.field,.field.sm,.label-config .field,.label-config .field.sm{grid-column:1/-1}.section{padding:17px}.section>h2{font-size:20px}.project-summary{grid-template-columns:repeat(3,1fr)}.asset-summary{grid-template-columns:repeat(2,1fr)}.actions{align-items:stretch}.actions>.btn,.actions>.btns{width:100%}.actions>.btns .btn{flex:1}.input-action{flex-direction:column}.input-action .btn{width:auto}.split-controls{grid-template-columns:1fr}.quick-help{grid-template-columns:1fr 1fr}.toast{left:12px;right:12px;bottom:12px;max-width:none}}
@media(max-width:480px){.project-summary{grid-template-columns:1fr}.quick-help{grid-template-columns:1fr}.section{padding:15px}.nav button span{display:none}.nav button{padding-inline:12px}.asset-summary{grid-template-columns:1fr 1fr}}
/* v3.2.6 full-viewport desktop workspace */
@media(min-width:981px){
html,body{height:100%;overflow:hidden}
body:before{display:none}
.wrap{width:100%;height:100vh;margin:0;padding:0}
.hero{height:64px;margin:0}
.hero .title{display:grid;grid-template-columns:auto minmax(210px,auto) minmax(0,1fr) auto;gap:14px;align-items:center;width:100%;height:64px;min-height:64px;padding:8px 16px;border-width:0 0 1px;border-radius:0;box-shadow:0 1px 8px rgba(31,43,58,.07)}
.brand-mark{width:40px;height:40px;border-radius:11px;font-size:16px}
.brand-kicker{display:none}
.hero h1{font-size:21px}
.hero .subtitle{font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hero .eyebrow{padding:6px 9px;font-size:11px}
.layout{height:calc(100vh - 64px);grid-template-columns:214px minmax(0,1fr);gap:0;transition:grid-template-columns .18s ease}
.side{position:relative;top:0;height:100%;max-height:none;padding:10px 9px;border-width:0 1px 0 0;border-radius:0;box-shadow:none;overflow:auto}
.main{display:block;height:100%;min-width:0;padding:16px 18px 24px;overflow:auto}
.section{min-height:calc(100vh - 104px);margin:0;padding:20px;border-radius:14px}
.nav button{display:grid;grid-template-columns:26px minmax(0,1fr);gap:9px;min-height:39px;padding:6px 9px;text-align:left}
.nav button .nav-icon{order:initial;display:grid;place-items:center;flex:none;width:26px;height:26px;color:#69778a;background:#f1f4f7;border:1px solid #e2e7ed;border-radius:8px}
.nav button .nav-icon svg{display:block;width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.nav button .nav-label{order:initial;display:block;width:auto;height:auto;min-width:0;color:inherit;background:none;border:0;border-radius:0;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.nav button.active .nav-icon{color:#fff;background:var(--brand);border-color:var(--brand)}
.nav button.active .nav-label{color:inherit;background:none;border:0}
.sidebar-toggle{all:unset;box-sizing:border-box;display:flex;width:100%;min-height:34px;margin-bottom:8px;padding:6px 9px;align-items:center;justify-content:space-between;gap:8px;cursor:pointer;color:#657286;border:1px solid #dfe5ec;border-radius:9px;background:#f7f9fc;font-size:11px;font-weight:700}
.sidebar-toggle:hover{color:#275fc8;border-color:#cbd8ee;background:#f1f5ff}
.sidebar-toggle span{display:grid;width:20px;height:20px;place-items:center;color:#275fc8;border-radius:6px;background:#e7efff;font-size:18px;line-height:1;transition:transform .18s ease}
body.sidebar-collapsed .layout{grid-template-columns:68px minmax(0,1fr)}
body.sidebar-collapsed .sidebar-toggle{justify-content:center;padding-inline:4px}
body.sidebar-collapsed .sidebar-toggle b{display:none}
body.sidebar-collapsed .sidebar-toggle span{transform:rotate(180deg)}
body.sidebar-collapsed .nav button{grid-template-columns:1fr;justify-content:center;padding-inline:5px;font-size:13px}
body.sidebar-collapsed .nav button .nav-icon{display:grid;justify-self:center;width:32px;height:32px;border-radius:9px}
body.sidebar-collapsed .nav button .nav-icon svg{width:20px;height:20px}
body.sidebar-collapsed .nav button .nav-label{display:none}
body.sidebar-collapsed .status{display:none}
}
@media(max-width:980px){
html,body{height:auto;overflow:auto}
.wrap{width:100%;height:auto;padding-top:0}
.hero{height:auto}
.hero .title{height:auto;min-height:58px;border-width:0 0 1px;border-radius:0}
.layout,body.sidebar-collapsed .layout{height:auto;grid-template-columns:1fr;gap:0}
.side{position:sticky;top:0;height:auto;max-height:none;border-width:0 0 1px;border-radius:0;box-shadow:0 4px 12px rgba(31,43,58,.06)}
.sidebar-toggle{display:none}
.main{height:auto;padding:12px;overflow:visible}
.section{min-height:0;margin:0;padding:18px}
body.sidebar-collapsed .nav button{grid-template-columns:26px minmax(0,1fr);justify-content:center;padding:7px 10px;font-size:13px}
body.sidebar-collapsed .nav button .nav-icon{display:grid}
body.sidebar-collapsed .nav button .nav-label{display:block}
body.sidebar-collapsed .status{display:none}
}
@media(max-width:980px){.nav button .nav-icon{display:grid}.nav button .nav-label{display:block}}
@media(max-width:720px){.main{padding:8px}.section{padding:16px}.hero .title{padding:10px 12px}.nav button{grid-template-columns:22px minmax(0,1fr);gap:6px}.nav button .nav-icon{width:22px;height:22px;border:0;background:transparent}.nav button.active .nav-icon{color:var(--brand);background:transparent}.nav button .nav-label{font-size:12px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*,*:before,*:after{transition:none!important;animation:none!important}}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <div class="card title">
      <div class="brand-mark" aria-hidden="true">YO</div>
      <div class="brand-copy"><span class="brand-kicker">LOCAL AI WORKBENCH</span><h1>YOLO团队训练平台</h1></div>
      <p class="subtitle">从数据准备、训练资产追踪到多设备部署都在这里完成。每次训练会记录数据集与模型的对应关系，方便后续测试、导出和复现。</p>
      <div id="connectionBadge" class="eyebrow"><span class="dot"></span><span>面板已连接 · 本机服务</span></div>
    </div>
    <div class="card guide">
      <div class="steps">
        <div class="step"><div class="num">1</div><div><b>选择数据</b><span>用“选择文件夹”按钮指定图片、XML 标注和输出目录。</span></div></div>
        <div class="step"><div class="num">2</div><div><b>检查是否就绪</b><span>自动检查图片/标注匹配、YOLO、CUDA 和显卡状态。</span></div></div>
        <div class="step"><div class="num">3</div><div><b>开始训练</b><span>实时查看进度；训练完成后可以直接测试或导出模型。</span></div></div>
      </div>
    </div>
  </div>

  <div class="layout">
    <aside class="card side">
      <svg class="nav-icon-defs" aria-hidden="true" width="0" height="0" style="position:absolute;overflow:hidden">
        <symbol id="nav-icon-projects" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1.5"></rect><rect x="14" y="3" width="7" height="7" rx="1.5"></rect><rect x="3" y="14" width="7" height="7" rx="1.5"></rect><rect x="14" y="14" width="7" height="7" rx="1.5"></rect></symbol>
        <symbol id="nav-icon-train" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"></circle><path d="m10 8 6 4-6 4z"></path></symbol>
        <symbol id="nav-icon-dataset" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="8" ry="3"></ellipse><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"></path></symbol>
        <symbol id="nav-icon-models" viewBox="0 0 24 24"><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9zM4.5 7.8 12 12l7.5-4.2M12 12v9"></path></symbol>
        <symbol id="nav-icon-assets" viewBox="0 0 24 24"><path d="M4 8h16v12H4zM3 4h18v4H3zM9 12h6"></path></symbol>
        <symbol id="nav-icon-test" viewBox="0 0 24 24"><path d="M9 3h6M10 3v5l-5.5 9.5A2.3 2.3 0 0 0 6.5 21h11a2.3 2.3 0 0 0 2-3.5L14 8V3M7.5 15h9"></path><circle cx="10" cy="17.5" r=".7"></circle></symbol>
        <symbol id="nav-icon-collab" viewBox="0 0 24 24"><circle cx="9" cy="8" r="3"></circle><circle cx="17" cy="9" r="2.5"></circle><path d="M3.5 20v-2a5.5 5.5 0 0 1 11 0v2M14.5 14.5a4.5 4.5 0 0 1 6 4.25V20"></path></symbol>
        <symbol id="nav-icon-label" viewBox="0 0 24 24"><path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5"></path><path d="m8 16 1.2-3.8L16.4 5 19 7.6l-7.2 7.2zM14.8 6.6l2.6 2.6"></path></symbol>
        <symbol id="nav-icon-convert" viewBox="0 0 24 24"><rect x="3" y="7" width="13" height="13" rx="2"></rect><path d="M8 3h13v13M14 10l7-7M16 3h5v5"></path></symbol>
        <symbol id="nav-icon-logs" viewBox="0 0 24 24"><path d="M6 3h9l4 4v14H6zM15 3v5h4M9 12h6M9 16h6"></path></symbol>
      </svg>
      <button id="sidebar-toggle" class="sidebar-toggle" type="button" onclick="toggleSidebar()" aria-expanded="true" aria-label="收起导航"><span aria-hidden="true">‹</span><b>收起导航</b></button>
      <div class="nav" role="navigation" aria-label="平台功能导航">
        <button data-tab="projects" class="active" title="项目中心"><span class="nav-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><use href="#nav-icon-projects"></use></svg></span><span class="nav-label">项目中心</span></button>
        <button data-tab="train" title="开始训练"><span class="nav-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><use href="#nav-icon-train"></use></svg></span><span class="nav-label">开始训练</span></button>
        <button data-tab="dataset" title="准备数据"><span class="nav-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><use href="#nav-icon-dataset"></use></svg></span><span class="nav-label">准备数据</span></button>
        <button data-tab="models" title="基础模型"><span class="nav-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><use href="#nav-icon-models"></use></svg></span><span class="nav-label">基础模型</span></button>
        <button data-tab="assets" title="模型资产"><span class="nav-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><use href="#nav-icon-assets"></use></svg></span><span class="nav-label">模型资产</span></button>
        <button data-tab="test" title="测试模型"><span class="nav-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><use href="#nav-icon-test"></use></svg></span><span class="nav-label">测试模型</span></button>
        <button data-tab="collab" title="协作标注"><span class="nav-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><use href="#nav-icon-collab"></use></svg></span><span class="nav-label">协作标注</span></button>
        <button data-tab="label" title="快速标注"><span class="nav-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><use href="#nav-icon-label"></use></svg></span><span class="nav-label">快速标注</span></button>
        <button data-tab="convert" title="部署与导出"><span class="nav-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><use href="#nav-icon-convert"></use></svg></span><span class="nav-label">部署与导出</span></button>
        <button data-tab="logs" title="运行记录"><span class="nav-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><use href="#nav-icon-logs"></use></svg></span><span class="nav-label">运行记录</span></button>

      </div>
      <div class="status">
        <div id="runPill" class="pill idle"><span class="dot"></span><span>空闲</span></div>
        <div class="mini" id="jobInfo">暂无任务</div>
        <div id="lastError" class="last-error" hidden></div>
        <div class="markers" id="markers"></div>
      </div>
    </aside>

    <main class="main">
      <section id="tab-projects" class="tab active card section">
        <h2>项目中心</h2><p class="hint">每个项目统一关联数据集、类别、训练参数和模型资产。先激活项目，再进入训练、标注或导出页面。</p>
        <div class="project-summary"><div><span>项目</span><b id="project-count">0</b></div><div><span>图片</span><b id="project-image-count">0</b></div><div><span>待处理问题</span><b id="project-issue-count">0</b></div></div>
        <div class="project-create">
          <h3 style="margin-top:0">新建项目</h3>
          <div class="grid">
            <div class="field"><label>项目名称</label><input id="new-project-name" placeholder="例如：产线零件缺陷检测"></div>
            <div class="field"><label>任务类型</label><select id="new-project-task"><option value="detect">目标检测</option><option value="classify">图像分类</option></select></div>
            <div class="field"><label>类别名称</label><input id="new-project-labels" placeholder="多个类别用逗号分隔，例如 defect, normal"></div>
            <div class="field"><label>数据集目录（可选）</label><div class="input-action"><input id="new-project-root" placeholder="留空则自动创建规范目录"><button class="btn" onclick="pickProjectDatasetRoot()">选择文件夹</button></div></div>
          </div>
          <div class="actions"><span class="mini">默认位置由平台数据目录管理；也可以关联已有 YOLO/VOC/分类数据集。</span><button class="btn primary" onclick="createPlatformProject()">创建并激活</button></div>
        </div>
        <div class="actions"><span class="mini" id="project-active-note">尚未激活项目</span><button class="btn" onclick="loadProjects(true)">重新检查数据集</button></div>
        <div id="project-list" class="project-list"><div class="empty">正在读取项目...</div></div>
      </section>

      <section id="tab-train" class="tab card section">
        <h2>开始训练</h2><p class="hint">可直接导入下载好的 YOLO 数据集，也可继续使用图片 + XML。其他参数已按 640×480 稳定训练配置准备好。</p>
        <div class="onboarding">
          <div class="onboarding-head"><div><b>选择训练方式</b><span>不确定时使用“640×480 推荐”；训练不会自动早停，由你手动结束。</span></div><button id="advanced-toggle" class="preset advanced-toggle" onclick="toggleAdvancedSettings()">更多设置</button></div>
          <div class="preset-row">
            <button class="preset" onclick="pickRawDatasetRoot(true)">导入下载的数据集</button>
            <button class="preset" onclick="applyTrainPreset('smoke')">首次试跑 · 5 轮</button>
            <button class="preset" onclick="applyTrainPreset('camera4060')">640×480 推荐</button>
            <button class="preset advanced-setting" onclick="applyTrainPreset('balanced')">固定训练 · 100 轮</button>
            <button class="preset" onclick="applyTrainPreset('quality')">持续训练 · 手动停止</button>
            <button class="preset" onclick="checkTrainReady(true)">只检查，不启动</button>
          </div>
        </div>
        <div id="prepared-dataset-banner" class="readiness" hidden></div>
        <div id="train-readiness" class="readiness" hidden></div>
        <div class="grid">
          <input id="prepared_dataset_yaml" type="hidden">
          <div class="field full advanced-setting"><label>训练任务</label><div class="choice"><label><input name="train_task" type="radio" value="detect"><span>目标检测（图片 + XML 标注）</span></label><label><input name="train_task" type="radio" value="classify"><span>图像分类（类别子文件夹）</span></label></div><div class="mini" id="train-task-hint">检测：Images Dir 与 Annotations Dir 一一对应；分类：Images Dir 下每个子文件夹即一个类别，例如 normal/、defect/。</div></div>
          <div class="field full"><label>训练输出目录</label><div class="input-action"><input id="dataset_root" placeholder="训练结果和模型将保存在这里"><button class="btn" onclick="pickTrainDirectory('dataset_root')">选择文件夹</button></div></div>
          <div class="field"><label>训练图片目录</label><div class="input-action"><input id="train_images_dir" placeholder="例如 D:/dataset/images"><button class="btn" onclick="pickTrainDirectory('train_images_dir')">选择文件夹</button></div></div>
          <div class="field" id="annotations-field"><label>XML 标注目录</label><div class="input-action"><input id="train_annotations_dir" placeholder="例如 D:/dataset/annotations"><button class="btn" onclick="pickTrainDirectory('train_annotations_dir')">选择文件夹</button></div></div>
          <div class="field full split-setting"><label>数据集划分：<span id="split_ratio_text" aria-live="polite">训练 80% / 验证 10% / 测试 10%</span></label><div class="split-controls"><div><span>训练集 <b id="split_train_value">80%</b></span><input id="train_ratio_percent" type="range" min="1" max="99" step="1" aria-label="训练集比例"></div><div><span>验证集 <b id="split_val_value">10%</b></span><input id="val_ratio_percent" type="range" min="1" max="99" step="1" aria-label="验证集比例"></div><div class="test-share"><span>测试集（自动计算）</span><b id="split_test_value">10%</b></div></div><div class="mini">推荐 80% / 10% / 10%。测试集只用于训练完成后的独立评估；将训练与验证之和调到 100% 可关闭测试集。</div></div>
          <div class="field advanced-setting"><label>基础模型</label><div class="input-action"><input id="base_model" placeholder="推荐 yolo11n.pt"><button class="btn" onclick="pickBaseModel()">选择模型</button></div><div class="mini">8GB 显存建议优先使用 yolo11n.pt。</div></div>

          <div class="field advanced-setting"><label>Conda 环境（通常留空）</label><input id="conda_env"></div>
          <div class="field advanced-setting"><label>PyTorch CUDA</label><select id="torch_cuda"><option value="cu128">CUDA 12.8（本机已安装）</option><option value="cu124">CUDA 12.4</option><option value="cu121">CUDA 12.1</option><option value="cu118">CUDA 11.8</option><option value="cpu">CPU 版 PyTorch</option><option value="none">不自动安装/更新 PyTorch</option></select></div>
          <div class="field advanced-setting"><label>训练设备</label><div class="choice"><label><input name="train_device" type="radio" value="cuda"><span>GPU（推荐）</span></label><label><input name="train_device" type="radio" value="cpu"><span>CPU</span></label></div></div>
          <div class="field advanced-setting"><label>数据缓存</label><select id="train_cache"><option value="False">关闭缓存（默认）</option><option value="True">缓存到内存</option><option value="disk">缓存到磁盘</option></select><div class="mini">内存足够时可加速训练；数据集较大建议选择磁盘或关闭。</div></div>
          <div class="field advanced-setting"><label>项目名称</label><input id="project_name"></div>

          <div class="field advanced-setting"><label>导出模型名称</label><input id="model_name"></div>
          <div class="field sm advanced-setting"><label>图片宽度</label><input id="img_width" type="number" min="32" step="32" inputmode="numeric"></div>
          <div class="field sm advanced-setting"><label>图片高度</label><input id="img_height" type="number" min="32" step="32" inputmode="numeric"></div>
          <div class="field advanced-setting"><label>图片适配方式</label><select id="image_resize_mode"><option value="crop">裁剪（居中裁剪后缩放）</option><option value="letterbox">等比缩放（留边填充）</option><option value="stretch">拉伸（直接缩放）</option></select><div class="mini">会在生成训练集时同步变换图片与标注框；推荐使用等比缩放。</div></div>
          <div class="field sm advanced-setting"><label>训练轮数</label><input id="epochs"><div class="mini">持续训练模式使用10000作为技术上限，通常由你手动停止。</div></div>
          <div class="field sm advanced-setting"><label>批量大小</label><input id="batch"></div>
          <div class="field sm advanced-setting"><label>初始学习率</label><input id="lr0"></div>
          <div class="field sm advanced-setting"><label>数据加载进程</label><input id="train_workers" type="number" min="0" max="16" step="1"><div class="mini">RTX 4060 推荐4；0会让GPU频繁等待CPU。</div></div>
          <div class="field sm advanced-setting"><label>自动早停</label><input id="patience" type="number" min="0" step="1"><div class="mini">0=关闭自动早停；当前默认由你手动决定何时停止。</div></div>
          <div class="field full advanced-setting"><div class="onboarding"><div class="onboarding-head"><div><b>实际训练尺寸</b><span>宽高不同时自动启用 YOLO 矩形训练。640×480 输入将保持4:3张量，不再自动补成640×640。</span></div></div></div></div>
          <div class="field advanced-setting"><label>导出算子模式</label><div class="choice"><label><input name="operator_mode" type="radio" value="recommended"><span>推荐算子</span></label><label><input name="operator_mode" type="radio" value="maixcam"><span>仅 MaixCAM 支持</span></label></div></div>
          <div class="field advanced-setting"><label>训练位置</label><div class="choice"><label><input name="train_mode" type="radio" value="local"><span>本机训练</span></label><label><input name="train_mode" type="radio" value="remote-windows"><span>远程 Windows</span></label></div></div>
          <div class="field advanced-setting"><label>Remote User</label><input id="remote_train_user" placeholder="如 Administrator"></div>
          <div class="field advanced-setting"><label>Remote Host</label><input id="remote_train_host"></div>
          <div class="field advanced-setting"><label>Remote Port</label><input id="remote_train_port"></div>
          <div class="field advanced-setting"><label>Remote Work Dir</label><input id="remote_train_work_dir"></div>
        </div>
        <div class="train-board">
          <div class="progress-card wide">
            <div class="progress-head"><b>资源预估</b><span id="resource-note">配置变化后自动估算</span></div>
            <div class="metrics-grid">
              <div><span>训练图片</span><b id="estimate-images">-</b></div>
              <div><span>内存</span><b id="estimate-ram">-</b></div>
              <div><span>显存</span><b id="estimate-vram">-</b></div>
              <div><span>Cache 额外</span><b id="estimate-cache">-</b></div>
              <div><span>图片宽 × 高</span><b id="estimate-imgsz">-</b></div>
              <div><span>Batch</span><b id="estimate-batch">-</b></div>
            </div>
            <div class="mini" id="resource-detail">估算值仅供参考，实际峰值会随模型、增强策略、驱动和环境波动。</div>
          </div>
        </div>
        <div class="actions"><div class="btns"><button class="btn advanced-setting" onclick="runAction('train_ssh')">测试训练 SSH</button><button class="btn advanced-setting" onclick="copyCommand('train')">复制训练命令</button><button class="btn" onclick="saveDefaults('训练配置')">保存设置（重启保留）</button><button class="btn red" onclick="stopTrainExport()">停止训练并导出最佳模型</button></div><button id="start-train-button" class="btn green" onclick="runAction('train')">检查并开始训练</button></div>
        <div class="train-board">

          <div class="progress-card wide">
            <div class="progress-head"><b>训练进度</b><span id="train-phase">等待开始</span></div>
            <div class="bar"><div id="epoch-bar" style="width:0%"></div></div>
            <div class="metrics-grid">
              <div><span>Epoch</span><b id="epoch-text">-</b></div>
              <div><span>Batch</span><b id="batch-text">-</b></div>
              <div><span>GPU</span><b id="gpu-text">-</b></div>
              <div><span>速度</span><b id="speed-text">-</b></div>
              <div><span>耗时</span><b id="elapsed-text">-</b></div>
              <div><span>剩余</span><b id="eta-text">-</b></div>
            </div>
          </div>
          <div class="progress-card">
            <div class="progress-head"><b>Loss</b><span id="loss-title">训练损失</span></div>
            <div class="metrics-grid loss-grid">
              <div id="loss-item"><span id="loss-label">loss</span><b id="loss-value">-</b></div>
              <div id="box-loss-item"><span>box</span><b id="box-loss">-</b></div>
              <div id="cls-loss-item"><span>cls</span><b id="cls-loss">-</b></div>
              <div id="dfl-loss-item"><span>dfl</span><b id="dfl-loss">-</b></div>
            </div>
            <canvas id="loss-chart" width="520" height="180"></canvas>
          </div>
          <div class="progress-card">
            <div class="progress-head"><b>验证指标</b><span id="val-text">-</span></div>
            <div id="detect-metrics" class="metrics-grid loss-grid">
              <div><span>Precision</span><b id="precision-text">-</b></div>
              <div><span>Recall</span><b id="recall-text">-</b></div>
              <div><span>mAP50</span><b id="map50-text">-</b></div>
              <div><span>mAP50-95</span><b id="map5095-text">-</b></div>
            </div>
            <div id="classify-metrics" class="metrics-grid loss-grid" hidden>
              <div><span>Top-1 Accuracy</span><b id="top1-text">-</b></div>
              <div><span>Top-5 Accuracy</span><b id="top5-text">-</b></div>
            </div>
            <canvas id="metric-chart" width="520" height="180"></canvas>
          </div>
        </div>
        <div class="cmd" id="cmd-train"></div>
      </section>

      <section id="tab-dataset" class="tab card section">
        <h2>准备数据</h2>
        <p class="hint">Roboflow/YOLO 导出的 <b>data.yaml + train/valid/test</b> 可直接导入，不需要 XML。旧式平铺 TXT 数据仍可转换。</p>
        <div class="onboarding">
          <div class="onboarding-head"><div><b>优先：直接导入下载目录</b><span>选择最外层文件夹，系统自动检查 data.yaml、图片、TXT 标签和原始数据划分，不复制也不改动文件。</span></div></div>
        </div>
        <div id="raw-dataset-summary" class="readiness" hidden></div>
        <div class="grid">
          <div class="field full"><label>原始数据集根目录</label><div class="input-action"><input id="raw_dataset_root" placeholder="例如 C:/Users/YourName/Downloads/dataset"><button class="btn" onclick="pickRawDatasetRoot()">选择文件夹</button></div></div>
          <div class="field"><label>图片目录（可自动识别）</label><input id="raw_images_dir" placeholder="例如 .../moxin/Main"></div>
          <div class="field"><label>YOLO TXT 标签目录（可自动识别）</label><input id="raw_labels_dir" placeholder="例如 .../moxin/Main_labels"></div>
          <div class="field full"><label>转换输出目录</label><input id="raw_output_dir" placeholder="留空则输出到原始目录/converted_voc"></div>
          <div class="field full"><label>类别名称（按编号顺序，用逗号分隔）</label><input id="raw_class_names" placeholder="例如 product；多类别填写 product, defect, background"><div class="mini">如果留空，类别 0、1、2 会自动命名为 class_0、class_1、class_2。</div></div>
          <div class="field full"><div class="choice"><label><input id="raw_overwrite" type="checkbox"><span>覆盖输出目录中的同名图片和 XML</span></label></div></div>
        </div>
        <div class="actions">
          <div class="btns"><button class="btn" onclick="inspectRawDataset()">识别并检查</button></div>
          <div class="btns"><button id="raw-import-button" class="btn green" onclick="importRawDataset()">直接导入训练配置</button><button id="raw-convert-button" class="btn advanced-setting" onclick="convertRawDataset()">旧式 TXT 转 XML</button></div>
        </div>
      </section>

      <section id="tab-models" class="tab card section">
        <h2>官方基础模型中心</h2><p class="hint">选择适合任务和设备的 Ultralytics 官方预训练权重。模型只下载到本机 Workspace，完成后可直接设为训练基础模型。</p>
        <div class="asset-summary">
          <div><span>可选模型</span><b id="base-model-total">0</b></div>
          <div><span>已下载</span><b id="base-model-downloaded">0</b></div>
          <div><span>支持任务</span><b>检测 / 分类</b></div>
          <div><span>本地目录</span><b id="base-model-root" style="font-size:12px;overflow-wrap:anywhere">-</b></div>
        </div>
        <div class="model-license" id="base-model-license">官方模型遵循 Ultralytics 许可证。用于公开分发或商业产品前，请确认许可证要求。</div>
        <div class="model-toolbar">
          <div><label for="base-model-family-filter">模型系列</label><select id="base-model-family-filter" onchange="renderBaseModels()"><option value="all">全部系列</option></select></div>
          <div><label for="base-model-task-filter">训练任务</label><select id="base-model-task-filter" onchange="renderBaseModels()"><option value="detect">目标检测</option><option value="classify">图像分类</option></select></div>
        </div>
        <div id="base-model-catalog" class="model-family-list"><div class="empty">正在读取基础模型目录...</div></div>
      </section>

      <section id="tab-assets" class="tab card section">
        <h2>数据集与模型资产</h2><p class="hint">独立查看“哪个数据集训练出了哪些模型”。每次训练都写入规范化运行目录和可追溯清单。</p>
        <div class="asset-summary">
          <div><span>数据集</span><b id="asset-dataset-count">0</b></div>
          <div><span>训练记录</span><b id="asset-run-count">0</b></div>
          <div><span>模型文件</span><b id="asset-model-count">0</b></div>
          <div><span>部署产物</span><b id="asset-deployment-count">0</b></div>
        </div>
        <div class="grid" style="margin-top:18px">
          <div class="field full"><label>补充扫描目录</label><div class="input-action"><input id="asset_scan_root" placeholder="选择包含 training-manifest.json 的训练运行根目录"><button class="btn" onclick="pickAssetRoot()">选择文件夹</button><button class="btn blue" onclick="scanModelAssets()">扫描并记住</button></div><div class="mini">平台工作区会自动扫描；这里只用于补充外部训练运行目录。</div></div>
          <div class="field full"><label>登记已有模型</label><div class="input-action"><input id="external-model-path" placeholder="选择已有 .pt 或 .onnx 文件"><button class="btn" onclick="pickExternalModel()">选择模型</button></div></div>
          <div class="field"><label>关联数据集 / 项目名称</label><input id="external-dataset-name" placeholder="例如：零件缺陷数据集 v2"></div>
          <div class="field sm"><label>任务</label><select id="external-model-task"><option value="detect">目标检测</option><option value="classify">图像分类</option><option value="unknown">未知</option></select></div>
          <div class="field sm"><label>类别</label><input id="external-model-labels" placeholder="defect, normal"></div>
          <div class="field full"><label>备注</label><div class="input-action"><input id="external-model-notes" placeholder="来源、训练设置或适用设备"><button class="btn primary" onclick="registerExternalModel()">登记到资产库</button></div></div>
          <div class="field full"><label>搜索</label><input id="asset-search" placeholder="按数据集、模型、类别、任务或路径筛选" oninput="renderModelAssets()"></div>
        </div>
        <div class="actions"><span class="mini" id="asset-roots-note">尚未读取资产目录</span><div class="btns"><button class="btn" onclick="compareSelectedModels()">比较已选模型</button><button class="btn" onclick="loadModelAssets(true)">刷新资产</button></div></div>
        <div id="model-compare" class="panel" style="margin-top:14px" hidden></div>
        <div id="model-assets" class="asset-library"><div class="empty">正在读取模型资产...</div></div>
      </section>

      <section id="tab-convert" class="tab card section">
        <h2>多平台部署与导出</h2><p class="hint">训练结束后会自动填入规范化的 model-best.pt。先选择目标设备，再生成适合该平台的模型和部署清单；厂商专用工具链与训练环境相互隔离。</p>
        <div class="grid">
          <div class="field full"><label>训练模型（.pt 或已有 .onnx）</label><div class="input-action"><input id="deploy_model" placeholder="训练完成后自动填入 model-best.pt"><button class="btn" onclick="pickDeployModel()">选择模型</button></div></div>
          <div class="field"><label>目标平台</label><select id="deployment_target" onchange="updateDeploymentProfile()"><option value="generic_onnx">通用平台 / ONNX Runtime</option><option value="raspberry_pi">树莓派（CPU）</option><option value="rockchip_rknn">香橙派 / Rockchip RKNN</option><option value="drobotics_rdk">地瓜机器人 RDK X3 / X5</option><option value="maixcam">Sipeed MaixCAM</option><option value="nvidia_jetson">NVIDIA Jetson / TensorRT</option><option value="intel_openvino">Intel / OpenVINO</option></select></div>
          <div class="field"><label>导出格式</label><select id="export_format"><option value="auto">自动推荐</option><option value="onnx">ONNX</option><option value="ncnn">NCNN</option><option value="openvino">OpenVINO</option><option value="engine">TensorRT engine</option><option value="rknn">RKNN</option></select></div>
          <div class="field"><label>目标芯片（可选）</label><input id="export_chip" placeholder="例如 rk3588、x5"></div>
          <div class="field"><label>data.yaml（INT8 时必填）</label><input id="export_data" placeholder="代表性校准数据"></div>
          <div class="field full"><label>输出目录</label><div class="input-action"><input id="export_output_dir"><button class="btn" onclick="pickDeployOutputDir()">选择文件夹</button></div></div>
          <div class="field full"><div class="choice"><label><input id="export_int8" type="checkbox"><span>INT8 量化（需要 data.yaml；部署前必须复测精度）</span></label></div></div>
        </div>
        <div id="deployment-profile-note" class="check-list"></div>
        <div class="actions"><div class="btns"><button class="btn" onclick="copyCommand('export')">复制导出命令</button><button class="btn" onclick="saveDefaults('部署设置')">存为默认</button></div><button class="btn blue" onclick="runAction('export')">生成部署模型</button></div>
        <div class="cmd" id="cmd-export"></div>

        <div class="advanced-setting" style="margin-top:28px">
          <h2>MaixCAM 专用转换（兼容旧流程）</h2><p class="hint">仅在需要 .cvimodel + .mud 时使用。其他平台不要填写此区域。</p>
          <div class="grid">
            <div class="field full"><label>ONNX Model</label><input id="model_path"></div>
            <div class="field"><label>Classes</label><input id="classes_path"></div>
            <div class="field"><label>Calib Images</label><input id="calib_dir"></div>
            <div class="field"><label>Test Image</label><input id="test_image"></div>
            <div class="field"><label>VM User</label><input id="vm_user"></div>
            <div class="field"><label>VM Host/IP</label><input id="vm_host"></div>
            <div class="field full"><label>VM Work Dir</label><input id="vm_work_dir"></div>
            <div class="field full"><div class="choice"><label><input id="skip_vm_convert" type="checkbox"><span>只上传转换包，跳过 VM 转换</span></label></div></div>
          </div>
          <div class="actions"><div class="btns"><button class="btn" onclick="runAction('vm_ssh')">测试 VM SSH</button><button class="btn" onclick="copyCommand('convert')">复制 MaixCAM 命令</button></div><button class="btn" onclick="runAction('convert')">运行 MaixCAM 转换</button></div>
        </div>
        <div class="cmd" id="cmd-convert"></div>
      </section>

      <section id="tab-test" class="tab card section">
        <h2>模型测试</h2><p class="hint">自动兼容目标检测与图像分类 `.pt` 模型。单张图片会保存带标注的预测图；文件夹测试会递归处理所有图片并输出预测图与 `predictions.csv`。</p>
        <div class="grid">
          <div class="field full"><label>Test Model .pt</label><input id="test_model"></div>
          <div class="field full"><label>Source</label><div class="choice"><label><input name="test_source" type="radio" value="camera"><span>Camera</span></label><label><input name="test_source" type="radio" value="image"><span>单张图片</span></label><label><input name="test_source" type="radio" value="folder"><span>图片文件夹</span></label></div></div>
          <div class="field full"><label>图片路径</label><div class="input-action"><input id="test_image_file" placeholder="选择单张图片时填写"><button class="btn" onclick="pickTestImage()">选择图片</button></div></div>
          <div class="field full"><label>图片文件夹路径</label><div class="input-action"><input id="test_image_folder" placeholder="选择图片文件夹时填写，支持递归扫描"><button class="btn" onclick="pickTestImageFolder()">选择文件夹</button></div></div>
          <div class="field full"><label>批量测试输出根目录</label><div class="input-action"><input id="test_output_dir" placeholder="默认保存到 workspace/test-results"><button class="btn" onclick="pickTestOutputDir()">选择文件夹</button></div></div>
          <div class="field"><label>Camera Index</label><input id="camera_index"></div>
          <div class="field"><label>Confidence（仅检测模型生效）</label><input id="conf"></div>
        </div>
        <div class="actions"><button class="btn" onclick="copyCommand('test')">复制测试命令</button><button class="btn primary" onclick="runAction('test')">开始测试</button></div>
        <div class="cmd" id="cmd-test"></div>
      </section>

      <section id="tab-collab" class="tab card section">
        <h2>本地优先的协作标注中心</h2><p class="hint">每台电脑都能独立标注；需要合作时，任意成员都可以临时开启局域网共享。伙伴只用浏览器，不需要 Docker、Python 或额外客户端。</p>
        <div class="asset-summary">
          <div><span>服务状态</span><b id="annotation-service-state">未启动</b></div>
          <div><span>当前模式</span><b id="annotation-service-mode">-</b></div>
          <div><span>服务端口</span><b>9000</b></div>
          <div><span>项目工作区</span><b id="annotation-workspace-state">本机</b></div>
        </div>
        <div class="onboarding" style="margin-top:18px">
          <div class="onboarding-head"><div><b>个人模式</b><span>仅当前电脑可以访问。适合离线整理、个人框选和制作可移植项目包。</span></div><button class="btn primary" onclick="runAnnotationService('annotation_personal')">启动个人标注</button></div>
        </div>
        <div class="onboarding" style="margin-top:12px">
          <div class="onboarding-head"><div><b>局域网共享模式</b><span>同一 Wi-Fi 或网线网络中的伙伴可使用浏览器登录、领取任务、提交和审核。</span></div><button class="btn green" onclick="runAnnotationService('annotation_share')">开启团队共享</button></div>
        </div>
        <div class="grid" style="margin-top:16px">
          <div class="field full"><label>本机入口</label><div class="input-action"><input id="annotation-local-url" readonly value="http://127.0.0.1:9000/"><button class="btn blue" onclick="openAnnotationCenter()">打开标注中心</button></div></div>
          <div class="field full"><label>伙伴访问地址</label><div id="annotation-lan-urls" class="cmd">共享模式启动后显示局域网地址</div></div>
        </div>
        <div class="quick-help">
          <div><b>1. 每个人独立</b><span>使用个人模式建立自己的项目；数据保存在各自工作区的 annotation-hub。</span></div>
          <div><b>2. 任意电脑共享</b><span>谁开启共享，谁就是当前团队主机；其他成员访问页面即可协作。</span></div>
          <div><b>3. 项目包交换</b><span>下载 .ytp-project.zip 后可导入另一台电脑；已有标注进入待审核，不覆盖原项目。</span></div>
          <div><b>4. 进入训练</b><span>审核通过后导出 Ultralytics YOLO，也可下载 COCO、VOC 或 LabelMe。</span></div>
        </div>
        <div class="actions"><div class="btns"><button class="btn" onclick="loadAnnotationService(true)">刷新状态</button><button class="btn red" onclick="runAnnotationService('annotation_stop')">停止标注服务</button></div><a class="btn" href="docs/COLLABORATIVE_ANNOTATION.md" target="_blank">查看使用说明</a></div>
      </section>

      <section id="tab-label" class="tab card section">
        <h2>单机快速标注（需人工复核）</h2><p class="hint">适合临时采样和视频跟踪；正式多人项目请使用“协作标注”。跟踪只负责把上一帧的框带到下一帧，不替代人工检查。</p>
        <div class="grid">
          <div class="field full"><label>标注来源</label><div class="choice"><label><input name="label_source_type" type="radio" value="video"><span>视频</span></label><label><input name="label_source_type" type="radio" value="camera"><span>摄像头</span></label><label><input name="label_source_type" type="radio" value="images"><span>图片集</span></label></div></div>
        </div>
        <div id="label-video-source" class="label-source">
          <div class="grid">
            <div class="field full"><label>视频文件夹</label><input id="label_video_dir" placeholder="例如 D:/videos 或 E:/datasets/raw_videos"></div>
          </div>
          <div class="actions"><div class="btns"><button class="btn" onclick="pickLabelVideoDir()">选择视频文件夹</button><button class="btn blue" onclick="loadLabelVideos()">读取文件夹视频</button></div><div class="btns"><button class="btn" onclick="selectPrevVideo()">上一个</button><button class="btn" onclick="selectNextVideo()">下一个</button></div></div>
          <div class="label-workspace">
            <div class="panel">
              <div class="queue-head"><h3>视频队列</h3><span class="count" id="label-video-count">0 个视频</span></div>
              <div id="label-video-list" class="video-list"><div class="empty">填写视频文件夹后点击“读取文件夹视频”。</div></div>
              <div class="video-preview">
                <div class="queue-head"><h3>视频预览</h3><span class="count" id="label-preview-name">未选择</span></div>
                <div id="label-video-preview" class="video-preview-box"><span>选择左侧视频后显示首帧预览</span></div>
              </div>
            </div>
            <div class="panel" id="label-video-config">
              <h3>当前视频与标注参数</h3>
              <div class="current-video"><span class="count">当前待标注视频</span><b id="label-current-video">未选择视频</b></div>
              <div class="label-config">
                <div class="field full"><label>视频路径</label><input id="label_video" placeholder="从队列选择，或手动输入视频文件路径"></div>
                <div class="field"><label>类别名称</label><input id="label_name" placeholder="多个标签用英文逗号分隔，如 w,person,car"></div>
                <div class="field"><label>文件名前缀</label><input id="label_prefix"></div>
                <div class="field"><label>图片输出目录</label><input id="label_images_dir"></div>
                <div class="field"><label>标注输出目录</label><input id="label_annotations_dir"></div>
                <div class="field"><label>跟踪策略</label><select id="label_tracker"><option value="csrt">稳健跟踪（推荐，较慢）</option><option value="kcf">快速跟踪（画面稳定）</option><option value="template">兼容模式（无需额外跟踪器）</option><option value="multi_template">多视角实验模式</option></select></div>

                <div class="field sm"><label>Save Every N Frames</label><input id="label_interval"></div>
                <div class="field sm"><label>Start Frame</label><input id="label_start_frame"></div>
                <div class="field sm"><label>Max Frames</label><input id="label_max_frames"></div>
                <div class="field sm"><label>Display Scale</label><input id="label_display_scale"></div>
                <div class="field sm"><label>JPEG Quality</label><input id="label_jpeg_quality"></div>
              </div>
            </div>
          </div>
        </div>
        <div id="label-camera-source" class="label-source" hidden>
          <div class="panel">
            <h3>摄像头实时标注</h3>
            <p class="hint">启动后在网页画面框选目标。跟踪过程中按保存间隔写入 JPEG 和同名 VOC XML；发现丢失或漂移时会暂停，修正后再继续。</p>
            <div class="label-config">
              <div class="field"><label>Camera Index</label><input id="label_camera_index" placeholder="0"></div>
              <div class="field"><label>类别名称</label><input id="label_name_camera" placeholder="多个标签用英文逗号分隔，如 w,person,car"></div>
              <div class="field"><label>文件名前缀</label><input id="label_prefix_camera" placeholder="camera"></div>
              <div class="field"><label>图片输出目录</label><input id="label_images_dir_camera"></div>
              <div class="field"><label>标注输出目录</label><input id="label_annotations_dir_camera"></div>
              <div class="field"><label>跟踪策略</label><select id="label_tracker_camera"><option value="csrt">稳健跟踪（推荐，较慢）</option><option value="kcf">快速跟踪（画面稳定）</option><option value="template">兼容模式（无需额外跟踪器）</option><option value="multi_template">多视角实验模式</option></select></div>
              <div class="field sm"><label>Save Every N Frames</label><input id="label_interval_camera"></div>
              <div class="field sm"><label>Max Frames</label><input id="label_max_frames_camera" placeholder="0 为持续采集"></div>
              <div class="field sm"><label>Display Scale</label><input id="label_display_scale_camera"></div>
              <div class="field sm"><label>JPEG Quality</label><input id="label_jpeg_quality_camera"></div>
            </div>
          </div>
        </div>
        <div id="label-images-source" class="label-source" hidden>
          <div class="panel">
            <h3>图片集与标注参数</h3>
            <p class="hint">选择已按时间顺序命名的帧图片文件夹。将按文件名排序跟踪，原图保留不动，仅在标注目录生成同名 XML。</p>
            <div class="label-config">
              <div class="field full"><label>图片集文件夹</label><div class="input-action"><input id="label_images_input_dir" placeholder="例如 E:/datasets/selected_frames（支持 jpg、png、bmp、webp）"><button class="btn" onclick="pickLabelImagesDir()">选择文件夹</button></div></div>
              <div class="field"><label>类别名称</label><input id="label_name_images" placeholder="多个标签用英文逗号分隔，如 w,person,car"></div>
              <div class="field"><label>标注输出目录</label><input id="label_annotations_dir_images"></div>
              <div class="field"><label>跟踪策略</label><select id="label_tracker_images"><option value="csrt">稳健跟踪（推荐，较慢）</option><option value="kcf">快速跟踪（画面稳定）</option><option value="template">兼容模式（无需额外跟踪器）</option><option value="multi_template">多视角实验模式</option></select></div>
              <div class="field sm"><label>每 N 张保存</label><input id="label_interval_images"></div>
              <div class="field sm"><label>起始图片序号</label><input id="label_start_frame_images"></div>
              <div class="field sm"><label>最多处理图片</label><input id="label_max_frames_images"></div>
              <div class="field sm"><label>显示缩放</label><input id="label_display_scale_images"></div>
            </div>
            <div class="quick-help"><div><b>1. 选择图片集</b>选择已筛选并按时间顺序命名的帧图片。</div><div><b>2. 添加首帧框</b>在首张图片拖动框选目标并选择类别。</div><div><b>3. 逐段复核</b>播放几帧后暂停检查；异常时选中目标并点“修正选中框”。</div><div><b>4. 保存结果</b>只保存所有目标都有效的帧，每张原图对应同名 XML。</div></div>
          </div>
        </div>
        <div id="label-browser-studio" class="label-studio panel" hidden>
          <div class="queue-head"><div><h3>网页标注工作台</h3><span class="count" id="label-session-tip">创建会话后，在画面上拖动鼠标框选目标。</span></div><button class="btn red" onclick="endBrowserLabelSession()">结束标注</button></div>
          <div class="label-studio-grid">
            <div>
              <div id="label-stage" class="label-stage empty-stage"><span>选择来源并点击“在网页中开始标注”后，此处显示实时画面。</span><img id="label-frame-image" hidden alt="当前标注画面"><canvas id="label-frame-canvas" hidden></canvas></div>
              <div id="label-session-status" class="label-status"></div>
              <div class="actions"><div class="btns"><button class="btn green" id="label-play-button" onclick="toggleBrowserLabelPlay()">播放跟踪</button><button class="btn" onclick="advanceBrowserLabelFrame()">下一帧</button><button class="btn" onclick="saveBrowserLabelFrame()">保存当前帧</button></div><div class="btns"><button class="btn" onclick="setLabelDrawMode('add')">添加框</button><button class="btn" onclick="setLabelDrawMode('edit')">修正选中框</button><button class="btn" onclick="setLabelDrawMode('sample')">追加选中目标视角</button><button class="btn red" onclick="deleteBrowserLabelObject()">删除选中框</button></div></div>
              <div class="label-help">建议每推进一小段就暂停抽查。红框或低质量提示不会自动保存；请选中目标后点“修正选中框”。多视角模式仍属实验功能，适合外观变化明显且背景较稳定的目标。</div>
            </div>
            <div class="panel"><h3>当前目标</h3><div id="label-object-list" class="label-object-list"><div class="empty">尚未框选目标。</div></div></div>
          </div>
        </div>
        <div class="actions"><div class="btns"><button class="btn advanced-setting" onclick="copyCommand('label')">复制兼容命令</button><button class="btn" onclick="saveDefaults('半自动标注')">存为默认</button></div><div class="btns"><button class="btn" onclick="loadLabelResults()">刷新标注结果</button><button class="btn primary" id="label-start-button" onclick="runLabelCurrent()">在网页中开始标注</button></div></div>
        <div class="cmd" id="cmd-label"></div>
        <h2 style="margin-top:24px">标注结果</h2><p class="hint">这里显示当前标注样本。视频模式下“删除废图”会同时删除导出图片和 XML；图片集模式下仅删除 XML，保留用户原始图片。</p>
        <div id="label-results" class="gallery"></div>

      </section>


      <section id="tab-logs" class="tab card section">
        <h2>运行日志</h2><p class="hint">这里实时显示训练、转换、SSH 测试、模型测试和视频打标输出。</p>
        <div class="actions"><div class="btns"><button class="btn" onclick="refreshState()">刷新</button><button class="btn" onclick="copyLogs()">复制日志</button></div><button class="btn red" onclick="stopJob()">停止当前任务</button></div>
        <div class="log" id="log"></div>
      </section>

    </main>
  </div>
</div>
<div class="toast" id="toast"></div>
<script>
const fields=['dataset_root','train_task','train_images_dir','train_annotations_dir','prepared_dataset_yaml','train_ratio_percent','val_ratio_percent','img_width','img_height','image_resize_mode','epochs','batch','train_workers','patience','lr0','conda_env','base_model','torch_cuda','train_cache','raw_dataset_root','raw_images_dir','raw_labels_dir','raw_output_dir','raw_class_names','asset_scan_root','project_name','model_name','remote_train_user','remote_train_host','remote_train_port','remote_train_work_dir','deploy_model','deployment_target','export_format','export_chip','export_output_dir','export_data','model_path','classes_path','calib_dir','test_image','vm_user','vm_host','vm_work_dir','test_model','test_image_file','test_image_folder','test_output_dir','camera_index','conf','label_video_dir','label_video','label_camera_index','label_source_type','label_images_input_dir','label_name','label_interval','label_images_dir','label_annotations_dir','label_prefix','label_tracker','label_start_frame','label_max_frames','label_display_scale','label_jpeg_quality'];



let values={};
let projectCatalog={active_project_id:'',projects:[],summary:{}};
let deploymentProfiles=[];
let modelAssetCatalog={summary:{},roots:[],datasets:[]};
let baseModelCatalog={summary:{},families:[],models:[]};
let modelDownloadWasRunning=false;
const modelCompareSelection=new Set();
let annotationServiceStatus={running:false,shared:false,lan_urls:[]};
let labelVideos=[];
let labelVideoIndex=-1;
let labelVisibleCount=150;
const LABEL_VIDEO_PAGE_SIZE=150;
let labelPreviewToken=0;
let labelResultsTimer=null;
let resourceEstimateTimer=null;
let lastResourceEstimateKey='';
let deleteConfirmUntil=0;


const rawLabelPrefix={manual:false,value:''};
function toast(msg){const el=document.getElementById('toast');el.textContent=msg;el.classList.add('show');setTimeout(()=>el.classList.remove('show'),2600)}
function applySidebarState(collapsed){document.body.classList.toggle('sidebar-collapsed',collapsed);const toggle=document.getElementById('sidebar-toggle');if(!toggle)return;toggle.setAttribute('aria-expanded',String(!collapsed));toggle.setAttribute('aria-label',collapsed?'展开导航':'收起导航');const label=toggle.querySelector('b');if(label)label.textContent=collapsed?'展开导航':'收起导航'}
function toggleSidebar(){const collapsed=!document.body.classList.contains('sidebar-collapsed');localStorage.setItem('yoloTeamPlatformSidebarCollapsed',collapsed?'1':'0');applySidebarState(collapsed)}
function escapeHtml(value){return String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]))}
function renderProjects(){const catalog=projectCatalog||{projects:[],summary:{}},summary=catalog.summary||{};setText('project-count',summary.project_count||0);setText('project-image-count',summary.image_count||0);setText('project-issue-count',summary.issue_count||0);const active=(catalog.projects||[]).find(item=>item.active);setText('project-active-note',active?`当前项目：${active.name}`:'尚未激活项目');const box=document.getElementById('project-list');if(!box)return;box.innerHTML='';if(!(catalog.projects||[]).length){box.innerHTML='<div class="empty">还没有项目。创建第一个项目后，数据、训练和模型会按项目归档。</div>';return}for(const project of catalog.projects){const dataset=project.dataset||{},card=document.createElement('article');card.className='project-card'+(project.active?' active':'');const labels=(project.labels||[]).map(label=>`<span class="asset-badge">${escapeHtml(label)}</span>`).join('');const issues=(dataset.issues||[]).map(issue=>`<span>• ${escapeHtml(issue)}</span>`).join('');const preview=(dataset.preview||[]).slice(0,12).map(item=>`<figure class="${item.has_label?'':'missing'}"><img loading="lazy" src="/api/project-image?project_id=${encodeURIComponent(project.id)}&path=${encodeURIComponent(item.relative_path)}" alt="${escapeHtml(item.name)}"><figcaption>${escapeHtml(item.name)}${item.has_label?'':' · 缺标注'}</figcaption></figure>`).join('');card.innerHTML=`<div class="project-head"><div><h3>${escapeHtml(project.name)}</h3><div class="asset-path">${escapeHtml(project.id)} · ${escapeHtml(project.dataset_root||project.root||'')}</div></div><div class="asset-badges"><span class="asset-badge ${project.active?'ok':''}">${project.active?'当前项目':'未激活'}</span><span class="asset-badge">${project.task==='classify'?'图像分类':'目标检测'}</span>${labels}</div></div><div class="project-health"><span class="health ${escapeHtml(dataset.health||'error')}">${dataset.health==='ok'?'数据就绪':dataset.health==='warning'?'需要检查':'暂无有效数据'}</span><span class="health">${dataset.image_count||0} 张图片</span><span class="health">${dataset.label_count||0} 个标注</span><span class="health">${dataset.box_count||0} 个框</span></div>${issues?`<div class="project-issues">${issues}</div>`:''}<div class="field full"><label>关联数据集目录</label><div class="input-action"><input class="project-root-input" value="${escapeHtml(project.dataset_root||project.root||'')}"><button class="btn choose-root">选择</button><button class="btn save-root">保存</button></div></div>${preview?`<div class="project-preview">${preview}</div>`:'<div class="empty">目录中还没有可预览图片。</div>'}<div class="actions"><span class="mini">更新于 ${escapeHtml(assetTime(project.updated_at))}</span><div class="btns"><button class="btn inspect-project">检查数据</button><button class="btn primary activate-project">${project.active?'进入训练':'激活项目'}</button></div></div>`;const rootInput=card.querySelector('.project-root-input');card.querySelector('.choose-root').onclick=async()=>{const selected=await chooseProjectRoot(rootInput.value);if(selected)rootInput.value=selected};card.querySelector('.save-root').onclick=()=>updatePlatformProject(project.id,{dataset_root:rootInput.value});card.querySelector('.inspect-project').onclick=()=>inspectPlatformProject(project.id);card.querySelector('.activate-project').onclick=()=>activatePlatformProject(project.id,true);box.appendChild(card)}}
async function loadProjects(showMessage=false){try{const response=await fetch('/api/projects');if(!response.ok)throw new Error(`HTTP ${response.status}`);projectCatalog=await response.json();renderProjects();if(showMessage)toast('项目和数据集检查已刷新')}catch(error){toast(error.message)}}
async function chooseProjectRoot(initial=''){const result=await api('/api/pick-project-dataset-root',{initial});return result.path||''}
async function pickProjectDatasetRoot(){try{const input=document.getElementById('new-project-root');const selected=await chooseProjectRoot(input.value);if(selected)input.value=selected}catch(error){toast(error.message)}}
async function createPlatformProject(){try{const result=await api('/api/projects/create',{name:document.getElementById('new-project-name').value,task:document.getElementById('new-project-task').value,labels:document.getElementById('new-project-labels').value,dataset_root:document.getElementById('new-project-root').value});projectCatalog=result.catalog;apply(result.values||{});renderProjects();await saveValues();document.getElementById('new-project-name').value='';document.getElementById('new-project-labels').value='';document.getElementById('new-project-root').value='';toast('项目已创建并激活')}catch(error){toast(error.message)}}
async function activatePlatformProject(projectId,openTrain=false){try{const result=await api('/api/projects/activate',{project_id:projectId});projectCatalog=result.catalog;apply(result.values||{});renderProjects();await saveValues();toast(`已激活 ${result.project.name}`);if(openTrain)showTab('train')}catch(error){toast(error.message)}}
async function updatePlatformProject(projectId,updates){try{const result=await api('/api/projects/update',{project_id:projectId,updates});projectCatalog=result.catalog;apply(result.values||{});renderProjects();await saveValues();toast('项目已更新')}catch(error){toast(error.message)}}
async function inspectPlatformProject(projectId){try{const result=await api('/api/projects/inspect',{project_id:projectId});projectCatalog=result.catalog;renderProjects();toast((result.dataset.issues||[]).length?`发现 ${result.dataset.issues.length} 类数据问题`:'数据集检查通过')}catch(error){toast(error.message)}}
function assetTime(value){return String(value||'-').replace('T',' ').replace(/([+-]\d\d:\d\d|Z)$/,'')}
function metricValue(metrics,patterns,exclude=[]){for(const [key,value] of Object.entries(metrics||{})){const lower=key.toLowerCase(); if(patterns.some(x=>lower.includes(x))&&!exclude.some(x=>lower.includes(x))) return fmt(value,4)} return ''}
function assetMetricText(metrics){const parts=[]; const map50=metricValue(metrics,['map50'],['map50-95','map50_95']); const map95=metricValue(metrics,['map50-95','map50_95']); const precision=metricValue(metrics,['precision']); const recall=metricValue(metrics,['recall']); const top1=metricValue(metrics,['top1','accuracy_top1']); if(map50) parts.push(`mAP50 ${map50}`); if(map95) parts.push(`mAP50-95 ${map95}`); if(precision) parts.push(`P ${precision}`); if(recall) parts.push(`R ${recall}`); if(top1) parts.push(`Top-1 ${top1}`); return parts.join(' · ')}
function useModelAsset(path,action,classesPath=''){if(action==='test'){setInputValue('test_model',path); showTab('test'); toast('已填入模型测试页面')}else if(action==='deploy'){setInputValue('deploy_model',path); if(classesPath) setInputValue('classes_path',classesPath); showTab('convert'); toast('已填入部署与导出页面')} saveValues(); updateCommands()}
async function copyAssetPath(path){try{await navigator.clipboard.writeText(path);toast('路径已复制')}catch(e){toast('无法复制路径')}}
function renderBaseModels(){const catalog=baseModelCatalog||{summary:{},families:[],models:[]},summary=catalog.summary||{};setText('base-model-total',summary.model_count||0);setText('base-model-downloaded',summary.downloaded_count||0);setText('base-model-root',catalog.root||'-');const license=document.getElementById('base-model-license');if(license){const info=catalog.license||{};license.innerHTML=`${escapeHtml(info.notice||'请确认官方模型许可证要求。')} <a href="${escapeHtml(info.url||'https://www.ultralytics.com/license')}" target="_blank" rel="noopener">查看 ${escapeHtml(info.label||'许可证')}</a>`}const familyFilter=document.getElementById('base-model-family-filter');if(familyFilter&&familyFilter.options.length===1){for(const family of catalog.families||[]){const option=document.createElement('option');option.value=family.id;option.textContent=`${family.label} · ${family.status}`;familyFilter.appendChild(option)}}const familyId=familyFilter?.value||'all',task=document.getElementById('base-model-task-filter')?.value||'detect',box=document.getElementById('base-model-catalog');if(!box)return;box.innerHTML='';const families=(catalog.families||[]).filter(item=>familyId==='all'||item.id===familyId);for(const family of families){const models=(catalog.models||[]).filter(item=>item.family===family.id&&item.task===task);if(!models.length)continue;const section=document.createElement('section');section.className='model-family-card';section.innerHTML=`<div class="asset-dataset-head"><div class="model-family-copy"><h3>${escapeHtml(family.label)} <span class="asset-badge">${escapeHtml(family.year)}</span></h3><div class="asset-path">${escapeHtml(family.description)}</div></div><div class="asset-badges"><span class="asset-badge ${family.recommended?'ok':''}">${escapeHtml(family.status)}</span><a class="btn" href="${escapeHtml(family.docs_url)}" target="_blank" rel="noopener">官方文档</a></div></div>`;const grid=document.createElement('div');grid.className='model-grid';for(const model of models){const card=document.createElement('article');card.className='base-model-card'+(model.downloaded?' downloaded':'');card.innerHTML=`<div class="base-model-title"><div><h4>${escapeHtml(model.name)}</h4><div class="mini">${escapeHtml(model.scale_label)} · ${escapeHtml(model.guidance)}</div></div><div class="resource-dots" title="资源等级 ${model.resource_level}/5">${[1,2,3,4,5].map(level=>`<i class="${level<=model.resource_level?'on':''}"></i>`).join('')}</div></div><div class="asset-badges"><span class="asset-badge">${escapeHtml(model.task_label)}</span>${model.recommended?'<span class="asset-badge ok">推荐</span>':''}${model.downloaded?`<span class="asset-badge ok">已下载 · ${model.size_mb} MB</span>`:'<span class="asset-badge warn">需要联网</span>'}</div>`;const actions=document.createElement('div');actions.className='asset-actions';if(model.downloaded){const use=document.createElement('button');use.className='btn primary';use.textContent='用于训练';use.onclick=()=>useDownloadedBaseModel(model.path,model.task);const again=document.createElement('button');again.className='btn';again.textContent='重新下载';again.onclick=()=>downloadBaseModel(model.name,true);actions.append(use,again)}else{const download=document.createElement('button');download.className='btn primary';download.textContent='下载并用于训练';download.onclick=()=>downloadBaseModel(model.name,false);actions.appendChild(download)}card.appendChild(actions);grid.appendChild(card)}section.appendChild(grid);box.appendChild(section)}if(!box.children.length)box.innerHTML='<div class="empty">当前筛选条件下没有可用模型。</div>'}
async function loadBaseModels(showMessage=false){try{const response=await fetch('/api/base-models');if(!response.ok)throw new Error(`HTTP ${response.status}`);baseModelCatalog=await response.json();renderBaseModels();if(showMessage)toast(`已读取 ${baseModelCatalog.summary?.model_count||0} 个官方模型选项`)}catch(error){toast(error.message)}}
async function downloadBaseModel(name,force=false){if(force&&!confirm(`确定重新下载 ${name} 吗？\n现有文件会在新文件完整下载后才被替换。`))return;try{await api('/api/base-models/download',{name,force,values:collect()});modelDownloadWasRunning=true;showTab('logs');toast(`${name} 下载任务已启动，完成后会自动设为基础模型`);refreshState()}catch(error){toast(error.message)}}
async function useDownloadedBaseModel(path,task){setInputValue('base_model',path);const taskInput=document.querySelector(`input[name="train_task"][value="${task}"]`);if(taskInput)taskInput.checked=true;collect();updateTrainTaskUI();await saveValues();updateCommands();scheduleResourceEstimate();showTab('train');toast('已设为当前训练基础模型')}
function renderModelAssets(){const catalog=modelAssetCatalog||{summary:{},datasets:[]},summary=catalog.summary||{}; setText('asset-dataset-count',summary.dataset_count||0);setText('asset-run-count',summary.run_count||0);setText('asset-model-count',summary.model_count||0);setText('asset-deployment-count',summary.deployment_count||0); const roots=document.getElementById('asset-roots-note'); if(roots) roots.textContent=(catalog.roots||[]).length?`已索引 ${(catalog.roots||[]).length} 个目录`:'尚未登记扫描目录'; const box=document.getElementById('model-assets'); if(!box) return; const query=String(document.getElementById('asset-search')?.value||'').trim().toLowerCase(); const datasets=(catalog.datasets||[]).filter(item=>!query||JSON.stringify(item).toLowerCase().includes(query)); box.innerHTML=''; if(!datasets.length){box.innerHTML=`<div class="empty">${query?'没有匹配的模型资产。':'尚未找到模型资产。完成一次新训练，或登记已有模型。'}</div>`;return} for(const dataset of datasets){const section=document.createElement('section');section.className='asset-dataset'; const classes=(dataset.classes||[]).slice(0,12).map(name=>`<span class="asset-badge">${escapeHtml(name)}</span>`).join(''); section.innerHTML=`<div class="asset-dataset-head"><div><h3>${escapeHtml(dataset.name||'未命名数据集')}</h3><div class="asset-path">${escapeHtml(dataset.root||'')}</div>${dataset.source?`<div class="asset-path">来源：${escapeHtml(dataset.source)}</div>`:''}</div><div class="asset-badges"><span class="asset-badge ok">${(dataset.runs||[]).length} 条模型记录</span>${dataset.image_count?`<span class="asset-badge">${dataset.image_count} 张训练图片</span>`:''}${dataset.version?`<span class="asset-badge">版本 ${escapeHtml(dataset.version)}</span>`:''}${(dataset.tasks||[]).map(task=>`<span class="asset-badge">${escapeHtml(task==='detect'?'目标检测':task==='classify'?'图像分类':task)}</span>`).join('')}${classes}</div></div>`; const grid=document.createElement('div');grid.className='asset-run-grid'; for(const run of dataset.runs||[]){const card=document.createElement('article');card.className='asset-run'; const t=run.training||{},metricText=assetMetricText(run.metrics),runKey=run.manifest||run.artifacts?.find(item=>['pt','onnx'].includes(item.kind))?.path||run.run_id,testEvaluated=(run.artifacts||[]).some(item=>item.kind==='test_evaluation'&&item.exists); card.innerHTML=`<div class="asset-dataset-head"><h4>${escapeHtml(run.model_name)}</h4><div class="asset-badges"><label class="asset-badge"><input class="compare-model" type="checkbox" ${modelCompareSelection.has(runKey)?'checked':''}> 对比</label><span class="asset-badge ${run.association==='manual'?'warn':'ok'}">${run.association==='manual'?'手动登记':'训练清单'}</span>${testEvaluated?'<span class="asset-badge ok">独立测试已完成</span>':''}</div></div><div class="asset-run-meta"><span>${assetTime(run.created_at)} · ${escapeHtml(run.status)}</span><span>${t.input_size?`${t.input_size[1]}×${t.input_size[0]}`:'尺寸未知'}${t.base_model?' · 基础 '+escapeHtml(t.base_model):''}${t.epochs_requested?' · '+t.epochs_requested+' epochs':''}${t.batch?' · batch '+t.batch:''}</span>${metricText?`<span>${escapeHtml(metricText)}</span>`:''}${t.notes?`<span>${escapeHtml(t.notes)}</span>`:''}<span>${escapeHtml(run.output_dir)}</span></div>`;card.querySelector('.compare-model').onchange=event=>event.target.checked?modelCompareSelection.add(runKey):modelCompareSelection.delete(runKey); const classesArtifact=(run.artifacts||[]).find(item=>item.kind==='classes'&&item.exists); for(const artifact of (run.artifacts||[]).filter(item=>['pt','onnx'].includes(item.kind))){const row=document.createElement('div');row.className='artifact-row';row.innerHTML=`<div class="artifact-name"><b>${escapeHtml(artifact.name)}</b><span>${artifact.exists?artifact.size_mb+' MB':'文件缺失'}</span></div>`; const actions=document.createElement('div');actions.className='asset-actions'; if(artifact.exists&&artifact.kind==='pt'){const test=document.createElement('button');test.className='btn';test.textContent='用于测试';test.onclick=()=>useModelAsset(artifact.path,'test',classesArtifact?.path||'');actions.appendChild(test)} if(artifact.exists){const deploy=document.createElement('button');deploy.className='btn';deploy.textContent='用于部署';deploy.onclick=()=>useModelAsset(artifact.path,'deploy',classesArtifact?.path||''); const copy=document.createElement('button');copy.className='btn';copy.textContent='复制路径';copy.onclick=()=>copyAssetPath(artifact.path);actions.append(deploy,copy)} row.appendChild(actions);card.appendChild(row)} if((run.deployments||[]).length){const deployments=document.createElement('div');deployments.className='asset-deployments';deployments.textContent='已部署：'+run.deployments.map(item=>`${item.target_label} / ${String(item.format).toUpperCase()}${item.chip?' / '+item.chip:''}`).join('；');card.appendChild(deployments)} grid.appendChild(card)} section.appendChild(grid);box.appendChild(section)}}
async function loadModelAssets(showMessage=false){try{const r=await fetch('/api/model-assets');if(!r.ok) throw new Error(`HTTP ${r.status}`);modelAssetCatalog=await r.json();renderModelAssets();if(showMessage) toast(`已读取 ${modelAssetCatalog.summary?.run_count||0} 条训练记录`)}catch(e){toast(e.message)}}
async function pickExternalModel(){try{const input=document.getElementById('external-model-path');const result=await api('/api/pick-external-model',{initial:input.value});if(result.path)input.value=result.path}catch(error){toast(error.message)}}
async function registerExternalModel(){try{const active=(projectCatalog.projects||[]).find(item=>item.active);const result=await api('/api/model-assets/register',{model_path:document.getElementById('external-model-path').value,dataset_name:document.getElementById('external-dataset-name').value||active?.name||'未关联数据集',dataset_root:active?.dataset_root||'',project_id:active?.id||'',task:document.getElementById('external-model-task').value,labels:document.getElementById('external-model-labels').value||active?.labels||[],notes:document.getElementById('external-model-notes').value});modelAssetCatalog=result.catalog;renderModelAssets();toast('已有模型已登记到资产库')}catch(error){toast(error.message)}}
function compareSelectedModels(){const runs=[];for(const dataset of modelAssetCatalog.datasets||[])for(const run of dataset.runs||[]){const key=run.manifest||run.artifacts?.find(item=>['pt','onnx'].includes(item.kind))?.path||run.run_id;if(modelCompareSelection.has(key))runs.push({...run,datasetName:dataset.name})}const box=document.getElementById('model-compare');if(runs.length<2){box.hidden=true;toast('请至少勾选两个模型');return}box.hidden=false;box.innerHTML=`<div class="queue-head"><h3>模型对比</h3><span class="count">${runs.length} 个模型</span></div><div style="overflow:auto"><table style="width:100%;border-collapse:collapse;font-size:12px"><thead><tr><th style="text-align:left;padding:8px">模型</th><th>数据集</th><th>任务</th><th>输入尺寸</th><th>轮次</th><th>Batch</th><th>指标</th><th>文件</th></tr></thead><tbody>${runs.map(run=>{const t=run.training||{},artifact=(run.artifacts||[]).find(item=>['pt','onnx'].includes(item.kind));return `<tr style="border-top:1px solid var(--line)"><td style="padding:9px"><b>${escapeHtml(run.model_name)}</b></td><td>${escapeHtml(run.datasetName)}</td><td>${escapeHtml(run.task)}</td><td>${t.input_size?`${t.input_size[1]}×${t.input_size[0]}`:'-'}</td><td>${escapeHtml(t.epochs_requested||'-')}</td><td>${escapeHtml(t.batch||'-')}</td><td>${escapeHtml(assetMetricText(run.metrics)||'-')}</td><td>${artifact?.exists?artifact.size_mb+' MB':'缺失'}</td></tr>`}).join('')}</tbody></table></div>`;box.scrollIntoView({behavior:'smooth',block:'nearest'})}
async function pickAssetRoot(){try{const j=await api('/api/pick-asset-root',{values:collect()});if(j.path){setInputValue('asset_scan_root',j.path);await saveValues()}else toast('未选择文件夹')}catch(e){toast(e.message)}}
async function scanModelAssets(){try{const v=collect();const j=await api('/api/model-assets/scan',{root:v.asset_scan_root,values:v});apply(j.values||{});modelAssetCatalog=j.catalog||modelAssetCatalog;renderModelAssets();await saveValues();toast(`扫描完成：${modelAssetCatalog.summary?.run_count||0} 条训练记录`)}catch(e){toast(e.message)}}
function renderAnnotationService(){const s=annotationServiceStatus||{};setText('annotation-service-state',s.running?'运行中':'未启动');setText('annotation-service-mode',s.running?(s.shared?'局域网共享':'个人模式'):'-');setText('annotation-workspace-state',s.running?'已连接':'本机');const local=document.getElementById('annotation-local-url');if(local)local.value=s.local_url||'http://127.0.0.1:9000/';const urls=document.getElementById('annotation-lan-urls');if(urls)urls.textContent=s.running&&s.shared&&(s.lan_urls||[]).length?(s.lan_urls||[]).join('\n'):s.running?'当前仅本机访问；切换到共享模式后伙伴才能连接。':'共享模式启动后显示局域网地址'}
async function loadAnnotationService(showMessage=false){try{const response=await fetch('/api/annotation-service');if(!response.ok)throw new Error(`HTTP ${response.status}`);annotationServiceStatus=await response.json();renderAnnotationService();if(showMessage)toast(annotationServiceStatus.running?'标注服务运行正常':'标注服务尚未启动')}catch(error){toast(error.message)}}
async function runAnnotationService(action){try{await api('/api/run',{action,values:collect()});toast(action==='annotation_stop'?'正在停止标注服务':action==='annotation_share'?'正在开启局域网共享':'正在启动个人标注');showTab('logs');setTimeout(()=>loadAnnotationService(),1800)}catch(error){toast(error.message)}}
async function openAnnotationCenter(){await loadAnnotationService();if(!annotationServiceStatus.running){toast('请先启动个人标注或团队共享');return}window.open(annotationServiceStatus.local_url||'http://127.0.0.1:9000/','_blank','noopener')}
function updateSplitRatio(changed=''){const trainEl=document.getElementById('train_ratio_percent'),valEl=document.getElementById('val_ratio_percent'),text=document.getElementById('split_ratio_text');if(!trainEl||!valEl||!text)return;let train=Math.round(Number(trainEl.value||80)),val=Math.round(Number(valEl.value||10));if(!Number.isFinite(train))train=80;if(!Number.isFinite(val))val=10;train=Math.max(1,Math.min(99,train));val=Math.max(1,Math.min(99,val));if(train+val>100){if(changed==='val_ratio_percent')train=100-val;else val=100-train}const test=Math.max(0,100-train-val);trainEl.value=String(train);valEl.value=String(val);trainEl.max=String(100-val);valEl.max=String(100-train);if(typeof values==='object'){values.train_ratio_percent=String(train);values.val_ratio_percent=String(val)}text.textContent=`训练 ${train}% / 验证 ${val}% / 测试 ${test}%`;setText('split_train_value',`${train}%`);setText('split_val_value',`${val}%`);setText('split_test_value',`${test}%`)}
function fmt(v,d=3){return v===null||v===undefined||v===''||Number.isNaN(Number(v))?'-':Number(v).toFixed(d).replace(/\.0+$/,'').replace(/(\.\d*?)0+$/,'$1')}
function setText(id,value){const el=document.getElementById(id); if(el) el.textContent=value}
function drawChart(id,history,series){const canvas=document.getElementById(id); if(!canvas) return; const ctx=canvas.getContext('2d'); const w=canvas.width,h=canvas.height; ctx.clearRect(0,0,w,h); ctx.fillStyle='#fffafa'; ctx.fillRect(0,0,w,h); const rows=(history||[]).filter(x=>series.some(s=>x[s.key]!==null&&x[s.key]!==undefined)); ctx.strokeStyle='rgba(128,80,80,.12)'; ctx.lineWidth=1; for(let i=1;i<4;i++){const y=i*h/4; ctx.beginPath(); ctx.moveTo(34,y); ctx.lineTo(w-10,y); ctx.stroke()} if(rows.length<2){ctx.fillStyle='#8b7474'; ctx.font='13px sans-serif'; ctx.fillText('等待更多 epoch 数据...',18,34); return} let vals=[]; for(const r of rows){for(const s of series){const v=Number(r[s.key]); if(Number.isFinite(v)) vals.push(v)}} let min=Math.min(...vals),max=Math.max(...vals); if(min===max){min-=1;max+=1} const pad=(max-min)*0.08; min-=pad; max+=pad; const xOf=i=>34+i*(w-48)/Math.max(1,rows.length-1); const yOf=v=>h-22-(Number(v)-min)*(h-42)/(max-min); ctx.font='11px sans-serif'; series.forEach((s,si)=>{ctx.strokeStyle=s.color; ctx.lineWidth=2; ctx.beginPath(); let started=false; rows.forEach((r,i)=>{const v=Number(r[s.key]); if(!Number.isFinite(v)) return; const x=xOf(i),y=yOf(v); if(!started){ctx.moveTo(x,y); started=true}else ctx.lineTo(x,y)}); ctx.stroke(); ctx.fillStyle=s.color; ctx.fillText(s.label,38+si*82,16)}); ctx.fillStyle='#806e6e'; ctx.fillText(`E${rows[0].epoch}`,34,h-7); ctx.fillText(`E${rows[rows.length-1].epoch}`,w-52,h-7)}
function updateTrainProgress(p){p=p||{}; const task=p.task||document.querySelector('input[name="train_task"]:checked')?.value||'detect'; const classify=task==='classify'; const pct=Math.max(0,Math.min(100,Number(p.percent||0))); const totalEpochs=Number(p.total_epochs||0); const epoch=Number(p.epoch||0); const totalBatches=Number(p.total_batches||0); const batch=Number(p.batch||0); const phaseMap={idle:'等待开始',pending:'准备训练',train:'训练中',val:'验证中',metrics:'指标已更新',export:'正在停止训练并导出'}; setText('train-phase',`${phaseMap[p.phase]||p.phase||'等待开始'}${p.updated_at?' · '+p.updated_at:''}`); const bar=document.getElementById('epoch-bar'); if(bar) bar.style.width=pct+'%'; setText('epoch-text',epoch&&totalEpochs?`${epoch}/${totalEpochs}`:'-'); setText('batch-text',totalBatches?`${batch}/${totalBatches} (${fmt(pct,1)}%)`:'-'); setText('gpu-text',p.gpu_mem||'-'); const speedNumber=parseFloat(String(p.speed||'')); const configuredBatch=Math.max(1,Number(values.batch||1)); setText('speed-text',Number.isFinite(speedNumber)?`${p.speed} · ≈${fmt(speedNumber*configuredBatch,1)}图/秒`:(p.speed||'-')); setText('elapsed-text',p.elapsed||'-'); setText('eta-text',p.eta||'-'); const lossItem=document.getElementById('loss-item'); const boxItem=document.getElementById('box-loss-item'); const clsItem=document.getElementById('cls-loss-item'); const dflItem=document.getElementById('dfl-loss-item'); if(lossItem) lossItem.hidden=!classify; if(boxItem) boxItem.hidden=classify; if(clsItem) clsItem.hidden=classify; if(dflItem) dflItem.hidden=classify; setText('loss-title',classify?'分类训练损失':'检测训练损失'); setText('loss-value',fmt(p.loss)); setText('box-loss',fmt(p.box_loss)); setText('cls-loss',fmt(p.cls_loss)); setText('dfl-loss',fmt(p.dfl_loss)); const detectMetrics=document.getElementById('detect-metrics'); const classifyMetrics=document.getElementById('classify-metrics'); if(detectMetrics) detectMetrics.hidden=classify; if(classifyMetrics) classifyMetrics.hidden=!classify; const m=p.metrics||{}; setText('val-text',p.val_total?`${p.val_batch||0}/${p.val_total} (${fmt(p.val_percent||0,1)}%)`:'-'); setText('precision-text',fmt(m.precision)); setText('recall-text',fmt(m.recall)); setText('map50-text',fmt(m.map50)); setText('map5095-text',fmt(m.map50_95)); setText('top1-text',fmt(m.top1_acc)); setText('top5-text',fmt(m.top5_acc)); const lossSeries=classify?[{key:'loss',label:'loss',color:'#dc7a32'}]:[{key:'box_loss',label:'box',color:'#d92f2f'},{key:'cls_loss',label:'cls',color:'#ef6b68'},{key:'dfl_loss',label:'dfl',color:'#a64f4f'}]; const metricSeries=classify?[{key:'top1_acc',label:'Top-1',color:'#21885a'},{key:'top5_acc',label:'Top-5',color:'#d92f2f'}]:[{key:'precision',label:'P',color:'#21885a'},{key:'recall',label:'R',color:'#d92f2f'},{key:'map50',label:'mAP50',color:'#dc7a32'},{key:'map50_95',label:'mAP50-95',color:'#a64f4f'}]; drawChart('loss-chart',p.history||[],lossSeries); drawChart('metric-chart',p.history||[],metricSeries)}
function updatePreparedDatasetUI(){const active=!!String(values.prepared_dataset_yaml||'').trim(); const banner=document.getElementById('prepared-dataset-banner'); const annotations=document.getElementById('annotations-field'); const split=document.getElementById('train_ratio_percent')?.closest('.field'); if(banner){banner.hidden=!active; banner.innerHTML=active?`<div class="check-item ok"><span class="check-icon">✓</span><b>已直接导入 YOLO 数据集</b><span class="check-detail">${values.prepared_dataset_yaml}；沿用原 train / valid / test，不需要 XML，也不会重新随机划分。</span></div>`:''} if(annotations) annotations.hidden=active||(document.querySelector('input[name="train_task"]:checked')?.value||'detect')==='classify'; if(split) split.hidden=active||!document.body.classList.contains('show-advanced')}
function updateTrainTaskUI(){const task=document.querySelector('input[name="train_task"]:checked')?.value||'detect'; const annotations=document.getElementById('annotations-field'); const hint=document.getElementById('train-task-hint'); const prepared=!!String(values.prepared_dataset_yaml||'').trim(); if(annotations) annotations.hidden=task==='classify'||prepared; if(hint) hint.textContent=task==='classify'?'分类数据集结构：Images Dir/类别名/图片。启用测试集时每个类别至少 3 张图片；Annotations Dir 不参与分类训练；Base Model 请使用分类权重，例如 yolo11n-cls.pt。':prepared?'已导入标准 YOLO 数据集，将直接使用 TXT 标签和原始 train/valid/test 划分。':'检测数据集结构：可直接导入 data.yaml 数据集，或使用同名图片 + VOC XML。启用测试集时至少需要 3 对有效样本。'; updatePreparedDatasetUI()}
async function loadDeviceProfiles(){try{const r=await fetch('/api/device-profiles'); const j=await r.json(); deploymentProfiles=j.items||[]; updateDeploymentProfile()}catch(e){}}
function updateDeploymentProfile(){const target=document.getElementById('deployment_target')?.value||values.deployment_target||'generic_onnx'; const profile=deploymentProfiles.find(item=>item.id===target); if(!profile) return; const format=document.getElementById('export_format'); if(format){for(const option of format.options) option.disabled=option.value!=='auto'&&!profile.formats.includes(option.value); if(format.selectedOptions[0]?.disabled) format.value='auto'} const chip=document.getElementById('export_chip'); if(chip) chip.placeholder=profile.chips?.length?`可选：${profile.chips.join('、')}；默认 ${profile.default_chip||profile.chips[0]}`:'该平台不需要填写'; const box=document.getElementById('deployment-profile-note'); if(box) box.innerHTML=`<div class="check-item ok"><span class="check-icon">✓</span><b>推荐 ${String(profile.recommended_format).toUpperCase()}</b><span class="check-detail">${profile.summary}</span></div><div class="check-item ${profile.vendor_toolchain?'warn':'ok'}"><span class="check-icon">${profile.vendor_toolchain?'!':'✓'}</span><b>${profile.vendor_toolchain?'需要设备厂商工具链':'可直接进入运行时验证'}</b><span class="check-detail">${profile.next_step}</span></div>`}
function syncLabelFields(prefix,toCanonical){const pairs=prefix==='camera'?[['label_name_camera','label_name'],['label_prefix_camera','label_prefix'],['label_images_dir_camera','label_images_dir'],['label_annotations_dir_camera','label_annotations_dir'],['label_tracker_camera','label_tracker'],['label_interval_camera','label_interval'],['label_max_frames_camera','label_max_frames'],['label_display_scale_camera','label_display_scale'],['label_jpeg_quality_camera','label_jpeg_quality']]:[['label_name_images','label_name'],['label_annotations_dir_images','label_annotations_dir'],['label_tracker_images','label_tracker'],['label_interval_images','label_interval'],['label_start_frame_images','label_start_frame'],['label_max_frames_images','label_max_frames'],['label_display_scale_images','label_display_scale']]; for(const [sourceId,canonicalId] of pairs){const from=document.getElementById(toCanonical?sourceId:canonicalId); const to=document.getElementById(toCanonical?canonicalId:sourceId); if(from&&to) to.value=from.value}}
function updateLabelSourceUI(){const source=document.querySelector('input[name="label_source_type"]:checked')?.value||'video'; const video=document.getElementById('label-video-source'); const camera=document.getElementById('label-camera-source'); const images=document.getElementById('label-images-source'); const start=document.getElementById('label-start-button'); if(video) video.hidden=source!=='video'; if(camera) camera.hidden=source!=='camera'; if(images) images.hidden=source!=='images'; if(source==='camera') syncLabelFields('camera',false); if(source==='images') syncLabelFields('images',false); if(start) start.textContent=source==='camera'?'在网页中开始摄像头标注':source==='images'?'在网页中开始图片集标注':'在网页中开始视频标注'}
function collect(){const source=document.querySelector('input[name="label_source_type"]:checked')?.value||'video'; if(source==='camera') syncLabelFields('camera',true); if(source==='images') syncLabelFields('images',true); for(const id of fields){const el=document.getElementById(id); if(el) values[id]=el.value} values.skip_vm_convert=!!document.getElementById('skip_vm_convert')?.checked; values.export_int8=!!document.getElementById('export_int8')?.checked; values.raw_overwrite=!!document.getElementById('raw_overwrite')?.checked; for(const n of ['operator_mode','train_mode','train_device','train_task','test_source','label_source_type']){const el=document.querySelector(`input[name="${n}"]:checked`); if(el) values[n]=el.value} return values}
function resourceEstimateKey(v){return [v.train_task||'detect',v.train_images_dir||'',v.base_model||'',v.img_width||'',v.img_height||'',v.image_resize_mode||'',v.batch||'',v.train_cache||'',v.train_device||''].join('|')}
function showResourceEstimate(e){e=e||{}; setText('estimate-images',e.image_count!==undefined?`${e.image_count} 张`:'-'); setText('estimate-ram',e.ram_text||'-'); setText('estimate-vram',e.vram_text||'-'); setText('estimate-cache',e.cache_text||'-'); setText('estimate-imgsz',e.img_size||'-'); setText('estimate-batch',e.batch||'-'); const riskMap={safe:'资源预估',warning:'资源预估 · 警告',danger:'资源预估 · 风险'}; const model=e.model_size?` · YOLO-${e.model_size}`:''; setText('resource-note',`${riskMap[e.risk]||'资源预估'}${model} · cache=${e.cache_mode||'-'}`); setText('resource-detail',e.note||'估算值仅供参考，实际峰值会随模型、增强策略、驱动和环境波动。')}
async function updateResourceEstimate(){try{const v=collect(); const key=resourceEstimateKey(v); if(key===lastResourceEstimateKey) return; lastResourceEstimateKey=key; setText('resource-note','正在估算...'); const j=await api('/api/train-estimate',{values:v}); showResourceEstimate(j.estimate||{})}catch(e){setText('resource-note','估算失败'); setText('resource-detail',e.message)}}
function scheduleResourceEstimate(){clearTimeout(resourceEstimateTimer); resourceEstimateTimer=setTimeout(updateResourceEstimate,450)}
function apply(v){const before=JSON.stringify(values); values={...values,...v}; for(const id of fields){const el=document.getElementById(id); if(el && document.activeElement!==el && values[id]!==undefined && el.value!==String(values[id])) el.value=values[id]} updateSplitRatio(); const skipVm=document.getElementById('skip_vm_convert'); if(skipVm) skipVm.checked=!!values.skip_vm_convert; const exportInt8=document.getElementById('export_int8'); if(exportInt8) exportInt8.checked=!!values.export_int8; const rawOverwrite=document.getElementById('raw_overwrite'); if(rawOverwrite) rawOverwrite.checked=!!values.raw_overwrite; for(const n of ['operator_mode','train_mode','train_device','train_task','test_source','label_source_type']){if(values[n]!==undefined){const el=document.querySelector(`input[name="${n}"][value="${values[n]}"]`); if(el && document.activeElement!==el) el.checked=true}} updateTrainTaskUI(); updatePreparedDatasetUI(); updateLabelSourceUI(); updateDeploymentProfile(); updateCurrentVideo(); if(JSON.stringify(values)!==before) updateCommands(); scheduleResourceEstimate()}


function setConnectionState(online){const badge=document.getElementById('connectionBadge'); if(!badge) return; badge.classList.toggle('offline',!online); const text=badge.querySelector('span:last-child'); if(text) text.textContent=online?'面板已连接 · 本机服务':'面板连接已断开 · 请运行启动脚本'}
async function api(path,body){let r; try{r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})})}catch(e){setConnectionState(false);throw new Error('无法连接训练面板。请双击 start_train_panel.cmd 后刷新页面。')} setConnectionState(true); let j; try{j=await r.json()}catch(e){throw new Error(`面板返回了无法解析的结果（HTTP ${r.status}）`)} if(!r.ok||j.error) throw new Error(j.error||r.statusText); return j}
let valuesSaveQueue=Promise.resolve();
function saveValues(){const snapshot={...collect()}; valuesSaveQueue=valuesSaveQueue.then(()=>api('/api/values',{values:snapshot})).catch(()=>{}); return valuesSaveQueue}
async function saveDefaults(scope){const snapshot={...collect()}; try{await valuesSaveQueue; const j=await api('/api/defaults',{values:snapshot}); apply(j.values||{}); toast(`${scope||'当前配置'}已保存为默认`)}catch(e){toast(e.message)}}
async function command(action){const j=await api('/api/command',{action,values:collect()}); return j.command}
async function updateCommands(){try{document.getElementById('cmd-train').textContent=await command('train');document.getElementById('cmd-export').textContent=await command('export');document.getElementById('cmd-convert').textContent=await command('convert');document.getElementById('cmd-test').textContent=await command('test');document.getElementById('cmd-label').textContent=await command('label')}catch(e){}}
function setInputValue(id,value){const el=document.getElementById(id); if(el){el.value=value; values[id]=value}}
function toggleAdvancedSettings(){const show=!document.body.classList.contains('show-advanced'); document.body.classList.toggle('show-advanced',show); localStorage.setItem('yoloTeamPlatformAdvanced',show?'1':'0'); setText('advanced-toggle',show?'收起更多设置':'更多设置')}
function applyTrainPreset(name){const task=document.querySelector('input[name="train_task"]:checked')?.value||'detect'; const recommendedBatch=String(values.batch||'16'),recommendedWorkers=String(values.train_workers||'4'); const presets={smoke:{epochs:'5',batch:'8',train_workers:'2',patience:'0',img_width:'640',img_height:'480',image_resize_mode:'letterbox',train_cache:'False'},camera4060:{epochs:'10000',batch:recommendedBatch,train_workers:recommendedWorkers,patience:'0',img_width:'640',img_height:'480',image_resize_mode:'letterbox',train_cache:'disk',lr0:'0.005'},balanced:{epochs:'100',batch:recommendedBatch,train_workers:recommendedWorkers,patience:'0',img_width:'640',img_height:'480',image_resize_mode:'letterbox',train_cache:'disk'},quality:{epochs:'10000',batch:recommendedBatch,train_workers:recommendedWorkers,patience:'0',img_width:'640',img_height:'480',image_resize_mode:'letterbox',train_cache:'disk'}}; const preset=presets[name]||presets.camera4060; for(const [key,value] of Object.entries(preset)) setInputValue(key,value); const model=document.getElementById('base_model'); if(model&&!model.value.trim()) setInputValue('base_model',task==='classify'?'yolo11n-cls.pt':'yolo11n.pt'); saveValues(); updateCommands(); scheduleResourceEstimate(); document.getElementById('train-readiness').hidden=true; toast(name==='smoke'?'已应用640×480首次验证配置':name==='camera4060'?'已应用本机推荐的640×480配置，由你手动停止':name==='quality'?'已应用640×480持续高质量配置，由你手动停止':'已应用推荐配置')}
function renderTrainChecks(result){const box=document.getElementById('train-readiness'); if(!box) return; box.hidden=false; box.innerHTML=''; for(const check of result.checks||[]){const item=document.createElement('div'); item.className='check-item '+check.status; const icon=document.createElement('span'); icon.className='check-icon'; icon.textContent=check.status==='ok'?'✓':check.status==='warn'?'!':'×'; const label=document.createElement('b'); label.textContent=check.label; const detail=document.createElement('span'); detail.className='check-detail'; detail.textContent=check.detail; item.append(icon,label,detail); box.appendChild(item)}}
async function checkTrainReady(showMessage=false){try{const result=await api('/api/train-check',{values:collect()}); renderTrainChecks(result); if(showMessage||!result.ready) toast(result.summary); return result.ready}catch(e){renderTrainChecks({checks:[{label:'面板检查',status:'error',detail:e.message}]}); toast(e.message); return false}}
async function pickTrainDirectory(field){try{const j=await api('/api/pick-train-directory',{field,values:collect()}); if(!j.path){toast('未选择文件夹');return} setInputValue(field,j.path); if(field==='train_images_dir'||field==='train_annotations_dir'){setInputValue('prepared_dataset_yaml',''); updatePreparedDatasetUI()} if(field==='train_images_dir'&&!String(values.dataset_root||'').trim()){const parent=j.path.replace(/[\\/][^\\/]+[\\/]?$/,''); if(parent) setInputValue('dataset_root',parent)} await saveValues(); updateCommands(); scheduleResourceEstimate(); document.getElementById('train-readiness').hidden=true}catch(e){toast(e.message)}}
function renderRawDatasetSummary(info,converted=false){const box=document.getElementById('raw-dataset-summary'); if(!box) return; box.hidden=false; box.innerHTML=''; const classes=(info.class_names||[]).map((name,index)=>`${index}=${name}`).join('，')||'尚未识别'; const warnings=[]; if(info.images_without_labels) warnings.push(`${info.images_without_labels} 张图片没有标签`); if(info.labels_without_images) warnings.push(`${info.labels_without_images} 个标签没有图片`); if(info.empty_labels) warnings.push(`${info.empty_labels} 个空标签`); if(info.invalid_lines) warnings.push(`${info.invalid_lines} 行格式无效`); let layout=`图片：${info.images_dir}；标签：${info.labels_dir}`; if(info.format==='yolo-split'){layout=Object.entries(info.splits||{}).map(([name,item])=>`${name} ${item.matched_count} 对`).join(' / '); warnings.push('将保留原 train / valid / test，不重新划分')} const checks=[{status:'ok',label:info.format==='yolo-split'?'标准 YOLO 数据集':'目录识别',detail:layout},{status:info.matched_count?'ok':'error',label:'图片/标签匹配',detail:`${info.image_count} 张图片，${info.label_count} 个 TXT，${info.matched_count} 对同名文件`},{status:info.box_count?'ok':'error',label:'检测框与类别',detail:`${info.box_count} 个框；${classes}`},{status:warnings.length?'warn':'ok',label:'数据提醒',detail:warnings.length?warnings.join('；'):'没有发现缺失或无效文件'},{status:'ok',label:converted?'转换完成':info.format==='yolo-split'?'可直接导入':'转换输出',detail:converted?`${info.converted_count} 张图片、${info.boxes_written} 个框已生成；${info.output_dir}`:info.format==='yolo-split'?`${info.yaml_path}；无需 XML`:info.output_dir}]; if(converted&&info.skipped_existing) checks.push({status:'warn',label:'已存在文件',detail:`${info.skipped_existing} 对文件未覆盖；勾选“覆盖同名文件”可重新生成`}); for(const check of checks){const item=document.createElement('div'); item.className='check-item '+check.status; const icon=document.createElement('span'); icon.className='check-icon'; icon.textContent=check.status==='ok'?'✓':check.status==='warn'?'!':'×'; const label=document.createElement('b'); label.textContent=check.label; const detail=document.createElement('span'); detail.className='check-detail'; detail.textContent=check.detail; item.append(icon,label,detail); box.appendChild(item)}}
async function pickRawDatasetRoot(importNow=false){try{const j=await api('/api/pick-raw-dataset-root',{values:collect()}); if(!j.path){toast('未选择文件夹');return} setInputValue('raw_dataset_root',j.path); setInputValue('raw_images_dir',''); setInputValue('raw_labels_dir',''); setInputValue('raw_output_dir',''); await inspectRawDataset(); if(importNow) await importRawDataset()}catch(e){toast(e.message)}}
async function inspectRawDataset(){try{const j=await api('/api/dataset-convert/inspect',{values:collect()}); const info=j.info||{}; setInputValue('raw_images_dir',info.images_dir||''); setInputValue('raw_labels_dir',info.labels_dir||''); setInputValue('raw_output_dir',info.output_dir||''); if(!String(document.getElementById('raw_class_names')?.value||'').trim()&&info.class_names?.length) setInputValue('raw_class_names',info.class_names.join(', ')); renderRawDatasetSummary(info,false); await saveValues(); toast(`已识别 ${info.matched_count||0} 对图片和标签`)}catch(e){renderRawDatasetSummary({images_dir:'-',labels_dir:'-',image_count:0,label_count:0,matched_count:0,box_count:0,class_names:[],output_dir:'-',invalid_lines:1},false); toast(e.message)}}
async function importRawDataset(){const button=document.getElementById('raw-import-button'); if(button){button.disabled=true;button.textContent='正在导入...'} try{const j=await api('/api/dataset-import/run',{values:collect()}); apply(j.values||{}); renderRawDatasetSummary(j.result||{},false); await saveValues(); updateCommands(); scheduleResourceEstimate(); toast('已直接导入，保留原数据划分且无需 XML')}catch(e){toast(e.message)}finally{if(button){button.disabled=false;button.textContent='直接导入训练配置'}}}
async function convertRawDataset(){const button=document.getElementById('raw-convert-button'); if(button){button.disabled=true;button.textContent='正在转换...'} try{const j=await api('/api/dataset-convert/run',{values:collect()}); apply(j.values||{}); renderRawDatasetSummary(j.result||{},true); await saveValues(); updateCommands(); scheduleResourceEstimate(); toast('转换完成，训练目录已经自动填入')}catch(e){toast(e.message)}finally{if(button){button.disabled=false;button.textContent='转换并填入训练配置'}}}
async function pickBaseModel(){try{const j=await api('/api/pick-base-model',{values:collect()}); if(j.path){setInputValue('base_model',j.path); await saveValues(); updateCommands(); scheduleResourceEstimate(); document.getElementById('train-readiness').hidden=true}else toast('未选择模型')}catch(e){toast(e.message)}}
async function pickDeployModel(){try{const j=await api('/api/pick-deploy-model',{values:collect()}); if(j.path){setInputValue('deploy_model',j.path); await saveValues(); updateCommands()}else toast('未选择模型')}catch(e){toast(e.message)}}
async function pickDeployOutputDir(){try{const j=await api('/api/pick-deploy-output-dir',{values:collect()}); if(j.path){setInputValue('export_output_dir',j.path); await saveValues(); updateCommands()}else toast('未选择文件夹')}catch(e){toast(e.message)}}
function videoPrefix(video){return video.stem.replace(/[^\w\u4e00-\u9fa5-]+/g,'_').replace(/^_+|_+$/g,'')||'track'}
function videoUrl(video,path='/api/video-file'){return `${path}?path=${encodeURIComponent(video.path)}&t=${Date.now()}`}
function videoSizeText(video){const size=Number(video.size||0); if(!size) return ''; const units=['B','KB','MB','GB','TB']; let n=size,i=0; while(n>=1024&&i<units.length-1){n/=1024;i++} return `${n>=10||i===0?n.toFixed(0):n.toFixed(1)} ${units[i]}`}
function updateVideoPreview(video){const box=document.getElementById('label-video-preview'); const name=document.getElementById('label-preview-name'); if(!box||!name) return; const token=++labelPreviewToken; box.classList.remove('clickable'); box.onclick=null; if(!video){name.textContent='未选择'; box.innerHTML='<span>选择左侧视频后显示首帧预览</span>'; return} name.textContent=video.name; box.classList.add('clickable'); box.innerHTML=`<img src="${videoUrl(video,'/api/video-preview')}" alt="" loading="eager"><div class="play-overlay"><div class="play-button">▶</div></div>`; const img=box.querySelector('img'); if(img){img.onerror=()=>{if(token!==labelPreviewToken) return; box.classList.remove('clickable'); box.onclick=null; box.innerHTML='<span>首帧预览读取失败：可能是视频编码不受当前 OpenCV 支持，但仍可尝试开始标注。</span>'}} box.onclick=()=>playVideoPreview(video)}
function playVideoPreview(video){const box=document.getElementById('label-video-preview'); if(!box) return; ++labelPreviewToken; box.classList.remove('clickable'); box.onclick=null; box.innerHTML=`<video src="${videoUrl(video)}" controls playsinline preload="metadata"></video>`; const player=box.querySelector('video'); if(player){player.onerror=()=>{box.innerHTML='<span>浏览器无法直接播放该视频编码或容器格式；仍可尝试通过网页标注工作台读取并标注。</span>'}; player.play().catch(()=>{})}}
function updateCurrentVideo(){const cur=document.getElementById('label-current-video'); if(!cur) return; const val=(document.getElementById('label_video')?.value||values.label_video||'').trim(); if(labelVideos.length){const matched=labelVideos.findIndex(v=>v.path===val); if(matched!==labelVideoIndex){labelVideoIndex=matched; renderLabelVideos(); updateVideoPreview(labelVideos[labelVideoIndex]||null); return}} cur.textContent=val||'未选择视频'; if(!labelVideos.length) updateVideoPreview(null)}
function renderLabelVideos(){const list=document.getElementById('label-video-list'); const count=document.getElementById('label-video-count'); if(!list||!count) return; const done=labelVideos.filter(v=>v.done).length; const visible=Math.min(labelVisibleCount,labelVideos.length); count.textContent=labelVideos.length?`${done}/${labelVideos.length} 已完成 · 显示 ${visible} 个`:'0 个视频'; list.innerHTML=''; if(!labelVideos.length){list.innerHTML='<div class="empty">当前文件夹没有找到视频。支持 mp4、avi、mov、mkv、wmv、webm 等格式。</div>'; const cur=document.getElementById('label-current-video'); if(cur) cur.textContent=(document.getElementById('label_video')?.value||values.label_video||'').trim()||'未选择视频'; updateVideoPreview(null); return} const fragment=document.createDocumentFragment(); labelVideos.slice(0,visible).forEach((video,idx)=>{const btn=document.createElement('button'); btn.className='video-item'+(idx===labelVideoIndex?' active':'')+(video.done?' done':''); const status=video.done?'已完成':'待标注'; const size=videoSizeText(video); btn.innerHTML=`<b>${idx+1}. ${video.name}</b><span>${status}${size?' · '+size:''} · ${video.rel}</span>`; btn.onclick=()=>selectLabelVideo(idx); fragment.appendChild(btn)}); list.appendChild(fragment); if(visible<labelVideos.length){const more=document.createElement('button'); more.className='video-item'; more.innerHTML=`<b>加载更多视频</b><span>继续显示 ${Math.min(LABEL_VIDEO_PAGE_SIZE,labelVideos.length-visible)} 个，剩余 ${labelVideos.length-visible} 个</span>`; more.onclick=()=>{labelVisibleCount=Math.min(labelVisibleCount+LABEL_VIDEO_PAGE_SIZE,labelVideos.length); renderLabelVideos()}; list.appendChild(more)} const cur=document.getElementById('label-current-video'); const video=labelVideos[labelVideoIndex]; if(cur){cur.textContent=video?video.path:((document.getElementById('label_video')?.value||values.label_video||'').trim()||'未选择视频')}}
function selectLabelVideo(index){if(index<0||index>=labelVideos.length) return; labelVideoIndex=index; if(index>=labelVisibleCount){labelVisibleCount=Math.min(labelVideos.length,Math.ceil((index+1)/LABEL_VIDEO_PAGE_SIZE)*LABEL_VIDEO_PAGE_SIZE)} const video=labelVideos[index]; setInputValue('label_video',video.path); if(!rawLabelPrefix.manual){setInputValue('label_prefix',videoPrefix(video))} renderLabelVideos(); updateVideoPreview(video); saveValues(); updateCommands()}
async function loadLabelVideos(){try{await saveValues(); const list=document.getElementById('label-video-list'); if(list) list.innerHTML='<div class="empty">正在读取视频文件夹，请稍候...</div>'; const r=await fetch('/api/label-videos'); const j=await r.json(); if(j.error) throw new Error(j.error); labelVideos=j.items||[]; labelVisibleCount=LABEL_VIDEO_PAGE_SIZE; const current=(document.getElementById('label_video')?.value||'').trim(); labelVideoIndex=labelVideos.findIndex(v=>v.path===current); if(labelVideoIndex<0&&labelVideos.length) labelVideoIndex=0; if(labelVideos.length) selectLabelVideo(labelVideoIndex); else renderLabelVideos(); toast(`已读取 ${labelVideos.length} 个视频${labelVideos.length>=2000?'，已自动限制前 2000 个':''}`)}catch(e){toast(e.message)}}

async function pickTestImage(){try{const j=await api('/api/pick-test-image',{values:collect()}); if(j.path){setInputValue('test_image_file',j.path); await saveValues(); updateCommands()}else{toast('未选择图片')}}catch(e){toast(e.message)}}
async function pickTestImageFolder(){try{const j=await api('/api/pick-test-image-folder',{values:collect()}); if(j.path){setInputValue('test_image_folder',j.path); await saveValues(); updateCommands()}else{toast('未选择文件夹')}}catch(e){toast(e.message)}}
async function pickTestOutputDir(){try{const j=await api('/api/pick-test-output-dir',{values:collect()}); if(j.path){setInputValue('test_output_dir',j.path); await saveValues(); updateCommands()}else{toast('未选择文件夹')}}catch(e){toast(e.message)}}
async function pickLabelVideoDir(){try{const j=await api('/api/pick-label-video-dir',{values:collect()}); if(j.path){setInputValue('label_video_dir',j.path); await saveValues(); await loadLabelVideos()}else{toast('未选择文件夹')}}catch(e){toast(e.message)}}
async function pickLabelImagesDir(){try{const j=await api('/api/pick-label-images-dir',{values:collect()}); if(j.path){setInputValue('label_images_input_dir',j.path); await saveValues(); toast('已选择图片集文件夹')}else{toast('未选择文件夹')}}catch(e){toast(e.message)}}
function selectNextVideo(){if(!labelVideos.length){toast('请先读取文件夹视频'); return} if(labelVideoIndex>=0&&labelVideos[labelVideoIndex]) labelVideos[labelVideoIndex].done=true; const next=Math.min(labelVideos.length-1,labelVideoIndex+1); selectLabelVideo(next); toast(next===labelVideos.length-1?'已到最后一个视频':'已切换到下一个视频')}

function selectPrevVideo(){if(!labelVideos.length){toast('请先读取文件夹视频'); return} selectLabelVideo(Math.max(0,labelVideoIndex-1))}
let labelSessionId=sessionStorage.getItem('yoloTeamPlatformLabelSessionId')||'';
let labelSessionState=null;
let labelPlaying=false;
let labelAdvanceBusy=false;
let labelPlayTimer=null;
let labelActiveObjectId=null;
let labelDrawMode='add';
let labelDragStart=null;
function labelFrameUrl(){return `/api/label-session/frame?session_id=${encodeURIComponent(labelSessionId)}&t=${Date.now()}`}
function labelPalette(i){return ['#50dc78','#50b4ff','#e678ff','#ffbe46','#78dcff','#b4a0ff','#78ffd2','#ff8c8c'][i%8]}
function showLabelStudio(show){const studio=document.getElementById('label-browser-studio'); if(studio) studio.hidden=!show}
function renderLabelSession(){
  const state=labelSessionState,status=document.getElementById('label-session-status'),list=document.getElementById('label-object-list'),tip=document.getElementById('label-session-tip');
  if(!state||!status||!list) return;
  const alertText=state.last_warning||((state.lost||false)?'跟踪异常，请修正或删除':''),autoText=state.last_auto_saved?'<span>本帧已自动保存</span>':'';
  status.innerHTML=`<span>来源：${state.source_type==='camera'?'摄像头':state.source_type==='images'?'图片集':'视频'}</span><span>帧：${state.frame_index}${state.frame_count?'/'+Math.max(0,state.frame_count-1):''}</span><span>已保存：${state.saved}</span><span>待复核未保存：${state.review_skipped||0}</span><span>目标：${state.objects.filter(x=>x.ok).length}/${state.objects.length}</span>${autoText}${state.ended?'<span>来源已结束</span>':''}${alertText?`<span>${alertText}</span>`:''}`;
  if(tip) tip.textContent=alertText?`${alertText} 已暂停播放。`:'拖动添加或修正目标框；建议每推进一小段就暂停抽查。';
  list.innerHTML='';
  if(!state.objects.length){list.innerHTML='<div class="empty">尚未框选目标。点击“添加框”后在画面拖动鼠标。</div>'}
  else state.objects.forEach((obj,index)=>{const item=document.createElement('button'); const quality=Math.round(Number(obj.quality??1)*100); item.className='label-object'+(obj.id===labelActiveObjectId?' active':'')+(!obj.ok?' lost':''); item.innerHTML=`<b>#${obj.id} ${obj.label}${obj.ok?'':' · 需复核'}</b><span>${obj.w} × ${obj.h} · 质量 ${quality}% · 参考视角 ${obj.sample_count||1}${obj.warning?' · '+obj.warning:''}</span>`; item.onclick=()=>{labelActiveObjectId=obj.id; renderLabelSession(); drawLabelCanvas()}; list.appendChild(item)});
  const button=document.getElementById('label-play-button'); if(button) button.textContent=labelPlaying?'暂停跟踪':'播放跟踪'; drawLabelCanvas();
}

function refreshLabelFrame(){const image=document.getElementById('label-frame-image'); const stage=document.getElementById('label-stage'); if(!image||!labelSessionId) return; image.hidden=false; stage?.classList.remove('empty-stage'); image.onload=()=>drawLabelCanvas(); image.src=labelFrameUrl()}
function setupLabelCanvas(){const canvas=document.getElementById('label-frame-canvas'); if(!canvas||canvas.dataset.ready) return; canvas.dataset.ready='1'; canvas.hidden=false; canvas.addEventListener('pointerdown',event=>{if(!labelSessionState) return; const point=labelCanvasPoint(event); if(!point) return; labelDragStart=point; canvas.setPointerCapture(event.pointerId); drawLabelCanvas(point)}); canvas.addEventListener('pointermove',event=>{if(!labelDragStart) return; drawLabelCanvas(labelCanvasPoint(event))}); canvas.addEventListener('pointerup',async event=>{if(!labelDragStart) return; const end=labelCanvasPoint(event); const start=labelDragStart; labelDragStart=null; canvas.releasePointerCapture(event.pointerId); drawLabelCanvas(); if(!end) return; const x=Math.min(start.x,end.x),y=Math.min(start.y,end.y),w=Math.abs(end.x-start.x),h=Math.abs(end.y-start.y); if(w<5||h<5){toast('标注框过小');return} const action=labelDrawMode==='edit'?'update':labelDrawMode==='sample'?'add_sample':'add'; if(action==='add_sample'&&!labelActiveObjectId){toast('请先在右侧选择要追加视角的目标');return} let label=''; if(action==='add'){label=await chooseBrowserLabel(); if(!label) return} try{const j=await api('/api/label-session/object',{session_id:labelSessionId,action,object_id:labelActiveObjectId,bbox:{x,y,w,h},label}); labelSessionState=j.state; if(action==='add') labelActiveObjectId=labelSessionState.objects.at(-1)?.id||null; renderLabelSession()}catch(e){toast(e.message)}}); window.addEventListener('resize',()=>drawLabelCanvas())}

function labelCanvasPoint(event){const image=document.getElementById('label-frame-image'); if(!image||!labelSessionState) return null; const rect=image.getBoundingClientRect(); if(!rect.width||!rect.height) return null; return {x:(event.clientX-rect.left)*labelSessionState.width/rect.width,y:(event.clientY-rect.top)*labelSessionState.height/rect.height}}
function drawLabelCanvas(dragEnd=null){const canvas=document.getElementById('label-frame-canvas'); const image=document.getElementById('label-frame-image'); if(!canvas||!image||!labelSessionState||image.hidden) return; const rect=image.getBoundingClientRect(); const dpr=window.devicePixelRatio||1; canvas.style.width=rect.width+'px'; canvas.style.height=rect.height+'px'; canvas.style.left=(rect.left-canvas.parentElement.getBoundingClientRect().left)+'px'; canvas.style.top=(rect.top-canvas.parentElement.getBoundingClientRect().top)+'px'; canvas.width=Math.max(1,Math.round(rect.width*dpr)); canvas.height=Math.max(1,Math.round(rect.height*dpr)); const ctx=canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,rect.width,rect.height); const sx=rect.width/labelSessionState.width,sy=rect.height/labelSessionState.height; labelSessionState.objects.forEach((obj,index)=>{const active=obj.id===labelActiveObjectId; ctx.strokeStyle=obj.ok?labelPalette(index):'#ff6678'; ctx.lineWidth=active?3:2; ctx.strokeRect(obj.x*sx,obj.y*sy,obj.w*sx,obj.h*sy); ctx.font='600 13px Microsoft YaHei UI'; const text=`#${obj.id} ${obj.label}${obj.ok?'':' LOST'}`; const tx=obj.x*sx,ty=Math.max(18,obj.y*sy-5); ctx.fillStyle='rgba(3,8,19,.85)'; const tw=ctx.measureText(text).width+10; ctx.fillRect(tx,ty-16,tw,20); ctx.fillStyle='#fff'; ctx.fillText(text,tx+5,ty)}); if(labelDragStart&&dragEnd){const x=Math.min(labelDragStart.x,dragEnd.x)*sx,y=Math.min(labelDragStart.y,dragEnd.y)*sy,w=Math.abs(labelDragStart.x-dragEnd.x)*sx,h=Math.abs(labelDragStart.y-dragEnd.y)*sy; ctx.strokeStyle='#fff';ctx.setLineDash([6,4]);ctx.lineWidth=2;ctx.strokeRect(x,y,w,h);ctx.setLineDash([])}}
async function chooseBrowserLabel(){const labels=(labelSessionState?.labels||[]); const choices=labels.length?labels:(collect().label_name||'object').split(/[,;\n]+/).map(x=>x.trim()).filter(Boolean); if(choices.length===1) return choices[0]; const answer=prompt(`输入类别名称：\n${choices.map((x,i)=>`${i+1}. ${x}`).join('\n')}`,choices[0]); if(answer===null) return ''; const index=Number(answer); const label=Number.isInteger(index)&&index>=1&&index<=choices.length?choices[index-1]:answer.trim(); if(!choices.includes(label)){toast('请输入标签列表中的类别');return ''} return label}
function setLabelDrawMode(mode){if(!labelSessionId){toast('请先开始网页标注');return} if((mode==='edit'||mode==='sample')&&!labelActiveObjectId){toast('请先在右侧选择目标');return} if(mode==='sample'&&labelSessionState?.tracker!=='multi_template'){toast('追加视角需要在开始会话前选择“多视角实验模式”');return} labelDrawMode=mode; toast(mode==='edit'?'请拖动绘制选中目标的新框':mode==='sample'?'请拖动绘制该目标在新角度下的框':'请拖动绘制新目标框')}

async function startBrowserLabelSession(){const v=collect(); if(v.label_source_type==='images'&&!v.label_images_input_dir.trim()){toast('请填写图片集文件夹路径');return} if(v.label_source_type==='camera'&&!/^\d+$/.test(v.label_camera_index.trim())){toast('请输入非负整数摄像头索引，例如 0');return} if(v.label_source_type==='video'&&!v.label_video.trim()){toast('请先从队列选择视频或填写视频路径');return} if(labelSessionId) await endBrowserLabelSession(); try{const j=await api('/api/label-session/start',{values:v}); labelSessionId=j.state.session_id; sessionStorage.setItem('yoloTeamPlatformLabelSessionId',labelSessionId); labelSessionState={...j.state,labels:(v.label_name||'object').split(/[,;\n]+/).map(x=>x.trim()).filter(Boolean)}; labelActiveObjectId=null; labelPlaying=false; showLabelStudio(true); setupLabelCanvas(); renderLabelSession(); refreshLabelFrame(); toast('半自动标注已就绪，请先添加目标框')}catch(e){toast(e.message)}}
async function advanceBrowserLabelFrame(){if(!labelSessionId||labelAdvanceBusy) return; labelAdvanceBusy=true; try{const j=await api('/api/label-session/advance',{session_id:labelSessionId}); labelSessionState={...j.state,labels:labelSessionState?.labels||[]}; if(labelSessionState.lost||labelSessionState.ended) stopBrowserLabelPlay(); renderLabelSession(); refreshLabelFrame()}catch(e){stopBrowserLabelPlay();toast(e.message)}finally{labelAdvanceBusy=false}}
function toggleBrowserLabelPlay(){if(labelPlaying) stopBrowserLabelPlay();else startBrowserLabelPlay()}
function startBrowserLabelPlay(){if(!labelSessionId){toast('请先开始网页标注');return} if(!labelSessionState?.objects.length){toast('请先添加至少一个目标框');return} if(labelSessionState.lost){toast('请先修正或删除丢失目标');return} labelPlaying=true; renderLabelSession(); const tick=async()=>{if(!labelPlaying) return; await advanceBrowserLabelFrame(); if(labelPlaying) labelPlayTimer=setTimeout(tick,45)}; tick()}
function stopBrowserLabelPlay(){labelPlaying=false;clearTimeout(labelPlayTimer);labelPlayTimer=null;renderLabelSession()}
async function saveBrowserLabelFrame(){if(!labelSessionId) return; try{const j=await api('/api/label-session/save',{session_id:labelSessionId}); labelSessionState={...j.state,labels:labelSessionState?.labels||[]}; renderLabelSession(); if(j.saved){toast('当前帧已保存');loadLabelResults()}else toast(j.state?.last_warning||'当前帧还不能安全保存')}catch(e){toast(e.message)}}
async function deleteBrowserLabelObject(){if(!labelSessionId||!labelActiveObjectId){toast('请先选择目标');return} try{const j=await api('/api/label-session/object',{session_id:labelSessionId,action:'delete',object_id:labelActiveObjectId,bbox:{x:0,y:0,w:3,h:3}}); labelSessionState={...j.state,labels:labelSessionState?.labels||[]}; labelActiveObjectId=labelSessionState.objects[0]?.id||null; renderLabelSession()}catch(e){toast(e.message)}}
async function endBrowserLabelSession(){stopBrowserLabelPlay(); if(!labelSessionId){showLabelStudio(false);return} try{await api('/api/label-session/end',{session_id:labelSessionId})}catch(e){} labelSessionId='';labelSessionState=null;labelActiveObjectId=null;sessionStorage.removeItem('yoloTeamPlatformLabelSessionId');showLabelStudio(false);loadLabelResults()}
async function runLabelCurrent(){await startBrowserLabelSession()}
async function copyCommand(action){try{const cmd=await command(action); await navigator.clipboard.writeText(cmd); toast('命令已复制')}catch(e){toast(e.message)}}
async function runAction(action){const startButton=document.getElementById('start-train-button'); try{if(action==='train'){if(startButton){startButton.setAttribute('disabled','');startButton.textContent='正在检查...'} const ready=await checkTrainReady(false); if(!ready) return; if(startButton) startButton.textContent='正在启动...'} await api('/api/run',{action,values:collect()}); showTab(action==='train'?'train':'logs'); toast(action==='train'?'训练已启动，请查看下方进度':'任务已启动'); refreshState()}catch(e){toast(e.message)}finally{if(action==='train'&&startButton){startButton.removeAttribute('disabled');startButton.textContent='检查并开始训练'}}}
async function stopJob(){try{const j=await api('/api/stop',{}); toast(j.stopped?'已请求停止':'当前没有正在运行的任务'); refreshState()}catch(e){toast(e.message)}}
async function stopTrainExport(){try{const j=await api('/api/stop-train-export',{}); toast(j.stopped?'已请求停止训练并导出当前 best':'当前没有正在训练的任务'); showTab('logs'); refreshState()}catch(e){toast(e.message)}}
async function loadLabelResults(){try{await saveValues(); const r=await fetch('/api/label-results'); const j=await r.json(); const box=document.getElementById('label-results'); box.innerHTML=''; const items=j.items||[]; if(!items.length){box.innerHTML='<div class="empty">当前图片目录和标注目录中还没有可显示的标注结果。</div>'; return} for(const it of items){const card=document.createElement('div'); card.className='sample'; const src='/api/label-preview?image='+encodeURIComponent(it.image)+'&xml='+encodeURIComponent(it.xml)+'&t='+Date.now(); card.innerHTML=`<img src="${src}" loading="lazy"><div class="meta"><b>${it.stem}</b><span>${(it.boxes||[]).length} 个框</span><button class="delete">删除标注</button></div>`; card.querySelector('.delete').onclick=async()=>{const imageMode=collect().label_source_type==='images'; const target=imageMode?'对应 XML 标注':'这张图片和对应 XML'; if(Date.now()>deleteConfirmUntil){if(!confirm(`确定删除${target}吗？\n确认后 5 分钟内删除标注不再重复询问。`)) return; deleteConfirmUntil=Date.now()+5*60*1000} await api('/api/delete-label-sample',{image:it.image,xml:it.xml}); card.remove(); toast(imageMode?'已删除 XML 标注':'已删除废图和 XML')}; box.appendChild(card)}}catch(e){toast(e.message)}}
async function loadTrainPlots(){try{const r=await fetch('/api/train-plots'); const j=await r.json(); const box=document.getElementById('train-plots'); if(!box) return; const note=document.getElementById('train-plots-note'); const items=j.items||[]; if(note) note.textContent=items.length?`已发现 ${items.length} 张图片`:'训练开始后自动刷新'; if(!items.length){box.innerHTML='<div class="empty">训练进行中或尚未生成可视化图。</div>'; return} box.innerHTML=''; for(const item of items){const card=document.createElement('div'); card.className='sample'; card.innerHTML=`<img src="/api/train-plot?name=${encodeURIComponent(item.name)}&t=${Date.now()}" loading="lazy"><div class="meta"><b>${item.name}</b></div>`; box.appendChild(card)}}catch(e){}}
function scheduleLabelResultsRefresh(){clearTimeout(labelResultsTimer); labelResultsTimer=setTimeout(()=>loadLabelResults(),350)}
async function refreshState(){try{const r=await fetch('/api/state'); if(!r.ok) throw new Error(`HTTP ${r.status}`); const s=await r.json(); setConnectionState(true); apply(s.values||{}); updateTrainProgress(s.train_progress||{}); const log=document.getElementById('log'); log.textContent=(s.logs||[]).join(''); log.scrollTop=log.scrollHeight; const pill=document.getElementById('runPill'); pill.className='pill '+(s.running?'run':'idle'); pill.querySelector('span:last-child').textContent=s.running?'运行中':'空闲'; const jobNames={train:'模型训练',model_download:'基础模型下载',export:'多平台导出',convert:'MaixCAM 专用转换',test:'模型测试',label:'单机快速标注',annotation_personal:'启动个人标注中心',annotation_share:'开启局域网协作标注',annotation_stop:'停止协作标注中心',train_ssh:'训练 SSH 检查',vm_ssh:'转换 SSH 检查'}; document.getElementById('jobInfo').textContent=s.job?`${jobNames[s.job]||s.job} · 开始 ${s.started_at||'-'}${s.finished_at?' · 结束 '+s.finished_at:''}${s.exit_code!==null&&s.exit_code!==undefined?' · 退出码 '+s.exit_code:''}`:'暂无任务';if(s.job==='model_download'&&s.running)modelDownloadWasRunning=true;if(modelDownloadWasRunning&&s.job==='model_download'&&!s.running){modelDownloadWasRunning=false;loadBaseModels();if(Number(s.exit_code)===0)toast('基础模型下载完成，已设为当前训练模型')} const errorBox=document.getElementById('lastError'); const failed=!s.running&&s.job&&s.exit_code!==null&&Number(s.exit_code)!==0; if(s.last_error||failed){errorBox.hidden=false; errorBox.textContent=s.last_error||`上次任务失败（退出码 ${s.exit_code}），请打开“运行日志”查看具体原因。`}else errorBox.hidden=true; const box=document.getElementById('markers'); box.innerHTML=''; for(const [k,v] of Object.entries(s.markers||{})){const div=document.createElement('div'); div.className='marker'; div.innerHTML=`<b>${k}</b><span>${v}</span>`; box.appendChild(div)}}catch(e){setConnectionState(false); const pill=document.getElementById('runPill'); if(pill){pill.className='pill idle';pill.querySelector('span:last-child').textContent='未连接'} const errorBox=document.getElementById('lastError'); if(errorBox){errorBox.hidden=false;errorBox.textContent='训练面板未运行。请双击 start_train_panel.cmd，然后刷新此页面。'}}}
function copyLogs(){navigator.clipboard.writeText(document.getElementById('log').textContent);toast('日志已复制')}
function showTab(name){const target=document.getElementById('tab-'+name); if(!target) name='projects'; document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));document.getElementById('tab-'+name).classList.add('active');let activeButton=null;document.querySelectorAll('.nav button').forEach(x=>{const active=x.dataset.tab===name;x.classList.toggle('active',active);if(active){x.setAttribute('aria-current','page');activeButton=x}else x.removeAttribute('aria-current')});const pageTitle=document.querySelector('#tab-'+name+'>h2')?.textContent?.trim();document.title=pageTitle?`${pageTitle} · YOLO团队训练平台`:'YOLO团队训练平台';localStorage.setItem('yoloTeamPlatformTab',name);window.scrollTo({top:0,behavior:'auto'});requestAnimationFrame(()=>{if(activeButton&&matchMedia('(max-width:980px)').matches)activeButton.scrollIntoView({block:'nearest',inline:'center',behavior:'smooth'})});if(name==='projects') loadProjects(); if(name==='models') loadBaseModels(); if(name==='label') loadLabelResults(); if(name==='assets') loadModelAssets(); if(name==='collab') loadAnnotationService()}
function enhanceFormAccessibility(){let index=0;document.querySelectorAll('.field').forEach(field=>{const label=field.querySelector(':scope > label');const control=field.querySelector(':scope > input, :scope > select, :scope > .input-action input, :scope > .input-action select');if(!label||!control)return;if(!control.id)control.id=`field-control-${++index}`;if(!label.htmlFor)label.htmlFor=control.id});document.querySelectorAll('button').forEach(button=>{if(!button.getAttribute('type'))button.setAttribute('type','button')})}

document.querySelectorAll('.nav button').forEach(b=>b.onclick=()=>showTab(b.dataset.tab));
document.querySelectorAll('input,select').forEach(el=>{const handler=()=>{if(el.id==='label_prefix') rawLabelPrefix.manual=true; if((el.id==='train_images_dir'||el.id==='train_annotations_dir')&&String(values.prepared_dataset_yaml||'').trim()){setInputValue('prepared_dataset_yaml',''); updatePreparedDatasetUI()} if(el.name==='train_task') updateTrainTaskUI(); if(el.name==='label_source_type') updateLabelSourceUI(); if(el.id==='train_ratio_percent'||el.id==='val_ratio_percent')updateSplitRatio(el.id); collect(); updateCurrentVideo(); saveValues(); updateCommands(); if(['train_images_dir','base_model','img_width','img_height','image_resize_mode','batch','train_cache'].includes(el.id)||el.name==='train_device'||el.name==='train_task') scheduleResourceEstimate(); if(['label_images_dir','label_annotations_dir','label_annotations_dir_images','label_images_input_dir'].includes(el.id)) scheduleLabelResultsRefresh()}; el.addEventListener('input',handler); el.addEventListener('change',handler)});
if(localStorage.getItem('yoloTeamPlatformAdvanced')==='1'){document.body.classList.add('show-advanced');setText('advanced-toggle','收起更多设置')}
enhanceFormAccessibility();
applySidebarState(localStorage.getItem('yoloTeamPlatformSidebarCollapsed')==='1');
showTab(localStorage.getItem('yoloTeamPlatformTab')||'projects');
updateSplitRatio();
loadDeviceProfiles();
loadProjects();
refreshState(); setInterval(refreshState,1400);



</script>
</body>
</html>'''


class PanelHandler(BaseHTTPRequestHandler):
    server_version = "YOLOTeamTrainingPlatform/3.2"

    def log_message(self, format: str, *args: Any) -> None:
        return


    def send_json(self, data: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            raw = HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if parsed.path == "/docs/COLLABORATIVE_ANNOTATION.md":
            document = SCRIPT_ROOT / "docs" / "COLLABORATIVE_ANNOTATION.md"
            if not document.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            raw = document.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if parsed.path == "/api/state":
            with STATE_LOCK:
                self.send_json({
                    "values": STATE["values"],
                    "logs": STATE["logs"],
                    "markers": STATE["markers"],
                    "train_progress": STATE["train_progress"],
                    "running": STATE["running"],
                    "job": STATE["job"],
                    "exit_code": STATE["exit_code"],
                    "started_at": STATE["started_at"],
                    "finished_at": STATE["finished_at"],
                    "last_error": STATE["last_error"],
                })
            return
        if parsed.path == "/api/device-profiles":
            self.send_json({"items": public_device_profiles()})
            return
        if parsed.path == "/api/annotation-service":
            self.send_json(annotation_service_status())
            return
        if parsed.path == "/api/projects":
            self.send_json(project_catalog())
            return
        if parsed.path == "/api/project-image":
            try:
                params = parse_qs(parsed.query)
                project_id = params.get("project_id", [""])[0]
                relative_path = params.get("path", [""])[0]
                catalog = project_catalog(include_health=False)
                project = next((item for item in catalog["projects"] if item.get("id") == project_id), None)
                if project is None:
                    raise ValueError("项目不存在。")
                dataset_root = Path(project.get("dataset_root") or project.get("root") or "").expanduser().resolve()
                image_path = resolve_under(relative_path, dataset_root)
                if not image_path.is_file() or image_path.suffix.lower() not in TRAIN_IMAGE_EXTENSIONS:
                    raise ValueError("图片不存在或格式不受支持。")
                raw = image_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mimetypes.guess_type(str(image_path))[0] or "application/octet-stream")
                self.send_header("Cache-Control", "private, max-age=60")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except Exception as exc:
                message = str(exc).encode("utf-8", errors="replace")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
            return
        if parsed.path == "/api/model-assets":
            with STATE_LOCK:
                values = STATE["values"].copy()
            self.send_json(model_asset_catalog(values))
            return
        if parsed.path == "/api/base-models":
            self.send_json(base_model_catalog(MODEL_ASSETS_DIR / "base-models"))
            return
        if parsed.path == "/api/train-plots":
            _, items = list_train_plots()
            self.send_json({"items": items})
            return
        if parsed.path == "/api/train-plot":
            try:
                params = parse_qs(parsed.query)
                name = params.get("name", [""])[0]
                plot_dir, _ = list_train_plots()
                if not plot_dir.is_dir():
                    raise ValueError("当前没有可用的训练图片目录。")
                send_train_plot(self, plot_dir, name)
            except Exception as exc:
                message = str(exc).encode("utf-8", errors="replace")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
            return
        if parsed.path == "/api/label-results":
            with STATE_LOCK:
                values = STATE["values"].copy()
            self.send_json({"items": list_label_results(values)})
            return
        if parsed.path == "/api/label-videos":
            with STATE_LOCK:
                values = STATE["values"].copy()
            self.send_json({"items": list_label_videos(values)})
            return
        if parsed.path == "/api/video-preview":
            try:
                params = parse_qs(parsed.query)
                video = params.get("path", [""])[0]
                with STATE_LOCK:
                    values = STATE["values"].copy()
                video_dir = Path(values.get("label_video_dir", "")).expanduser().resolve()
                if not video_dir.is_dir():
                    raise ValueError("请先选择有效的视频文件夹。")
                video_path = resolve_under(video, video_dir)
                raw = render_video_preview(video_path)
            except Exception as exc:
                message = str(exc).encode("utf-8", errors="replace")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if parsed.path == "/api/video-file":
            try:
                params = parse_qs(parsed.query)
                video = params.get("path", [""])[0]
                with STATE_LOCK:
                    values = STATE["values"].copy()
                video_dir = Path(values.get("label_video_dir", "")).expanduser().resolve()
                if not video_dir.is_dir():
                    raise ValueError("请先选择有效的视频文件夹。")
                video_path = resolve_under(video, video_dir)
                send_video_file(self, video_path)
            except Exception as exc:
                message = str(exc).encode("utf-8", errors="replace")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
            return
        if parsed.path == "/api/label-session/frame":
            try:
                params = parse_qs(parsed.query)
                session = get_label_session(params.get("session_id", [""])[0])
                with session["lock"]:
                    ok, encoded = cv2.imencode(".jpg", session["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    if not ok:
                        raise ValueError("当前标注帧编码失败。")
                    raw = encoded.tobytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except Exception as exc:
                message = str(exc).encode("utf-8", errors="replace")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
            return
        if parsed.path == "/api/label-preview":


            params = parse_qs(parsed.query)
            image = params.get("image", [""])[0]
            xml = params.get("xml", [""])[0]
            with STATE_LOCK:
                values = STATE["values"].copy()
            image_path = resolve_under(image, label_result_images_dir(values))
            xml_path = resolve_under(xml, Path(values["label_annotations_dir"]))
            raw = render_label_preview(image_path, xml_path)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_error(404)



    def do_POST(self) -> None:
        try:
            body = self.read_json()
            if self.path == "/api/command":
                action = str(body.get("action", ""))
                values = clean_values(body.get("values"))
                cmd = command_for(action, values)
                self.send_json({"command": quote_cmd(cmd)})
                return
            if self.path == "/api/train-estimate":
                values = clean_values(body.get("values"))
                self.send_json({"estimate": estimate_train_resources(values)})
                return
            if self.path == "/api/train-check":
                values = clean_values(body.get("values"))
                self.send_json(train_preflight(values))
                return
            if self.path == "/api/values":

                values = clean_values(body.get("values"))
                with STATE_LOCK:
                    STATE["values"] = values.copy()
                self.send_json({"ok": True})
                return
            if self.path == "/api/defaults":
                values = save_user_defaults(body.get("values") or {})
                with STATE_LOCK:
                    STATE["values"] = values.copy()
                self.send_json({"ok": True, "values": values, "path": str(USER_DEFAULTS_FILE)})
                return
            if self.path == "/api/projects/create":
                raw_labels = body.get("labels")
                labels = raw_labels if isinstance(raw_labels, list) else str(raw_labels or "").replace("，", ",").split(",")
                project = create_project(
                    str(body.get("name") or ""),
                    task=str(body.get("task") or "detect"),
                    labels=labels,
                    dataset_root=str(body.get("dataset_root") or ""),
                )
                with STATE_LOCK:
                    values = apply_project_to_values(project, STATE["values"])
                    STATE["values"] = values.copy()
                self.send_json({"ok": True, "project": project, "values": values, "catalog": project_catalog()})
                return
            if self.path == "/api/projects/activate":
                project = activate_project(str(body.get("project_id") or ""))
                with STATE_LOCK:
                    values = apply_project_to_values(project, STATE["values"])
                    STATE["values"] = values.copy()
                self.send_json({"ok": True, "project": project, "values": values, "catalog": project_catalog()})
                return
            if self.path == "/api/projects/update":
                updates = body.get("updates") if isinstance(body.get("updates"), dict) else {}
                project = update_project(str(body.get("project_id") or ""), updates)
                with STATE_LOCK:
                    values = apply_project_to_values(project, STATE["values"])
                    STATE["values"] = values.copy()
                self.send_json({"ok": True, "project": project, "values": values, "catalog": project_catalog()})
                return
            if self.path == "/api/projects/inspect":
                catalog = project_catalog(include_health=False)
                project_id = str(body.get("project_id") or "")
                project = next((item for item in catalog["projects"] if item.get("id") == project_id), None)
                if project is None:
                    raise ValueError("项目不存在。")
                self.send_json({"ok": True, "dataset": inspect_dataset(project.get("dataset_root") or project.get("root"), task=str(project.get("task") or "detect")), "catalog": project_catalog()})
                return
            if self.path == "/api/pick-project-dataset-root":
                initial = str(body.get("initial") or "")
                self.send_json({"path": pick_directory(initial, "选择项目数据集目录")})
                return
            if self.path == "/api/pick-test-image":
                values = clean_values(body.get("values"))
                self.send_json({"path": pick_image_file(values.get("test_image_file", ""))})
                return
            if self.path == "/api/pick-train-directory":
                values = clean_values(body.get("values"))
                field = str(body.get("field", ""))
                titles = {
                    "dataset_root": "选择训练输出目录",
                    "train_images_dir": "选择训练图片目录",
                    "train_annotations_dir": "选择 XML 标注目录",
                }
                if field not in titles:
                    raise ValueError("不支持的训练目录字段。")
                self.send_json({"path": pick_directory(values.get(field, ""), titles[field])})
                return
            if self.path == "/api/pick-raw-dataset-root":
                values = clean_values(body.get("values"))
                selected = pick_directory(values.get("raw_dataset_root", ""), "选择原始 YOLO TXT 数据集")
                self.send_json({"path": selected})
                return
            if self.path == "/api/dataset-convert/inspect":
                values = clean_values(body.get("values"))
                self.send_json({"info": inspect_yolo_txt_dataset(values)})
                return
            if self.path == "/api/dataset-import/run":
                values = clean_values(body.get("values"))
                result = inspect_prepared_yolo_dataset(values.get("raw_dataset_root", ""))
                train_split = result["splits"]["train"]
                values.update({
                    "raw_dataset_root": result["root"],
                    "raw_images_dir": train_split["images_dir"],
                    "raw_labels_dir": train_split["labels_dir"],
                    "raw_output_dir": result["root"],
                    "raw_class_names": ", ".join(result["class_names"]),
                    "dataset_root": result["root"],
                    "train_task": "detect",
                    "train_images_dir": train_split["images_dir"],
                    "train_annotations_dir": "",
                    "prepared_dataset_yaml": result["yaml_path"],
                })
                with STATE_LOCK:
                    STATE["values"] = values.copy()
                self.send_json({"ok": True, "result": result, "values": values})
                return
            if self.path == "/api/dataset-convert/run":
                values = clean_values(body.get("values"))
                with STATE_LOCK:
                    panel_busy = bool(STATE["running"])
                if panel_busy:
                    raise ValueError("当前训练或其他任务正在运行。请等任务结束后再转换，避免抢占磁盘和处理器资源。")
                result = convert_yolo_txt_to_voc(values)
                values.update({
                    "raw_dataset_root": result["root"],
                    "raw_images_dir": result["images_dir"],
                    "raw_labels_dir": result["labels_dir"],
                    "raw_output_dir": result["output_dir"],
                    "raw_class_names": ", ".join(result["class_names"]),
                    "dataset_root": result["output_dir"],
                    "train_task": "detect",
                    "train_images_dir": result["output_images_dir"],
                    "train_annotations_dir": result["output_annotations_dir"],
                    "prepared_dataset_yaml": "",
                })
                with STATE_LOCK:
                    STATE["values"] = values.copy()
                self.send_json({"ok": True, "result": result, "values": values})
                return
            if self.path == "/api/pick-base-model":
                values = clean_values(body.get("values"))
                self.send_json({"path": pick_model_file(values.get("base_model", ""))})
                return
            if self.path == "/api/pick-deploy-model":
                values = clean_values(body.get("values"))
                self.send_json({"path": pick_model_file(values.get("deploy_model", ""), include_onnx=True)})
                return
            if self.path == "/api/pick-deploy-output-dir":
                values = clean_values(body.get("values"))
                self.send_json({"path": pick_directory(values.get("export_output_dir", ""), "选择部署模型输出目录")})
                return
            if self.path == "/api/pick-asset-root":
                values = clean_values(body.get("values"))
                initial = values.get("asset_scan_root", "") or values.get("dataset_root", "")
                self.send_json({"path": pick_directory(initial, "选择包含 training-manifest.json 的训练运行根目录")})
                return
            if self.path == "/api/model-assets/scan":
                values = clean_values(body.get("values"))
                raw_root = str(body.get("root") or values.get("asset_scan_root") or values.get("dataset_root") or "").strip()
                root = Path(raw_root).expanduser().resolve()
                if not root.is_dir():
                    raise ValueError("模型资产扫描目录必须是有效文件夹。")
                with MODEL_REGISTRY_LOCK:
                    register_asset_root(MODEL_REGISTRY_FILE, root)
                values["asset_scan_root"] = str(root)
                with STATE_LOCK:
                    STATE["values"] = values.copy()
                self.send_json({"ok": True, "catalog": model_asset_catalog(values), "values": values})
                return
            if self.path == "/api/model-assets/register":
                raw_labels = body.get("labels")
                labels = raw_labels if isinstance(raw_labels, list) else str(raw_labels or "").replace("，", ",").split(",")
                with MODEL_REGISTRY_LOCK:
                    record = register_external_model(
                        MODEL_REGISTRY_FILE,
                        str(body.get("model_path") or ""),
                        dataset_name=str(body.get("dataset_name") or "未关联数据集"),
                        dataset_root=str(body.get("dataset_root") or ""),
                        task=str(body.get("task") or "unknown"),
                        project_id=str(body.get("project_id") or ""),
                        labels=labels,
                        notes=str(body.get("notes") or ""),
                    )
                self.send_json({"ok": True, "record": record, "catalog": model_asset_catalog(clean_values({}))})
                return
            if self.path == "/api/pick-external-model":
                self.send_json({"path": pick_model_file(str(body.get("initial") or ""), include_onnx=True)})
                return
            if self.path == "/api/pick-test-image-folder":
                values = clean_values(body.get("values"))
                self.send_json({"path": pick_directory(values.get("test_image_folder", ""))})
                return
            if self.path == "/api/pick-test-output-dir":
                values = clean_values(body.get("values"))
                self.send_json({"path": pick_directory(values.get("test_output_dir", ""))})
                return
            if self.path == "/api/pick-label-video-dir":
                values = clean_values(body.get("values"))
                selected = pick_directory(values.get("label_video_dir", ""))
                if selected:
                    values["label_video_dir"] = selected
                    with STATE_LOCK:
                        STATE["values"] = values.copy()
                self.send_json({"path": selected})
                return
            if self.path == "/api/pick-label-images-dir":
                values = clean_values(body.get("values"))
                selected = pick_directory(values.get("label_images_input_dir", ""))
                if selected:
                    values["label_images_input_dir"] = selected
                    with STATE_LOCK:
                        STATE["values"] = values.copy()
                self.send_json({"path": selected})
                return
            if self.path == "/api/label-session/start":
                values = clean_values(body.get("values"))
                state = start_label_session(values)
                self.send_json({"ok": True, "state": state})
                return
            if self.path == "/api/label-session/advance":
                session = get_label_session(str(body.get("session_id", "")))
                with session["lock"]:
                    state = advance_label_session(session)
                self.send_json({"ok": True, "state": state})
                return
            if self.path == "/api/label-session/object":
                session = get_label_session(str(body.get("session_id", "")))
                action = str(body.get("action", "add"))
                bbox_raw = body.get("bbox") or {}
                with session["lock"]:
                    frame = session["frame"]
                    bbox = sanitize_label_bbox(
                        (bbox_raw.get("x", 0), bbox_raw.get("y", 0), bbox_raw.get("w", 0), bbox_raw.get("h", 0)),
                        frame.shape[1], frame.shape[0],
                    )
                    if action == "delete":
                        object_id = int(body.get("object_id", 0))
                        session["objects"] = [obj for obj in session["objects"] if obj.obj_id != object_id]
                    else:
                        if bbox[2] < 3 or bbox[3] < 3:
                            raise ValueError("标注框过小，请重新绘制。")
                        if action in {"update", "add_sample"}:
                            object_id = int(body.get("object_id", 0))
                            obj = next((item for item in session["objects"] if item.obj_id == object_id), None)
                            if obj is None:
                                raise ValueError("要修正的目标不存在。")
                            if action == "add_sample":
                                add_sample = getattr(obj.tracker, "add_sample", None)
                                if not callable(add_sample):
                                    raise ValueError("追加视角仅支持多视角实验模式，请重新开始会话后选择该模式。")
                                if not add_sample(frame, bbox):
                                    raise ValueError("参考视角采集失败，请重新绘制更大的框。")
                                obj.bbox = bbox
                                obj.ok = True
                                obj.sample_count += 1
                                obj.quality = 1.0
                                obj.warning = ""
                            else:
                                obj.bbox = bbox
                                obj.tracker = make_label_tracker(session["tracker"])
                                obj.ok = init_label_tracker(obj.tracker, frame, bbox)
                                obj.sample_count = 1
                                obj.quality = 1.0 if obj.ok else 0.0
                                obj.warning = "" if obj.ok else "跟踪器重新初始化失败"
                        else:
                            label = str(body.get("label", "")).strip()
                            if label not in session["labels"]:
                                raise ValueError("请选择当前标签列表中的类别。")
                            tracker = make_label_tracker(session["tracker"])
                            if not init_label_tracker(tracker, frame, bbox):
                                raise ValueError("跟踪器初始化失败，请换一个更大的标注框。")
                            session["objects"].append(LabelTrackObject(session["next_object_id"], label, bbox, tracker))
                            session["next_object_id"] += 1
                    if not any(not obj.ok for obj in session["objects"]):
                        session["last_warning"] = ""
                    session["last_auto_saved"] = False
                    state = label_session_state(session)
                self.send_json({"ok": True, "state": state})
                return
            if self.path == "/api/label-session/save":
                session = get_label_session(str(body.get("session_id", "")))
                with session["lock"]:
                    saved = save_label_session_sample(session)
                    state = label_session_state(session)
                self.send_json({"ok": True, "saved": saved, "state": state})
                return
            if self.path == "/api/label-session/end":
                end_label_session(str(body.get("session_id", "")))
                self.send_json({"ok": True})
                return



            if self.path == "/api/run":
                action = str(body.get("action", ""))
                values = clean_values(body.get("values"))
                start_job(action, values)
                self.send_json({"ok": True})
                return
            if self.path == "/api/base-models/download":
                values = clean_values(body.get("values"))
                values["model_download_name"] = str(body.get("name") or "").strip().lower()
                values["model_download_force"] = bool(body.get("force"))
                start_job("model_download", values)
                self.send_json({"ok": True})
                return
            if self.path == "/api/stop":
                self.send_json({"stopped": stop_job()})
                return
            if self.path == "/api/stop-train-export":
                self.send_json({"stopped": stop_train_and_export()})
                return

            if self.path == "/api/delete-label-sample":
                image = str(body.get("image", ""))
                xml = str(body.get("xml", ""))
                with STATE_LOCK:
                    values = STATE["values"].copy()
                image_path = resolve_under(image, label_result_images_dir(values))
                xml_path = resolve_under(xml, Path(values["label_annotations_dir"]))
                paths = (xml_path,) if values.get("label_source_type") == "images" else (image_path, xml_path)
                deleted = []
                for path in paths:
                    if path.exists() and path.is_file():
                        path.unlink()
                        deleted.append(str(path))
                self.send_json({"ok": True, "deleted": deleted})
                return
            self.send_error(404)

        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO Team Training Platform web panel")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8989)


    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    with STATE_LOCK:
        STATE["values"] = load_user_defaults()
        try:
            previous_log = LATEST_JOB_LOG_FILE.read_text(encoding="utf-8", errors="replace")
            STATE["logs"] = previous_log.splitlines(keepends=True)[-MAX_LOG_LINES:]
        except OSError:
            STATE["logs"] = []

    url = f"http://127.0.0.1:{args.port}"
    try:
        server = ThreadingHTTPServer((args.host, args.port), PanelHandler)
    except OSError as exc:
        print(f"无法监听 {args.host}:{args.port}：{exc}", file=sys.stderr)
        print(f"请确认 {args.port} 端口未被占用，并允许 Python 通过防火墙。", file=sys.stderr)


        raise SystemExit(1) from exc
    print(f"YOLO团队训练平台：{url}")
    print(f"Listening on {args.host}:{args.port}")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_job()
        server.server_close()


if __name__ == "__main__":
    main()
