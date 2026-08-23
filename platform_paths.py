# -*- coding: utf-8 -*-
"""Product identity, workspace layout, and artifact naming rules."""

from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path
from typing import Iterable


PRODUCT_NAME = "YOLO团队训练平台"
PRODUCT_CODE = "YOLOTeamTrainingPlatform"
PRODUCT_SLUG = "yolo-team-training-platform"
PRODUCT_VERSION_FILE = "VERSION.txt"

APP_ROOT = Path(__file__).resolve().parent
INSTALL_ROOT = Path(os.environ.get("YOLO_TEAM_PLATFORM_HOME") or APP_ROOT).expanduser().resolve()
WORKSPACE_ROOT = Path(os.environ.get("YOLO_TEAM_PLATFORM_DATA") or (INSTALL_ROOT / "workspace")).expanduser().resolve()

CONFIG_DIR = WORKSPACE_ROOT / "config"
LOG_DIR = WORKSPACE_ROOT / "logs"
STATE_DIR = WORKSPACE_ROOT / "state"
DATASETS_DIR = WORKSPACE_ROOT / "datasets"
ANNOTATION_HUB_DIR = WORKSPACE_ROOT / "annotation-hub"
TRAINING_RUNS_DIR = WORKSPACE_ROOT / "training-runs"
MODEL_ASSETS_DIR = WORKSPACE_ROOT / "model-assets"
EXPORTS_DIR = WORKSPACE_ROOT / "exports"
DATASET_EXPORTS_DIR = EXPORTS_DIR / "datasets"
DEPLOYMENT_EXPORTS_DIR = EXPORTS_DIR / "deployments"
TEST_RESULTS_DIR = WORKSPACE_ROOT / "test-results"
CACHE_DIR = WORKSPACE_ROOT / "cache"
TEMP_DIR = WORKSPACE_ROOT / "temp"
BACKUPS_DIR = WORKSPACE_ROOT / "backups"

WORKSPACE_DIRECTORIES = (
    CONFIG_DIR,
    LOG_DIR,
    STATE_DIR,
    DATASETS_DIR,
    ANNOTATION_HUB_DIR,
    TRAINING_RUNS_DIR,
    MODEL_ASSETS_DIR,
    DATASET_EXPORTS_DIR,
    DEPLOYMENT_EXPORTS_DIR,
    TEST_RESULTS_DIR,
    CACHE_DIR,
    TEMP_DIR,
    BACKUPS_DIR,
)

_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul", *(f"com{index}" for index in range(1, 10)), *(f"lpt{index}" for index in range(1, 10)),
}


def ensure_workspace() -> Path:
    for directory in WORKSPACE_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_ROOT


def safe_identifier(value: object, fallback: str = "project", max_length: int = 48) -> str:
    """Return a portable lowercase identifier for directories and artifacts."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", text, flags=re.UNICODE).strip("-.")
    text = re.sub(r"-{2,}", "-", text)
    if not text or text in _WINDOWS_RESERVED:
        text = fallback
    return text[:max_length].rstrip("-.") or fallback


def local_timestamp(moment: dt.datetime | None = None) -> str:
    value = moment or dt.datetime.now().astimezone()
    return value.strftime("%Y%m%d-%H%M%S")


def artifact_stem(parts: Iterable[object], timestamp: str | None = None) -> str:
    identifiers = [safe_identifier(part, "item") for part in parts if str(part or "").strip()]
    identifiers.append(timestamp or local_timestamp())
    return "__".join(identifiers)


def unique_directory(parent: Path, stem: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / stem
    index = 2
    while candidate.exists():
        candidate = parent / f"{stem}__{index:02d}"
        index += 1
    return candidate


def default_windows_install_root() -> str:
    """Documented installer default; the installer applies the same D-drive fallback rule."""
    return r"D:\YOLOTeamTrainingPlatform"
