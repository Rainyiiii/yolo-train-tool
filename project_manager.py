# -*- coding: utf-8 -*-
"""Project registry and lightweight dataset inspection for the local platform."""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from platform_paths import CONFIG_DIR, DATASETS_DIR, safe_identifier


PROJECTS_FILE = CONFIG_DIR / "projects.json"
PROJECT_SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_EXTENSIONS = {".txt", ".xml", ".json"}


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def load_projects(path: Path = PROJECTS_FILE) -> dict[str, Any]:
    raw = _read_json(path)
    projects = [item for item in raw.get("projects", []) if isinstance(item, dict) and item.get("id")]
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "active_project_id": str(raw.get("active_project_id") or ""),
        "projects": projects,
    }


def save_projects(registry: dict[str, Any], path: Path = PROJECTS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "active_project_id": str(registry.get("active_project_id") or ""),
        "projects": registry.get("projects", []),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _unique_project_id(name: str, registry: dict[str, Any]) -> str:
    base = safe_identifier(name, "project")
    used = {str(item.get("id")) for item in registry["projects"]}
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}-{index}"
        index += 1
    return candidate


def create_project(
    name: str,
    task: str = "detect",
    labels: list[str] | None = None,
    dataset_root: str | Path | None = None,
    path: Path = PROJECTS_FILE,
) -> dict[str, Any]:
    project_name = str(name or "").strip()
    if not project_name:
        raise ValueError("项目名称不能为空。")
    if task not in {"detect", "classify"}:
        raise ValueError("当前项目任务仅支持目标检测或图像分类。")
    registry = load_projects(path)
    project_id = _unique_project_id(project_name, registry)
    project_root = (DATASETS_DIR / project_id).resolve()
    for directory in (
        project_root / "images",
        project_root / "labels",
        project_root / "annotations",
        project_root / "imports",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    chosen_dataset = Path(dataset_root).expanduser().resolve() if str(dataset_root or "").strip() else project_root
    normalized_labels = list(dict.fromkeys(str(item).strip() for item in (labels or []) if str(item).strip()))
    now = _now()
    project = {
        "id": project_id,
        "name": project_name,
        "task": task,
        "labels": normalized_labels,
        "root": str(project_root),
        "dataset_root": str(chosen_dataset),
        "created_at": now,
        "updated_at": now,
        "notes": "",
    }
    registry["projects"].append(project)
    registry["active_project_id"] = project_id
    save_projects(registry, path)
    return project


def activate_project(project_id: str, path: Path = PROJECTS_FILE) -> dict[str, Any]:
    registry = load_projects(path)
    project = next((item for item in registry["projects"] if item.get("id") == project_id), None)
    if project is None:
        raise ValueError("项目不存在。")
    registry["active_project_id"] = project_id
    project["updated_at"] = _now()
    save_projects(registry, path)
    return project


def update_project(project_id: str, values: dict[str, Any], path: Path = PROJECTS_FILE) -> dict[str, Any]:
    registry = load_projects(path)
    project = next((item for item in registry["projects"] if item.get("id") == project_id), None)
    if project is None:
        raise ValueError("项目不存在。")
    if "name" in values:
        name = str(values.get("name") or "").strip()
        if not name:
            raise ValueError("项目名称不能为空。")
        project["name"] = name
    if "dataset_root" in values:
        raw_root = str(values.get("dataset_root") or "").strip()
        project["dataset_root"] = str(Path(raw_root).expanduser().resolve()) if raw_root else project["root"]
    if "labels" in values:
        raw_labels = values.get("labels") if isinstance(values.get("labels"), list) else []
        project["labels"] = list(dict.fromkeys(str(item).strip() for item in raw_labels if str(item).strip()))
    if "notes" in values:
        project["notes"] = str(values.get("notes") or "")[:4000]
    project["updated_at"] = _now()
    save_projects(registry, path)
    return project


def _find_files(root: Path, extensions: set[str]) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def _inspect_yolo_label(path: Path, class_counts: Counter[str]) -> tuple[int, int]:
    invalid = 0
    boxes = 0
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError:
        return 1, 0
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            invalid += 1
            continue
        try:
            class_id = int(float(parts[0]))
            coordinates = [float(value) for value in parts[1:5]]
        except ValueError:
            invalid += 1
            continue
        if class_id < 0 or any(value < 0 or value > 1 for value in coordinates) or coordinates[2] <= 0 or coordinates[3] <= 0:
            invalid += 1
            continue
        class_counts[str(class_id)] += 1
        boxes += 1
    return invalid, boxes


def inspect_dataset(root: str | Path, preview_limit: int = 60, verify_limit: int = 2000, task: str = "detect") -> dict[str, Any]:
    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.is_dir():
        return {
            "root": str(dataset_root),
            "exists": False,
            "health": "error",
            "issues": ["数据集目录不存在。"],
            "image_count": 0,
            "label_count": 0,
            "preview": [],
            "class_counts": {},
        }
    images = _find_files(dataset_root, IMAGE_EXTENSIONS)
    labels = _find_files(dataset_root, LABEL_EXTENSIONS) if task != "classify" else []
    image_stems = {path.stem.casefold() for path in images}
    label_stems = {path.stem.casefold() for path in labels}
    missing_labels = sum(1 for path in images if path.stem.casefold() not in label_stems) if task != "classify" else 0
    orphan_labels = sum(1 for path in labels if path.stem.casefold() not in image_stems) if task != "classify" else 0
    bad_images: list[str] = []
    for image_path in images[:verify_limit]:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, ValueError):
            bad_images.append(str(image_path.relative_to(dataset_root)))
    class_counts: Counter[str] = Counter()
    invalid_boxes = 0
    box_count = 0
    for label_path in (path for path in labels if path.suffix.lower() == ".txt"):
        invalid, boxes = _inspect_yolo_label(label_path, class_counts)
        invalid_boxes += invalid
        box_count += boxes
    if task == "classify":
        for image_path in images:
            relative = image_path.relative_to(dataset_root)
            if len(relative.parts) > 1:
                class_counts[relative.parts[-2]] += 1
    issues: list[str] = []
    if not images:
        issues.append("没有找到支持的图片。")
    if missing_labels:
        issues.append(f"{missing_labels} 张图片没有同名标注文件。")
    if orphan_labels:
        issues.append(f"{orphan_labels} 个标注文件没有同名图片。")
    if bad_images:
        issues.append(f"发现 {len(bad_images)} 张无法读取的图片。")
    if invalid_boxes:
        issues.append(f"YOLO 标签中有 {invalid_boxes} 行格式或坐标无效。")
    health = "ok" if images and not issues else "warning" if images else "error"
    preview = []
    for image_path in images[:preview_limit]:
        try:
            size = image_path.stat().st_size
        except OSError:
            size = 0
        preview.append({
            "name": image_path.name,
            "path": str(image_path),
            "relative_path": str(image_path.relative_to(dataset_root)).replace("\\", "/"),
            "size_kb": round(size / 1024, 1),
            "has_label": task == "classify" or image_path.stem.casefold() in label_stems,
        })
    return {
        "root": str(dataset_root),
        "exists": True,
        "health": health,
        "issues": issues,
        "image_count": len(images),
        "label_count": len(labels),
        "box_count": box_count,
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "bad_images": bad_images[:30],
        "invalid_boxes": invalid_boxes,
        "class_counts": dict(sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))),
        "preview": preview,
        "truncated": len(images) > preview_limit,
    }


def project_catalog(path: Path = PROJECTS_FILE, include_health: bool = True) -> dict[str, Any]:
    registry = load_projects(path)
    projects: list[dict[str, Any]] = []
    for item in registry["projects"]:
        project = dict(item)
        project["active"] = item.get("id") == registry["active_project_id"]
        if include_health:
            project["dataset"] = inspect_dataset(item.get("dataset_root") or item.get("root"), task=str(item.get("task") or "detect"))
        projects.append(project)
    return {
        "active_project_id": registry["active_project_id"],
        "projects": projects,
        "summary": {
            "project_count": len(projects),
            "image_count": sum(int(item.get("dataset", {}).get("image_count", 0)) for item in projects),
            "issue_count": sum(len(item.get("dataset", {}).get("issues", [])) for item in projects),
        },
    }
