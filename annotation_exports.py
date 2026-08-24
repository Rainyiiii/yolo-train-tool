# -*- coding: utf-8 -*-
"""Dataset and portable project-package exporters for collaborative annotation."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from annotation_store import IMAGE_EXTENSIONS, AnnotationError, AnnotationStore, _json, _now
from platform_paths import local_timestamp, safe_identifier


EXPORT_FORMATS = {"yolo", "coco", "voc", "labelme", "project"}
MAX_PACKAGE_ITEMS = 100_000
MAX_PACKAGE_UNCOMPRESSED_BYTES = 100 * 1024 * 1024 * 1024


def _safe_name(value: str, fallback: str) -> str:
    clean = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", str(value), flags=re.UNICODE).strip("._")
    return clean[:100] or fallback


def _item_name(item: dict[str, Any]) -> str:
    source = Path(item["original_name"])
    return f"{item['id']:06d}_{_safe_name(source.stem, 'image')}{source.suffix.lower()}"


def _split_name(item: dict[str, Any]) -> str:
    value = int(hashlib.sha256(item["relative_source"].encode("utf-8", errors="replace")).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "val" if value < 90 else "test"


def _write_text(archive: zipfile.ZipFile, name: str, value: str) -> None:
    archive.writestr(name, value.encode("utf-8"))


def _add_image(archive: zipfile.ZipFile, item: dict[str, Any], target: str) -> None:
    path = Path(item["path"])
    if not path.is_file():
        raise AnnotationError(f"导出时找不到图片：{item['original_name']}")
    archive.write(path, target)


def _manifest(project: dict[str, Any], items: list[dict[str, Any]], version: str, export_format: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "yolo_team_dataset_export",
        "project": {"id": project["id"], "name": project["name"], "task_type": project["task_type"], "labels": project["labels"]},
        "dataset_version": version,
        "format": export_format,
        "image_count": len(items),
        "box_count": sum(len(item["boxes"]) for item in items),
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def export_dataset(store: AnnotationStore, actor: dict[str, Any], project_id: int, export_format: str) -> Path:
    export_format = str(export_format).lower()
    if export_format not in EXPORT_FORMATS:
        raise AnnotationError("不支持的数据集导出格式。")
    if export_format == "project":
        return export_project_package(store, actor, project_id)
    project, items = store.export_rows(project_id, actor)
    version = store.dataset_version(project, items)
    stamp = local_timestamp()
    base = safe_identifier(project["name"], f"project-{project_id}")
    project_export_dir = store.exports_dir / base
    project_export_dir.mkdir(parents=True, exist_ok=True)
    output = project_export_dir / f"{base}__dataset__{export_format}__v{version}__{stamp}.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        if export_format == "yolo":
            _export_yolo(archive, project, items)
        elif export_format == "coco":
            _export_coco(archive, project, items)
        elif export_format == "voc":
            _export_voc(archive, project, items)
        else:
            _export_labelme(archive, project, items)
        _write_text(archive, "dataset-manifest.json", json.dumps(_manifest(project, items, version, export_format), ensure_ascii=False, indent=2) + "\n")
    return output


def _export_yolo(archive: zipfile.ZipFile, project: dict[str, Any], items: list[dict[str, Any]]) -> None:
    labels = project["labels"]
    class_ids = {name: index for index, name in enumerate(labels)}
    split_counts = {"train": 0, "val": 0, "test": 0}
    for item in items:
        split = _split_name(item)
        split_counts[split] += 1
        name = _item_name(item)
        _add_image(archive, item, f"images/{split}/{name}")
        lines = []
        for box in item["boxes"]:
            cx = (float(box["x"]) + float(box["w"]) / 2) / item["width"]
            cy = (float(box["y"]) + float(box["h"]) / 2) / item["height"]
            width = float(box["w"]) / item["width"]
            height = float(box["h"]) / item["height"]
            lines.append(f"{class_ids[box['label']]} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}")
        _write_text(archive, f"labels/{split}/{Path(name).stem}.txt", "\n".join(lines) + ("\n" if lines else ""))
    yaml_lines = ["path: .", "train: images/train", "val: images/val"]
    if split_counts["test"]:
        yaml_lines.append("test: images/test")
    yaml_lines += [f"nc: {len(labels)}", "names: " + json.dumps(labels, ensure_ascii=False)]
    _write_text(archive, "data.yaml", "\n".join(yaml_lines) + "\n")


def _export_coco(archive: zipfile.ZipFile, project: dict[str, Any], items: list[dict[str, Any]]) -> None:
    labels = project["labels"]
    class_ids = {name: index + 1 for index, name in enumerate(labels)}
    images = []
    annotations = []
    annotation_id = 1
    for item in items:
        name = _item_name(item)
        _add_image(archive, item, f"images/{name}")
        images.append({"id": item["id"], "file_name": name, "width": item["width"], "height": item["height"]})
        for box in item["boxes"]:
            width, height = float(box["w"]), float(box["h"])
            annotations.append({
                "id": annotation_id,
                "image_id": item["id"],
                "category_id": class_ids[box["label"]],
                "bbox": [float(box["x"]), float(box["y"]), width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": [],
            })
            annotation_id += 1
    payload = {
        "info": {"description": project["name"], "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index + 1, "name": label, "supercategory": "object"} for index, label in enumerate(labels)],
    }
    _write_text(archive, "annotations/instances_default.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _export_voc(archive: zipfile.ZipFile, project: dict[str, Any], items: list[dict[str, Any]]) -> None:
    split_stems = {"train": [], "val": [], "test": []}
    for item in items:
        name = _item_name(item)
        stem = Path(name).stem
        split_stems[_split_name(item)].append(stem)
        _add_image(archive, item, f"JPEGImages/{name}")
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "JPEGImages"
        ET.SubElement(root, "filename").text = name
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(item["width"])
        ET.SubElement(size, "height").text = str(item["height"])
        ET.SubElement(size, "depth").text = "3"
        for box in item["boxes"]:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = box["label"]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bbox, "xmin").text = str(max(0, round(float(box["x"]))))
            ET.SubElement(bbox, "ymin").text = str(max(0, round(float(box["y"]))))
            ET.SubElement(bbox, "xmax").text = str(min(item["width"], round(float(box["x"]) + float(box["w"]))))
            ET.SubElement(bbox, "ymax").text = str(min(item["height"], round(float(box["y"]) + float(box["h"]))))
        _write_text(archive, f"Annotations/{stem}.xml", ET.tostring(root, encoding="unicode"))
    for split, stems in split_stems.items():
        if stems:
            _write_text(archive, f"ImageSets/Main/{split}.txt", "\n".join(stems) + "\n")


def _export_labelme(archive: zipfile.ZipFile, project: dict[str, Any], items: list[dict[str, Any]]) -> None:
    for item in items:
        name = _item_name(item)
        _add_image(archive, item, f"images/{name}")
        shapes = []
        for box in item["boxes"]:
            x, y, width, height = float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"])
            shapes.append({"label": box["label"], "points": [[x, y], [x + width, y + height]], "group_id": None, "description": "", "shape_type": "rectangle", "flags": {}})
        payload = {"version": "5.0.0", "flags": {}, "shapes": shapes, "imagePath": f"images/{name}", "imageData": None, "imageHeight": item["height"], "imageWidth": item["width"]}
        _write_text(archive, f"annotations/{Path(name).stem}.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def export_project_package(store: AnnotationStore, actor: dict[str, Any], project_id: int) -> Path:
    if actor["role"] not in {"admin", "reviewer"}:
        raise AnnotationError("只有管理员或审核员可以导出项目包。", 403)
    project = store.get_project(project_id, actor)
    with store.connect() as connection:
        rows = connection.execute(
            """SELECT i.*,u.username AS assignee_name FROM items i
               LEFT JOIN users u ON u.id=i.assignee_id WHERE i.project_id=? ORDER BY i.id""",
            (project_id,),
        ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["boxes"] = json.loads(item.pop("annotations_json") or "[]")
        item["path"] = store.projects_dir / str(project_id) / "images" / item["stored_name"]
        items.append(item)
    version = store.dataset_version(project, items)
    stamp = local_timestamp()
    project_slug = safe_identifier(project["name"], "project")
    project_export_dir = store.exports_dir / project_slug
    project_export_dir.mkdir(parents=True, exist_ok=True)
    output = project_export_dir / f"{project_slug}__annotation-project__v{version}__{stamp}.ytp-project.zip"
    manifest_items = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for item in items:
            package_name = _item_name(item)
            _add_image(archive, item, f"images/{package_name}")
            manifest_items.append({
                "image": f"images/{package_name}",
                "original_name": item["original_name"],
                "relative_source": item["relative_source"],
                "width": item["width"],
                "height": item["height"],
                "status": item["status"],
                "revision": item["revision"],
                "assignee": item.get("assignee_name") or "",
                "review_comment": item["review_comment"],
                "boxes": item["boxes"],
            })
        manifest = {
            "schema_version": 1,
            "kind": "yolo_team_annotation_project",
            "project": {
                "name": project["name"],
                "task_type": project["task_type"],
                "labels": project["labels"],
                "review_enabled": project.get("review_enabled", False),
            },
            "dataset_version": version,
            "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "items": manifest_items,
        }
        _write_text(archive, "project-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return output


def import_project_package(store: AnnotationStore, actor: dict[str, Any], package_path: str | Path) -> dict[str, Any]:
    if actor["role"] not in {"admin", "reviewer"}:
        raise AnnotationError("只有管理员或审核员可以导入项目包。", 403)
    package = Path(package_path).expanduser().resolve()
    if not package.is_file() or package.suffix.lower() != ".zip":
        raise AnnotationError("请选择有效的 .ytp-project.zip 项目包。")
    try:
        archive = zipfile.ZipFile(package, "r")
    except zipfile.BadZipFile as exc:
        raise AnnotationError("项目包不是有效的 ZIP 文件。") from exc
    with archive:
        try:
            manifest = json.loads(archive.read("project-manifest.json").decode("utf-8-sig"))
        except (KeyError, ValueError, UnicodeError) as exc:
            raise AnnotationError("项目包缺少有效的 project-manifest.json。") from exc
        if manifest.get("kind") != "yolo_team_annotation_project" or manifest.get("schema_version") != 1:
            raise AnnotationError("项目包版本不受支持。")
        project_data = manifest.get("project") if isinstance(manifest.get("project"), dict) else {}
        labels = [str(value).strip() for value in project_data.get("labels", []) if str(value).strip()]
        raw_items = manifest.get("items")
        if not isinstance(raw_items, list):
            raise AnnotationError("项目包中的图片清单无效。")
        if len(raw_items) > MAX_PACKAGE_ITEMS:
            raise AnnotationError(f"项目包超过 {MAX_PACKAGE_ITEMS} 张图片的安全上限。")
        if sum(info.file_size for info in archive.infolist()) > MAX_PACKAGE_UNCOMPRESSED_BYTES:
            raise AnnotationError("项目包解压后的总大小超过 100GB，已停止导入。")
        project = store.create_project(
            actor,
            f"{project_data.get('name') or package.stem}（导入）",
            labels,
            review_enabled=bool(project_data.get("review_enabled", True)),
        )
        project_id = project["id"]
        destination = store.projects_dir / str(project_id) / "images"
        destination.mkdir(parents=True, exist_ok=True)
        imported = 0
        now = _now()
        try:
            with store.connect() as connection:
                for raw in raw_items:
                    if not isinstance(raw, dict):
                        continue
                    member_name = str(raw.get("image") or "").replace("\\", "/")
                    member_path = Path(member_name)
                    if not member_name.startswith("images/") or ".." in member_path.parts or member_path.is_absolute():
                        raise AnnotationError("项目包中存在不安全的图片路径。")
                    try:
                        info = archive.getinfo(member_name)
                    except KeyError as exc:
                        raise AnnotationError(f"项目包缺少图片：{member_name}") from exc
                    if info.is_dir() or info.file_size > 250 * 1024 * 1024:
                        raise AnnotationError("项目包中存在无效图片或超过 250MB 的单张图片。")
                    suffix = member_path.suffix.lower()
                    if suffix not in IMAGE_EXTENSIONS:
                        raise AnnotationError(f"项目包中存在不支持的图片格式：{suffix or '无扩展名'}")
                    stored_name = f"{uuid.uuid4().hex}{suffix}"
                    target_path = destination / stored_name
                    with archive.open(info) as source, target_path.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    try:
                        with Image.open(target_path) as image:
                            width, height = image.size
                            image.verify()
                    except (OSError, UnidentifiedImageError) as exc:
                        raise AnnotationError(f"项目包中的文件不是有效图片：{member_name}") from exc
                    boxes = store._validate_boxes(raw.get("boxes", []), {"width": width, "height": height}, labels)
                    status = "submitted" if boxes else "unassigned"
                    connection.execute(
                        """INSERT INTO items(project_id,original_name,relative_source,stored_name,width,height,status,
                           annotations_json,revision,review_comment,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            project_id,
                            str(raw.get("original_name") or member_path.name),
                            str(raw.get("relative_source") or member_path.name),
                            stored_name,
                            width,
                            height,
                            status,
                            _json(boxes),
                            max(0, int(raw.get("revision") or 0)),
                            "从项目包导入，包含标注的图片需要审核。" if boxes else "",
                            now,
                            now,
                        ),
                    )
                    imported += 1
                connection.execute("UPDATE projects SET updated_at=? WHERE id=?", (now, project_id))
                store._event(connection, project_id, None, actor["id"], "project_package_imported", {"source": str(package), "count": imported})
        except Exception:
            with store.connect() as connection:
                connection.execute("DELETE FROM projects WHERE id=?", (project_id,))
            shutil.rmtree(destination.parent, ignore_errors=True)
            raise
    return {"id": project_id, "name": project["name"], "labels": labels, "imported": imported}
