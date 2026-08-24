from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


SCALES = (
    ("n", "Nano", "边缘设备优先", 1),
    ("s", "Small", "速度与精度平衡", 2),
    ("m", "Medium", "需要较强 GPU", 3),
    ("l", "Large", "高精度与高显存", 4),
    ("x", "X-Large", "最高精度与最高资源占用", 5),
)

MODEL_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "id": "yolo26",
        "label": "YOLO26",
        "prefix": "yolo26",
        "year": "2026",
        "status": "最新 · 边缘优先",
        "description": "原生端到端、默认免 NMS，面向 CPU 和低功耗边缘部署优化。",
        "docs_url": "https://docs.ultralytics.com/models/yolo26/",
        "recommended": True,
    },
    {
        "id": "yolo11",
        "label": "YOLO11",
        "prefix": "yolo11",
        "year": "2024",
        "status": "成熟稳定",
        "description": "当前平台长期使用的稳定系列，训练、导出和设备适配经验较多。",
        "docs_url": "https://docs.ultralytics.com/models/yolo11/",
        "recommended": False,
    },
    {
        "id": "yolov8",
        "label": "YOLOv8",
        "prefix": "yolov8",
        "year": "2023",
        "status": "兼容性优先",
        "description": "生态成熟，适合需要兼容既有脚本、模型或部署工具链的项目。",
        "docs_url": "https://docs.ultralytics.com/models/yolov8/",
        "recommended": False,
    },
)


def _model_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        for task, suffix, task_label in (
            ("detect", "", "目标检测"),
            ("classify", "-cls", "图像分类"),
        ):
            for scale, scale_label, guidance, resource_level in SCALES:
                filename = f"{family['prefix']}{scale}{suffix}.pt"
                entries.append({
                    "name": filename,
                    "family": family["id"],
                    "family_label": family["label"],
                    "task": task,
                    "task_label": task_label,
                    "scale": scale,
                    "scale_label": scale_label,
                    "guidance": guidance,
                    "resource_level": resource_level,
                    "docs_url": family["docs_url"],
                    "recommended": bool(family["recommended"] and scale == "n"),
                })
    return entries


MODEL_ENTRIES = tuple(_model_entries())
MODEL_INDEX = {entry["name"]: entry for entry in MODEL_ENTRIES}


def validate_model_name(name: str) -> dict[str, Any]:
    normalized = Path(str(name or "").strip()).name.lower()
    if normalized != str(name or "").strip().lower() or normalized not in MODEL_INDEX:
        raise ValueError("请选择模型中心列出的官方基础模型。")
    return MODEL_INDEX[normalized]


def base_model_catalog(model_root: str | Path) -> dict[str, Any]:
    root = Path(model_root).expanduser().resolve()
    models: list[dict[str, Any]] = []
    downloaded = 0
    for entry in MODEL_ENTRIES:
        path = root / entry["name"]
        exists = path.is_file() and path.stat().st_size >= 1024 * 1024
        if exists:
            downloaded += 1
        models.append({
            **entry,
            "downloaded": exists,
            "path": str(path),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 1) if exists else 0.0,
        })
    return {
        "root": str(root),
        "families": [dict(item) for item in MODEL_FAMILIES],
        "models": models,
        "summary": {"model_count": len(models), "downloaded_count": downloaded},
        "license": {
            "label": "Ultralytics AGPL-3.0 / Enterprise License",
            "url": "https://www.ultralytics.com/license",
            "notice": "下载和使用官方代码及权重前，请根据你的分发和商业使用方式确认 Ultralytics 许可证要求。",
        },
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_base_model(name: str, output_dir: str | Path, *, force: bool = False) -> dict[str, Any]:
    entry = validate_model_name(name)
    destination_root = Path(output_dir).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / entry["name"]
    if destination.is_file() and destination.stat().st_size >= 1024 * 1024 and not force:
        print(f"模型已存在，跳过下载：{destination}", flush=True)
        result = {"path": str(destination), "sha256": file_sha256(destination), "downloaded": False}
        print(f"MODEL_SHA256={result['sha256']}", flush=True)
        print(f"BASE_MODEL={destination}", flush=True)
        return result

    temp_parent = destination_root.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{entry['family']}-download-", dir=str(temp_parent)))
    partial = destination.with_suffix(destination.suffix + ".part")
    previous_cwd = Path.cwd()
    try:
        print(f"正在通过 Ultralytics 官方下载：{entry['name']}", flush=True)
        print(f"保存目录：{destination_root}", flush=True)
        os.chdir(temp_dir)
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("当前 Ultralytics 运行环境不可用，请先完整修复运行环境。") from exc
        model = YOLO(entry["name"])
        checkpoint = Path(str(getattr(model, "ckpt_path", entry["name"])))
        if not checkpoint.is_absolute():
            checkpoint = (temp_dir / checkpoint).resolve()
        if not checkpoint.is_file():
            fallback = temp_dir / entry["name"]
            checkpoint = fallback if fallback.is_file() else checkpoint
        if not checkpoint.is_file() or checkpoint.stat().st_size < 1024 * 1024:
            raise RuntimeError("官方模型下载结果不完整，未替换已有模型。")

        shutil.copy2(checkpoint, partial)
        os.replace(partial, destination)
        digest = file_sha256(destination)
        print(f"下载完成：{destination}（{destination.stat().st_size / (1024 * 1024):.1f} MB）", flush=True)
        print(f"MODEL_SHA256={digest}", flush=True)
        print(f"BASE_MODEL={destination}", flush=True)
        return {"path": str(destination), "sha256": digest, "downloaded": True}
    finally:
        os.chdir(previous_cwd)
        partial.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download an approved Ultralytics base model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_base_model(args.model, args.output_dir, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
