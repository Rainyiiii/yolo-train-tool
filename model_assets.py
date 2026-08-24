# -*- coding: utf-8 -*-
"""Persistent training/deployment asset registry and catalog scanner."""

from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Any, Iterable


REGISTRY_SCHEMA_VERSION = 1
TRAINING_MANIFEST_NAME = "training-manifest.json"


def _resolved_text(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_registry(registry_path: Path) -> dict[str, Any]:
    data = _load_json(registry_path)
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "roots": [str(item) for item in data.get("roots", []) if str(item).strip()],
        "manifests": [str(item) for item in data.get("manifests", []) if str(item).strip()],
        "external_models": [item for item in data.get("external_models", []) if isinstance(item, dict) and item.get("path")],
    }


def save_registry(registry_path: Path, registry: dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "roots": sorted(dict.fromkeys(registry.get("roots", [])), key=str.casefold),
        "manifests": sorted(dict.fromkeys(registry.get("manifests", [])), key=str.casefold),
        "external_models": registry.get("external_models", []),
    }
    temporary = registry_path.with_suffix(registry_path.suffix + ".tmp")
    temporary.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(registry_path)


def register_asset_root(registry_path: Path, root: str | Path) -> str:
    resolved = _resolved_text(root)
    registry = load_registry(registry_path)
    if resolved not in registry["roots"]:
        registry["roots"].append(resolved)
        save_registry(registry_path, registry)
    return resolved


def register_asset_manifest(registry_path: Path, manifest: str | Path) -> str:
    resolved = _resolved_text(manifest)
    registry = load_registry(registry_path)
    if resolved not in registry["manifests"]:
        registry["manifests"].append(resolved)
    path = Path(resolved)
    if path.name == TRAINING_MANIFEST_NAME:
        root = str(path.parent)
        if root not in registry["roots"]:
            registry["roots"].append(root)
    save_registry(registry_path, registry)
    return resolved


def register_external_model(
    registry_path: Path,
    model_path: str | Path,
    dataset_name: str = "未关联数据集",
    dataset_root: str | Path | None = None,
    task: str = "unknown",
    project_id: str = "",
    labels: list[str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    path = Path(model_path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() not in {".pt", ".onnx"}:
        raise ValueError("请选择存在的 .pt 或 .onnx 模型文件。")
    registry = load_registry(registry_path)
    record = {
        "path": str(path),
        "name": path.stem,
        "dataset_name": str(dataset_name or "未关联数据集").strip(),
        "dataset_root": _resolved_text(dataset_root) if str(dataset_root or "").strip() else str(path.parent),
        "task": task if task in {"detect", "classify"} else "unknown",
        "project_id": str(project_id or ""),
        "labels": list(dict.fromkeys(str(item).strip() for item in (labels or []) if str(item).strip())),
        "notes": str(notes or "")[:2000],
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    existing = next((item for item in registry["external_models"] if str(item.get("path", "")).casefold() == str(path).casefold()), None)
    if existing is None:
        registry["external_models"].append(record)
    else:
        existing.update(record)
        record = existing
    save_registry(registry_path, registry)
    return record


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _artifact(kind: str, raw_path: str | Path | None, base_dir: Path | None = None) -> dict[str, Any] | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    path = path.resolve()
    exists = path.exists()
    size = path.stat().st_size if exists and path.is_file() else 0
    return {
        "kind": kind,
        "path": str(path),
        "name": path.name,
        "exists": exists,
        "size_mb": round(size / (1024 * 1024), 2),
    }


def _run_from_manifest(manifest_path: Path, data: dict[str, Any]) -> dict[str, Any] | None:
    if data.get("kind") != "training_run":
        return None
    dataset = data.get("dataset") if isinstance(data.get("dataset"), dict) else {}
    training = data.get("training") if isinstance(data.get("training"), dict) else {}
    artifacts_raw = data.get("artifacts") if isinstance(data.get("artifacts"), dict) else {}
    classes_raw = dataset.get("classes") if isinstance(dataset.get("classes"), list) else []
    artifacts = [
        item for item in (
            _artifact("pt", artifacts_raw.get("pt"), manifest_path.parent),
            _artifact("onnx", artifacts_raw.get("onnx"), manifest_path.parent),
            _artifact("classes", artifacts_raw.get("classes"), manifest_path.parent),
            _artifact("results", artifacts_raw.get("results_csv"), manifest_path.parent),
            _artifact("test_evaluation", artifacts_raw.get("test_evaluation"), manifest_path.parent),
        ) if item is not None
    ]
    configured_output = Path(str(data.get("output_dir") or manifest_path.parent)).expanduser()
    output_dir = _resolved_text(configured_output if configured_output.exists() else manifest_path.parent)
    configured_dataset_root = Path(str(dataset.get("root") or Path(output_dir).parent)).expanduser()
    dataset_root = _resolved_text(configured_dataset_root if configured_dataset_root.exists() else Path(output_dir).parent)
    return {
        "run_id": str(data.get("run_id") or manifest_path.parent.name),
        "created_at": str(data.get("created_at") or ""),
        "status": str(data.get("status") or "completed"),
        "association": "manifest",
        "manifest": str(manifest_path.resolve()),
        "output_dir": output_dir,
        "model_name": str(training.get("model_name") or next((item["name"] for item in artifacts if item["kind"] == "pt"), manifest_path.parent.name)),
        "task": str(dataset.get("task") or training.get("task") or "unknown"),
        "classes": [str(item) for item in classes_raw],
        "dataset": {
            "id": str(dataset.get("id") or Path(output_dir).parent.name),
            "name": str(dataset.get("name") or Path(output_dir).parent.name),
            "root": dataset_root,
            "source": str(dataset.get("source") or ""),
            "image_count": _safe_int(dataset.get("image_count")),
            "version": str(dataset.get("version") or ""),
        },
        "training": training,
        "metrics": data.get("metrics") if isinstance(data.get("metrics"), dict) else {},
        "artifacts": artifacts,
        "deployments": [],
    }


def _candidate_output_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    if (root / TRAINING_MANIFEST_NAME).is_file():
        return [root]
    return sorted((path.parent for path in root.rglob(TRAINING_MANIFEST_NAME) if path.is_file()), reverse=True)[:1000]


def _deployment_record(manifest_path: Path, data: dict[str, Any]) -> dict[str, Any] | None:
    if "target" not in data or "source_model" not in data:
        return None
    artifact = _artifact(str(data.get("format") or "model"), data.get("artifact"), manifest_path.parent)
    source_model = Path(str(data["source_model"])).expanduser()
    if not source_model.is_absolute():
        source_model = manifest_path.parent / source_model
    return {
        "target": str(data.get("target") or ""),
        "target_label": str(data.get("target_label") or data.get("target") or ""),
        "format": str(data.get("format") or ""),
        "chip": data.get("chip"),
        "source_model": _resolved_text(source_model),
        "artifact": artifact,
        "manifest": str(manifest_path.resolve()),
    }


def collect_model_assets(
    registry_path: Path,
    extra_roots: Iterable[str | Path] = (),
    deployment_roots: Iterable[str | Path] = (),
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    roots: list[Path] = []
    for raw in [*registry["roots"], *extra_roots]:
        if not str(raw).strip():
            continue
        path = Path(raw).expanduser().resolve()
        if path not in roots:
            roots.append(path)

    manifest_paths: list[Path] = []
    for raw in registry["manifests"]:
        path = Path(raw).expanduser().resolve()
        if path.is_file() and path not in manifest_paths:
            manifest_paths.append(path)
    output_dirs: list[Path] = []
    for root in roots:
        for output_dir in _candidate_output_dirs(root):
            if output_dir not in output_dirs:
                output_dirs.append(output_dir)
            manifest = output_dir / TRAINING_MANIFEST_NAME
            if manifest.is_file() and manifest not in manifest_paths:
                manifest_paths.append(manifest)

    training_by_dir: dict[str, dict[str, Any]] = {}
    deployments: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        data = _load_json(manifest_path)
        run = _run_from_manifest(manifest_path, data)
        if run is not None:
            training_by_dir[str(Path(run["output_dir"]).resolve()).casefold()] = run
            continue
        deployment = _deployment_record(manifest_path, data)
        if deployment is not None:
            deployments.append(deployment)

    for raw_root in deployment_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            continue
        for path in root.rglob("*.manifest.json"):
            deployment = _deployment_record(path, _load_json(path))
            if deployment is not None and all(item["manifest"] != deployment["manifest"] for item in deployments):
                deployments.append(deployment)

    runs = list(training_by_dir.values())
    for index, record in enumerate(registry["external_models"], start=1):
        model_path = Path(str(record.get("path") or "")).expanduser().resolve()
        dataset_name = str(record.get("dataset_name") or "未关联数据集")
        dataset_root = str(record.get("dataset_root") or model_path.parent)
        runs.append({
            "run_id": f"external-{index}",
            "created_at": str(record.get("created_at") or ""),
            "status": "registered" if model_path.is_file() else "missing",
            "association": "manual",
            "manifest": "",
            "output_dir": str(model_path.parent),
            "model_name": str(record.get("name") or model_path.stem),
            "task": str(record.get("task") or "unknown"),
            "classes": [str(item) for item in record.get("labels", [])],
            "dataset": {
                "id": str(record.get("project_id") or dataset_name),
                "name": dataset_name,
                "root": dataset_root,
                "source": "手动登记",
                "image_count": 0,
                "version": "",
            },
            "training": {"notes": str(record.get("notes") or "")},
            "metrics": {},
            "artifacts": [item for item in [_artifact(model_path.suffix.lower().lstrip("."), model_path)] if item is not None],
            "deployments": [],
        })
    for run in runs:
        model_paths = {item["path"].casefold() for item in run["artifacts"] if item["kind"] in {"pt", "onnx"}}
        run["deployments"] = [item for item in deployments if item["source_model"].casefold() in model_paths]

    grouped: dict[str, dict[str, Any]] = {}
    for run in sorted(runs, key=lambda item: item.get("created_at", ""), reverse=True):
        dataset = run["dataset"]
        key = f"{dataset['id']}|{str(dataset['root']).casefold()}"
        group = grouped.setdefault(key, {
            "id": dataset["id"], "name": dataset["name"], "root": dataset["root"],
            "source": dataset.get("source", ""), "image_count": dataset.get("image_count", 0),
            "version": dataset.get("version", ""), "tasks": [], "classes": [], "runs": [],
        })
        group["runs"].append(run)
        if run["task"] not in group["tasks"]:
            group["tasks"].append(run["task"])
        for class_name in run["classes"]:
            if class_name not in group["classes"]:
                group["classes"].append(class_name)

    datasets = list(grouped.values())
    model_count = sum(sum(item["exists"] for item in run["artifacts"] if item["kind"] in {"pt", "onnx"}) for run in runs)
    return {
        "summary": {
            "dataset_count": len(datasets),
            "run_count": len(runs),
            "model_count": model_count,
            "deployment_count": sum(len(run["deployments"]) for run in runs),
        },
        "roots": [str(path) for path in roots],
        "datasets": datasets,
    }
