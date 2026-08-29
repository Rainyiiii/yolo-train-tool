# -*- coding: utf-8 -*-
"""Export a trained Ultralytics model for a selected deployment target."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from device_profiles import DEVICE_PROFILES, EXPORT_FORMATS, get_device_profile, resolve_export_format
from platform_paths import DEPLOYMENT_EXPORTS_DIR, artifact_stem, local_timestamp, safe_identifier, unique_directory
from rdk_x5_deployment import calibration_images, create_rdk_x5_bundle


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_stdio()


def parse_imgsz(value: str) -> int | tuple[int, int]:
    parts = [item.strip() for item in value.replace("x", ",").split(",") if item.strip()]
    if len(parts) not in {1, 2}:
        raise argparse.ArgumentTypeError("输入尺寸应为 640 或 480,640（高,宽）")
    try:
        numbers = [int(item) for item in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("输入尺寸必须是整数") from exc
    if any(item < 32 or item % 32 for item in numbers):
        raise argparse.ArgumentTypeError("输入尺寸必须是大于等于 32 的 32 倍数")
    return numbers[0] if len(numbers) == 1 else (numbers[0], numbers[1])


def load_names(data_path: str, classes_path: str = "", model_path: Path | None = None) -> dict[str, str]:
    if data_path:
        path = Path(data_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"数据配置不存在：{path}")
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        names = data.get("names", {})
        if isinstance(names, list):
            return {str(index): str(name) for index, name in enumerate(names)}
        if isinstance(names, dict):
            return {str(key): str(value) for key, value in names.items()}
    candidates = []
    if classes_path:
        candidates.append(Path(classes_path).expanduser().resolve())
    if model_path is not None:
        candidates.extend((model_path.parent / "dataset-classes.txt", model_path.parent / "classes.txt"))
    for path in candidates:
        if path.is_file():
            lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if lines:
                return {str(index): name for index, name in enumerate(lines)}
    return {}


def copy_export_artifact(source: Path, output_dir: Path, destination_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / destination_name
    if source.resolve() == destination.resolve():
        return source
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)
    return destination


def inspect_onnx_runtime(model_path: Path) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("缺少 onnxruntime，请重新运行安装器或执行 pip install -r requirements.txt") from exc
    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise RuntimeError(f"ONNX Runtime 无法加载导出的模型：{exc}") from exc
    describe = lambda value: {"name": value.name, "shape": value.shape, "type": value.type}
    return {
        "provider": "CPUExecutionProvider",
        "inputs": [describe(value) for value in session.get_inputs()],
        "outputs": [describe(value) for value in session.get_outputs()],
    }


def export_model(args: argparse.Namespace) -> tuple[Path, Path]:
    source = Path(args.model).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"模型文件不存在：{source}")
    if source.suffix.lower() not in {".pt", ".onnx"}:
        raise ValueError("部署入口目前支持 .pt 和 .onnx 模型。")

    profile = get_device_profile(args.target)
    vendor_ptq = bool(profile.get("vendor_ptq"))
    export_format = resolve_export_format(args.target, args.format)
    chip = (args.chip or profile.get("default_chip") or "").strip().lower()
    allowed_chips = [str(item) for item in profile.get("chips", [])]
    if chip and allowed_chips and chip not in allowed_chips:
        raise ValueError(f"{profile['label']} 的芯片可选值：{'、'.join(allowed_chips)}")
    if export_format != "onnx" and source.suffix.lower() != ".pt":
        raise ValueError(f"导出 {export_format} 需要原始 .pt 权重；ONNX 仅可复制为通用/厂商工具链输入。")
    if args.int8 and not args.data and not vendor_ptq:
        raise ValueError("INT8 导出需要 --data 指向 data.yaml，用于代表性校准数据。")
    calibration_dir_text = str(getattr(args, "calibration_images", "") or "").strip()
    calibration_dir = Path(calibration_dir_text).expanduser().resolve() if calibration_dir_text else None
    if vendor_ptq:
        if calibration_dir is None or not calibration_dir.is_dir():
            raise ValueError("RDK X5 NPU 转换需要 --calibration-images 指向代表性图片目录。")
        image_count = len(calibration_images(calibration_dir))
        if not image_count:
            raise ValueError("RDK X5 校准目录中没有支持的图片。")
        if image_count < 20:
            print(f"警告：RDK X5 校准图片只有 {image_count} 张；官方建议 20–50 张。")

    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else DEPLOYMENT_EXPORTS_DIR
    timestamp = local_timestamp()
    model_slug = safe_identifier(source.stem.removesuffix("-best"), "model")
    export_stem = artifact_stem([model_slug, args.target, export_format], timestamp)
    output_dir = unique_directory(output_root / safe_identifier(args.target, "target"), export_stem)
    names = load_names(args.data, getattr(args, "classes", ""), source)
    artifact_source = source

    if not (source.suffix.lower() == ".onnx" and export_format == "onnx"):
        from ultralytics import YOLO

        model = YOLO(str(source))
        model_names = getattr(model, "names", None)
        if not names and isinstance(model_names, dict):
            names = {str(key): str(value) for key, value in model_names.items()}
        kwargs: dict[str, Any] = {
            "format": export_format,
            "imgsz": args.imgsz,
            "dynamic": False,
        }
        if export_format == "onnx":
            kwargs.update({"opset": int(profile["opset"]), "simplify": True})
        if export_format == "rknn":
            kwargs["name"] = chip or "rk3588"
        if args.int8 and not vendor_ptq:
            kwargs.update({"int8": True, "data": str(Path(args.data).expanduser().resolve())})
        exported = model.export(**kwargs)
        if isinstance(exported, (list, tuple)):
            exported = exported[0]
        artifact_source = Path(str(exported)).expanduser().resolve()
        if not artifact_source.exists():
            raise RuntimeError(f"导出命令已结束，但未找到产物：{artifact_source}")

    if artifact_source.is_dir():
        destination_name = f"{export_stem}.{safe_identifier(export_format, 'model')}"
    else:
        destination_name = f"{export_stem}{artifact_source.suffix.lower()}"
    artifact = copy_export_artifact(artifact_source, output_dir, destination_name)
    runtime_check = inspect_onnx_runtime(artifact) if artifact.is_file() and artifact.suffix.lower() == ".onnx" else None
    vendor_conversion = None
    deployment_artifact = artifact
    if vendor_ptq:
        vendor_conversion = create_rdk_x5_bundle(
            output_dir=output_dir,
            source_model=source,
            onnx_artifact=artifact,
            calibration_dir=calibration_dir,
            input_size=args.imgsz,
            class_names=names,
        )
        deployment_artifact = Path(vendor_conversion["bundle"])
    manifest = {
        "schema_version": 3,
        "kind": "yolo_team_deployment_export",
        "export_id": output_dir.name,
        "target": args.target,
        "target_label": profile["label"],
        "family": profile["family"],
        "format": export_format,
        "chip": chip or None,
        "source_model": str(source),
        "artifact": str(deployment_artifact),
        "intermediate_artifact": str(artifact) if vendor_ptq else None,
        "final_artifact": profile.get("final_artifact") or str(deployment_artifact),
        "input_size": args.imgsz if isinstance(args.imgsz, int) else list(args.imgsz),
        "dynamic_shape": False,
        "int8": bool(args.int8 or profile.get("forced_int8")),
        "class_names": names,
        "vendor_toolchain_required": bool(profile["vendor_toolchain"]),
        "next_step": profile["next_step"],
        "documentation": profile["docs_url"],
        "runtime_check": runtime_check,
        "vendor_conversion": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in (vendor_conversion or {}).items()
            if key != "plan"
        } or None,
    }
    manifest_path = output_dir / f"{export_stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"部署平台：{profile['label']}")
    print(f"导出格式：{export_format}")
    print(f"模型产物：{deployment_artifact}")
    if vendor_ptq:
        print(f"中间 ONNX：{artifact}")
        print(f"最终目标：{profile['final_artifact']}（当前状态：等待 WSL/OpenExplorer 转换）")
        print(f"转换脚本：{vendor_conversion['conversion_script']}")
    print(f"部署清单：{manifest_path}")
    print(f"下一步：{profile['next_step']}")
    print(f"DEPLOY_ARTIFACT={deployment_artifact}", flush=True)
    print(f"DEPLOY_MANIFEST={manifest_path}", flush=True)
    return deployment_artifact, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO团队训练平台 multi-platform model exporter")
    parser.add_argument("--model", required=True, help="trained .pt model or an existing .onnx")
    parser.add_argument("--target", choices=DEVICE_PROFILES, default="generic_onnx")
    parser.add_argument("--format", choices=EXPORT_FORMATS, default="auto")
    parser.add_argument("--imgsz", type=parse_imgsz, default=parse_imgsz("480,640"), help="640 or height,width")
    parser.add_argument("--chip", default="", help="target chip, for example rk3588 or x5")
    parser.add_argument("--data", default="", help="data.yaml, required for INT8 calibration")
    parser.add_argument("--calibration-images", default="", help="representative images for vendor PTQ, required for RDK X5")
    parser.add_argument("--classes", default="", help="optional classes.txt for ONNX hand-off")
    parser.add_argument("--output-dir", default="", help="deployment export root")
    parser.add_argument("--int8", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_model(args)


if __name__ == "__main__":
    main()
