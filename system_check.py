from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from platform_paths import LOG_DIR, PRODUCT_NAME

ROOT = Path(__file__).resolve().parent
REPORT_PATH = LOG_DIR / "system-check.json"


def configure_stdio() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_stdio()


def module_status(name: str, *, distribution: str = "", requirement: str = "") -> dict[str, object]:
    try:
        available = importlib.util.find_spec(name) is not None
    except Exception as exc:
        return {"ok": False, "reason": "missing", "error": str(exc)}
    if not available:
        return {"ok": False, "reason": "missing", "error": f"找不到 Python 模块 {name}。"}
    detected = "installed"
    if distribution:
        try:
            detected = package_version(distribution)
        except PackageNotFoundError:
            return {
                "ok": False,
                "reason": "wrong-package",
                "version": detected,
                "error": f"需要安装 {distribution}{requirement}。",
            }
    if requirement:
        try:
            compatible = Version(detected) in SpecifierSet(requirement)
        except InvalidVersion:
            compatible = False
        if not compatible:
            return {
                "ok": False,
                "reason": "incompatible",
                "version": detected,
                "error": f"当前版本 {detected}，要求 {requirement}。",
            }
    return {"ok": True, "version": detected, "requirement": requirement}


def nvidia_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return ""


def build_report() -> dict[str, object]:
    modules = {
        "torch": module_status("torch", distribution="torch"),
        "ultralytics": module_status("ultralytics", distribution="ultralytics", requirement=">=8.4,<9"),
        "opencv": module_status("cv2", distribution="opencv-contrib-python", requirement=">=4.10,<5"),
        "Pillow": module_status("PIL", distribution="Pillow", requirement=">=10,<13"),
        "onnx": module_status("onnx", distribution="onnx", requirement=">=1.17,<2"),
        "onnxsim": module_status("onnxsim", distribution="onnxsim", requirement=">=0.4.36,<1"),
        "onnxslim": module_status("onnxslim", distribution="onnxslim", requirement=">=0.1.71,<1"),
        "onnxruntime": module_status("onnxruntime", distribution="onnxruntime", requirement=">=1.18,<2"),
        "yaml": module_status("yaml", distribution="PyYAML", requirement=">=6,<7"),
        "psutil": module_status("psutil", distribution="psutil", requirement=">=6,<8"),
        "packaging": module_status("packaging", distribution="packaging", requirement=">=24,<27"),
    }
    cuda_available = False
    cuda_version = ""
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_version = str(torch.version.cuda or "")
    except Exception:
        pass
    missing = [name for name, value in modules.items() if not value["ok"]]
    return {
        "ok": not missing,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "nvidia_gpu": nvidia_name(),
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "training_device": "cuda" if cuda_available else "cpu",
        "modules": modules,
        "missing": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Check whether {PRODUCT_NAME} is ready.")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = build_report()
    if args.write_report:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Python {report['python']}")
    if report["cuda_available"]:
        print(f"训练设备：{report['nvidia_gpu']} (CUDA {report['cuda_version']})")
    else:
        print("训练设备：CPU（未发现可用的 NVIDIA CUDA 显卡）")
    if report["missing"]:
        print("异常组件：" + ", ".join(report["missing"]))
        return 1
    print("系统自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
