from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "logs" / "system_check.json"


def module_status(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
        return {"ok": True, "version": getattr(module, "__version__", "installed")}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


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
        "torch": module_status("torch"),
        "ultralytics": module_status("ultralytics"),
        "opencv": module_status("cv2"),
        "Pillow": module_status("PIL"),
        "onnx": module_status("onnx"),
        "onnxsim": module_status("onnxsim"),
        "onnxslim": module_status("onnxslim"),
        "onnxruntime": module_status("onnxruntime"),
        "yaml": module_status("yaml"),
        "psutil": module_status("psutil"),
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
    parser = argparse.ArgumentParser(description="Check whether MyAutoTrain is ready.")
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
        print("缺少组件：" + ", ".join(report["missing"]))
        return 1
    print("系统自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
