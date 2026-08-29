# -*- coding: utf-8 -*-
"""Build a portable RDK X5 NPU conversion hand-off bundle.

The Windows application can prepare and validate the source model and
calibration images.  The final Bayes-e ``.bin`` must be compiled in the
D-Robotics x86 Linux OpenExplorer environment, so this module makes that
boundary explicit instead of presenting the intermediate ONNX as deployable.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OFFICIAL_MODEL_ZOO = "https://github.com/D-Robotics/rdk_model_zoo.git"
OFFICIAL_MODEL_ZOO_BRANCH = "rdk_x5"


def calibration_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: str(path).lower(),
    )


def _copy_calibration_images(paths: list[Path], destination: Path, limit: int = 50) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for index, source in enumerate(paths[:limit], start=1):
        target = destination / f"calibration_{index:03d}{source.suffix.lower()}"
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def _conversion_script(pt_name: str, onnx_name: str) -> str:
    source_selection = (
        f'''python3 -c "import ultralytics, torch, onnx" >/dev/null 2>&1 || {{ echo "缺少 ONNX 导出依赖；请执行 pip install -r requirements-rdk-x5.txt" >&2; exit 1; }}
PT_PATH="$MODEL_DIR/{pt_name}"
python3 "$CONVERSION_DIR/export_monkey_patch.py" --pt "$PT_PATH"
ONNX_PATH="${{PT_PATH%.pt}}.onnx"
test -f "$ONNX_PATH" || {{ echo "未找到官方导出的 ONNX：$ONNX_PATH" >&2; exit 1; }}'''
        if pt_name
        else f'''ONNX_PATH="$MODEL_DIR/{onnx_name}"
echo "注意：正在使用已有 ONNX。它必须遵循 D-Robotics Ultralytics YOLO 输出协议。"'''
    )
    return f'''#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
MODEL_DIR="$ROOT/model"
CAL_DIR="$ROOT/calibration_images"
OUTPUT_DIR="$ROOT/output"
WORK_DIR="$ROOT/.work"
MODEL_ZOO_DIR="${{RDK_MODEL_ZOO_DIR:-$ROOT/.rdk_model_zoo}}"

command -v python3 >/dev/null || {{ echo "缺少 python3" >&2; exit 1; }}
command -v git >/dev/null || {{ echo "缺少 git" >&2; exit 1; }}
command -v hb_mapper >/dev/null || {{ echo "缺少 hb_mapper；请先进入 RDK X5 OpenExplorer 环境" >&2; exit 1; }}

IMAGE_COUNT="$(find "$CAL_DIR" -maxdepth 1 -type f | wc -l | tr -d ' ')"
if [ "$IMAGE_COUNT" -lt 20 ]; then
  echo "校准图片只有 $IMAGE_COUNT 张；官方建议 20–50 张" >&2
  exit 1
fi

if [ ! -f "$MODEL_ZOO_DIR/samples/vision/ultralytics_yolo/conversion/mapper.py" ]; then
  if [ -n "${{RDK_MODEL_ZOO_DIR:-}}" ]; then
    echo "RDK_MODEL_ZOO_DIR 不包含 rdk_x5 转换工具：$MODEL_ZOO_DIR" >&2
    exit 1
  fi
  if [ -e "$MODEL_ZOO_DIR" ]; then
    echo "转换包中的 .rdk_model_zoo 不完整；请手动移走后重试：$MODEL_ZOO_DIR" >&2
    exit 1
  fi
  git clone --depth 1 --branch {OFFICIAL_MODEL_ZOO_BRANCH} {OFFICIAL_MODEL_ZOO} "$MODEL_ZOO_DIR"
fi

CONVERSION_DIR="$MODEL_ZOO_DIR/samples/vision/ultralytics_yolo/conversion"
mkdir -p "$OUTPUT_DIR"
{source_selection}

python3 "$CONVERSION_DIR/mapper.py" \
  --onnx "$ONNX_PATH" \
  --cal-images "$CAL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --jobs "${{RDK_JOBS:-4}}" \
  --ws "$WORK_DIR"

BIN_PATH="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*_bayese_*_nv12.bin' -print -quit)"
test -n "$BIN_PATH" || {{ echo "转换结束但未找到 RDK X5 .bin" >&2; exit 1; }}
if command -v hb_model_info >/dev/null; then hb_model_info "$BIN_PATH"; fi
echo "RDK_X5_BIN=$BIN_PATH"
echo "转换完成。把 output 目录和 classes.txt 复制到 RDK X5。"
'''


def _verify_script() -> str:
    return '''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-$(find "$ROOT/output" -maxdepth 1 -type f -name '*.bin' -print -quit)}"
test -n "$MODEL" && test -f "$MODEL" || { echo "未找到 .bin；可传入模型路径" >&2; exit 1; }
command -v hrt_model_exec >/dev/null || { echo "缺少 hrt_model_exec，请在 RDK X5 或匹配工具链中运行" >&2; exit 1; }
hrt_model_exec model_info --model_file "$MODEL"
hrt_model_exec perf --model_file "$MODEL" --thread_num "${RDK_THREADS:-1}"
echo "验证完成：$MODEL"
'''


def create_rdk_x5_bundle(
    output_dir: Path,
    source_model: Path,
    onnx_artifact: Path,
    calibration_dir: Path,
    input_size: int | tuple[int, int],
    class_names: dict[str, str],
) -> dict[str, Any]:
    images = calibration_images(calibration_dir)
    if not images:
        raise ValueError("RDK X5 NPU 转换需要代表性校准图片目录。")

    bundle = output_dir / "rdk-x5-npu-bundle"
    model_dir = bundle / "model"
    calibration_target = bundle / "calibration_images"
    output_target = bundle / "output"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_target.mkdir(parents=True, exist_ok=True)

    pt_name = ""
    if source_model.suffix.lower() == ".pt":
        pt_name = "trained-model.pt"
        shutil.copy2(source_model, model_dir / pt_name)
    onnx_name = "intermediate-model.onnx"
    shutil.copy2(onnx_artifact, model_dir / onnx_name)
    copied = _copy_calibration_images(images, calibration_target)

    def class_order(item: tuple[str, str]) -> tuple[int, int | str]:
        key = str(item[0])
        return (0, int(key)) if key.isdigit() else (1, key)

    classes = [name for _, name in sorted(class_names.items(), key=class_order)]
    (bundle / "classes.txt").write_text("\n".join(classes) + ("\n" if classes else ""), encoding="utf-8")
    (bundle / "requirements-rdk-x5.txt").write_text(
        "ultralytics>=8.4,<9\nonnx>=1.17,<2\nonnxruntime>=1.18,<2\nopencv-python>=4.10,<5\nnumpy>=1.24\n",
        encoding="utf-8",
        newline="\n",
    )
    convert_script = bundle / "convert_rdk_x5.sh"
    verify_script = bundle / "verify_rdk_x5.sh"
    convert_script.write_text(_conversion_script(pt_name, onnx_name), encoding="utf-8", newline="\n")
    verify_script.write_text(_verify_script(), encoding="utf-8", newline="\n")

    height, width = (input_size, input_size) if isinstance(input_size, int) else input_size
    plan = {
        "schema_version": 1,
        "target": "drobotics_rdk_x5",
        "soc": "Sunrise 5",
        "bpu_arch": "Bayes-e",
        "march": "bayes-e",
        "intermediate_format": "onnx",
        "final_format": "bin",
        "runtime_input": "nv12",
        "training_input": "rgb_nchw_float32",
        "input_size": [height, width],
        "quantization": "int8_ptq",
        "calibration_images": len(copied),
        "expected_output_pattern": "output/*_bayese_*_nv12.bin",
        "compile_host": "x86_64 Ubuntu 22.04 / D-Robotics OpenExplorer",
        "board_runtime": "RDK OS >= 3.5.0 / matching libdnn and hbm_runtime",
        "official_model_zoo": f"{OFFICIAL_MODEL_ZOO}#{OFFICIAL_MODEL_ZOO_BRANCH}",
    }
    (bundle / "conversion-plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (bundle / "README.md").write_text(
        f"""# RDK X5 NPU 转换包

此目录的最终目标是 RDK X5 Bayes-e BPU 可运行的 `.bin`，当前 ONNX 只是中间文件。

## 已准备

- 原始权重：{'`model/trained-model.pt`（转换时会使用官方 monkey patch 重新导出）' if pt_name else '`model/intermediate-model.onnx`（请确认符合官方输出协议）'}
- 校准图片：{len(copied)} 张（官方建议 20–50 张）
- 输入尺寸：{width}×{height}，固定形状
- 运行时输入：NV12；训练侧输入：RGB / NCHW / float32

## 1. 在 x86 Linux 编译

推荐 Ubuntu 22.04 + Python 3.10。进入官方 RDK X5 OpenExplorer 环境，或安装轻量工具链：

```bash
pip install rdkx5-yolo-mapper
pip install -r requirements-rdk-x5.txt
hb_mapper --version
cd /path/to/rdk-x5-npu-bundle
bash convert_rdk_x5.sh
```

脚本会固定拉取官方 `rdk_model_zoo` 的 `rdk_x5` 分支，调用其 Ultralytics YOLO 导出/Mapper 流程，最终文件位于 `output/*_bayese_*_nv12.bin`。

## 2. 校验产物

在 OpenExplorer 或 RDK X5 上执行：

```bash
bash verify_rdk_x5.sh output/你的模型.bin
```

检查通过后，再把 `output` 和 `classes.txt` 复制到板卡。工具链与板端 `libdnn` 应保持匹配；实际上线前必须用留出的测试集复测精度，并测量完整摄像头前处理、推理和后处理链路。

官方参考：<https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5/samples/vision/ultralytics_yolo>
""",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "bundle": bundle,
        "calibration_image_count": len(copied),
        "calibration_source_count": len(images),
        "conversion_script": convert_script,
        "verification_script": verify_script,
        "expected_final_artifact": str(output_target / "*_bayese_*_nv12.bin"),
        "status": "conversion_required",
        "plan": plan,
    }
