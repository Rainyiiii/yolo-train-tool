# -*- coding: utf-8 -*-
"""Deployment target profiles used by the CLI and the web panel.

Keep vendor-specific conversion details here instead of coupling them to the
training workflow. ONNX is the common hand-off format when conversion must run
inside a vendor toolchain.
"""

from __future__ import annotations

from typing import Any


DEVICE_PROFILES: dict[str, dict[str, Any]] = {
    "generic_onnx": {
        "label": "通用平台 / ONNX Runtime",
        "family": "generic",
        "recommended_format": "onnx",
        "formats": ["onnx"],
        "opset": 17,
        "chips": [],
        "vendor_toolchain": False,
        "summary": "兼容 Windows、Linux 和多数支持 ONNX Runtime 的平台。",
        "next_step": "在目标设备安装对应架构的 ONNX Runtime，并先用少量样本核对预处理和类别顺序。",
        "docs_url": "https://onnxruntime.ai/docs/install/",
    },
    "raspberry_pi": {
        "label": "树莓派（CPU）",
        "family": "raspberry-pi",
        "recommended_format": "ncnn",
        "formats": ["ncnn", "onnx"],
        "opset": 17,
        "chips": [],
        "vendor_toolchain": False,
        "summary": "CPU 部署优先尝试 NCNN，也保留 ONNX 作为通用回退。",
        "next_step": "把导出目录复制到树莓派，使用 Ultralytics/NCNN 运行时验证速度和精度。",
        "docs_url": "https://docs.ultralytics.com/guides/raspberry-pi/",
    },
    "rockchip_rknn": {
        "label": "香橙派 / Rockchip RKNN",
        "family": "rockchip",
        "recommended_format": "rknn",
        "formats": ["rknn", "onnx"],
        "opset": 17,
        "chips": ["rk3588", "rk3576", "rk3566", "rk3568"],
        "default_chip": "rk3588",
        "vendor_toolchain": True,
        "summary": "适用于采用 RK35xx NPU 的香橙派等开发板；具体支持范围以板卡 SoC 为准。",
        "next_step": "在 Rockchip 官方支持的环境中验证 RKNN，并在板端使用匹配版本的 RKNN Runtime。",
        "docs_url": "https://github.com/airockchip/rknn-toolkit2",
    },
    "drobotics_rdk": {
        "label": "地瓜机器人 RDK X3 / X5",
        "family": "d-robotics",
        "recommended_format": "onnx",
        "formats": ["onnx"],
        "opset": 17,
        "chips": ["x5", "x3"],
        "default_chip": "x5",
        "vendor_toolchain": True,
        "summary": "先生成 ONNX，再交给 D-Robotics PTQ 工具链编译为板端模型。",
        "next_step": "按目标板选择 march：RDK X5 使用 bayes-e，RDK X3 使用 bernoulli2，并准备 PTQ 校准数据。",
        "docs_url": "https://developer.d-robotics.cc/rdk_x_doc/en/Advanced_development/toolchain_development/intermediate/ptq_process",
    },
    "maixcam": {
        "label": "Sipeed MaixCAM",
        "family": "maixcam",
        "recommended_format": "onnx",
        "formats": ["onnx"],
        "opset": 11,
        "chips": ["cv181x"],
        "default_chip": "cv181x",
        "vendor_toolchain": True,
        "summary": "生成固定输入尺寸 ONNX，随后使用 TPU-MLIR 转换为 .cvimodel 并配套 .mud。",
        "next_step": "需要完整 MaixCAM 包时，继续使用本页下方的 MaixCAM 专用转换；部署前核对 .mud 的输入尺寸和类别。",
        "docs_url": "https://wiki.sipeed.com/maixpy/doc/en/ai_model_converter/maixcam.html",
    },
    "nvidia_jetson": {
        "label": "NVIDIA Jetson / TensorRT",
        "family": "nvidia",
        "recommended_format": "engine",
        "formats": ["engine", "onnx"],
        "opset": 17,
        "chips": [],
        "vendor_toolchain": True,
        "summary": "优先 TensorRT engine；engine 应在与目标 JetPack/TensorRT 匹配的设备或容器中生成。",
        "next_step": "在目标 Jetson 或匹配的 NVIDIA 容器中测试 engine；跨机器复制前确认 TensorRT 与 CUDA 版本一致。",
        "docs_url": "https://docs.ultralytics.com/integrations/tensorrt/",
    },
    "intel_openvino": {
        "label": "Intel CPU / GPU / NPU（OpenVINO）",
        "family": "intel",
        "recommended_format": "openvino",
        "formats": ["openvino", "onnx"],
        "opset": 17,
        "chips": [],
        "vendor_toolchain": False,
        "summary": "面向 Intel CPU、核显和受支持的 NPU，输出 OpenVINO 模型目录。",
        "next_step": "在目标机安装 OpenVINO Runtime，核对 AUTO/CPU/GPU/NPU 设备选择与精度。",
        "docs_url": "https://docs.ultralytics.com/integrations/openvino/",
    },
}


EXPORT_FORMATS = ("auto", "onnx", "ncnn", "openvino", "engine", "rknn")


def get_device_profile(profile_id: str) -> dict[str, Any]:
    try:
        profile = DEVICE_PROFILES[profile_id]
    except KeyError as exc:
        choices = ", ".join(DEVICE_PROFILES)
        raise ValueError(f"未知部署平台 {profile_id!r}；可选值：{choices}") from exc
    return {"id": profile_id, **profile}


def resolve_export_format(profile_id: str, requested: str = "auto") -> str:
    profile = get_device_profile(profile_id)
    export_format = requested.strip().lower() or "auto"
    if export_format == "auto":
        return str(profile["recommended_format"])
    if export_format not in profile["formats"]:
        supported = "、".join(str(item) for item in profile["formats"])
        raise ValueError(f"{profile['label']} 不支持 {export_format}；可选：{supported}")
    return export_format


def public_device_profiles() -> list[dict[str, Any]]:
    return [get_device_profile(profile_id) for profile_id in DEVICE_PROFILES]
