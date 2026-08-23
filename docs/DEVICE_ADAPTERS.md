# 设备适配与模型导出

MyAutoTrain 的训练流程与设备转换流程相互独立。训练统一保留 `best.pt`，部署时再根据目标运行时导出；遇到必须使用厂商编译器的平台，则以固定尺寸 ONNX 作为交接格式。

## 当前设备配置档

| 配置档 | 推荐产物 | 适用范围 | 当前状态 |
| --- | --- | --- | --- |
| 通用平台 / ONNX Runtime | ONNX | Windows、Linux、x86_64、ARM64 | 已提供统一导出与部署清单 |
| 树莓派 CPU | NCNN；ONNX 回退 | Raspberry Pi 4/5 等 CPU 推理 | 已提供导出入口，待建立真机基准 |
| 香橙派 / Rockchip | RKNN；ONNX 交接 | 采用 RK3588、RK3576、RK3566、RK3568 的板卡 | 已提供 RKNN 导出入口，待逐板验证 |
| 地瓜机器人 RDK | ONNX | RDK X5、RDK X3 | 已提供 ONNX 与后续步骤清单，厂商 PTQ 尚未集成 |
| Sipeed MaixCAM | ONNX → `.cvimodel` + `.mud` | CV181x | 保留原专用转换流程 |
| NVIDIA Jetson | TensorRT engine；ONNX 回退 | Jetson Orin 等 | 已提供导出入口；应在匹配目标环境中生成 engine |
| Intel | OpenVINO | Intel CPU、GPU、受支持的 NPU | 已提供导出入口，待设备矩阵验证 |

“香橙派”是板卡品牌，不代表所有型号都使用 Rockchip NPU。只有确认 SoC 属于受支持的 RK35xx 系列时才选择 RKNN；Allwinner 等其他型号应先选择 NCNN 或通用 ONNX。

## 使用方式

网页进入“部署与导出”：

1. 选择训练得到的 `best.pt`；训练结束后通常会自动填入。
2. 选择目标平台，格式保留“自动推荐”即可。
3. RKNN 等平台按需填写芯片名；INT8 必须提供 `data.yaml` 和有代表性的校准图片。
4. 点击“生成部署模型”。输出目录中会同时生成模型和 `*.manifest.json` 部署清单。
5. 在目标设备上复测预处理、类别顺序、置信度、速度、内存和精度。

也可以直接使用命令行：

```bash
python export_model.py \
  --model runs/my_yolo_project/my_yolo_model/weights/best.pt \
  --target raspberry_pi \
  --format auto \
  --imgsz 480,640 \
  --output-dir deploy/raspberry_pi
```

Rockchip 示例：

```bash
python export_model.py \
  --model best.pt \
  --target rockchip_rknn \
  --chip rk3588 \
  --output-dir deploy/rk3588
```

## 为什么保留 ONNX

ONNX 是本项目的通用中间格式，不等于每块板上的最终最优格式：

- Raspberry Pi CPU 可进一步使用 NCNN。Ultralytics 的树莓派指南列出了 NCNN、ONNX、OpenVINO、MNN、LiteRT 等路线，并将 NCNN 作为 CPU 性能优先方案。
- Rockchip 使用 RKNN Toolkit2 将模型交给 NPU；训练环境与 RKNN 工具链的 Python/驱动版本不应强行混装。
- D-Robotics RDK 的 PTQ 流程接收 ONNX 与 YAML 配置；X5 和 X3 的 `march` 不同。
- MaixCAM 需要固定输入尺寸，并由 TPU-MLIR 生成 `.cvimodel`，同时提供 `.mud` 描述文件。

参考资料：

- [Ultralytics 模型导出格式](https://docs.ultralytics.com/modes/export/)
- [Ultralytics Raspberry Pi 指南](https://docs.ultralytics.com/guides/raspberry-pi/)
- [Rockchip RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [D-Robotics RDK PTQ 流程](https://developer.d-robotics.cc/rdk_x_doc/en/Advanced_development/toolchain_development/intermediate/ptq_process)
- [Sipeed MaixCAM 模型转换](https://wiki.sipeed.com/maixpy/doc/en/ai_model_converter/maixcam.html)

## 新增设备适配器

设备配置集中在 `device_profiles.py`。新增配置时至少要明确：

- 稳定的配置 ID、用户可读名称和芯片/运行时边界；
- 推荐导出格式和允许的回退格式；
- 是否依赖厂商工具链、支持的芯片参数和官方文档；
- 固定/动态输入、量化校准、预处理、后处理和类别信息要求；
- 至少一台真机上的版本、延迟、内存、功耗和精度回归结果。

不要只因为某块板“能运行 Linux”就标记为已适配。适配完成应同时满足模型可生成、板端可加载、结果正确、性能可重复。
