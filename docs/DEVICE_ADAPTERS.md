# 设备适配与模型导出

YOLO团队训练平台把“训练模型”“转换中间件”“目标设备可运行模型”分开记录。页面会显示最终产物、运行时和连续步骤；厂商编译器未执行时只标记为“等待转换”，不会把中间 ONNX 当作 NPU 模型。

## 当前设备配置档

| 配置档 | 起步建议 | 最终产物 | 当前边界 |
| --- | --- | --- | --- |
| 通用平台 | 固定尺寸 | `.onnx` | Windows/Linux/x86_64/ARM64 通用交接 |
| Raspberry Pi 4 | YOLO n 规格、416×416 | NCNN 模型目录 | ARM64 CPU；待本项目真机基准 |
| Raspberry Pi 5 | YOLO n 规格、640×640 | NCNN 模型目录 | ARM64 CPU；待本项目真机基准 |
| 香橙派 / Rockchip | 按实际 SoC 选择 | `.rknn` | 仅适用于支持的 RK35xx NPU |
| D-Robotics RDK X5 | YOLO n、640×640、INT8 PTQ | Bayes-e `.bin` | Windows 准备包；x86 Linux 编译；X5 运行 |
| Sipeed MaixCAM | 固定输入 ONNX | `.cvimodel` + `.mud` | 保留专用 TPU-MLIR 流程 |
| NVIDIA Jetson | TensorRT | `.engine` | 应在匹配 JetPack/TensorRT 环境生成 |
| Intel | OpenVINO | OpenVINO 模型目录 | Intel CPU/GPU/受支持 NPU |

“香橙派”是板卡品牌，不代表所有型号都使用 Rockchip NPU。只有确认 SoC 属于受支持的 RK35xx 系列时才选择 RKNN；Allwinner 等其他型号应先选择 NCNN 或通用 ONNX。

## 使用方式

网页进入“部署与导出”：

1. 选择训练资产中的 `model-best.pt`；训练结束后通常会自动填入。
2. 选择目标平台，格式保留“自动推荐”即可。
3. 可点击“将设备建议带到训练页”带入下一次训练的模型和尺寸；这不会改变已经训练好的权重。
4. RDK X5 选择 20–50 张代表性校准图片；其他 INT8 路线按要求提供 `data.yaml`。
5. 点击“生成部署模型 / 转换包”。部署清单会区分中间产物、最终产物和当前状态。
6. 在目标设备上复测预处理、类别顺序、置信度、速度、内存和精度。

也可以直接使用命令行：

```bash
python export_model.py \
  --model workspace/training-runs/default-project/default-project__yolo-model__train__20260824-120000/model-best.pt \
  --target raspberry_pi_4 \
  --format auto \
  --imgsz 416 \
  --output-dir deploy/raspberry-pi-4
```

Rockchip 示例：

```bash
python export_model.py \
  --model model-best.pt \
  --target rockchip_rknn \
  --chip rk3588 \
  --output-dir deploy/rk3588
```

## RDK X5 Bayes-e NPU

RDK X5 使用 Sunrise 5 / Bayes-e BPU。X 系列运行模型是 `.bin`，不是 `.hbm`。官方流程使用固定形状 float32 NCHW ONNX、20–50 张代表性图片做 PTQ，运行时输入转换为 NV12，最终由 `hb_mapper` 生成 `*_bayese_*_nv12.bin`。

```bash
python export_model.py \
  --model model-best.pt \
  --target drobotics_rdk_x5 \
  --imgsz 640 \
  --calibration-images /path/to/representative-images \
  --output-dir deploy/rdk-x5
```

输出的 `rdk-x5-npu-bundle` 包含原始 `.pt`、中间 ONNX、最多 50 张校准图片、`classes.txt`、`conversion-plan.json`、转换脚本和校验脚本。有 `.pt` 时，脚本会调用官方 `export_monkey_patch.py` 重新导出兼容 ONNX，不直接假定普通 Ultralytics ONNX 满足板端输出协议。

在 x86_64 Ubuntu 22.04 + Python 3.10 的官方 OpenExplorer 环境中执行：

```bash
pip install rdkx5-yolo-mapper
pip install -r requirements-rdk-x5.txt
hb_mapper --version
cd rdk-x5-npu-bundle
bash convert_rdk_x5.sh
bash verify_rdk_x5.sh output/你的模型_bayese_640x640_nv12.bin
```

工具链和板端 `libdnn` / `hbm_runtime` 应保持匹配。板端建议 RDK OS 3.5.0 或更新版本；发布前还要用留出的测试集复测量化精度。当前 Windows 自动化验证转换包结构和参数，但不替代真实 OpenExplorer 编译与 X5 真机运行。

## 为什么保留 ONNX

ONNX 是本项目的通用中间格式，不等于每块板上的最终最优格式：

- Raspberry Pi CPU 可进一步使用 NCNN。Ultralytics 的树莓派指南列出了 NCNN、ONNX、OpenVINO、MNN、LiteRT 等路线，并将 NCNN 作为 CPU 性能优先方案。
- Rockchip 使用 RKNN Toolkit2 将模型交给 NPU；训练环境与 RKNN 工具链的 Python/驱动版本不应强行混装。
- D-Robotics RDK X5 的 ONNX 只是 PTQ 中间件；`march` 为 `bayes-e`，最终产物必须是可由板端运行时加载的 `.bin`。
- MaixCAM 需要固定输入尺寸，并由 TPU-MLIR 生成 `.cvimodel`，同时提供 `.mud` 描述文件。

参考资料：

- [Ultralytics 模型导出格式](https://docs.ultralytics.com/modes/export/)
- [Ultralytics Raspberry Pi 指南](https://docs.ultralytics.com/guides/raspberry-pi/)
- [Rockchip RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [D-Robotics RDK X5 Model Zoo](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5)
- [D-Robotics Ultralytics YOLO 转换](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5/samples/vision/ultralytics_yolo/conversion)
- [Sipeed MaixCAM 模型转换](https://wiki.sipeed.com/maixpy/doc/en/ai_model_converter/maixcam.html)

## 新增设备适配器

设备配置集中在 `device_profiles.py`。新增配置时至少要明确：

- 稳定的配置 ID、用户可读名称和芯片/运行时边界；
- 推荐导出格式和允许的回退格式；
- 是否依赖厂商工具链、支持的芯片参数和官方文档；
- 固定/动态输入、量化校准、预处理、后处理和类别信息要求；
- 至少一台真机上的版本、延迟、内存、功耗和精度回归结果。

不要只因为某块板“能运行 Linux”就标记为已适配。适配完成应同时满足模型可生成、板端可加载、结果正确、性能可重复。
