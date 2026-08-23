# MyAutoTrain 使用说明

## 1. 获取项目

从 GitHub 下载 ZIP，或使用 Git 克隆：

```bash
git clone https://github.com/Rainyiiii/yolo-train-tool.git
cd yolo-train-tool
```

不要直接在压缩包预览窗口中运行脚本，先完整解压到一个有写入权限的目录。

## 2. Windows 首次安装

双击：

```text
一键安装并启动.cmd
```

安装器会自动：

1. 检查 Python 3.10–3.14。
2. 创建项目专用 `.venv`。
3. 检测 NVIDIA GPU 并选择 CUDA 或 CPU PyTorch。
4. 安装训练、ONNX 和推理依赖。
5. 下载默认的 `yolo11n.pt`。
6. 运行系统自检并启动网页面板。

以后使用：

- `启动训练面板.cmd`：启动面板。
- `关闭训练面板.cmd`：停止面板，不会删除数据和模型。

## 3. Ubuntu 首次安装

在项目目录执行：

```bash
bash ubuntu_install_and_start.sh
```

只安装依赖、不启动面板：

```bash
bash ubuntu_install_and_start.sh --no-start
```

以后使用：

```bash
bash ubuntu_start_train_panel.sh
bash ubuntu_stop_train_panel.sh
```

打开浏览器访问：<http://127.0.0.1:8989/>。

## 4. 本地或局域网协作标注

训练面板左侧进入“协作标注”，或直接运行：

- Windows 个人使用：`启动个人标注中心.cmd`
- Windows 局域网共享：`开启局域网协作标注.cmd`
- Ubuntu 个人使用：`bash ubuntu_start_annotation.sh`
- Ubuntu 局域网共享：`bash ubuntu_start_annotation.sh --share`

个人模式只能从本机打开 <http://127.0.0.1:9000/>。共享模式会显示局域网地址，伙伴只需要浏览器，不需要安装 MyAutoTrain、Python 或 Docker。每台电脑都能保留自己的 `annotation_workspace`；需要交换独立成果时导出、导入 `.matproj.zip` 项目包。实时多人编辑则连接同一台共享主机，由编辑锁避免互相覆盖。

项目支持成员账号、按数量分配图片、提交与审核；审核通过后可导出 Ultralytics YOLO、COCO、Pascal VOC 和 LabelMe。完整流程见[本地优先协作标注](COLLABORATIVE_ANNOTATION.md)。

## 5. 准备数据

推荐使用标准 YOLO 数据集：

```text
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

`data.yaml` 中应包含训练、验证路径和类别名称。进入网页后点击“导入下载的数据集”，选择包含 `data.yaml` 的最外层目录。

旧式 VOC 数据也可以使用：

```text
dataset/
├── images/
└── annotations/
```

其中图片和 XML 标注应同名，并且 XML 中包含有效检测框。

## 6. 开始训练

1. 打开“准备数据”或“训练”页面。
2. 选择数据集目录。
3. 选择任务类型，例如目标检测。
4. 选择基础模型，默认是 `yolo11n.pt`。
5. 根据显存调整 Batch。
6. 点击“检查并开始训练”。

建议先使用较小模型和较小 Batch 验证数据格式，再增加训练轮数或模型规模。

训练完成后，平台会保留最佳模型和训练结果。具体输出路径以页面提示为准。

每次新训练还会生成 `training_manifest.json`，记录数据集、类别、训练参数和模型文件之间的对应关系。进入独立的“模型资产”页面即可按数据集查看所有训练模型；旧版输出可选择原数据集目录后扫描。详见[数据集与模型资产](MODEL_ASSETS.md)。

## 7. 测试模型

进入“测试模型”页面，可选择：

- 摄像头实时测试
- 单张图片测试
- 图片文件夹批量测试
- 视频或辅助标注流程

测试时可调整置信度阈值 `conf`。阈值较低会得到更多候选框，但误检可能增加。

## 8. 多平台导出

训练完成后进入“部署与导出”，选择 `best.pt`、目标平台和输出目录。平台会生成目标模型以及 `*.manifest.json` 部署清单。当前内置通用 ONNX、树莓派 NCNN、Rockchip RKNN、地瓜 RDK ONNX 交接、MaixCAM、TensorRT 和 OpenVINO 配置档，详见[设备适配文档](DEVICE_ADAPTERS.md)。

ONNX 仍是厂商工具链之间的主要交接格式。项目安装器会自动安装：

- `onnx`
- `onnxsim`
- `onnxslim`
- `onnxruntime`

如果使用远程训练环境，远程工具会根据 CPU/CUDA 配置选择 `onnxruntime` 或 `onnxruntime-gpu`。

INT8 导出需要填写 `data.yaml` 并使用有代表性的校准数据。导出成功不代表板端适配已经完成；部署前必须在目标设备核对预处理、类别顺序、输出张量、速度和精度。

## 9. 半自动标注

网页标注会在跟踪失败、位置/尺寸突变或模板匹配质量较低时暂停。只要当前帧有一个目标需要复核，整帧都不会自动保存，以避免多目标漏标。推荐逐段播放、频繁抽查，具体见[半自动标注说明](SEMI_AUTO_LABELING.md)。

## 10. 配置文件

首次安装会根据设备生成：

```text
train_panel_defaults.json
```

这是本机配置，不建议上传到 GitHub。发布包使用 `train_panel_defaults.example.json` 作为模板。

常用设置包括：

- 数据集路径
- 基础模型
- 训练设备
- Batch
- 数据加载进程
- 输入尺寸
- 训练轮数
- 远程训练主机

## 11. 日志和诊断

优先查看：

```text
logs/install.log
logs/install_ubuntu.log
logs/launcher.log
logs/panel.log
logs/system_check.json
```

如果面板无法启动，先运行系统自检，再确认 8989 端口没有被其他服务占用。

## 12. 分享给其他人

Windows 维护者可以双击：

```text
制作队友部署包.cmd
```

生成的 ZIP 会排除本机 `.venv`、日志、个人配置、训练数据和模型输出。分享前请确认 README、兼容性说明和使用说明已经符合你的实际发布方式。
