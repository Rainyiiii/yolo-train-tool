# MyAutoTrain 使用说明

## 1. 获取项目

从 GitHub 下载 ZIP，或使用 Git 克隆：

```bash
git clone https://github.com/Rainyiiii/yolo-train-tool.git
cd myautotrain-team
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

## 4. 准备数据

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

## 5. 开始训练

1. 打开“准备数据”或“训练”页面。
2. 选择数据集目录。
3. 选择任务类型，例如目标检测。
4. 选择基础模型，默认是 `yolo11n.pt`。
5. 根据显存调整 Batch。
6. 点击“检查并开始训练”。

建议先使用较小模型和较小 Batch 验证数据格式，再增加训练轮数或模型规模。

训练完成后，平台会保留最佳模型和训练结果。具体输出路径以页面提示为准。

## 6. 测试模型

进入“测试模型”页面，可选择：

- 摄像头实时测试
- 单张图片测试
- 图片文件夹批量测试
- 视频或辅助标注流程

测试时可调整置信度阈值 `conf`。阈值较低会得到更多候选框，但误检可能增加。

## 7. 导出与 ONNX

训练完成后可以导出 ONNX 模型，用于跨平台推理或后续设备转换。项目安装器会自动安装：

- `onnx`
- `onnxsim`
- `onnxslim`
- `onnxruntime`

如果使用远程训练环境，远程工具会根据 CPU/CUDA 配置选择 `onnxruntime` 或 `onnxruntime-gpu`。

## 8. 配置文件

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

## 9. 日志和诊断

优先查看：

```text
logs/install.log
logs/install_ubuntu.log
logs/launcher.log
logs/panel.log
logs/system_check.json
```

如果面板无法启动，先运行系统自检，再确认 8989 端口没有被其他服务占用。

## 10. 分享给其他人

Windows 维护者可以双击：

```text
制作队友部署包.cmd
```

生成的 ZIP 会排除本机 `.venv`、日志、个人配置、训练数据和模型输出。分享前请确认 README、兼容性说明和使用说明已经符合你的实际发布方式。
