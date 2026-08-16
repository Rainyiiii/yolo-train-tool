# MyAutoTrain 团队版

MyAutoTrain 是一个面向 Windows 和 Ubuntu 的 YOLO 训练工具，把数据转换、辅助标注、模型训练、导出和测试放在同一个网页中。队友不需要配置命令行环境，也不需要知道 CUDA 和 Python 的安装细节。

## 队友第一次使用

1. 解压收到的 ZIP 文件。不要直接在压缩包预览窗口里运行。
2. 双击 `一键安装并启动.cmd`。
3. 等待自动安装完成，网页会自动打开。
4. 点击“导入下载的数据集”，选择包含 `data.yaml` 的最外层目录；也可继续手动选择图片与 XML。
5. 选择 `640×480 推荐`，点击“检查并开始训练”。

第一次安装需要下载 Python、PyTorch 和训练组件，所需时间取决于网络速度。安装记录保存在 `logs/install.log`。

以后只需双击：

- `启动训练面板.cmd`：打开平台。
- `关闭训练面板.cmd`：关闭平台，不会删除数据或模型。

网页地址固定为：<http://127.0.0.1:8989/>

## Ubuntu 部署

Ubuntu 22.04/24.04 建议使用 Bash 运行。部署工具支持 Python 3.10–3.14，会自动判断 NVIDIA GPU，安装对应的 CUDA 12.8 或 CPU 版 PyTorch，并安装包括 `onnxruntime` 在内的项目依赖。

```bash
cd MyAutoTrain-Team-日期时间
bash ubuntu_install_and_start.sh
```

如果只安装、不启动网页：

```bash
bash ubuntu_install_and_start.sh --no-start
```

日常启动和停止：

```bash
bash ubuntu_start_train_panel.sh
bash ubuntu_stop_train_panel.sh
```

Ubuntu 上访问：<http://127.0.0.1:8989/>。如果要从其他电脑访问，请自行配置 SSH 端口转发或反向代理，不建议直接把训练面板暴露到公网。

## 发给队友

维护者双击 `制作队友部署包.cmd`，程序会在 `dist` 文件夹生成：

```text
MyAutoTrain-Team-日期时间.zip
```

分享包会自动排除本机虚拟环境、私人路径配置、训练数据、模型权重、日志和训练结果，避免把个人文件发给队友。

## GitHub 文档

- [兼容性说明](docs/COMPATIBILITY.md)：系统、Python、GPU、依赖和已知限制。
- [使用说明](docs/USAGE.md)：Windows/Ubuntu 安装、训练、测试、导出和排错。
- `.gitignore` 已排除虚拟环境、日志、个人配置、训练输出和部署压缩包。

上传 GitHub 前请根据自己的发布方式补充 `LICENSE`，并确认不包含个人数据、训练图片、私有路径、日志或模型权重。

## 推荐操作流程

### 1. 准备数据

推荐直接使用 Roboflow/YOLO 导出的标准目录：

```text
dataset/
├─ data.yaml
├─ train/
│  ├─ images/
│  └─ labels/
├─ valid/
│  ├─ images/
│  └─ labels/
└─ test/
   ├─ images/
   └─ labels/
```

网页会从 `data.yaml` 动态读取任意单类别或多类别名称，检查图片/TXT 是否匹配，并沿用原来的 train/valid/test 划分；不需要 XML，也不会重新随机分组。

旧式数据仍支持同名图片和 VOC XML：

```text
dataset/
├─ images/
│  ├─ 001.jpg
│  └─ 002.jpg
└─ annotations/
   ├─ 001.xml
   └─ 002.xml
```

如果手里是没有 `data.yaml` 的平铺图片与 YOLO TXT，进入“准备数据”，可使用兼容转换功能生成 XML。

### 2. 开始训练

默认稳定配置：

| 项目 | 默认值 |
| --- | --- |
| 模型 | YOLO11n |
| 输入尺寸 | 640×480 |
| Batch | 16 |
| 数据加载进程 | 4 |
| 自动早停 | 关闭 |
| 停止方式 | 用户手动停止 |

点击“更多设置”可以调整模型、尺寸、Batch、缓存、远程训练和导出参数。日常使用不需要修改这些内容。

### 3. 停止训练

点击“停止训练并导出最佳模型”。系统会停止训练并保留当前 `best.pt`，不会擅自替用户提前停止。

### 4. 测试模型

进入“测试模型”，可测试摄像头实时画面、单张图片或整个图片文件夹。

## 硬件与环境

- 有 NVIDIA 显卡：安装器自动安装 CUDA 12.8 PyTorch，并使用 GPU 训练。
- 没有 NVIDIA 显卡：自动安装 CPU 版，功能完整但训练较慢。
- Intel AI Boost NPU：可用于后续 OpenVINO 推理扩展，不能用于 PyTorch YOLO 训练。
- Python：支持 3.10–3.14；电脑没有 Python 时安装器会尝试通过 winget 安装 Python 3.14。

系统自检报告保存在 `logs/system_check.json`。

## 常见问题

### 双击后没有网页

再次双击 `启动训练面板.cmd`，或手动访问 <http://127.0.0.1:8989/>。仍打不开时查看 `logs/launcher.log` 和 `logs/panel.log`。

### 安装中断

检查网络后重新双击 `一键安装并启动.cmd`。安装器会复用已经完成的环境，不需要从头删除。

### 找不到 CUDA

先更新 NVIDIA 驱动，再重新运行安装程序。网页里的“只检查，不启动”会显示当前训练设备。

### 显存不足

打开“更多设置”，把 Batch 从 16 调整为 8 或 4。不要为了提高显存占用强行增大 Batch；出现 NaN 会损害模型质量。

### 数据检查不通过

标准 YOLO 数据集请确认最外层存在 `data.yaml`，每个划分中都有 `images/labels`；旧式 XML 数据请确认图片与 XML 同名并含有效检测框。

## 项目文件

```text
一键安装并启动.cmd       队友第一次使用
启动训练面板.cmd         日常启动
关闭训练面板.cmd         日常关闭
制作队友部署包.cmd       生成分享 ZIP
ubuntu_install_and_start.sh Ubuntu 第一次部署
ubuntu_start_train_panel.sh Ubuntu 日常启动
ubuntu_stop_train_panel.sh  Ubuntu 日常停止
train_panel.py           网页与本地服务
host_train_export.py     数据准备、训练与导出
model_test.py            模型测试
video_track_label.py     辅助标注
```

用户路径保存在 `train_panel_defaults.json`，该文件不会进入团队分享包。
