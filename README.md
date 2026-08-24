# YOLO团队训练平台

[![Latest Release](https://img.shields.io/github/v/release/Rainyiiii/yolo-train-tool?include_prereleases&label=release)](https://github.com/Rainyiiii/yolo-train-tool/releases)

YOLO团队训练平台是一个面向 Windows、Ubuntu 和多种边缘部署平台的本地优先训练工具，把协作标注、数据检查、模型训练、模型资产、多平台导出和测试连接起来。每台电脑都能独立工作，也能临时成为局域网团队主机；部署端通过设备配置档适配树莓派、Rockchip/香橙派、地瓜机器人 RDK、MaixCAM、NVIDIA Jetson、Intel OpenVINO 等运行时。

> 当前仍处于公开发布前的工程化阶段。设备配置档表示“已有导出路线”，不等于所有板卡均已完成真机验证；请查看[设备适配文档](docs/DEVICE_ADAPTERS.md)。

## 下载应用

普通用户请从 [GitHub Releases](https://github.com/Rainyiiii/yolo-train-tool/releases) 下载最新版 Windows 安装器。仓库同时保留桌面应用、Python 训练/标注核心和安装器的完整源码，两种使用方式对应同一套功能和版本。

## 队友第一次使用

1. 双击 `YOLO-Team-Training-Platform-Setup-v3.2.8-beta.exe`。
2. 保持默认安装目录 `D:\YOLOTeamTrainingPlatform`，点击安装。
3. 安装器自动准备 .NET、WebView2、Python、PyTorch、ONNX Runtime 和平台依赖。
4. 安装完成后从桌面打开“YOLO团队训练平台”。
5. 点击“导入下载的数据集”，选择包含 `data.yaml` 的最外层目录；也可继续手动选择图片与 XML。
6. 选择 `640×480 推荐`，点击“检查并开始训练”。

平台新建数据集时默认按照训练 80% / 验证 10% / 测试 10% 划分。训练集用于学习，验证集用于选择最佳模型，测试集只在训练完成后对 `best.pt` 做独立评估；导入已有 YOLO 数据集时保留其原始 `train / valid / test`。

第一次安装需要联网下载训练组件，所需时间取决于网络速度。以后直接运行新版安装包会默认执行增量更新：保留健康的 Python、PyTorch、ONNX Runtime 和其他依赖，仅更新程序或补齐变化的组件。只有在安装向导中主动勾选“完整修复运行环境”才会删除并重建 Runtime；两种模式都不会改动 Workspace。安装器会实时显示并自动滚动安装日志，完整记录同时保存在 `D:\YOLOTeamTrainingPlatform\Workspace\logs\installation.log`。

开发源码模式仍保留以下脚本：

- `启动训练面板.cmd`：打开平台。
- `关闭训练面板.cmd`：关闭平台，不会删除数据或模型。
- `启动个人标注中心.cmd`：本机离线标注。
- `开启局域网协作标注.cmd`：让伙伴通过浏览器协作。
- `关闭协作标注中心.cmd`：停止标注服务，不删除数据。

网页模式默认使用 <http://127.0.0.1:8989/>。Windows 桌面版检测到已有 YOLO 训练面板时，会询问是否关闭旧服务；选择“否”会保留旧服务并退出，绝不会同时启动第二个训练面板。若默认端口仅被其他软件占用，桌面版会使用其他仅限本机访问的端口。实际地址记录在 `Workspace/logs/launcher.log`。

## Ubuntu 部署

Ubuntu 22.04/24.04 建议使用 Bash 运行。部署工具支持 Python 3.10–3.14，会自动判断 NVIDIA GPU，安装对应的 CUDA 12.8 或 CPU 版 PyTorch，并安装包括 `onnxruntime` 在内的项目依赖。

```bash
cd yolo-train-tool
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

维护者运行 `installer/windows/build-installer.ps1`，程序会在 `dist` 文件夹生成：

```text
YOLO-Team-Training-Platform-Setup-v3.2.8-beta.exe
```

安装器不会打包本机 Workspace、私人路径、训练数据、日志和训练结果。

## 应用源码与核心源码

本项目采用单仓库共存方式：

| 目录/文件 | 内容 |
| --- | --- |
| `desktop/YOLOTeamTrainingPlatform.Desktop/` | WebView2 Windows 应用源码 |
| `installer/windows/` | 一键安装器及自动构建源码 |
| `train_panel.py`、`annotation_*.py` | 训练平台与协作标注核心 |
| `host_train_export.py`、`export_model.py` | 训练和多设备模型导出 |
| `tests/` | 自动测试 |

WebView2 只是桌面承载层，不复制训练逻辑；桌面应用与浏览器模式共享同一个本地服务。详细说明见[开发与仓库结构](docs/DEVELOPMENT.md)。

## GitHub 文档

- [兼容性说明](docs/COMPATIBILITY.md)：系统、Python、GPU、依赖和已知限制。
- [目录与命名规范](docs/DIRECTORY_AND_NAMING_STANDARD.md)：安装布局、工作区和所有新资产命名。
- [Windows WebView2 安装与发布](docs/WINDOWS_INSTALLER.md)：一键安装、运行时依赖和维护者构建。
- [开发与仓库结构](docs/DEVELOPMENT.md)：应用源码、Python 核心和 GitHub 自动发布方式。
- [使用说明](docs/USAGE.md)：Windows/Ubuntu 安装、训练、测试、导出和排错。
- [设备适配与模型导出](docs/DEVICE_ADAPTERS.md)：树莓派、Rockchip、RDK、MaixCAM、Jetson、OpenVINO。
- [数据集与模型资产](docs/MODEL_ASSETS.md)：规范训练清单和测试/部署快捷操作。
- [项目中心](docs/PROJECT_CENTER.md)：项目、数据集检查、训练与模型关联。
- [本地优先协作标注](docs/COLLABORATIVE_ANNOTATION.md)：个人/共享模式、账号任务、审核、项目包和数据集导出。
- [半自动标注说明](docs/SEMI_AUTO_LABELING.md)：质量检查、复核流程和已知边界。
- [优化路线图](docs/ROADMAP.md)：公开发布前、设备适配、数据与工程化计划。
- [参与贡献](CONTRIBUTING.md) 与 [安全说明](SECURITY.md)：设备验证要求、代码来源和本地服务边界。
- `.gitignore` 已排除虚拟环境、日志、个人配置、训练输出和部署压缩包。

历史工程的准确来源与授权目前无法确认，因此仓库不提供猜测性来源链接，也不宣称未知部分为原创。当前采用保留权利的 [LICENSE](LICENSE)，详细边界与第三方组件见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。发布前仍需确认仓库不包含个人数据、训练图片、私有路径、日志或模型权重。

## 推荐操作流程

### 1. 准备数据

不再强制依赖 Roboflow。“项目中心”的目标检测项目会自动同步到个人/团队标注中心；选择类别、拖动画框，再点击“完成并下一张”即可。团队成员可以直接从公共队列自动领取下一张，不必预先分配；审核默认关闭，需要双人复核时再按项目开启。已完成数据可导出 Ultralytics YOLO、COCO、VOC 或 LabelMe。详见[协作标注文档](docs/COLLABORATIVE_ANNOTATION.md)。

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

点击“停止训练并导出最佳模型”。系统会停止训练并把当前最佳权重归档为 `model-best.pt`，不会擅自替用户提前停止。

### 4. 测试模型

进入“测试模型”，可测试摄像头实时画面、单张图片或整个图片文件夹。

### 5. 部署到目标设备

进入“部署与导出”，选择训练资产中的 `model-best.pt` 和目标平台：

| 目标 | 默认路线 |
| --- | --- |
| 通用 Linux / Windows | ONNX |
| 树莓派 CPU | NCNN |
| Rockchip RK35xx / 对应香橙派 | RKNN |
| 地瓜机器人 RDK X3/X5 | ONNX → 厂商 PTQ |
| MaixCAM | ONNX → `.cvimodel` + `.mud` |
| NVIDIA Jetson | TensorRT engine |
| Intel CPU/GPU/NPU | OpenVINO |

导出目录会生成模型产物和 `*.manifest.json` 部署清单。任何 INT8 模型都要使用代表性校准数据，并在目标设备上重新验证精度。

## 硬件与环境

- 有 NVIDIA 显卡：安装器自动安装 CUDA 12.8 PyTorch，并使用 GPU 训练。
- 没有 NVIDIA 显卡：自动安装 CPU 版，功能完整但训练较慢。
- Intel AI Boost NPU：可用于后续 OpenVINO 推理扩展，不能用于 PyTorch YOLO 训练。
- Python：支持 3.10–3.14；电脑没有 Python 时安装器会尝试通过 winget 安装 Python 3.14。
- 依赖包含 `onnxruntime`，用于 ONNX 模型验证；厂商转换工具链建议使用独立环境或容器。

系统自检报告保存在 `workspace/logs/system-check.json`。

## 常见问题

### 双击后没有网页

再次双击 `启动训练面板.cmd`，或手动访问 <http://127.0.0.1:8989/>。仍打不开时查看 `workspace/logs/launcher.log` 和 `workspace/logs/panel.log`。

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
启动个人标注中心.cmd      Windows 本机离线标注
开启局域网协作标注.cmd    Windows 团队共享标注
关闭协作标注中心.cmd      Windows 停止标注服务
ubuntu_start_annotation.sh Ubuntu 个人/共享标注
ubuntu_stop_annotation.sh  Ubuntu 停止标注服务
train_panel.py           网页与本地服务
annotation_server.py     原生协作标注服务端
annotation_store.py      账号、项目、任务、锁和审核数据库
annotation_exports.py    项目包与多格式数据集导出
host_train_export.py     数据准备、训练与导出
model_test.py            模型测试
video_track_label.py     辅助标注
device_profiles.py       设备配置档
export_model.py          多平台模型导出
```

用户设置保存在 `Workspace/config/settings.json`，该文件不会进入安装器或 Git 仓库。
