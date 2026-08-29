# YOLO团队训练平台使用说明

## 1. 获取项目

从 GitHub 下载 ZIP，或使用 Git 克隆：

```bash
git clone https://github.com/Rainyiiii/yolo-train-tool.git
cd yolo-train-tool
```

不要直接在压缩包预览窗口中运行脚本，先完整解压到一个有写入权限的目录。

## 2. Windows 首次安装

普通队友直接双击发布页中的安装器：

```text
YOLO-Team-Training-Platform-Setup-v3.2.15-beta.exe
```

安装器会自动：

1. 默认安装到 `D:\YOLOTeamTrainingPlatform`；
2. 静默安装 .NET 8 Desktop Runtime 和 WebView2 Runtime；
3. 检查或安装 Python 3.10–3.14；
4. 创建隔离 Python 环境；
5. 检测 NVIDIA GPU 并选择 CUDA 或 CPU PyTorch；
6. 安装 ONNX、`onnxruntime` 和训练依赖；
7. 创建 Workspace 并启动 WebView2 桌面程序。

以后直接运行新版安装包即可覆盖升级。默认模式会保留并检查现有 Runtime，依赖没有变化时不联网执行 pip 下载；需要新增依赖时只补装缺失或不兼容项。如果运行环境确实损坏，可在安装向导中主动勾选“完整修复运行环境”，重新创建 Python、PyTorch、ONNX Runtime 等组件。完整修复同样不会删除 Workspace 中的数据集、标注、模型和配置。

以后双击桌面上的“YOLO团队训练平台”。源码开发者仍可使用 `启动YOLO团队训练平台.cmd`。

打开后默认进入“工作台总览”。平台按“项目 → 数据 → 训练 → 验证 → 部署”显示当前进度并给出下一步。按 `Ctrl+K` 可快速查找页面；自动跟踪标注等实验能力默认隐藏，可在左侧导航底部主动开启。

“项目中心”支持搜索、数据状态筛选、编辑、复制配置和删除。点击删除时，默认只把项目从列表移除，不动数据、训练模型和导出产物；只有主动勾选“同时删除平台托管的项目数据”并输入完整项目名称后，平台才会删除该项目的专用托管目录。外部关联的数据集始终不会由项目删除功能处理。

需要部署 RDK X5 时，在“部署与导出”选择“地瓜机器人 RDK X5”。“可选编译环境”会显示 Windows WSL、Ubuntu 22.04、Python 3.10 和 RDK Mapper 四层状态：缺少 WSL 时点击“安装 WSL + Ubuntu”，安装后按提示重启，再点击“配置编译环境”。不用 RDK X5 的成员可以完全跳过。移除时只清理平台专用工具链和缓存，保留 Ubuntu、其他 Linux 数据、训练数据和已生成 `.bin`。

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

个人模式只能从本机打开 <http://127.0.0.1:9000/>。共享模式会显示局域网地址，伙伴只需要浏览器，不需要安装平台、Python 或 Docker。每台电脑都能保留自己的 `Workspace/annotation-hub`；需要交换独立成果时导出、导入 `.ytp-project.zip` 项目包。实时多人编辑则连接同一台共享主机，由编辑锁避免互相覆盖。

“项目中心”的目标检测项目和图片会自动同步到标注中心，无需重复创建。个人与团队模式都默认使用“完成并下一张”，团队成员打开公共图片时会自动领取，无需预先分配；完成数据可直接导出，只有管理员为项目开启“双人审核”后才进入审核流程。“项目与团队”中的成员管理、任务分配、项目包、数据集导出和危险操作默认折叠，需要时再展开。完整流程见[本地优先协作标注](COLLABORATIVE_ANNOTATION.md)。

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

每次新训练都会建立不可覆盖的运行目录并生成 `training-manifest.json`，记录数据集、类别、训练参数和模型文件之间的对应关系。进入“模型资产”页面即可按数据集查看训练模型。详见[数据集与模型资产](MODEL_ASSETS.md)。

## 7. 测试模型

进入“测试模型”页面，可选择：

- 摄像头实时测试
- 单张图片测试
- 图片文件夹批量测试
- 视频或辅助标注流程

测试时可调整置信度阈值 `conf`。阈值较低会得到更多候选框，但误检可能增加。

## 8. 多平台导出

训练完成后进入“部署与导出”，选择训练资产中的 `model-best.pt`、目标平台和输出目录。树莓派 4、树莓派 5 和 RDK X5 已拆成独立配置，页面会显示每一步、最终产物和目标运行时；“将设备建议带到训练页”可一次带入起步模型与固定输入尺寸。平台会生成目标模型以及 `*.manifest.json` 部署清单，详见[设备适配文档](DEVICE_ADAPTERS.md)。

ONNX 仍是厂商工具链之间的主要交接格式。项目安装器会自动安装：

- `onnx`
- `onnxsim`
- `onnxslim`
- `onnxruntime`

如果使用远程训练环境，远程工具会根据 CPU/CUDA 配置选择 `onnxruntime` 或 `onnxruntime-gpu`。

通用 INT8 导出需要填写 `data.yaml`。RDK X5 会强制使用厂商 PTQ，并要求选择 20–50 张代表性图片。生成转换包后，在同页按顺序操作：

1. “检查 WSL”确认本机存在 Ubuntu 22.04、x86_64 与 Python 3.10。
2. 首次点击“配置编译环境”，在 WSL 用户目录建立隔离工具链；无需 Docker。
3. 点击“编译 `.bin`”，成功后模型路径会自动回填，部署清单状态更新为 `compiled`。
4. 填写官方 RDK X5 系统的 SSH 用户名与局域网 IP；推荐提前配置 SSH 密钥。
5. “检查板卡 SSH”确认 aarch64、`hbm_runtime` 和可选的 `hrt_model_exec`。
6. 点击“上传并真机测试”；选择测试图片时会运行官方示例并把结果图回传到转换包。

平台不保存 SSH 密码，也不会要求 root。编译成功不代表板端适配已经完成；部署前必须在目标设备核对预处理、类别顺序、输出张量、速度和留出测试集精度。

## 9. 半自动标注

网页标注会在跟踪失败、位置/尺寸突变或模板匹配质量较低时暂停。只要当前帧有一个目标需要复核，整帧都不会自动保存，以避免多目标漏标。推荐逐段播放、频繁抽查，具体见[半自动标注说明](SEMI_AUTO_LABELING.md)。

## 10. 配置文件

首次安装会根据设备生成：

```text
Workspace/config/settings.json
```

这是本机配置，不应上传到 GitHub。完整目录见[目录与命名规范](DIRECTORY_AND_NAMING_STANDARD.md)。

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
Workspace/logs/installation.log
Workspace/logs/installation-ubuntu.log
Workspace/logs/panel.log
Workspace/logs/annotation-server.log
Workspace/logs/system-check.json
```

工作台总览中的“运行系统诊断”会检查组件是否缺失、版本是否符合 `requirements.txt`，以及 OpenCV 是否为标注跟踪所需的 Contrib 版本。点击“复制诊断报告”可以带走当前版本、GPU/CUDA、组件状态和错误信息；“查看运行日志”可直接跳到实时日志页。进入“开始训练”后，平台会自动检查数据、模型、运行环境和设备，修改关键参数后自动刷新检查结果。如果面板无法启动，再查看安装器生成的报告并确认 8989 端口没有被其他服务占用。

## 12. 分享给其他人

Windows 维护者执行：

```powershell
.\installer\windows\build-installer.ps1
```

生成的 Setup 不包含本机 `.venv`、Workspace、日志、个人配置、训练数据或模型输出。`制作队友部署包.cmd` 只用于生成源码 ZIP，不是普通队友的首选安装方式。分享前请确认 README、兼容性说明和使用说明已经符合你的实际发布方式。
