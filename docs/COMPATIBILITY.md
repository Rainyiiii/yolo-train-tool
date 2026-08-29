# YOLO团队训练平台兼容性说明

本文档说明 YOLO团队训练平台的运行环境、硬件要求、依赖选择和已知限制。

## 支持范围

| 项目 | 支持情况 |
| --- | --- |
| Windows | Windows 10/11，使用 Windows PowerShell 5.1 |
| Ubuntu | Ubuntu 22.04/24.04，使用 Bash |
| Python | 3.10–3.14 |
| NVIDIA GPU | 自动使用 CUDA 12.8 PyTorch 训练组件 |
| 无 NVIDIA GPU | 自动使用 CPU 版 PyTorch，功能完整但训练较慢 |
| 浏览器 | Chrome、Edge、Firefox 等现代浏览器 |
| macOS | 当前未提供一键部署脚本，未做完整验证 |
| ARM/树莓派 | Raspberry Pi 4/5 独立 NCNN/ONNX 配置；训练仍建议使用 x86_64 主机，真机矩阵持续补充 |
| 地瓜机器人 RDK X5 | 本机 Ubuntu 22.04 WSL2 一键编译 Bayes-e INT8 `.bin`，再通过 SSH 上传官方系统验证；板端建议 RDK OS 3.5+ |
| Rockchip / MaixCAM | 已提供设备配置或专用转换入口；不同板型仍需按文档使用厂商工具链并真机验证 |

Python 3.14 是当前最新稳定功能版本；Python 3.15 仍是预发布版本，因此暂不列入保证范围。即使个别依赖已经提供对应 wheel，也可能出现 PyTorch、OpenCV 或 ONNX 组件不兼容。

RDK X5 编译工具链是例外：当前锁定 Ubuntu 22.04、x86_64、Python 3.10。它运行在独立 WSL 虚拟环境中，不受 Windows 主训练环境 Python 3.10–3.14 范围影响。平台不要求 Docker；板卡侧需要可用的 SSH 和 `hbm_runtime`，图片推理还需要 Git 或预先准备的官方 Model Zoo。

## NVIDIA GPU 要求

Windows 或 Ubuntu 机器需要：

1. 正常安装 NVIDIA 显卡驱动。
2. 在终端运行 `nvidia-smi` 能看到显卡信息。
3. 具备足够磁盘空间下载 PyTorch 和 CUDA 运行库。

安装器会安装 PyTorch 自带的 CUDA 运行时，不要求用户另外安装 CUDA Toolkit。驱动版本仍然需要满足 PyTorch/CUDA 12.8 的兼容要求。

建议显存：

| 显存 | 建议 Batch |
| --- | --- |
| 4 GB 以下 | 2–4 |
| 4–6 GB | 4–8 |
| 6–8 GB | 8 |
| 8 GB 以上 | 8–16 |

实际 Batch 还会受到图片尺寸、模型大小、数据增强和数据加载进程数量影响。显存不足时优先降低 Batch。

## 自动安装的核心组件

- `torch`、`torchvision`、`torchaudio`
- `ultralytics`
- `opencv-contrib-python`
- `onnx`、`onnxsim`、`onnxslim`
- `onnxruntime`
- `Pillow`、`PyYAML`、`psutil`

Windows 一键安装脚本和 Ubuntu 安装脚本都会创建项目自己的 `.venv`，不会把这些依赖写入系统 Python。

## 网络与磁盘

首次安装需要访问：

- PyPI 或可用的 Python 包镜像
- `download.pytorch.org`，下载 PyTorch
- GitHub Ultralytics Release，下载 `yolo11n.pt`

NVIDIA 环境首次安装可能需要数 GB 磁盘空间。建议预留至少 8–12 GB 可用空间；训练数据、缓存和输出模型需要额外空间。

## 已知限制

- 面板默认只监听 `127.0.0.1:8989`。Windows 桌面版只允许一个 YOLO 训练面板服务；发现旧服务时先询问是否关闭。默认端口被其他软件占用时可选择其他回环端口，仍只允许本机访问。
- 协作标注个人模式监听 `127.0.0.1:9000`；共享模式监听 `0.0.0.0:9000`，仅应在可信局域网中使用。
- 局域网伙伴只需要现代浏览器，不需要安装 Docker、Python 或本平台；作为主机的电脑需要完成项目安装。
- 如需远程使用，建议通过 SSH 端口转发或反向代理，不建议直接暴露到公网。
- Ubuntu 脚本会优先寻找本机已有的 Python 3.10–3.14；如果没有，会尝试通过 `apt` 安装 `python3`、`python3-venv` 和 `python3-pip`。如果发行版默认 Python 低于 3.10，需要手动安装受支持版本。
- Python、PyTorch、Ultralytics 和 ONNX 依赖会持续更新；发布新版本前建议在干净的 Windows 和 Ubuntu 环境各测试一次。
- TensorRT engine、RKNN、RDK X5 `.bin` 和 MaixCAM 模型与厂商运行时/驱动版本相关，不保证跨设备或跨版本直接复用。X5 属于 Bayes-e，最终运行模型扩展名是 `.bin`；`.hbm` 不能作为 X5 适配完成的判据。

## 常见排错

### 找不到 Python

检查版本：

```bash
python3 --version
```

必须是 Python 3.10–3.14。Ubuntu 可安装 `python3-venv` 后重新运行部署脚本。

### 找不到 CUDA

检查：

```bash
nvidia-smi
```

如果命令不存在或没有显卡信息，安装器会退回 CPU 模式。

### 缺少 onnxruntime

重新运行安装器，或在项目虚拟环境中执行：

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Windows 使用：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 查看诊断报告

工作台总览可直接点击“运行系统诊断”，检查 Python、PyTorch/CUDA、Ultralytics、OpenCV Contrib、ONNX Runtime 与导出组件。诊断同时核对安装包名称和版本范围，避免“模块能导入但版本或发行包不符合要求”的假正常。完成后可点击“复制诊断报告”，将平台时间、Python、GPU/CUDA、组件版本和错误整理成一段可直接提交的问题信息；安装器也会生成 `Workspace/logs/system-check.json`。面板启动失败时，重点查看：

- `Workspace/logs/installation.log` 或 `workspace/logs/installation-ubuntu.log`
- `Workspace/logs/launcher.log`
- `Workspace/logs/panel.log`
- `Workspace/logs/system-check.json`
