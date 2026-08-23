# Windows WebView2 安装与发布

## 队友安装

队友只需要运行：

```text
YOLO-Team-Training-Platform-Setup-v3.0.2-beta.exe
```

安装器默认选择 `D:\YOLOTeamTrainingPlatform`，并自动完成：

1. 安装 .NET 8 Desktop Runtime；
2. 安装 Microsoft Edge WebView2 Evergreen Runtime；
3. 安装或定位 Python 3.10–3.14；
4. 创建隔离运行环境；
5. 根据 NVIDIA GPU 选择 CUDA 12.8 或 CPU PyTorch；
6. 安装 `onnxruntime` 等平台依赖；
7. 创建规范化 Workspace；
8. 创建桌面和开始菜单快捷方式。

首次安装需要联网下载 Python/PyTorch/PyPI 依赖，耗时取决于网络和显卡版本。安装器内的“正在安装运行环境”页面会显示三个阶段和实时滚动输出；完整日志同时写入 `D:\YOLOTeamTrainingPlatform\Workspace\logs\installation.log`。安装失败时平台不会被启动，请保留该日志用于排错。

WebView2 桌面程序只加载本机 `127.0.0.1:8989` 平台页面；外部文档链接交给系统浏览器。下载文件保存到 `Workspace/exports/downloads`。关闭桌面窗口时会停止本机面板和协作标注服务，不删除工作区。

## 卸载与用户数据

从 Windows“已安装的应用”卸载时，程序会停止本机服务，并始终删除实际安装目录下由平台管理的 `App`、`Desktop` 和 `Runtime`。卸载器随后询问是否保留 `Workspace`：

- 选择“是”：仅保留数据集、标注、训练模型、导出结果和个人配置；
- 选择“否”：删除 Workspace，并在安装根目录为空后删除根目录；
- 静默卸载默认保留 Workspace；维护脚本可传入 `/PURGEDATA` 彻底删除，或传入 `/KEEPDATA` 明确保留。

卸载规则只使用本次安装记录的 `{app}` 路径和明确的子目录，不会通配删除安装根目录，也不会处理安装目录之外的内容。

## 维护者构建

构建机需要 .NET 8 SDK 与 Inno Setup 6，然后执行：

```powershell
.\installer\windows\build-installer.ps1
```

生成文件位于 `dist/`。脚本会从微软官方下载并缓存 .NET Desktop Runtime 与 WebView2 Evergreen Bootstrapper。

推送 `v*` 版本标签后，GitHub Actions 会执行同一个构建脚本，将 Setup 保存为工作流产物并自动创建 GitHub Release。应用源码、Python 核心源码和安装器源码始终保留在同一仓库、同一提交中。

公开发布前必须为 Setup 和桌面 EXE 配置可信代码签名证书。未签名安装器可能触发 Windows SmartScreen；当前仓库不会伪造或绕过该提示。
