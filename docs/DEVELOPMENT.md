# 开发与仓库结构

本仓库采用单仓库结构，同时保存可发布应用和完整源码。普通用户无需阅读源码，从 GitHub Releases 下载 Setup 即可；开发者可以在同一版本中核对桌面壳、训练服务、标注服务和安装逻辑。

## 源码组成

```text
yolo-train-tool/
├─ desktop/YOLOTeamTrainingPlatform.Desktop/  WebView2 Windows 桌面应用源码
├─ installer/windows/                         Inno Setup 与自动构建源码
├─ tests/                                     Python 自动测试
├─ docs/                                      使用、适配和维护文档
├─ train_panel.py                             训练平台 Web UI 与 API
├─ annotation_*.py                            协作标注服务
├─ host_train_export.py                       数据准备与训练流水线
├─ export_model.py                            多设备模型导出
├─ rdk_x5_deployment.py                       RDK X5 NPU 转换包与校验入口
├─ rdk_x5_remote.py                           WSL 编译与 RDK X5 SSH 真机验证
├─ model_assets.py                            数据集—训练—模型资产索引
├─ platform_paths.py                          产品目录与命名规范
└─ install_runtime.ps1                        Windows 运行环境安装逻辑
```

`desktop` 不是另一套业务实现。它负责执行桌面单实例和训练面板单服务检查、启动本地 Python 服务，并通过 WebView2 安全加载回环地址。默认端口为 `127.0.0.1:8989`；已有 YOLO 服务必须由用户确认关闭，其他软件造成的端口冲突才回退到其他本机端口。训练、标注和导出逻辑只有一份，浏览器模式、WebView2 模式共用。

## 本地验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\installer\windows\build-installer.ps1
```

生成的 `dist/`、`.venv/` 和 `workspace/` 不进入 Git。版本标签 `v*` 会触发 `.github/workflows/windows-installer.yml`，GitHub 自动构建 Windows Setup；带标签的构建还会创建 Release 并上传安装器。

## 发布规则

1. 更新 `VERSION.txt`、`CHANGELOG.md` 和用户文档。
2. 在 Windows 上通过测试并本地构建一次 Setup。
3. 推送源码提交。
4. 创建与版本一致的标签，例如 `v3.2.1-beta`。
5. 等待 GitHub Actions 成功，核对 Release 中的安装器。
6. 发布标签前复核 `LICENSE` 与第三方声明，并确认 Windows 代码签名 secrets 已配置。
