# 更新记录

## 3.0.0-beta

- 平台统一更名为“YOLO团队训练平台”。
- 增加 WebView2 Windows 桌面封装和 Inno Setup 一键安装器，默认安装到 `D:\YOLOTeamTrainingPlatform`。
- Windows 自动准备 .NET、WebView2、Python 3.10–3.14、PyTorch、ONNX Runtime 和全部平台依赖。
- 建立统一 `Workspace` 布局与不可覆盖的训练、导出、测试和项目包命名规范。
- 模型资产仅接受带 `training-manifest.json` 的可追溯训练记录，不再推断或迁移旧训练目录。
- 协作标注项目包升级为 `.ytp-project.zip`，并统一数据集导出清单。
- Ubuntu 安装与运行路径同步采用新版 Workspace 规范。

## 2.3.0-beta

- 增加不依赖 Docker 的本地优先协作标注中心。
- 同一套程序支持个人模式和局域网共享模式，任何电脑都可以临时成为团队主机。
- 增加管理员、审核员、标注员账号和任务分配。
- 增加五分钟编辑锁、修订冲突检查、提交、审核通过和驳回流程。
- 增加可移植 `.matproj.zip` 项目包，支持独立工作区之间交换图片和标注。
- 增加 Ultralytics YOLO、COCO、Pascal VOC 和 LabelMe 导出，只导出审核通过的数据。
- 增加 Windows 与 Ubuntu 的个人/共享标注启动和停止工具。

## 2.2.0-beta

- 增加独立“模型资产”页面，按数据集归组展示训练记录、PT/ONNX 和设备部署产物。
- 新训练自动生成 `training_manifest.json`，记录数据集、类别、参数、指标和模型路径。
- 支持扫描旧版 `outputs_*` 目录，并明确区分“清单关联”和“推断关联”。
- 模型资产可一键回填到测试页面或部署页面。
- 增加本机 `model_registry.json` 索引，记录已扫描目录和训练/部署清单。

## 2.1.0-beta

- 增加通用 ONNX、树莓派 NCNN、Rockchip RKNN、地瓜 RDK、MaixCAM、TensorRT 和 OpenVINO 设备配置档。
- 增加独立的多平台导出工具、部署清单和 ONNX Runtime 加载检查。
- 将 MaixCAM 专用转换降为兼容旧流程，不再作为项目唯一部署目标。
- 半自动标注增加漂移质量判断；任一目标异常时整帧停止保存，避免生成缺框标签。
- 网页跟踪器改为面向用户的策略名称，并增加待复核计数和异常原因。
- 增加设备适配、半自动标注、路线图和第三方来源核对文档。
- 分享包改名为 `MyAutoTrain-日期时间.zip`，并包含 `docs` 文档目录。
- 默认 SSH 端口修正为 22，面板直接运行时默认仅监听 `127.0.0.1`。

## 2.0.0-team

- 增加队友电脑一键安装与启动流程。
- 自动识别 NVIDIA CUDA 或 CPU 训练环境。
- 增加安装后系统自检和 JSON 检查报告。
- 增加一键制作团队分享 ZIP 包。
- 主界面改为红白浅色主题。
- 默认页面只保留数据选择、训练预设和开始/停止操作。
- 低频参数和设备转换收进“更多设置”。
- 默认使用 640×480、batch 16、4 个加载进程和手动停止模式。
- 增加 Ubuntu 部署、启动和停止脚本，支持 NVIDIA CUDA 12.8 或 CPU 环境。
- 安装依赖补充 `onnxruntime`，系统自检会报告其缺失状态。
- Python 支持范围扩展为 3.10–3.14。
