# 参与贡献

感谢帮助 YOLO团队训练平台适配更多训练与部署平台。提交代码前请先阅读 [设备适配文档](docs/DEVICE_ADAPTERS.md) 和 [路线图](docs/ROADMAP.md)。

## 开发检查

```bash
python -m unittest discover -s tests -v
python -m py_compile train_panel.py host_train_export.py export_model.py device_profiles.py
```

修改 Windows/Ubuntu 安装器时，还应分别做 PowerShell 与 Bash 语法检查，并在干净环境完成一次安装。

## 设备适配提交

设备支持不能只以“成功导出”为结论。PR 或 Issue 请提供：

- 板卡型号、SoC、系统镜像和架构；
- Python、驱动、厂商编译器和运行时版本；
- 源模型、输入尺寸、任务类型、导出参数和量化方式；
- 板端加载结果、至少一张样本的输出核对；
- 延迟/FPS、峰值内存、精度变化，最好包含可复现命令；
- 发现的限制和对应官方文档。

新增设备配置请修改 `device_profiles.py`，补充测试和 `docs/DEVICE_ADAPTERS.md`。厂商工具链应尽量放在隔离环境或容器中，不要把互相冲突的依赖塞进训练 `.venv`。

## 代码与来源

- 不要提交数据集、模型权重、日志、虚拟环境、IP、账号或密钥。
- 引入第三方代码前确认许可证，记录来源，并更新 `THIRD_PARTY_NOTICES.md`。
- 不要提交许可证不明的复制代码；必要时用可证明的独立实现替换。
- 保持面板默认仅监听本机地址，不在未经认证的情况下暴露到公网。
