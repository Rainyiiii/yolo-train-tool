# 目录与命名规范

本规范适用于 YOLO团队训练平台 3.2 及之后创建的目录、项目、训练运行、模型、数据集导出、部署导出和测试结果。

## 安装布局

Windows 默认安装根目录为 `D:\YOLOTeamTrainingPlatform`；没有 D 盘时回退到系统 `Program Files\YOLOTeamTrainingPlatform`。

```text
YOLOTeamTrainingPlatform/
├─ App/                         只读程序和 Python 源码
├─ Desktop/                     WebView2 桌面程序
├─ Runtime/Python/              隔离 Python 虚拟环境
└─ Workspace/                   用户数据，卸载时默认保留
   ├─ config/                   设置和本机索引
   ├─ logs/                     安装、服务和任务日志
   ├─ state/                    PID、停止信号等瞬时状态
   ├─ datasets/                 训练数据集
   ├─ annotation-hub/           协作标注数据库、图片和项目导出
   ├─ training-runs/            可追溯训练运行
   ├─ model-assets/base-models/ 基础模型
   ├─ exports/datasets/         数据集格式导出
   ├─ exports/deployments/      设备部署导出
   ├─ exports/downloads/        WebView2 下载归档
   ├─ test-results/             图片/文件夹测试结果
   ├─ cache/                    可再生成缓存
   ├─ temp/                     可清理临时任务
   └─ backups/                  人工备份
```

程序更新不得覆盖 `Workspace`。卸载程序默认不删除该目录，避免误删数据集和模型。

## 标识符规则

- 目录和资产标识使用小写 ASCII、数字、中文及连字符 `-`。
- 空格、斜杠、冒号及其他特殊字符统一转换为 `-`。
- 项目标识和模型标识最长 48 个字符。
- 禁止使用 Windows 保留名称，如 `CON`、`NUL`、`COM1`。
- 时间使用本地时区，格式固定为 `YYYYMMDD-HHMMSS`。
- 同名目录绝不覆盖，追加 `__02`、`__03`。

## 资产命名

| 类型 | 规范 |
| --- | --- |
| 训练运行 | `<project>__<model>__train__<timestamp>` |
| 数据集导出 | `<project>__dataset__<format>__v<hash>__<timestamp>.zip` |
| 标注项目包 | `<project>__annotation-project__v<hash>__<timestamp>.ytp-project.zip` |
| 部署运行目录 | `<model>__<target>__<format>__<timestamp>` |
| 部署模型 | `<model>__<target>__<format>__<timestamp>.<ext>` |
| 部署清单 | `<model>__<target>__<format>__<timestamp>.manifest.json` |
| 测试运行 | `<model>__<source>__test__<timestamp>` |
| 发布安装器 | `YOLO-Team-Training-Platform-Setup-v<version>.exe` |

清单文件使用 UTF-8 JSON；路径字段保存绝对路径，清单内部资产字段优先使用相对路径。任何自动任务都不得静默覆盖已有资产。
