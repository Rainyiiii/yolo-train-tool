# 数据集与模型资产

## 官方基础模型中心

主导航中的“基础模型”用于下载训练起点，不等同于训练完成后的“模型资产”。当前目录收录 Ultralytics 官方 YOLO26、YOLO11 和 YOLOv8 的目标检测、图像分类权重，并按 Nano、Small、Medium、Large、X-Large 展示资源等级。

- 点击“下载并用于训练”后，下载任务进入统一运行日志；下载完成会自动写入当前训练配置。
- 已下载模型保存在 `Workspace/model-assets/base-models/`，升级应用不会重复下载或删除。
- 重新下载时先写入临时文件，只有完整模型准备好后才原子替换旧文件。
- 为避免任意地址下载和路径穿越，服务端只接受当前模型目录白名单中的官方权重名称。
- Ultralytics 官方模型与代码涉及 AGPL-3.0 / Enterprise License；公开分发、网络服务或商业集成前应根据实际使用方式确认授权。

当前训练链路支持目标检测与图像分类，因此模型中心暂不开放分割、姿态、OBB 和语义分割权重。后续应在对应数据格式、训练、验证、测试和导出链路完整实装后再开放。

“模型资产”页面用于建立可审计的数据集—训练运行—模型—部署产物关系。规范训练会自动写入 `training-manifest.json`；已有 `.pt` / `.onnx` 也可手动登记并明确关联项目或数据集，不再需要依赖目录猜测关系。

## 标准训练运行

```text
Workspace/training-runs/<project-id>/
└─ <project>__<model>__train__YYYYMMDD-HHMMSS/
   ├─ model-best.pt
   ├─ model-best.onnx
   ├─ dataset-classes.txt
   ├─ training-metrics.csv
   ├─ training-arguments.yaml
   ├─ training-manifest.json
   ├─ plots/
   ├─ calibration-images/
   └─ test-sample.jpg
```

清单记录数据集 ID、来源、版本指纹、类别、图片数量、训练参数、指标和资产相对路径。页面只把清单中明确声明的关系标记为可信关联。

## 部署导出

```text
Workspace/exports/deployments/<target>/
└─ <model>__<target>__<format>__YYYYMMDD-HHMMSS/
   ├─ <model>__<target>__<format>__YYYYMMDD-HHMMSS.<ext>
   └─ <model>__<target>__<format>__YYYYMMDD-HHMMSS.manifest.json
```

部署清单通过 `source_model` 回连训练模型，记录目标平台、格式、芯片、输入尺寸、量化和下一步厂商工具链要求。

## 本地索引

索引保存在 `Workspace/config/model-registry.json`。它记录清单、补充扫描根目录和手动登记模型的路径/关联信息，不复制模型本体。移动外部运行目录或模型后，需要在“模型资产”页面重新登记。

页面可勾选至少两个模型进行并排比较；有训练清单时会展示输入尺寸、轮次、Batch 和指标，手动登记模型只展示维护者明确填写的信息。
