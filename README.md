# Padding Your Kernel

用访存约束为 CUDA 算子直接推导一个行尾补量，并说明采用的规则、物理行距和额外分配。工作台包含 PolyBench/GPU 和 Rodinia 的 17 个算子预设，其中 MVT 提供两个尺寸。可以查看各配置的历史真机计时，在本机重新测指定预设，也可以输入满足模型条件的 FP32 相邻列复用布局。

[模型与适用范围](docs/theory.md) · [参与开发](CONTRIBUTING.md) · [版本记录](CHANGELOG.md)

![选定 SYRK 后查看现场测量和历史真机记录](docs/workbench.png)

## 快速开始

解析功能和历史结果需要 Python 3.10 或更新版本。在仓库根目录安装并启动。

```text
python -m pip install .
padding-your-kernel serve
```

打开 [本地工作台](http://127.0.0.1:8766/) 。选择预设或输入参数，点击「分析布局」查看结果。报告可下载为 JSON。

## 运行真机实验

在 Windows 上使用带 CUDA 驱动的 NVIDIA GPU、CUDA 12 Toolkit（用于 NVRTC 编译）、NumPy 和现行的 Experiment 23 研究项目。两个项目位于同一父目录时，工作台会自动找到 `Analytical GPU Padding Benefit Model`。也可在启动时指定目录。

```powershell
padding-your-kernel serve --research-dir "C:\SYSU\papers\Analytical GPU Padding Benefit Model"
```

在「算子预设」中选定算子和尺寸，点击「运行真机实验」。工作台先核对研究项目冻结的算子与补量，再调用已有的 CUDA 实现。一次运行一个配置。目标布局与 `p=0` 做预热配对，理论值为零时只计基线。新结果标注本机设备、CUDA 事件耗时、配对加速比、正确性和各进程的结果。运行日志和原始数据写入 `results/live/`，可用 `--results-dir` 改位置。

历史结果始终来自 Experiment 23 的 RTX 4060 Ti 预热记录。18 个配置的总表可在页面打开。配对加速比按实验中的轮次与进程计算，旁边的两列时间各自取中位数。只有三个独立进程的配对结果都大于 1 时才标记为通过。自定义相邻列输入使用解析规则。

[随包发布的预热记录](src/padding_your_kernel/data/experiment23_primed.json) · [从原始记录重建快照](tools/export_experiment23.py)

## 命令行

```text
padding-your-kernel presets
padding-your-kernel preset syrk
padding-your-kernel adjacent-columns --rows 2048 --width 2048
```

两个示例分别给出 (p=8) 和 (p=16)。也可用 `python -m padding_your_kernel` 调用相同命令。工作台监听本机回环地址，可用 `serve --port 8770` 改端口。

## 如何读报告

| 字段 | 含义 |
|---|---|
| `p` | 每行增加的元素数。零表示紧凑布局。 |
| `layouts` | 各操作数的逻辑宽、物理行距与额外字节。 |
| `rule`、`reason` | 采用的访存约束及选择依据。 |
| `scope`、`assumptions` | 推导所覆盖的阶段、访问顺序和理想缓存条件。 |
| `status` | 建议填充、保留紧凑布局，或输入不在当前模型覆盖范围。 |

预设对应已声明的源码访问模式和固定尺寸。自定义分析需要显式声明行宽、行数、输入基址对齐及同组干扰预算。报告中的补量是模型推导结果。实际缺失数与加速比需在目标 GPU 上分别验证。

## 仓库结构

| 目录 | 内容 |
|---|---|
| `src/padding_your_kernel/` | 同一分析核心、CLI、本地 API 和 Web 工作台。 |
| `tests/` | 公式、预设和 HTTP 接口的无 GPU 检查。 |
| `docs/` | 公式推导和各预设的适用范围。 |
| `tools/` | 从研究实验记录重建历史计时快照。 |

```text
python -m pip install -e .
python -B -m unittest discover -s tests -v
```
