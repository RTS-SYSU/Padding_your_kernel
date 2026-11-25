# Padding Your Kernel

用访存约束为 CUDA 算子直接推导一个行尾补量。工作台包含 PolyBench/GPU 和 Rodinia 的 17 个算子预设，其中 MVT 提供两个尺寸。可以在本机测量指定预设。

[模型与适用范围](docs/theory.md) · [参与开发](CONTRIBUTING.md) · [版本记录](CHANGELOG.md)

## 快速开始

解析功能需要 Python 3.10 或更新版本。在仓库根目录安装并启动。

```text
python -m pip install .
padding-your-kernel serve
```

打开 [本地工作台](http://127.0.0.1:8766/) 。选择算子预设，点击「分析布局」查看补量和行距。推导依据可通过 JSON 报告或命令行查看。

## 运行真机实验

在 Windows 上使用带 CUDA 驱动的 NVIDIA GPU、CUDA 12 Toolkit（用于 NVRTC 编译）、NumPy 和现行的 Experiment 23 研究项目。两个项目位于同一父目录时，工作台会自动找到 `Analytical GPU Padding Benefit Model`。也可在启动时指定目录。

```powershell
padding-your-kernel serve --research-dir "C:\SYSU\papers\Analytical GPU Padding Benefit Model"
```

在「算子预设」中选定算子和尺寸，点击「运行真机实验」。工作台先核对研究项目冻结的算子与补量，再调用已有的 CUDA 实现。一次运行一个配置。目标布局与 `p=0` 做预热配对，理论值为零时只计基线。新结果标注本机设备、CUDA 事件耗时、配对加速比、正确性和各进程的结果。运行日志和原始数据写入 `results/live/`，可用 `--results-dir` 改位置。

配对加速比按实验中的轮次与进程计算，旁边的两列时间各自取中位数。三个独立进程的配对结果都大于 1 时标记为通过。

## 命令行

```text
padding-your-kernel presets
padding-your-kernel preset syrk
```

SYRK 预设给出 `p=8`。也可用 `python -m padding_your_kernel` 调用相同命令。工作台监听本机回环地址，可用 `serve --port 8770` 改端口。

## 如何读报告

| 字段 | 含义 |
|---|---|
| `p` | 每行增加的元素数。零表示紧凑布局。 |
| `layouts` | 各操作数的逻辑宽、物理行距与额外字节。 |
| `rule`、`reason` | 采用的访存约束及选择依据。 |
| `scope`、`assumptions` | 推导所覆盖的阶段、访问顺序和理想缓存条件。 |
| `status` | 建议填充、保留紧凑布局，或输入不在当前模型覆盖范围。 |

预设对应已声明的源码访问模式和固定尺寸。报告中的补量是模型推导结果。实际缺失数与加速比需在目标 GPU 上分别验证。

## 仓库结构

| 目录 | 内容 |
|---|---|
| `src/padding_your_kernel/` | 同一分析核心、CLI、本地 API 和 Web 工作台。 |
| `tests/` | 公式、预设和 HTTP 接口的无 GPU 检查。 |
| `docs/` | 公式推导和各预设的适用范围。 |

```text
python -m pip install -e .
python -B -m unittest discover -s tests -v
```
