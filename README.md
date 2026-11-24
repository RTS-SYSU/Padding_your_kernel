# Padding Your Kernel

Padding Your Kernel 是一个 CUDA 行尾补量实验项目。它从 kernel 的逻辑访问和 GPU 的地址粒度推导最小补量，并用配对计时和 Nsight Compute 对比原布局与补后布局。两种布局运行同一个 kernel，保持有效数据、计算和逻辑访问顺序一致。

## 快速开始

本地界面只需要 Python 3.11 或更高版本。启动后打开 [本地工作台](http://127.0.0.1:8765/)。

```powershell
python -B ui/server.py
```

工作台默认进入「运行实验」。选择实验和输入配置后点击「运行实验」，服务会编译 CUDA kernel、事前冻结补量、执行配对计时并采集缓存计数。完成后页面会展示本次结果。使用「历史记录」查看已保存的结果。每份新记录包含所选配置和事前计划。

实时运行使用本机 GPU，一次执行一个实验。新记录写入 `results/live_*`。Pavia 实验需要原始 `PaviaU.mat`，服务会在配置的结果库中寻找已有影像，也可以通过 `python -B ui/server.py --pavia-input <文件>` 指定。实时工作台的 CUDA、Nsight Compute 和 Python 依赖与命令行实验相同。

结果页可以查看补量公式、行首相位、发出的读 sector、L1/L2 读 miss 和单次 kernel 时间。双窗口计算器支持自定义逻辑行宽和第二窗口起点。两个随附实验都使用 `.cg`，历史数据的 L1 读 miss 显示为未采集。时间显示为原布局与补后布局的计时中位数之比。

服务会扫描项目的 `results/` 和同级历史结果库 `../PADGain_results/results/`。如有其他结果目录，使用 `python -B ui/server.py --results <目录>`。每个实验记录放在该目录的一个子文件夹内，保留运行器生成的 `plan.json`、计时汇总和计数 JSON。也可以在没有实验结果时使用补量计算器。

## 实验

| 实验 | 补量构造目标 | 观察指标 |
|---|---|---|
| [双窗口基准](experiments/dual_window/README.md) | 根据第二窗口位置求能让每行两窗同处 128 B 地址块的最小补量 | 请求数、L2 读 miss、执行时间 |
| [Pavia University 最远光谱对](experiments/pavia/README.md) | 让 warp 的 32 元素加载恰好覆盖两个 32 B sector，103 列补 9 | 请求数、L2 读 miss、执行时间 |

两项实验均在 RTX 4060 Ti 上验证。计时范围是一次目标 CUDA kernel 启动，布局准备、传输和 CPU 校验在计时范围外。运行器先冻结补量、地址计数和预算，再做交替配对测量。事前真机 miss 与加速比条件不足时记为未判定。

## 运行环境与目录

运行 CUDA 实验需要 Windows 上的 CUDA Toolkit、MSVC v143 和兼容的 NVIDIA GPU。默认构建架构是 `sm_89`。采集缓存计数需要 Nsight Compute，处理 Pavia 影像另外需要 NumPy 与 SciPy。两项实验各自的 README 给出了构建、冻结计划和测量命令。

| 路径 | 内容 |
|---|---|
| `experiments/dual_window/` | 双窗口 CUDA 源码、运行器及缓存诊断 |
| `experiments/pavia/` | 光谱对搜索 CUDA 源码、数据准备及测量 |
| `dataset/` | 双窗口输入定义、生成与核验工具 |
| `tools/` | CUDA 构建入口 |
| `ui/` | 本地 Web 服务与前端 |
| `results/` | 本地生成的实验记录 |

```powershell
python -B -m unittest discover -s experiments/dual_window -p "test_*.py"
python -B -m unittest discover -s dataset -p "test_*.py"
```

