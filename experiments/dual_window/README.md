# 行尾 padding 的双窗口 GPU 实验

另一个可运行的实验是 [Pavia University 最远光谱对搜索](../pavia/README.md)。

矩阵元素为四字节无符号整数。每遍读取每行的第 0–7 列和第 `G` 至 `G+7` 列，列号从 0 起算。每个八线程小组依次处理分配给自己的行，在同一个目标 kernel 内重复扫描 40 遍。两种布局使用相同的逻辑数据、访问顺序、启动配置和 kernel。[CUDA 实现](bench.cu)

## 在运行前确定补量与覆盖

设逻辑行宽为 `W` 个元素，表首按 128 字节对齐。使所有行首都保持 128 字节对齐的最小补量为 `p=(-W) mod 32`。在支持的窗口位置，两种布局每行每遍都发出两个 32 字节读 sector。可逐个行首相位计算两个窗口触及的 128 字节地址块总数。运行器将补量、两侧覆盖和发出的 sector 数冻结在 `plan.json`。[几何计算](prediction.py)

| 输入 | 第二窗口起始列 | 补前 / 补后行距（元素） | 补前 / 补后每行平均覆盖的 128 字节块 |
|---|---:|---:|---:|
| original | 24 | 56 / 64 | 1.75 / 1 |
| family-104-16-80 | 16 | 104 / 128 | 1.5 / 1 |
| family-136-16-80 | 16 | 136 / 160 | 1.5 / 1 |

`family-W-G-F` 的行数按设备标称 L2 容量的 `F%` 除以 128 字节，再向下取整至 256 的倍数。这个比例定义输入大小。要预测真机重访 miss，还需要知道缓存映射、替换和重访之间的实际访问。只掌握双窗口几何时，事前 L2 miss 数和 kernel 加速比均记为未判定。

## 真机验证

原布局和补后布局分别分配，表的申请量按本基准的 2 MiB 粒度向上取整并核算预算。每侧都先初始化，读取两倍 L2 容量的独立缓冲区，再按相同窗口预热一遍。随后用 CUDA event 对目标 kernel 计时一次，逐项核对逻辑输出。实测加速比为 `T(0)/T(p)`。可在全部配对计时之后另外用 Nsight Compute 单 pass 核对发出的读 sector、到达 L2 的读 sector 和 L2 读 miss。[运行器](run.py)

从 Windows 项目根目录执行以下命令。构建示例适用于 RTX 4060 Ti。冻结输出会打印本次结果目录，在评价命令中使用该目录。

```powershell
$bench = & ./tools/build.ps1 -Arch sm_89 | Select-Object -Last 1
python -B experiments/dual_window/run.py $bench --family-case 104 16 80 --family-case 136 16 80 --freeze-only
python -B experiments/dual_window/run.py $bench --evaluate '<本次冻结目录>' --profile
```

不指定输入时运行 56 列的 `holdout` 和 `original`。`--family` 运行行宽 104、112、120 与第二窗口起始列 16、24 的六种组合。省略 `--profile` 可仅采集时间与逻辑输出校验。

[104 列与 136 列的同批真机记录](../../../PADGain_results/results/row_tail_v1_pair_20261120T0603232351399Z/row_window_family_20261120T060326748635Z/pair_analysis.md) · [双窗口数据集 v0.1](../../../PADGain_results/results/PADGain-DualWindow-v0.1/README.md) · [输入生成器](../../dataset/workloads.py)

## 按窗口间距构造最小补量

令物理行距为 `P=W+p`，行首相位按 32 个元素取模。完整周期的最大相位为 `32-gcd(P,32)`。在本实验使用的基址和 8 元素对齐约束下，要让每行的两个窗口都留在同一个 128 B 地址块，必须有 `gcd(P,32) >= G+8`。因此，令 `k` 为不小于 `G+8` 的最小二次幂，直接取 `p=(-W) mod k`。例如 `G=8` 时 `k=16`，104 列补 8 个元素即可。

两臂发出的读请求数相同，均为 `2 × 行数 × 40`。若预热读入的 sector 全部驻留，两臂目标 kernel 的 L2 读 miss 均为零，等代价串行模型给出 1×。实际驻留和耗时在事前记为未判定。可用如下命令先冻结方案，再做 12 对交替计时和两轮独立单 pass 采集。

```powershell
$bench = & ./tools/build.ps1 -Arch sm_89 | Select-Object -Last 1
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = Join-Path (Get-Location) "results/dual_window_current_theory_$stamp"
New-Item -ItemType Directory -Path $out -Force
python -B experiments/dual_window/current_theory.py freeze $bench $out
python -B experiments/dual_window/current_theory.py measure $bench $out
```

[RTX 4060 Ti 本轮实测](../../../PADGain_results/results/dual_window_current_theory_4060ti_20261124/report.md)

## 历史对照实验：由参考行宽估计收益

本节复现使用历史计时的跨行宽估计。当前理论预测入口及所需缓存条件见[求解器说明](../../../PADGain/docs/SOLVER_ENTRY.md)。给定已保存的设备标定文件，`prior_forecast.py freeze` 在 CPU 上根据行宽、第二窗口位置和行数计算补量、预计的两侧耗时及加速比。以下示例使用 4060 Ti 的历史标定文件，预测 232 列的输入，再用同一张卡验证。

```powershell
$experiment = '../PADGain_results/results/unseen_width_20261122T131402Z'
$forecast = Join-Path $env:TEMP ('padgain_' + [guid]::NewGuid() + '.json')
python -B experiments/dual_window/prior_forecast.py freeze --profile "$experiment/profile_4060ti.json" --case family-232-16-80 --out $forecast
# $bench 为上文构建的目标程序。
python -B experiments/dual_window/run.py $bench --forecast $forecast
```

跨行宽经验模型在每张卡上只使用固定的六种参考行宽（168、176、184、200、208、216），按地址块覆盖、行距和行数拟合每个读 sector 的时间成本。两种目标物理行距都须与校准行距分离，当前拟合范围为第二窗口从第 16 列开始、扫描 40 遍、行数在 75% 至 85% 的参考规模内。不同输入使用相同系数。输出同时保留固定 48 元素行距探针给出的两档粗基线；超出跨行宽模型范围时，`primary_model` 标明采用粗基线。两种方法均为真机时间估计，L2 miss 数单独记为未判定。[预测入口](prior_forecast.py) · [跨行宽模型](width_transfer.py) · [固定探针基线](device_cost.py) · [两卡的事前验证及误差](../../../PADGain_results/results/unseen_width_20261122T131402Z/report.md)

## 4060 Ti 的重访诊断

独立探针先读地址 A，插入一组已知地址，再读 A。默认模式比较同一 sector、相邻 sector 和稀疏步长。`--pressure` 在相同数量与跨度下比较等距和扰动地址。`--counter-check` 在同一次单 sector 加载上核对设备时钟与 L2 读 miss。`--matched-counter` 在同一分配上分别重新准备两次读取，交替执行顺序。用 NCU 时只选 `probe_read_anchor`，让配对的另一次读取正常执行。编译时给探针单独的输出目录。[探针源码](calibrate_l2.cu)

```powershell
$probe = & ./tools/build.ps1 -Arch sm_89 -SourceFile ./experiments/dual_window/calibrate_l2.cu -BuildDir (Join-Path $env:TEMP 'PADGain_l2_calibration') | Select-Object -Last 1
& $probe --pressure
```

CPU 端可以重放已选两种布局，输出每条 warp 加载的缺失 sector 分布。选组、相联度及 warp 交错均作为显式的待检验条件。时间诊断读取目标机器码，核对两次加载的共同依赖，再计算该 warp 内的数据就绪时间范围。[缓存重放](cache_estimate.py) · [地址压力评分](score_pressure.py) · [冷热及计数器评分](assess_calibration.py) · [指令依赖诊断](warp_timing.py) · [本轮数据和核验报告](../../../PADGain_results/results/double_window_cache_calibration_20261122T090433Z/report.md)
