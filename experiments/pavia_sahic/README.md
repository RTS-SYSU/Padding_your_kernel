# Pavia University 上的 SAHIC 六阶段实验

CUDA 阶段依据 Wu 等论文算法 1–2、图 2、图 4–5 及表 VIII 重建。论文完整可编译工程与中间数据未找到。全部输入来自公开影像和真实地物标注，空间尺寸 610×340，103 波段和 9 类。用固定种子的按类约 1% 抽样选出 423 个训练像素。使用 L1 多项逻辑回归和 RBF 特征，CPU 近端迭代求初始概率。外层在 CPU 重建三轮 ADMM，取第 3 轮各阶段的输入。FFT 采用周期边界，收缩按图 5 的标量阈值。V4 和 V5 分别测一次单类别平面启动。两维 launch 用 16×16，因为论文图使用方形 tile，未获得原始 launch 配置。该实验的输出以 NumPy 重建的对应阶段校验。

| Kernel | 事前选择依据 |
|---|---|
| q_kernel | 按元素更新 q，FP64 类别平面每行 207,400 元素，整除 4，p=0 |
| fixq_kernel | 按训练索引回写类别标签。读的标签与索引保持紧凑，稀疏写回未推出有益的非零候选 |
| nu_aux3_sum_kernel | 同一线程按类别扫一列，连续线程读相邻像素。FP64 平面行距整除 4，p=0 |
| Vs_kernel | 同一线程计算 V1、V2、V3 和 a，计算路径上的连续读已对齐，p=0 |
| V4_kernel | 输入为宽 610 的复数平面，复数占 16 B，宽整除每 sector 的 2 元素。写回的实数转置平面宽 340 也对齐，p=0 |
| V5_kernel | 宽 610 的 FP64 乘子数组每 sector 容纳 4 元素，周期内的奇数行错开 2 元素。使行距为 4 的倍数得最小 p=2。V5 和 D5 的双方向 FP64 数组统一采用行距 612，复数 FFT 平面保持原布局 |

## 复现

从仓库根目录运行，下列两个 MAT 参数分别指向 PaviaU.mat 和 PaviaU_gt.mat。实验目录需要为空。

```powershell
$out = Join-Path (Get-Location) 'results/sahic_new'
New-Item -ItemType Directory -Path $out -Force
python -B experiments/pavia_sahic/prepare.py <影像路径> <标注路径> $out
& ./tools/build.ps1 -SourceFile ./experiments/pavia_sahic/bench.cu -BuildDir (Join-Path $out 'build')
python -B experiments/pavia_sahic/run.py freeze $out
python -B experiments/pavia_sahic/run.py measure $out
```

[Wu 等原论文](https://www2.umbc.edu/rssipl/people/aplaza/Papers/Journals/2018.JSTARS.Adaptive.pdf) · [Pavia 数据发布页](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
