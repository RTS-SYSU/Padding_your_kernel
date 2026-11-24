# Pavia University 最远光谱对搜索

对左上和中心各一个 40×40 区域执行 1,279,200 个像素对的距离计算。每个 warp 读取两条光谱的 103 个原生 uint16 波段。事前由 32 B sector 对齐推导最小补量 p=9，比较物理行距 103 与 112。两侧使用同一个 CUDA kernel。运行器用独立 CPU 枚举核对最远像素对和距离，分别采集 kernel 执行时间和硬件计数。

在仓库根目录运行。需要 CUDA、MSVC、Python 3.11 或更新版本、NumPy、SciPy、Nsight Compute。修改 `run.py` 中的 NCU 路径以适配本机版本。

```powershell
$out = Join-Path (Get-Location) ('results/pavia_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $out -Force
$url = 'https://raw.githubusercontent.com/gokriznastic/HybridSN/master/data/PaviaU.mat'
curl.exe -fL $url -o (Join-Path $out 'PaviaU.mat')
Set-Content -LiteralPath (Join-Path $out 'source-url.txt') -Value $url -Encoding utf8NoBOM
& ./tools/build.ps1 -Arch sm_89 -SourceFile ./experiments/pavia/bench.cu -BuildDir (Join-Path $out 'build')
python -B experiments/pavia/run.py freeze $out
python -B experiments/pavia/run.py measure $out
```

[历史实测报告和原始记录](../../../PADGain_results/results/pavia_current_theory_4060ti_20261124/report.md)
