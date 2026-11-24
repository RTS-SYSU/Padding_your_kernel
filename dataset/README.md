# PADGain-DualWindow v0.1

这是一套用于研究**行尾补量如何改变同一 GPU kernel 的缓存行为和耗时**的合成微基准数据集。一个输入定义两段要读的列及矩阵尺寸规则。一条真机观测记录一次运行中原布局与补后布局的成对测量。同一输入可以在多次运行和多张卡上留下多条观测。

## 打开数据

| 文件（相对于解压目录） | 内容 |
|---|---|
| [workloads.jsonl](workloads.jsonl) | 197 条输入定义，每行一个 JSON 对象，主键为 `case_id` |
| [measured.jsonl](measured.jsonl) | 两侧都有原始计时和输出校验的成对观测，主键为 `observation_id`，用 `case_id` 关联输入 |
| [reported.jsonl](reported.jsonl) | 用户提供的两条 RTX 3060 汇总，保留报告精度 |
| [coverage.json](coverage.json) | 各设备观测数、不同输入数、可用计数器数及逐输入覆盖情况。某输入对应空对象表示待测 |
| [example/metadata.json](example/metadata.json) | 依据 V100 的 `original` 实测尺寸重建的完整矩阵，包括两种物理布局及对应观测 ID（见同目录的 `source.json`） |
| [evidence_index.jsonl](evidence_index.jsonl) | 每份证据的包内路径、原存储根类别、原相对路径、字节数和 SHA-256；文件内容在 `evidence/` |
| [folds.json](folds.json) | 按补后物理行距留出的五组输入 |
| [sha256.json](sha256.json) | 发布包内数据、文档与工具的文件指纹 |

197 个输入中，39 个已有原生真机计时，另 158 个可生成、待补测。下表按设备型号统计，同一个输入的跨批次复测会增加观测条数。

| 设备型号 | 有原生计时的不同输入 | 原生计时对 | 有单 pass L2 计数的计时对 | 仅有用户汇总 |
|---|---:|---:|---:|---:|
| RTX 4060 Ti | 39 | 49 | 14 | 0 |
| Tesla V100-PCIE-32GB | 39 | 43 | 0 | 0 |
| RTX 3060 | 0 | 0 | 0 | 2 |

## 一个输入长什么样

输入是一张按行存放的 `uint32` 矩阵。`W` 是逻辑列数，`G` 是第二窗口起始列，列号从 0 开始。每次读取同一行的 `0…7` 和 `G…G+7`。一个八线程小组分担一行，每个线程取两个元素。每个目标 kernel 启动一次，每个线程以相同顺序扫描自己负责的行 40 遍。

例如 `family-104-16-80` 表示 104 个逻辑列、读取第 0–7 和 16–23 列、补后地址覆盖的目标比例为标称 L2 容量的 80%。`F=80` 指用来确定行数的比例，80% 也不等于硬件测得的占用率或命中率。给定 L2 字节数 `C`，行数为 `256 × floor(C × F / (100 × 128 × 256))`。对同一个输入 ID，GPU 容量不同，实际行数可以不同。原布局行距为 104，补后为 128。两个矩阵的前 104 列逐元素相等，计算、逻辑读取序列和读取量一致。

128 字节相当于 32 个四字节元素。按现有实验规则取 `p=(-W) mod 32`，使每行起点对齐，再比较 `pad=0` 与 `pad=p`。基准程序把每张表的申请量向上取整至 2 MiB，按两种布局独立核算空间。每侧计时前按相同协议读取两倍 L2 容量的缓冲区，再预热一遍。普通 CUDA event 计时只覆盖目标 kernel。实测加速比为 `base_ms / pad_ms`，大于 1 表示补后较快。两个窗口读数使用 `ld.global.cg.u32`，按 PTX 缓存策略绕过 L1 数据缓存，在 L2 缓存。本版覆盖这种加载策略的实验。[NVIDIA PTX 缓存操作符](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)

| 输入来源（`design_stratum`） | 选择条件 | 数量 |
|---|---|---:|
| `main_grid` | W 从 104 至 216、步长 8，保留需要补齐的 12 种宽度；G 取 8、16、24；F 取 76、78、80、82、84 | 180 |
| `capacity_controls` | W 取 104、136；G 取 8、16、24；F 取 40、100。在相同宽度和窗口下比较较小覆盖量与标称容量附近的情况 | 12 |
| `historical_81` | F=81，历史 (W,G) 为 (104,24)、(112,16)、(176,16) | 3 |
| `legacy_56` | W=56，G=24；`original` 取 F=75，`holdout` 取 F=80 | 2 |

## 如何读观测

| 字段 | 定义 |
|---|---|
| `logical_words`、`second_window_words`、`pad_words` | W、G、每行的补量，单位都是 4 字节元素 |
| `l2_percent` / `target_l2_percent` | 输入表 / 观测表中的 F，单位为百分数 |
| `base_mean_128b_blocks_per_row`、`pad_mean_128b_blocks_per_row` | 对 128 字节对齐表首，在一个完整的行首偏移周期内统计两个窗口每行触及的不同 128 字节地址块，再取平均；用于研究的事前地址特征 |
| `gpu_name`、`l2_bytes`、`rows`、`blocks`、`threads`、`rounds` | 设备型号、标称 L2 字节数、实际行数、线程块数、每块线程数、目标扫描遍数 |
| `gpu_uuid` | 有协议或监控记录绑定时填入物理 GPU 标识。V100 记录至少来自两张同型号卡，跨物理卡分析时保留该字段 |
| `base_ms`、`pad_ms`、`speedup` | 两侧原生计时（毫秒）与配对比值；报告级数据可只有比值 |
| `base_verified`、`pad_verified` | 对应布局的逻辑输出与 CPU 参考是否一致 |
| `base_l2_read_miss_sectors`、`pad_l2_read_miss_sectors` | 两侧 L2 读查找 miss 的 sector 数。原始 NCU 名称为 `lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum` |
| `base_reported_l2_read_misses`、`pad_reported_l2_read_misses` | 3060 原表标为「L2 读 miss」的读数，未提供计数器原名、单位或采样配置 |
| `label_status`、`counter_status`、`evidence` | 原生核验或用户报告、计数依据、证据文件映射。JSON 的 `null` 表示缺失 |

原生计时来自 14 份带协议、冻结计划和评价记录的运行。纳入这些运行的全部成对目标结果。同一配置在不同批次的记录各自保留。设备校准探针、profiler 的耗时、事后重复性探针和镜像文件不充当新的标签。匹配的 `one_pass_ncu` 计数来自另一次启动，各布局分别单 pass 采集。在 `unavailable` 的记录上仍可分析时间收益。3060 的行数、加速比和 L2 读 miss 取自用户给出的表，两侧原始时间、校验和 profiler pass 数未知。

例如在 4060 Ti 的 `row_window_family_20261120T060326748635Z` 批次中，下列两个输入都有 209,664 行、第二窗口均从第 16 列开始、每行均补 24 个元素。两组补前每行平均覆盖 1.5 个 128 字节块，补后都为 1 个。完整行距仍有区别，同一批次的真实结果可按下表复查。

| 原行宽 → 补后行距 | 原布局 / 补后时间 (ms) | 加速比 | 原布局 / 补后 L2 读 miss (sector) |
|---|---:|---:|---:|
| 104 → 128 | 2.890304 / 0.594944 | 4.858× | 13,286,074 / 416,474 |
| 136 → 160 | 1.447712 / 0.792576 | 1.827× | 4,956,065 / 1,547,020 |

## 重建与核验

解压后，在数据包根目录运行下列命令。CPU 工具使用 Python 3.11+ 和 NumPy。第一条命令核验指纹、跨表关系、时间比值和样例矩阵。第二条按 V100 的 6 MiB L2 参数生成一对指定输入。生成目录须为新目录。

```powershell
python -B tools/verify_dataset.py .
python -B tools/workloads.py materialize family-104-16-80 --l2-bytes 6291456 --output sample_104_v100
```

`logical_base.npy` 保存原始逻辑矩阵，`padded.npy` 保存同样的列及行尾补区，两者均为小端四字节无符号整数、按行连续排列。逻辑值由行号与列号确定，补区填入 `0xdeadc0de`。NPY 文件给出已初始化的表内容，CUDA 额外分配的对齐空间记在设备计划中。可使用 `--rows 4` 替换 `--l2-bytes`，生成便于逐行查看的小样例。

在配置好 CUDA 和 MSVC 的 Windows 主机上可用包内源码复现目标入口。下例面向 RTX 4060 Ti，V100 使用支持 `sm_70` 的工具链，RTX 3060 桌面卡使用 `sm_86`。第一条产生所用二进制路径。第二条只打印计划，第三条运行同一目标两侧，分别输出记录。kernel 在设备上按同一公式初始化输入，运行时不需要加载 NPY。历史证据的源码版本以各运行记录的 `bench_cu_sha256` 为准。

```powershell
$bench = & ./tools/build.ps1 -Arch sm_89 | Select-Object -Last 1
& $bench info family-104-16-80
& $bench both family-104-16-80
```

## 用于模型评价时

若问题是“换一个未参与拟合的行距，模型还能否预测收益”，可轮流留出 `W+p` 为 64、128、160、192、224 的整组输入。每轮只在有原生标签的测试输入上评分；其余行距中有标签的输入用于拟合。这样，同一输入的两种布局、同一补后行距的其他配置、其他设备和各次复测都会留在该组。按设备及输入分别列出误差，并报告不同输入数和运行数，避免将复测当作更多独立几何。历史名称 `holdout` 本身不代表对新模型的盲测。

这版可用于复核双窗口的地址覆盖、观察真机波动、检验给定补量的收益预测。原生计时集中在历史选过的容量附近，只有部分输入有硬件计数。数据包含合成输入和 NVIDIA GPU 记录；有关真实应用分布、其他加载策略和其他架构的结论，需要对应的应用输入与测量。

## 学术做法与本版选择

- Spatter 把 gather/scatter 的下标与步长等访问参数写成可重放配置，并交代哪些模式来自应用轨迹。这里保留能重建访问的 W、G、F 和 kernel，明确记录合成来源。[Spatter 论文](https://doi.org/10.1145/3422575.3422794) · [作者的 JSON 示例](https://hpcgarage.github.io/spatter/)
- Filipovič 等公开了五个 CUDA 基准在多张 GPU 上的实测时间和硬件计数器。这里借鉴其逐设备保存观测的方式，保留两侧时间、计数单位及原始日志。PADGain 的补量仍在计时前由地址条件直接确定。[KTT 公开数据集](https://data.mendeley.com/datasets/nn53dskr7z/1)
- TenSet 在成本模型评价中把测试网络涉及的任务整体留出。这里采用类似的分组原则，具体按补后行距分组是本数据集的设计。[TenSet 论文 §5.2](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/a684eceee76fc522773286a895bc8436-Paper-round1.pdf)
