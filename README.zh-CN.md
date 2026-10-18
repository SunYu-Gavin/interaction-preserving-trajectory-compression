# 多智能体轨迹数据库的交互保真压缩

[English](README.md)

本项目提供面向自动驾驶场景库的CPU轨迹压缩研究实现。在约束同步位置误差的同时，保持原始共同有效采样时刻上的声明距离阈值关系及其连续区间。

## 快速运行

使用Python 3.12及以上版本，在仓库根目录执行：

```bash
python -m venv .venv
# 按当前终端的方式激活虚拟环境。
python -m pip install -e ".[test]"
python -m pytest -q
python -m iptc.cli import --input examples/synthetic.csv --output runs/raw.npz --ticks-per-second 10
python -m iptc.benchmark --config configs/demo.yaml
python -m iptc.cli query runs/demo/rec0_eps0.1_rep0_FULL-DAG-BYTE.npz --intervals
```

Windows PowerShell可使用`.\.venv\Scripts\Activate.ps1`激活环境；也可直接用`.\.venv\Scripts\python.exe`替代上述`python`。

示例是程序生成的90条合成观测，仅演示接口，不属于论文真实数据实验。重新运行请使用新的输出文件名或目录，程序不会覆盖已有压缩文件。

## 原方法与字节择优扩展

- **RAW-DAG**：从原始轨迹开始进行条件路径优化。
- **FULL-DAG**：从修复后的几何简化表示开始进行条件路径优化。
- **FULL-DAG-BYTE**：保留原FULL-DAG，再增加两轮实际字节择优；考虑等点数及比条件最少点路径多一个锚点的候选，仅接受总文件字节减少的更新。

在原12个录制或场景、四档误差组成的48组配置中，FULL-DAG相对强基线GEO-GREEDY的中位存储节省为1.108%；扩展版为1.986%，两者均为48/48组文件更小。扩展版相对原FULL-DAG进一步节省的中位数为0.866%。扩展通过额外搜索换取存储收益；其完整编码、查询和内存尚无同口径的受控对照。

发布版包含14种核心方法、数据导入、解码与查询、批量运行、功能测试和结果表。完整说明见：

- [方法定义及保证](docs/METHODS.md)
- [数据格式与获取入口](docs/DATA.md)
- [实验复现](docs/REPRODUCE.md)
- [已有结果与统计口径](results/README.md)

真实数据来自CitySim以及InterHub中的nuPlan、Lyft，不含正式INTERACTION数据。仓库不分发原始轨迹、缓存或可还原真实轨迹的压缩文件；请自行获取数据。论文稿件与历史研究过程记录不随代码发布。

本快照暂不附带LICENSE文件。
