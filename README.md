# Reproduce SuperMimic with LAFAN

**维护者：Jianglin Yan**
**本地整理日期：2026-10-28**

这是 SuperMimic 论文的本地复现项目，包含训练实现、LAFAN1/BONES-SEED 数据处理代码、实验评测脚本和复现笔记。当前算法实现基于第三方独立复现项目 `CuboiLeo/SuperMimic-Reproduced` 的固定版本整理；本项目不是论文作者发布的官方代码。来源、版本和许可说明见 [ATTRIBUTION.md](ATTRIBUTION.md) 与 [LICENSE](LICENSE)。

> 仓库只收录核心代码和文档。动作数据、模型权重、训练日志、缓存、论文 PDF 和生成的视频/图像不包含在 Git 上传内容中。数据集需按其原始许可单独获取。

## 当前内容

- 基于 mjlab / MuJoCo Warp 的 G1 运动模仿环境与训练循环。
- 可微分 G1 正向运动学、动力学世界模型和 imagination/BPTT 策略训练。
- LAFAN1 与 BONES-SEED 数据转换、筛选、紧凑库构建脚本。
- 评测、回放、漂移诊断及仿真速度脚本。
- 上游实现的实验记录，以及本地复现规格、4090 环境准备和数据来源笔记。

该仓库不包含本地训练输出，因此不在此报告本地 4090 的最终成功率或论文复现结论。`docs/EXPERIMENTS.md` 中保留的历史实验记录来自上游候选项目；请勿将其误认为 Jianglin Yan 在本机上的实验结果。

## 环境

依赖锁定在 Python 3.12 与 `uv.lock`。项目配置固定了 PyTorch CUDA、MuJoCo、Warp 和 mjlab 版本。Ubuntu + RTX 4090 环境建议使用项目独立 Python 环境，不需要修改系统 Python 或 CUDA 驱动。

```bash
uv sync --group dev
```

如果环境中 `PYTHONPATH` 指向其它项目，可在命令前使用 `env -u PYTHONPATH`。

## 数据准备

数据不随仓库分发。下载和处理前，请分别核对数据源当前许可及访问条款；特别是 BONES-SEED 需要接受数据集条款，LAFAN1 数据及其重定向数据也不能默认再分发。

```bash
uv run python scripts/data/csv_to_npz_batch.py --help
uv run python scripts/data/filter_bones_seed.py --help
uv run python scripts/data/build_library.py --help
```

本地准备记录及数据来源见 [`docs/local/SOURCES_AND_DATA.md`](docs/local/SOURCES_AND_DATA.md)。

## 训练与评测入口

以下示例展示入口形式；先准备对应数据，再根据显存和数据量调整环境数、批大小及运行时长。

```bash
# 单卡 LAFAN1 示例
# 将 data/npz/lafan1 替换为已处理数据所在目录
env -u PYTHONPATH uv run sm-train \
  --run-name lafan1 \
  --data.npz-dir data/npz/lafan1 \
  --env.num-envs 4096 \
  --policy-trainer.batch-size 4096 \
  --schedule.max-hours 3

# 查看实际参数
uv run sm-train --help
uv run python scripts/eval_benchmark.py --help
uv run python scripts/play.py --help
```

训练日志、权重与转换后的数据应保存在本地运行目录，并由 `.gitignore` 排除。不要将数据集、检查点或生成媒体文件提交到本仓库。

## 项目结构

```text
src/supermimic/       环境、数据管线、FK、世界模型、策略与训练实现
scripts/              数据准备、训练辅助、评估、回放和诊断脚本
tests/                数值一致性与环境冒烟测试

docs/                 上游实现文档与实验记录
docs/local/           本地论文规格、数据来源、路线和 4090 环境笔记
```

## 复现笔记

- [论文规格](docs/local/PAPER_SPEC.md)
- [复现路线](docs/local/REPRODUCTION_ROADMAP.md)
- [代码与数据来源](docs/local/SOURCES_AND_DATA.md)
- [4090 环境快照（2026-10-27）](docs/local/REMOTE_4090_BASELINE_2026-10-27.md)：历史检查记录，不代表当前机器实时状态
- [上游实验日志](docs/EXPERIMENTS.md)：其中结果属于来源项目，不是本地运行结果

## 许可与署名

本地维护者和整理日期见 [AUTHORS.md](AUTHORS.md)。保留来源代码的 MIT 许可及原始版权声明；具体实现、方法论文和数据集各自的署名与许可不同，详见 [ATTRIBUTION.md](ATTRIBUTION.md)。
