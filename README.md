# PJ2 图像分类

复旦大学智能信息处理课程项目，组号 **2-2**。在 Food-101 六类子集和 Imagenette 320px 上，使用全局颜色特征和局部特征分别完成分类。每套数据使用相同的官方划分和统一的评估口径，对照两种特征的结果。

**提交截止** 2026 年 12 月 2 日 23:59（北京时间）。组 2-2 的课堂汇报安排在 PJ4。

## 数据与实验协议

| 数据集 | 固定划分 | 类别 |
| --- | --- | --- |
| Food-101 课程六类子集 | `meta/train.txt` 和 `meta/test.txt`，每类 750/250 张，共 4500/1500 张 | sushi、tacos、takoyaki、tiramisu、tuna_tartare、waffles |
| Imagenette 320px | 课程包原有 `train/` 和 `val/`，合计 9469/3925 张；将 `val/` 作为最终测试集 | tench、English springer、cassette player、chain saw、church、French horn、garbage truck、gas pump、golf ball、parachute |

从 Elearning 下载两份课程包 `PJ2--food_101_dataset.tgz` 和 `PJ2--imagenette2-320.tgz`。照片由各成员从课程或数据集发布渠道获取。来源见 [ETH Zurich Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) 和 [fastai Imagenette](https://github.com/fastai/imagenette)（[320px 发布包](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz)）。优先使用 Elearning 同名课程包，以匹配校验值。课程包的 SHA-256 见 `code/data/archives.sha256`。

在仓库根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
python code/setup_data.py "/存放两个课程数据压缩包的目录"
python code/prepare_splits.py
```

成功后生成 `code/data/raw/food-101-coursework-6classes/` 和 `code/data/raw/imagenette2-320/`。原图压缩包及解包目录保存在各成员本地。数据准备脚本先校验压缩包，再解包到约定路径。第 1 章的已核对清单、类别映射和审计结果位于 `code/results/data_audit/`。后续模块各自提供复现命令。

两条实验路线分别使用 **HSV 全图颜色直方图 → LinearSVC** 和 **SIFT 局部描述子 → K-means 视觉词袋（BoVW）→ LinearSVC**。两套数据分别训练，共形成四组主实验。分类器使用相同的训练集内分层验证策略、C 候选（0.1、1、10）和随机种子 42。BoVW 词典只用训练样本；验证中如调整 K 或分类器，必须在每个训练折学习词典和变换，再用于该折验证数据。最终测试集只用于最终评估。建议固定三折分层验证，按 Macro-F1 选参数。BoVW 可先固定 K=128，每张图最多保留 128 个 SIFT 描述子，并限制聚类样本量以控制运算成本。统一报告 Accuracy、Macro-F1、逐类指标和混淆矩阵，记录实际提取耗时及失败样本处理规则。

## 六人分工

| 顺序与建议交付 | 组员 · 报告章节 | 读取 | 编写与实验 | 交付 |
| --- | --- | --- | --- | --- |
| 1 · 11 月 19 日 | 卢远卓 · 数据与划分 | 两个课程包及官方划分 | `prepare_splits.py`：生成两套清单，核对类别、图片解码、训练/测试交集，分析每类数量和尺寸分布 | 清单、类名映射、审计表及分布图，第 1 章 |
| 2 · 11 月 21 日 | 于智博 · 全局颜色特征 | 第 1 章清单和原图 | `extract_color.py`：计算 HSV 联合直方图与 L1 归一化；对比两种分箱配置的维数、耗时和特征示例 | 两套数据的颜色特征缓存、分箱配置与直方图示例，第 2 章 |
| 3 · 11 月 23 日 | 竺钱成 · 全局特征分类 | 第 1 章清单和第 2 章颜色特征 | `classify_color.py`：训练 LinearSVC，在训练集内选 C，评估两套数据；提供共用评分函数 | 全局预测 CSV、指标表、混淆矩阵及分类设置，第 3 章 |
| 4 · 11 月 25 日 | 张帅宇 · 局部描述子 | 第 1 章清单和原图，第 3 章确认的评估口径 | `extract_sift.py`：提取 SIFT 的 128 维描述子；比较关键点数量、无关键点比例和提取时间，展示代表图 | 描述子索引与缓存、关键点示例和数量统计，第 4 章 |
| 5 · 11 月 28 日 | 杨晓芳 · 局部特征分类 | 第 4 章描述子、第 1 章清单及第 3 章评分函数 | `classify_bovw.py`：每套数据各自用训练描述子拟合词典，量化并归一化词频，用同口径 LinearSVC 分类；记录词典规模和训练开销 | 局部预测 CSV、指标表、混淆矩阵和词袋设置，第 5 章 |
| 6 · 11 月 30 日 | 郑周锐 · 对比与误差分析 | 第 3、5 章对齐的测试预测、特征耗时和配置 | `compare_results.py`：校验样本一对一对应，比较两条路线的总体和逐类指标，统计互有胜负的样本并复核典型错例 | 四组结果汇总、对比图、错例及结论，第 6 章 |

每人交付可运行脚本、小体积统计表和代表图，以及同一 Word 对应章节。章节至少包含实验方法、关键函数原理和代码、实测结果、分析、运行命令。大型特征缓存与原图保存在各自本地，后续成员通过清单及脚本重建。

## 接口与交接

- 使用第 1 章交付的 `food101_split.csv` 和 `imagenette_split.csv`。每行含 `sample_id,split,path,label`，其中 `sample_id` 在同一数据集唯一，`path` 相对 `code/data/raw/`，`split` 统一为 `train` 或 `test`。
- 特征保存时保留 `sample_id`。预测 CSV 使用 `sample_id,y_true,y_pred`，每个最终测试样本恰好一行。比较脚本按 ID 对齐并验证真实标签。
- 实验脚本统一放在 `code/`，小体积实验产物放在 `code/results/`。每章结果记下运行命令、超参数、依赖版本、随机种子及测试样本数量。
- 接力顺序按分工表执行。上一位合并后，下一位同步最新 `main`，提交自己的代码、实验结果和对应 Word 章节。共享 Word 一次由一人编辑。

## 第 1 章核对结果

共验证 19,394 张图像，全部能解码。两个数据集均无跨划分的重复样本 ID 或相同文件字节。Food-101 的一张训练图和一张测试图解码后像素相同，详情见 `code/results/data_audit/exact_cross_split_matches.csv`。两条分类路线统一保留课程原划分。

## 报告与提交

组内维护一份汇总报告 `report/project2-report.docx`。各成员填写自己负责的章节，包含方法、所用函数的原理、关键代码、实测结果及分析。12 月 1 日复核四组实验结果。上传汇总 Word 时，按 Elearning 题面命名为 `Project2-学号-姓名-2-2.docx`，学号和姓名使用提交人信息。
