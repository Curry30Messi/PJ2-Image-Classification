第 2 章复现与交接

在仓库根目录执行（Python 3.12）
python -m pip install -r code/requirements.txt
python code/setup_data.py "E:/Downloads"
python code/extract_color.py

setup_data.py 仅首次解包时运行，已有 code/data/raw 时直接运行 extract_color.py。
使用第 1 章既有 CSV，不需要重新生成划分。原图不缩放。
脚本生成 8x8x8 与 16x8x8 两组 HSV 特征；每组均包含全部 train/test 行。
code/results/cache/color/ 下四个 NPZ 保留本地，由下一位通过脚本重建。
code/results/color/ 下统计、配置和示例图随代码提交。

读取示例
import numpy as np
with np.load("code/results/cache/color/food101_8x8x8.npz", allow_pickle=False) as data:
    train = data["split"] == "train"
    test = data["split"] == "test"
    X_train, y_train = data["X"][train], data["label"][train]
    X_test, y_test = data["X"][test], data["label"][test]
    test_ids = data["sample_id"][test]

X 为 float32，行顺序与 *_split.csv 完全一致；label 为原清单的类别字符串。
sample_id、split、label 与 X 的行一一对应；bins 记录 H/S/V 分箱数。
manifest_sha256 是输入清单的字节校验值。
例如 imagenette_16x8x8.npz 为 Imagenette 的 1024 维版本。
不要同时将两种配置拼接为主实验特征。
如需选分箱方案，应在训练集内联合验证分箱和 C，最终测试集留到选择结束后评估。
HSV 直方图不拟合跨图片统计量，可先缓存再按官方划分和交叉验证索引取行。

计时为本机单次遍历。读取、解码和 HSV 转换对两种配置共享，
summary.csv 的 combined_seconds 为共享阶段加对应直方图计算时间，
不含缓存压缩、图表生成；不可把两组 combined_seconds 相加当作总运行时间。
失败策略为报出 sample_id 并停止，不静默漏样本。
