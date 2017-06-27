# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analyse.basic import *

train = pd.read_csv("./data/train.csv")
shape = train.shape

# # y的分布情况
# # 1.散点图
# y = train["y"].sort_values(ascending=True)
# plt.scatter(np.arange(shape[0]), y)
# plt.xlabel("Index")
# plt.ylabel("y")
# plt.show()
# # 2.柱状图
# sns.distplot(train["y"], bins=50)
# plt.show()

# # 特征的整体类型
# dtype = train.dtypes.reset_index()
# dtype.columns = ["name", "type"]
# print(dtype.groupby("type").count().reset_index())

# 缺失情况
missing(train)
print(train["X0"].unique())