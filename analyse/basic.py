# -*- coding: utf-8 -*-

__all__ = ["missing"]

import pandas as pd

def missing(df):
    df_shape = df.shape
    df_samples = df_shape[0]
    df_columns = df_shape[1]
    missing_df = df.isnull()
    missing_columns = missing_df.sum(axis=0)
    missing_columns = missing_columns[missing_columns > 0].sort_values(ascending=False)
    if len(missing_columns) == 0:
        print("没有任何缺失数据")
        return
    else:
        missing_col_count = len(missing_columns)
        print("有{0}列数据缺失，占总列数的{1}%".format(missing_col_count, round(missing_col_count/df_columns, 6)))
        print("缺失的列为: {}".format(missing_columns.index.values))
    missing_samples = missing_df.sum(axis=1)
    missing_samples = missing_samples[missing_samples > 0]
    missing_spl_count = len(missing_samples)
    print("有{0}行数据缺失，占总行数的{1}%".format(missing_spl_count, round(missing_spl_count/df_samples, 6)))
    return