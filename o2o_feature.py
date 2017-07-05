# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import date

DATA_DIR = "D:/Code/data/O2O/"
TRAIN_FILE = "ccf_offline_stage1_train.csv"
TEST_FILE = "ccf_offline_stage1_test_revised.csv"
ON_TRAIN_FILE = "ccf_online_stage1_train.csv"

train = pd.read_csv(DATA_DIR+TRAIN_FILE, header=None)  # 1754884 * 7
train.columns = ["user_id", "merchant_id", "coupon_id", "discount_rate", "distance", "date_received", "date"]
test= pd.read_csv(DATA_DIR+TEST_FILE, header=None)  # 113640 * 6
test.columns = ["user_id", "merchant_id", "coupon_id", "discount_rate", "distance", "date_received"]
# on_train = pd.read_csv(DATA_DIR+ON_TRAIN_FILE, header=None)  # 11429826 * 7
# on_train.columns = ["user_id","merchant_id","action","coupon_id","discount_rate","date_received","date"]


# 训练数据分析
# # 1.优惠券接受日期
# train_date_received = list(train["date_received"].unique())
# train_date_received.remove("null")
# train_date_received = sorted(train_date_received)
# print(train_date_received[0])  # 开始日期是20160101
# print(train_date_received[-1])  # 结束日期是20160615
# print(len(train_date_received))  # 共167天数据
# print((train["date_received"]=="null").sum(), (train["date_received"]=="null").sum()/train.shape[0])  # date_received字段共701602条缺失, 缺失率为40%


feature1 = train[(train.date>="20160101")&(train.date<="20160413") | ((train.date_received>="20160101")&(train.date_received<="20160413")&(train.date=="null"))]  # 995240
dataset1 = train[(train.date>="20160414")&(train.date<="20160514")]  # 147745
feature2 = train[(train.date>="20160201")&(train.date<="20160514") | ((train.date_received>="20160201")&(train.date_received<="20160514")&(train.date=="null"))]  # 812779
dataset2 = train[(train.date>="20160515")&(train.date<="20160615")]  # 194432
feature3 = train[(train.date>="20160315")&(train.date<="20160630") | ((train.date_received>="20160315")&(train.date_received<="20160630")&(train.date=="null"))]  # 1036975
dataset3 = test  # 113640

# 预测集3最近一个月内每个用户领取优惠券的总和
t = dataset3[["user_id"]]
t['this_month_user_receive_all_coupon_count'] = 1
t = t.groupby("user_id").sum().reset_index()

# 同一用户领取同一优惠券的次数
t1 = dataset3[["user_id","coupon_id"]]
t1["this_month_user_receive_same_coupon_count"] = 1
t1 = t1.groupby(["user_id","coupon_id"]).sum().reset_index()

# 一个用户领取多张同一优惠券的最早最晚日期
t2 = dataset3[["user_id","coupon_id","date_received"]]
t2.date_received = t2.date_received.astype("str")
t2 = t2.groupby(['user_id','coupon_id']).agg(lambda x:":".join(x)).reset_index()  # aggregate
t2["receive_number"] = t2.date_received.apply(lambda s:len(s.split(":")))
t2 = t2[t2.receive_number>1]
t2["max_date_received"] = t2.date_received.apply(lambda s:max([int(d) for d in s.split(":")]))
t2["min_date_received"] = t2.date_received.apply(lambda s:min([int(d) for d in s.split(":")]))
t2 = t2[["user_id","coupon_id","max_date_received","min_date_received"]]

# 这条记录是本月第一次/最后一次接收优惠券
def is_firstlastone(x):
    if x==0:
        return 1
    elif x>0:
        return 0
    else:
        return -1 #those only receive once
t3 = dataset3[['user_id','coupon_id','date_received']]
t3 = pd.merge(t3, t2, on=['user_id','coupon_id'],how='left')
t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received
t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received - t3.min_date_received
t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(is_firstlastone)
t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(is_firstlastone)
t3 = t3[['user_id','coupon_id','date_received','this_month_user_receive_same_coupon_lastone','this_month_user_receive_same_coupon_firstone']]

# 用户对应日期接收的优惠券的数量
t4 = dataset3[['user_id','date_received']]
t4['this_day_user_receive_all_coupon_count'] = 1
t4 = t4.groupby(['user_id','date_received']).agg('sum').reset_index()

# 用户对应日期接收的同一张优惠券的数量
t5 = dataset3[['user_id','coupon_id','date_received']]
t5['this_day_user_receive_same_coupon_count'] = 1
t5 = t5.groupby(['user_id','coupon_id','date_received']).agg('sum').reset_index()

t6 = dataset3[['user_id','coupon_id','date_received']]
t6.date_received = t6.date_received.astype('str')
t6 = t6.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
t6.rename(columns={'date_received':'dates'},inplace=True)


def get_day_gap_before(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(date_received[0:4]), int(date_received[4:6]), int(date_received[6:8])) - date(int(d[0:4]), int(d[4:6]), int(d[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)


def get_day_gap_after(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(d[0:4]), int(d[4:6]), int(d[6:8])) - date(int(date_received[0:4]), int(date_received[4:6]), int(date_received[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)
t7 = dataset3[['user_id','coupon_id','date_received']]
t7 = pd.merge(t7,t6,on=['user_id','coupon_id'],how='left')
t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
t7 = t7[['user_id','coupon_id','date_received','day_gap_before','day_gap_after']]

other_feature3 = pd.merge(t1,t,on='user_id')
other_feature3 = pd.merge(other_feature3,t3,on=['user_id','coupon_id'])
other_feature3 = pd.merge(other_feature3,t4,on=['user_id','date_received'])
other_feature3 = pd.merge(other_feature3,t5,on=['user_id','coupon_id','date_received'])
other_feature3 = pd.merge(other_feature3,t7,on=['user_id','coupon_id','date_received'])
other_feature3.to_csv('data/other_feature3.csv',index=None)
print(other_feature3.shape)