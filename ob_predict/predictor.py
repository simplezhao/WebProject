#!/usr/bin/python
# -*- coding: UTF-8 -*-

#import MySQLdb

import datetime
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
import os
from matplotlib.pyplot import figure
from fastprogress import master_bar, progress_bar
import torch.nn as nn
import time 
from sklearn.metrics import mean_squared_error
import warnings
import itertools
from kalmanfilter import KalmanFilter

sns.set_style("darkgrid") 
warnings.filterwarnings("ignore")

class Predictor:
    def __init__(self, state_transition=1, process_noise=1, observation_model=1, observation_noise=5):
        # mysql db
        #self.db = MySQLdb.connect("localhost", "testuser", "test123", "TESTDB", charset='utf8' )

        self.kf = KalmanFilter(state_transition=state_transition, process_noise=process_noise,
           observation_model=observation_model, observation_noise=observation_noise)

    def preprocessing(self, ds):
        ds['进站时间'] = pd.to_datetime(ds['进站时间'])
        df = ds.set_index('进站时间')

        min_day = df.index.min().date().strftime('%Y-%m-%d')
        max_day = df.index.max().date().strftime('%Y-%m-%d')

        timl = pd.date_range(min_day, max_day, freq='1D')
        names = set(list(df['进站名称']))
        pairs = [i for i in itertools.combinations(names, 2)]
        result = {}
        print("processing data...")
        for dt in timl:
            record = dict.fromkeys(pairs , 0)
            fg = df[df.index.date == datetime.date(dt.year, dt.month, dt.day)]
            for n in names:
                x = fg[fg['进站名称'] == n].groupby('出站名称').count()['用户ID']
                if len(x) > 0:
                    for m,j in zip(x.to_frame().index, x):
                        if (m, n) in record:
                            record[(m,n)] += j
                        elif (n, m) in record:
                            record[(n,m)] += j
            result[dt] = record

        multiple_time_series = {}
        for k, v in result.items():
            for ke, ve in v.items():
                if ke not in multiple_time_series:
                    multiple_time_series[ke] = []
                multiple_time_series[ke].append(ve)
        return timl, multiple_time_series

    def do_predict(self, ob_pair, last_days=7, predict_days=3, sql_cmd="", mode="predict_all", output_dir="./output/"):
        """
        cursor = self.db.cursor()
        cursor.execute(sql_cmd) # SQL语句，用户自定义
        data = cursor.fetchone()
        # 做一些逻辑处理
        # ...
        # 数据格式pandas：id, 进站名称, 进站时间, 出战名称, 出站时间, 渠道编号, 价格
        """
        ds = pd.read_csv("trips.csv", encoding='gb18030')
        timl, multiple_time_series = self.preprocessing(ds)
        if ob_pair in multiple_time_series:
            pdata = multiple_time_series[ob_pair]
        else:
            ob_pair = (ob_pair[1], ob_pair[0])
            pdata = multiple_time_series[ob_pair]
        if mode == "predict_all":
            pred = []
            y = []
            index = timl[-(len(pdata)-last_days-1):]
            for i in range(len(pdata) - last_days - 1):
                x = self.kf.predict(pdata[:last_days + i], 1)
                r = x.observations.mean[0]
                pred.append(r)
                y.append(pdata[last_days + i + 1])
            plt.plot(index, y, 'b-', label="observation")
            plt.plot(index, pred, 'r-', label="predict")
            plt.title('Predict OD [{} and {}]'.format(ob_pair[0], ob_pair[1]),fontsize=21)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.ylabel('Passageer Volume', fontsize=18)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            plt.legend()
            plt.savefig(output_dir + '/predict_image_all.jpg')
        elif mode == "predict_future":
            if predict_days > len(pdata):
                raise Exception
            new_stamps = []
            for i in range(1, predict_days+1):
                new_stamps.append(timl[-1] + datetime.timedelta(days=i))
            x = self.kf.predict(pdata, predict_days)
            r = 0.5 * np.array(x.observations.mean) + 0.5 * np.array(pdata[-3:][::-1])
            plt.plot(timl, pdata, 'r-', label="observation")
            plt.plot(new_stamps, r, 'b-', label="prediction")
            plt.title('Predict OD [{} and {}]'.format(ob_pair[0], ob_pair[1]), fontsize=21)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.ylabel('Next {} Days Passageer Volume'.format(predict_days), fontsize=18)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            plt.legend()
            plt.savefig(output_dir + '/predict_image_future.jpg')
        else:
            raise Exception("Not support yet.")

if __name__ == "__main__":
    p_instance = Predictor()
    p_instance.do_predict(ob_pair=('Sta134', 'Sta63'))
