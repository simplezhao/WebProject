#!/usr/bin/python
# -*- coding: UTF-8 -*-

#import MySQLdb

import datetime
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import random
import os
from matplotlib.pyplot import figure
from fastprogress import master_bar, progress_bar
import torch.nn as nn
import time 
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error
import torch 
import warnings
from lstm import LSTM

sns.set_style("darkgrid") 
warnings.filterwarnings("ignore")

class Predictor:
    def __init__(self):
        # 加载模型
        self.lstm = torch.load('./best_model.pth', map_location=torch.device('cpu')) 
        self.lstm.eval()

        # mysql db
        #self.db = MySQLdb.connect("localhost", "testuser", "test123", "TESTDB", charset='utf8' )

        self.interval = 15 # 断面客流量统计间隔
        time_interal = pd.date_range('06:00', '23:00', freq='{}T'.format(self.interval))
        self.seq_length = len(time_interal)
        self.min_value = -3
        self.max_value = 620

    def data_processing(self, ds):
        ds['进站时间'] = pd.to_datetime(ds['进站时间'])
        df = ds.set_index('进站时间')
        dd_in = df.resample('{}min'.format(self.interval)).count()
        ds['出站时间'] = pd.to_datetime(ds['出站时间'])
        df2 = ds.set_index('出站时间')

        dd_out = df2.resample('{}min'.format(self.interval)).count()

        time_index = dd_out.index
        in_count = np.array(dd_in['用户ID'])
        out_count = np.array(dd_out['用户ID'])

        res = in_count[:2].tolist()
        cum_sum_in = sum(in_count[:2])
        cum_sum_out = 0
        flag = time_index[:-2][0].strftime('%Y-%m-%d')
        for t, inc, out in zip(time_index[:-2], in_count[2:], out_count[:-2]):
            tmp = t.strftime('%Y-%m-%d')
            if tmp != flag:
                cum_sum_in = 0
                cum_sum_out = 0
                flag = tmp
            cum_sum_in += inc
            cum_sum_out += out
            res.append(cum_sum_in - cum_sum_out)
        dd = pd.DataFrame(data={"客流量": np.array(res)}, index=time_index)
        dd = dd[(dd.index.hour >= 6) & (dd.index.hour <= 23)]
        dd = dd[dd.index.weekday < 5]
        data = np.asarray(dd['客流量'][-90*self.seq_length:])
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_normalized = self.scaler.fit_transform(data.reshape(-1, 1))
        x, y = self.sliding_windows(train_data_normalized, self.seq_length)
        dataX = Variable(torch.Tensor(np.array(x)))
        dataY = Variable(torch.Tensor(np.array(y)))

        return data, dataX, dataY


    def sliding_windows(self, data, seq_length):
        x = []
        y = []
        for i in range(len(data)-seq_length-1):
            _x = data[i:(i+seq_length)]
            _y = data[i+seq_length]
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y)

    def do_predict(self, sql_cmd="", mode="predict_future", output_dir="./output/"):
        """
        cursor = self.db.cursor()
        cursor.execute(sql_cmd) # SQL语句，用户自定义
        data = cursor.fetchone()
        # 做一些逻辑处理
        # ...
        # 数据格式pandas：id, 进站名称, 进站时间, 出战名称, 出站时间, 渠道编号, 价格
        """

        # 以读csv文件为例子
        ds = pd.read_csv("trips.csv", encoding='gb18030')

        data, dataX, dataY = self.data_processing(ds)

        if mode == "predict_all":
            train_predict = self.lstm(dataX.to('cpu'))
            data_predict = train_predict.cpu().data.numpy()
            dataY_plot = dataY.data.numpy()

            #data_predict = self.scaler.inverse_transform(data_predict)
            #dataY_plot = self.scaler.inverse_transform(dataY_plot)
            data_predict = data_predict * (self.max_value - self.min_value) + self.min_value
            dataY_plot = dataY_plot * (self.max_value - self.min_value) + self.min_value

            df_predict = pd.DataFrame(data_predict)
            df_labels = pd.DataFrame(dataY_plot)

            figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
            plt.plot(df_labels[0])
            plt.plot(df_predict[0])
            plt.legend(['Time Series','Prediction'],fontsize = 21)
            plt.suptitle('Predict Dataset',fontsize = 23)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.ylabel('Volume/{}min'.format(self.interval), fontsize=18)
            plt.xlabel('Date', fontsize=18)
            plt.savefig(output_dir + '/predict_image_all.jpg')

        elif mode == "predict_future":
            x = data[-2*self.seq_length:-self.seq_length]
            x = self.scaler.fit_transform(np.array(x).reshape(-1, 1)).reshape(1, -1, 1)
            y = data[-self.seq_length:]
            y = self.scaler.fit_transform(np.array(y).reshape(-1, 1)).reshape(-1, 1)

            pred = []
            for i in range(self.seq_length):
                sample_data = Variable(torch.Tensor(x))
                train_predict = self.lstm(sample_data.to('cpu'))
                data_predict = train_predict.cpu().data.numpy()
                k = np.append(x, [data_predict])
                x = np.reshape(k[1:], [1, self.seq_length, 1])
                pred.append(data_predict[0][0])

            pred = np.array(pred).reshape([-1, 1])

            data_predict = self.scaler.inverse_transform(pred).reshape(-1)
            dataY_plot = self.scaler.inverse_transform(y).reshape(-1)

            figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
            plt.plot(dataY_plot)
            plt.plot(data_predict)
            plt.legend(['Time Series','Prediction'],fontsize = 21)
            plt.suptitle('Predict The Next Day',fontsize = 23)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.ylabel('Volume/{}min'.format(self.interval), fontsize=18)
            plt.xlabel('Date', fontsize=18)
            plt.savefig(output_dir + '/predict_image_future.jpg')
        else:
            raise Exception("Not support yet.")

if __name__ == "__main__":
    p_instance = Predictor()
    p_instance.do_predict()
