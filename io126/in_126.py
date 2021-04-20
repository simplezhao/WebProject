#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.stats as stats
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.utils import plot_model
#from tensorflow.keras.utils import multi_gpu_model
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.style.use("ggplot")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.metrics import mean_absolute_error
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[67]:


df_weather = pd.read_excel(r'C:\Users\hyh\python代码\三种机器学习方法\站点预测\data\城市天气数据(1).xls')


# # 转换日期

# In[68]:


df_weather['week'] = df_weather['日期'].apply(lambda x: datetime.datetime.strptime(str(x)[:-9], "%Y-%m-%d").weekday() + 1)


# In[69]:


df_weather


# # 分割天气

# In[70]:


df_weather['天气状况'] = df_weather['天气状况'].apply(lambda x: x.split('/'))


# In[71]:


for i in range(df_weather.shape[0]):
    temp = []
    for j in df_weather['天气状况'][i]:
        j = j.strip()
        temp.append(j)
    df_weather['天气状况'][i] = temp


# In[72]:


df_weather['最高气温'] = df_weather['最高气温'].apply(lambda x: int(x[:-2]))


# In[73]:


l = list(range(275))
df_weather['天气编码'] = l


# In[74]:


for i in range(df_weather.shape[0]):
    if "小雨" in df_weather['天气状况'][i]:
        if df_weather['最高气温'][i] >= 25:
            df_weather['天气编码'][i] = 8
        else:
            df_weather['天气编码'][i] = 9
    if "中雨" in df_weather['天气状况'][i] or "雷阵雨" in df_weather['天气状况'][i]:
        df_weather['天气编码'][i] = 10
    if "暴雨" in df_weather['天气状况'][i] or "大雨" in df_weather['天气状况'][i]:
        df_weather['天气编码'][i] = 11
    if "小雨" not in df_weather['天气状况'][i] and "中雨" not in df_weather['天气状况'][i] and "雷阵雨" not in df_weather['天气状况'][i] and "暴雨" not in df_weather['天气状况'][i] and "大雨" not in df_weather['天气状况'][i]:
        if df_weather['最高气温'][i] >= 25:
            df_weather['天气编码'][i] = 12
        else:
            df_weather['天气编码'][i] = 13


# - 优先级1：天气中含有“暴雨”或“大雨”为类别11；
# - 优先级2：天气中含有“中雨”或“雷阵雨”为类别10；
# - 优先级3：天气中含有“小雨”且最高温度>=25，为类别8，天气中含有“小雨”且最高温度<25为类别9；
# - 优先级4：天气中不含有“小雨”，“中雨”，“雷阵雨”，“暴雨”，“大雨”，即非下雨天。此时若最高气温>=25为类别12，否则为类别13.

# In[75]:


df_tw = df_weather.drop(['天气状况',"最高气温","最低气温"],axis = 1)


# In[76]:


df_tw


# # 时间，空间维度选择

# In[77]:


df = pd.read_excel(r'C:\Users\hyh\python代码\三种机器学习方法\站点预测\data\data.xlsx')


# In[78]:


df.head()


# In[79]:


df = df.fillna(method = 'bfill')


# In[80]:


df_s_cor = df.iloc[:412,:].corr()


# In[81]:


df_s_cor


# In[82]:


arr = np.array(list(df_s_cor.iloc[1,:]))
index = list(arr.argsort()[-70:][::-1])  #选取相关性排名前70为的站点进行预测


# In[83]:


df_f = df.iloc[:,np.add(index,1)]
df_f = df.iloc[:,0:1].join(df_f)

df_f


# In[97]:


df_arima = df_f.copy()


# In[19]:


pacf = plot_pacf(df_f["126出"], lags=20)
plt.title("PACF")
pacf.show()


# - #通过计算偏自相关系数，决定使用前3个时段进行预测

# # 构造时空维度数据集

# In[20]:


seq_len = 4
data_ = []
df_f1 = df_f.iloc[:,1:]
for i in range(len(df_f1) - seq_len+1):   #构造训练数据
    data_.append(df_f1.iloc[i: i + seq_len])
data_ = np.array([df.values for df in data_])
x = data_[:, :-1, :]  #预测数据集
y = data_[:, -1, 0]   #被预测数据集


# # 构造日期，天气数据集

# In[21]:


df_f['日期'] = df_f['日期'].apply(lambda x: str(x)[:11].strip().replace('/','-'))
df_f['日期'] = pd.to_datetime(df_f['日期'],format = "%Y-%m-%d")
df_ww = pd.merge(df_f,df_tw).iloc[:,-2:]
df_ww['week'] = df_ww['week'].astype('category')
df_ww['天气编码'] = df_ww['天气编码'].astype('category')
df_embedding = df_ww[3:]
df_input1 = df_embedding.values
df_input1.shape


# # 构造测试集，训练集

# In[22]:


length = df_input1.shape[0]
boundary = int(length*0.7)
x_train1 = df_input1[:boundary]
x_train2 = x[:boundary]
y_train = y[:boundary]
x_test1 = df_input1[boundary:]
x_test2 = x[boundary:]
y_test = y[boundary:]
mean = x_train2.mean(axis=0)
std = x_train2.std(axis=0)
x_train2 = (x_train2 - mean)/std
x_test2 = (x_test2 - mean)/std
x_test2.shape


# In[23]:


seq_len = 2
max_features = 14
embedding_dims = 64


# In[51]:


class Embedding_Lstm(object):
    def __init__(self,seq_len,max_features,embedding_dims):
        self.maxlen = seq_len
        self.max_features = max_features
        self.embedding_dims = embedding_dims
    
    def creat_model(self):
        input1_= layers.Input((self.maxlen,),name='input1')
        input2_ = layers.Input(shape=(3, 70), name='input2')
        embedding = layers.Embedding(self.max_features,self.embedding_dims, input_length=self.maxlen)(input1_)
        x1 = layers.LSTM(64)(embedding) #x1.shape = [batch,64]
        x2 = layers.LSTM(32,return_sequences=True)(input2_)  # x2.shape = [batch,64]
        x2 = layers.LSTM(64,return_sequences=True)(x2)
        x2 = layers.Dropout(0.1)(x2)
        x2 = layers.LSTM(256,return_sequences=True)(x2)
        x2 = layers.Dropout(0.1)(x2)
        x2 = layers.LSTM(256,return_sequences=True)(x2)
        x2 = layers.Dropout(0.1)(x2)
        x2 = layers.LSTM(64)(x2)
        x = layers.concatenate([x1, x2])
        x = layers.Dense(32,activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(1)(x)
        model = Model(inputs=[input1_, input2_], outputs=[output])
        
        return model
    
embedding_lstm = Embedding_Lstm(seq_len, max_features, embedding_dims).creat_model()
        
        
        
        
        
        
    


# In[52]:


plot_model(embedding_lstm, to_file='embedding_lstm.png', show_shapes=True)


# In[53]:


embedding_lstm.compile(optimizer=keras.optimizers.Adam(), loss='mae')
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5, min_lr=0.00001)


# In[54]:


history1 = embedding_lstm.fit([x_train1,x_train2], y_train, epochs=500, batch_size=64,validation_data=([x_test1,x_test2], y_test))


# In[57]:


plt.plot(history1.epoch, history1.history.get('loss'), 'y', label='Training loss')  #绘制训练图
plt.plot(history1.epoch, history1.history.get('val_loss'), 'b', label='Test loss')
plt.legend()


# In[58]:


l_pred=list(embedding_lstm.predict([x_test1,x_test2]).reshape(154,))
plt.figure(figsize = (16,8),dpi = 100)  
plt.plot(y_test)
plt.plot(l_pred)
plt.title('LOAD per sample')
plt.xlabel('sample')
plt.ylabel('LOAD')
plt.legend()
plt.show()  


# In[59]:


# 模型保存
embedding_lstm.save('embedding_lstm.h5')
#模型读取
embedding_lstm_read = keras.models.load_model('embedding_lstm.h5')
l_pred_read=list(embedding_lstm_read.predict([x_test1,x_test2]).reshape(154,))


# In[ ]:





# In[ ]:





# # 单一lstm

# In[41]:


model = keras.Sequential()    #建立神经网络模型
model.add(layers.LSTM(32, input_shape=(x_train2.shape[1:]), return_sequences=True))
model.add(layers.LSTM(64, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.6))
model.add(layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.6))
model.add(layers.LSTM(64, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.6))
model.add(layers.LSTM(32))
model.add(layers.Dense(1))


# In[42]:


model.compile(optimizer=keras.optimizers.Adam(), loss='mae')  #定义优化器，损失函数
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=0.00001)  #对学习率进行调整


# In[43]:


#训练800个epochs
history = model.fit(x_train2, y_train,
                    batch_size = 64,
                    epochs=800, 
                    validation_data=(x_test2, y_test))


# In[44]:


plt.plot(history.epoch, history.history.get('loss'), 'y', label='Training loss')  #绘制训练图
plt.plot(history.epoch, history.history.get('val_loss'), 'b', label='Test loss')
plt.legend()


# In[60]:


l_pred_single=list(model.predict(x_test2).reshape(154,))
plt.figure(figsize = (16,8),dpi = 100)  
plt.plot(y_test,label = "真实值")
plt.plot(l_pred,label="embedding_lstm预测值")
plt.plot(l_pred_single,color = 'green',label = "单一lstm预测值")
plt.title('LOAD per sample')
plt.xlabel('sample')
plt.ylabel('LOAD')
plt.legend()
plt.show() 


# In[ ]:





# In[ ]:




