#!/usr/bin/env python
# coding: utf-8
#工作日和周末客流分析

#单独运行py文件时必加下面几行,否则找不到model
import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WebProject.settings")
django.setup()

# In[2]:


import pandas as pd
import numpy as np
import warnings
import scipy.stats as st
import seaborn as sns
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA 
from app.models import Trips


# ## 定义所要评价与可视化函数

# In[38]:


#平均绝对误差
def Score_MAE(x,y):
    return np.mean(np.abs(x-y))
#平均绝对误差比
def Score_MPAE(x,y):
    S=[]
    for i,j in zip(np.abs(x-y),y):
        if j!=0:
            S.append(i/j)
    return np.mean(np.array(S))
#可视化
def visual(name_list,func):
    plt.figure(figsize=[20,10])
    for name,way in zip(name_list,func):
        sns.lineplot(pd.date_range("2020-4-8","2020-7-16"),name,label=way)
    
    


# ## 数据读取与整合

# In[4]:


#df=pd.read_csv("../input/data2/trips-.csv")
df = pd.DataFrame(list(Trips.objects.all().values()))
#处理数据，数据计数
df.columns=["trip_id","user","in_name","date_in","out_name","date_out","channel_id","price"]
df["date_in"]=df["date_in"].map(lambda x : x.split(" ")[0])
df["date_out"]=df["date_out"].map(lambda x : x.split(" ")[0])


# In[5]:


#查看可重复计数的用户
a=df[df["date_out"]!=df["date_in"]]["user"]
#数据统计
groups1=df.groupby(["date_in"])
groups2=df.groupby(["date_out"])
time,num=[],[]
for (i,group1),(j,group2) in zip(groups1,groups2):
    if i==j:
        time.append(i)
        num.append(max(len(group1),len(group2)))
    else:
        print(i)
#统计后实际使用时间序列数据        
df_use=pd.DataFrame({"date":pd.to_datetime(time),"num":num})
df_use=df_use.sort_values("date")

#分割训练与测试集
df_use_train=df_use.iloc[:-100,:]
df_use_test=df_use.iloc[-100:,:]


# ## FaceBook Prophet 预测

# In[18]:


from fbprophet import Prophet
FB=Prophet()
FB.fit(pd.DataFrame({"ds":df_use_train["date"],"y":df_use_train["num"]}))


# In[19]:


pre_prophet=[]
for i in list(df_use_test["date"].values):
    pre_prophet+list(FB.predict(pd.DataFrame({"ds":[str(i).split("T")[0]]}))["yhat"].values)


# # ARIMA model

# In[20]:


train=df_use
use=train["num"].values
AR = ARIMA(use,(7,0,1)).fit()
pre_ARIMA=AR.predict(1)


# ## 神经网络的训练集数据

# In[22]:


new_data=pre_ARIMA-np.array(list(train["num"][1:]))
train_new=new_data[:-100]
test_new=new_data[-115:]


# ## RBF模型学习误差 （ARIMA-RBF）

# In[23]:


import torch, random
import torch.nn as nn
import torch.optim as optim
 
torch.manual_seed(42)
 
class RBFN(nn.Module):
    """
    以高斯核作为径向基函数
    """
    def __init__(self, centers, n_out=1):
        """
        :param centers: shape=[center_num,data_dim]
        :param n_out:
        """
        super(RBFN, self).__init__()
        self.n_out = n_out
        self.num_centers = centers.size(0) # 隐层节点的个数
        self.dim_centure = centers.size(1) # 
        self.centers = nn.Parameter(centers)
        self.beta = nn.Parameter(torch.ones(1, self.num_centers), requires_grad=True)
        
#         self.beta = torch.zeros(1, self.num_centers)*10
        # 对线性层的输入节点数目进行了修改
        self.linear = nn.Linear(self.num_centers+self.dim_centure, self.n_out, bias=True)
        self.initialize_weights()# 创建对象时自动执行
 
 
    def kernel_fun(self, batches):
        n_input = batches.size(0)  # number of inputs
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
#         print(A)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
#         print(B)
        C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdim=False)))
#         C = torch.exp(-(((A - B).pow(2).sum(2, keepdim=False))))
        return  C
                      
    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(torch.cat([batches, radial_val], dim=1))
        return class_score
 
    def initialize_weights(self, ):
        """
        网络权重初始化
        :return:
        """
        for m in self.modules():
            
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.2)
                m.bias.data.zero_()
 
    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)
# centers = torch.rand((5,8))
# rbf_net = RBFN(centers)
# rbf_net.print_network()
# rbf_net.initialize_weights()
 
if __name__ =="__main__":
    data_train=train_new
    data_test=test_new
    x_train,x_test=[],[]
    y_train,y_test=[],[]
    for i in range(len(data_train)-15):
        x_tr=data_train[i:14+i]
        y_tr=data_train[15+i]
        x_train.append(x_tr)
        y_train.append(y_tr)
    for j in range(len(data_test)-15):
        x_te=data_test[j:14+j]
        y_te=data_test[15+j]
        x_test.append(x_te)
        y_test.append(y_te)
    x_train,y_train,x_test,y_test=torch.tensor(x_train, dtype=torch.float32).T,torch.tensor(y_train, dtype=torch.float32),torch.tensor(x_test, dtype=torch.float32).T,torch.tensor(y_test, dtype=torch.float32)

    y_train.reshape((-1,1))
    data= x_train.T
    label=y_train

#     print(data.size())
 
    centers = data
    rbf = RBFN(centers,1)
    params = rbf.parameters()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params,lr=0.005)
 
    for i in range(500):
        optimizer.zero_grad()
 
        y = rbf.forward(data)
        loss = loss_fn(y,label)
        loss.backward()
        optimizer.step()
        print(f"epoch:{i}  loss:{loss.data}")
 
    # 加载使用
    pre_RBF = rbf.forward(x_test.T)
#     print(y.data)
#     print(label.data)


# ## BP神经网络学习误差（ARIMA-BP）

# In[24]:


import torch
import torch.nn.functional as F     # 激励函数都在这

class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # 隐藏层线性输出
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden2, n_output)   # 输出层线性输出

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.sigmoid(self.hidden1(x))      # 激励函数(隐藏层的线性值)
        x = F.sigmoid(self.hidden2(x))
        x = self.predict(x)             # 输出值
        return x

net = Net(n_feature=14, n_hidden1=30,n_hidden2=10, n_output=1)

print(net)  # net 的结构
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""
optimizer = torch.optim.SGD(net.parameters(), lr=0.005)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)


data_train=train_new
data_test=test_new
x_train,x_test=[],[]
y_train,y_test=[],[]
for i in range(len(data_train)-15):
    x_tr=data_train[i:14+i]
    y_tr=data_train[15+i]
    x_train.append(x_tr)
    y_train.append(y_tr)
for j in range(len(data_test)-15):
    x_te=data_test[j:14+j]
    y_te=data_test[15+j]
    x_test.append(x_te)
    y_test.append(y_te)
x_train,y_train,x_test,y_test=torch.tensor(x_train, dtype=torch.float32).T,torch.tensor(y_train, dtype=torch.float32),torch.tensor(x_test, dtype=torch.float32).T,torch.tensor(y_test, dtype=torch.float32)



x=x_train.T
y=y_train.reshape(-1,1)
for t in range(500):
    prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值
    
    loss = loss_func(prediction, y)     # 计算两者的误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
    print('\rEpoch: %d,  Loss: %f ' % (t,loss))
pre_BP=net(x_test.T)
#     print(loss)


# In[25]:


pre_end_ARIMA_RBF=pre_ARIMA[-100:]-pre_RBF.reshape(100).data.numpy()
pre_end_ARIMA_BP=pre_ARIMA[-100:]-pre_BP.reshape(100).data.numpy()


# ## 测试集图形对比

# In[32]:


pre_prophet=pre_p


# In[42]:


visual([df_use_test["num"],pre_ARIMA[-100:],pre_end_ARIMA_RBF,pre_end_ARIMA_BP],["real","ARIMA","ARIMA-RBF","ARIMA-BP"])


# ## 精度对比

# In[44]:


for real,pre,name in zip([pre_ARIMA[-100:],pre_end_ARIMA_RBF,pre_end_ARIMA_BP],[df_use_test["num"],df_use_test["num"],df_use_test["num"],df_use_test["num"]],["ARIMA","ARIMA-RBF","ARIMA-BP"]):
    real,pre=np.array(real),np.array(pre)
    print(f"{name}_MAE:{Score_MAE(pre,real)}  {name}_MAPE:{Score_MPAE(pre,real)}")


# In[ ]:




