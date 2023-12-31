import torch
import torch.nn as nn

# ------Concatenate two tensors
# import math

# a=torch.randn((10,1))
# b=torch.rand((10,19))

# c=torch.concat((a,b),dim=1)
# print(c)
# ******************************************************************************************

# ------Test PosEnc with Harvard Method
# seq_len=10
# d_model=512
# pos_encoding=torch.zeros((seq_len,d_model))
# pos=torch.arange(start=0, end=seq_len, step=1).unsqueeze(1)
# two_i=torch.arange(start=0, end=d_model-1, step=2).unsqueeze(0)
# # print(pos)
# # print(two_i)
# # div_term2=torch.exp(two_i*(-math.log(10000))/d_model)
# # print(torch.max(div_term-div_term2))
# theta=pos/(10000**(two_i/d_model))
# pos_encoding[:,::2]=torch.sin(theta)
# pos_encoding[:,1::2]=torch.cos(theta)
# print(pos_encoding)
# print(pos_encoding.shape)
# ******************************************************************************************

# ------MinMaxScaler
# import pandas as pd
# import numpy as np

# v=np.arange(100)
# crypto_name='Bitcoin'
# print(v)
# v=v.reshape((10,10))
# print(v)
# print(v.min())
# print(v.max())
# scaled_array= (v-v.min())/(v.max()-v.min())
# print(scaled_array)

# ******************************************************************************************

# ------MinMaxScaler Complete (with saving data)
# max_v=v.max()
# min_v=v.min()
# scaled_v= (v-min_v)/(max_v-min_v)
# print(max_v)
# print(scaled_v)

# min_maxes = pd.read_csv('../Processed Prices/Scaler.csv')
# print(min_maxes)
# print('================')

# new_row=pd.DataFrame({'CryptoName': crypto_name, 'Min': min_v, 'Max': max_v })

# new_min_maxes= pd.concat((min_maxes, new_row)).drop_duplicates(subset='CryptoName')
# new_min_maxes.to_csv('../Processed Prices/Scaler.csv', index=False)

# print(new_min_maxes)

# ******************************************************************************************

# ------Periodic Function (for time2vec)
# import math
# import matplotlib.pyplot as plt

# modulo = lambda x:torch.sin(3.14159893 * (x%1024**(0.4)))
# saw = lambda x: x-torch.floor(x)
# x = torch.arange(0,1,1/100)*1024
# y = saw(x)

# plt.plot(x,y)
# plt.show()

# ******************************************************************************************

# ------Convolution Feed Forward Network
# bs=128
# sql=50
# dm=256
# dff=512
# a=torch.randn((bs,sql,dm)) # N=bs, Cin=sql, Lin=dm
# conv1=nn.Conv1d(in_channels=dm, out_channels=dff, kernel_size=19, padding=9)
# conv2=nn.Conv1d(in_channels=dff, out_channels=dm, kernel_size=19, padding=9)
# res1=conv1(a.transpose(-1,-2)) # N=bs, Cin=dm, Lin=sql
# res2=conv2(res1).transpose(-1,-2)
# lin=nn.Linear(dm,dm)
# pool = nn.AvgPool1d((dm))
# print(res1.shape)
# print(res2.shape)
# print(pool(lin(res2)).shape)

# ******************************************************************************************

# self.encoder_stack=nn.ModuleList([
#             EncoderLayer(d_model, h, p, d_FF) for _ in range(N)
#         ])

# WHEN N=6 IS EQUAL TO:
# self.EncoderLayer1 = EncoderLayer(d_model, h, p, d_FF)
# self.EncoderLayer2 = EncoderLayer(d_model, h, p, d_FF)
# self.EncoderLayer3 = EncoderLayer(d_model, h, p, d_FF)
# self.EncoderLayer4 = EncoderLayer(d_model, h, p, d_FF)
# self.EncoderLayer5 = EncoderLayer(d_model, h, p, d_FF)
# self.EncoderLayer6 = EncoderLayer(d_model, h, p, d_FF)