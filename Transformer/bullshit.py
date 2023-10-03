# import torch
# import math

# a=torch.randn((10,1))
# b=torch.rand((10,19))

# c=torch.concat((a,b),dim=1)
# print()

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


import pandas as pd
import numpy as np

v=pd.DataFrame(np.random.random((100,1))*100)
crypto_name='Bitcoin'
print(v)

max_v=v.max()
min_v=v.min()
scaled_v= (v-min_v)/(max_v-min_v)
print(max_v)
print(scaled_v)

min_maxes = pd.read_csv('../Processed Prices/Scaler.csv')
print(min_maxes)
print('================')

new_row=pd.DataFrame({'CryptoName': crypto_name, 'Min': min_v, 'Max': max_v })

new_min_maxes= pd.concat((min_maxes, new_row)).drop_duplicates(subset='CryptoName')
new_min_maxes.to_csv('../Processed Prices/Scaler.csv', index=False)

print(new_min_maxes)

