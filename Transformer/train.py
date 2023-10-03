from Transformer import Transformer
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
import numpy as np

bitcoin_prices=pd.read_csv('../Processed Prices/Bitcoin.csv')
print(bitcoin_prices.head())
print('---------------------------------------------')

bitcoin_prices=np.array(bitcoin_prices)
print(bitcoin_prices)

d_model=512
h=8
N=6
d_FF=2048
seqlen_encoder=100 # in len_input hast
seqlen_decoder=20 # len output
seqlen_sum=seqlen_encoder+seqlen_decoder

f=torch.sin
p=0.1
epoch_number=10
batch_size=50
learning_rate=1e-3

device=torch.device('cuda:0')
torch.set_default_device(device)

transformer=Transformer(d_model, h, p, d_FF, N, seqlen_encoder, seqlen_decoder, f)
parameters=list(transformer.parameters())
print(parameters[0])
# for params in parameters:
#     print(params)

loss_function=nn.MSELoss()
optimizer=optim.Adam(
    params=transformer.parameters(),
    betas=(0.9,0.98),
    eps=1e-9,
    lr=learning_rate
)

transformer.train()

for epoch in range(epoch_number):
    for row in tqdm(range(0, bitcoin_prices.shape[0], batch_size)):
        optimizer.zero_grad()

        inputs=bitcoin_prices[row: row+batch_size, 0:seqlen_encoder]
        outputs=bitcoin_prices[row: row+batch_size, seqlen_encoder-1: seqlen_sum-1]
        targets=torch.tensor(bitcoin_prices[row: row+batch_size, seqlen_encoder:], dtype=torch.float32) #chizi hast ke bayad behesh beresim

        predicted_outputs=transformer.forward(inputs, outputs).squeeze()
        loss=loss_function.forward(predicted_outputs, targets)
        loss.backward() # x.grad += dloss/dx

        optimizer.step() # x += -lr*x.grad
    print(f'Epoch: {epoch}, Loss: {loss.item}')
    



