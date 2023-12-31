import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm


bitcoin_prices=pd.read_csv('Processed Prices/Bitcoin Test.csv').astype('float32')
print(bitcoin_prices.head())
print('---------------------------------------------')

bitcoin_prices=np.array(bitcoin_prices)
print(bitcoin_prices)
print('---------------------------------------------')

device=torch.device('cuda:0')
torch.set_default_device(device)

learned_transformer=torch.load('Models/Bitcoin Model.pth')
print(len(list(learned_transformer.parameters())))

seqlen_encoder=100 # in len_input hast
seqlen_decoder=20 # len output
seqlen_sum=seqlen_encoder+seqlen_decoder
batch_size=200

len_bitcoin_prices=bitcoin_prices.shape[0]
print(len_bitcoin_prices)
# batch_size=len_bitcoin_prices
bitcoin_prices=bitcoin_prices[0: len_bitcoin_prices-(len_bitcoin_prices%batch_size),:]
len_bitcoin_prices=bitcoin_prices.shape[0]
print(len_bitcoin_prices)

mse_loss_function=nn.MSELoss()
mae_loss_function=nn.L1Loss()

mse_loss_sum=0
mae_loss_sum=0
with torch.no_grad():
    for row in tqdm(range(0, len_bitcoin_prices, batch_size)):
        inputs=bitcoin_prices[row: row+batch_size, 0:seqlen_encoder]
        outputs=bitcoin_prices[row:row+batch_size, seqlen_encoder-1: seqlen_sum-1]
        targets=torch.tensor(bitcoin_prices[row:row+batch_size, seqlen_encoder:])

        predicted_outputs=learned_transformer(inputs, outputs).squeeze()
        mse_loss=mse_loss_function.forward(predicted_outputs, targets)
        mae_loss=mae_loss_function.forward(predicted_outputs, targets)

        mse_loss_sum+=mse_loss.item()*batch_size
        mae_loss_sum+=mae_loss.item()*batch_size
        
    print(f'MSE Loss:{mse_loss_sum/len_bitcoin_prices}')
    print(f'MAE Loss:{mae_loss_sum/len_bitcoin_prices}')
