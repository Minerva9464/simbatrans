from Transformer import Transformer
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
import numpy as np

def train_model(crypto_name, d_model, h, N, d_FF, seqlen_encoder, 
                seqlen_decoder, f, p, epoch_number, batch_size, 
                learning_rate, loss_function
                ):
    
    crypto_prices=pd.read_csv(f'Processed Prices/{crypto_name} Train.csv').astype('float32').iloc[:5000, :]

    crypto_prices=np.array(crypto_prices)

    len_crypto_prices=crypto_prices.shape[0]
    crypto_prices=crypto_prices[0: len_crypto_prices-(len_crypto_prices%batch_size),:] # bar batch_size bakhsh pazir bashe
    len_crypto_prices=crypto_prices.shape[0]

    device=torch.device('cuda:0')
    torch.set_default_device(device)

    transformer=Transformer(d_model, h, p, d_FF, N, seqlen_encoder, seqlen_decoder, f)

    optimizer=optim.Adam(
        params=transformer.parameters(),
        betas=(0.9,0.98),
        eps=1e-9,
        lr=learning_rate
    )

    transformer.train()

    for epoch in range(epoch_number):
        for row in tqdm(range(0, len_crypto_prices, batch_size)):
            optimizer.zero_grad()

            inputs=crypto_prices[row: row+batch_size, 0:seqlen_encoder]
            outputs=crypto_prices[row: row+batch_size, seqlen_encoder-1: -1]
            targets=torch.tensor(crypto_prices[row: row+batch_size, seqlen_encoder:]) #chizi hast ke bayad behesh beresim

            predicted_outputs=transformer.forward(inputs, outputs).squeeze()
            loss=loss_function.forward(predicted_outputs, targets)
            loss.backward() # x.grad += dloss/dx

            optimizer.step() # x += -lr*x.grad
            # print(loss.item())
        # print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

    torch.save(transformer, f'Models/{crypto_name} Model.pth')
        
if __name__=='__main__':
    # Model Parameters
    d_model=512
    h=8
    N=6
    d_FF=2048
    seqlen_encoder=100 # in len_input hast
    seqlen_decoder=20 # len output
    f=torch.sin
    p=0.1
    
    # Learning Parameters
    epoch_number=2
    batch_size=50
    learning_rate=1e-3
    loss_function=nn.L1Loss()

    train_model('Bitcoin', d_model, h, N, d_FF, seqlen_encoder, 
                seqlen_decoder, f, p, epoch_number, batch_size, 
                learning_rate, loss_function
                )

    train_model('Ethereum', d_model, h, N, d_FF, seqlen_encoder, 
                seqlen_decoder, f, p, epoch_number, batch_size, 
                learning_rate, loss_function
                )





