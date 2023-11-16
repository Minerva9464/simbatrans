from Transformer import Transformer
import torch
import torch.nn as nn
from torchmetrics.functional import r2_score, mean_absolute_percentage_error
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
import numpy as np

def train_model(crypto_name, d_model, h, N, d_FF, seqlen_encoder, 
                seqlen_decoder, f, p, epoch_number, batch_size, 
                loss_function
                ):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device=torch.device('cuda:0')
    else:
        device=torch.device('cpu')
        
    torch.set_default_device(device)
    
    crypto_prices=pd.read_csv(f'Processed Prices/{crypto_name} Train.csv').astype('float32').iloc[:, :]

    crypto_prices=torch.tensor(np.array(crypto_prices))

    len_crypto_prices=crypto_prices.shape[0]
    crypto_prices=crypto_prices[0: len_crypto_prices-(len_crypto_prices%batch_size),:] # bar batch_size bakhsh pazir bashe
    len_crypto_prices=crypto_prices.shape[0]

    transformer=Transformer(d_model, h, p, d_FF, N, seqlen_encoder, seqlen_decoder, f)

    optimizer=optim.Adam(
        params=transformer.parameters(),
        betas=(0.9,0.98),
        eps=1e-9,
        lr=0
    )

    transformer.train()
    # print(transformer.parameters().)
    for epoch in range(epoch_number):
        step_num = epoch+1
        for row in tqdm(range(0, len_crypto_prices, batch_size)):
            optimizer.param_groups[0]['lr'] = d_model**(-0.5)*min(step_num**(-0.5), step_num*4000**(-1.5))
            optimizer.zero_grad() # x-grad=0

            inputs=crypto_prices[row: row+batch_size, 0:seqlen_encoder]
            outputs=crypto_prices[row: row+batch_size, seqlen_encoder-1: -1]
            targets=crypto_prices[row: row+batch_size, seqlen_encoder:] # chizi hast ke bayad behesh beresim

            predicted_outputs=transformer.forward(inputs, outputs).squeeze()
            
            loss=loss_function(predicted_outputs, targets) if loss_function != r2_score else 1-loss_function(predicted_outputs, targets)
            loss.backward() # x.grad += dloss/dx
            optimizer.step() # x += -lr*x.grad
            
        r2_score_computed = r2_score(predicted_outputs, targets)
        print(f'Epoch: {step_num}, Loss: {loss.item()}, R2 Score: {r2_score_computed}')

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
    epoch_number=100
    batch_size=100
    loss_function=nn.MSELoss() # MSE
    # loss_function=nn.L1Loss() # MAE
    # loss_function=mean_absolute_percentage_error # MAPE
    # loss_function=r2_score # R2-Score

    train_model('Bitcoin', d_model, h, N, d_FF, seqlen_encoder, 
                seqlen_decoder, f, p, epoch_number, batch_size, 
                loss_function
                )

    train_model('Ethereum', d_model, h, N, d_FF, seqlen_encoder, 
                seqlen_decoder, f, p, epoch_number, batch_size, 
                loss_function
                )





