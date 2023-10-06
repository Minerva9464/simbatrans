import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

def test_model(crypto_name, seqlen_encoder, batch_size, h, N, f=None):
    crypto_prices=pd.read_csv(f'Processed Prices/{crypto_name} Test.csv').astype('float32')

    crypto_prices=np.array(crypto_prices)

    device=torch.device('cuda:0')
    torch.set_default_device(device)

    learned_transformer=torch.load(f'Models/{crypto_name} Model.pth')

    len_bitcoin_prices=crypto_prices.shape[0]
    crypto_prices=crypto_prices[0: len_bitcoin_prices-(len_bitcoin_prices%batch_size),:]
    len_bitcoin_prices=crypto_prices.shape[0]

    result=pd.read_csv('Processed Prices/Scaler.csv')
    # print(min_maxes)
    selected_crypto_min_max=result[result.CryptoName==crypto_name]
    # print(selected_crypto_min_max)
    min_crypto=selected_crypto_min_max.Min.iloc[0]
    # print(min_crypto)
    max_crypto=selected_crypto_min_max.Max.iloc[0]
    crypto_prices_scaled= (crypto_prices-min_crypto)/(max_crypto-min_crypto)
    # print(crypto_prices_scaled)

    mse_loss_function=nn.MSELoss()
    mae_loss_function=nn.L1Loss()

    mse_loss_sum=0
    mae_loss_sum=0
    with torch.no_grad():
        for row in tqdm(range(0, len_bitcoin_prices, batch_size)):
            inputs=crypto_prices_scaled[row: row+batch_size, 0:seqlen_encoder]
            outputs=crypto_prices_scaled[row:row+batch_size, seqlen_encoder-1:-1]
            targets=torch.tensor(crypto_prices_scaled[row:row+batch_size, seqlen_encoder:])

            predicted_outputs=learned_transformer(inputs, outputs).squeeze()
            mse_loss=mse_loss_function.forward(predicted_outputs, targets)
            mae_loss=mae_loss_function.forward(predicted_outputs, targets)

            mse_loss_sum+=mse_loss.item()*batch_size
            mae_loss_sum+=mae_loss.item()*batch_size
        mse = mse_loss_sum/len_bitcoin_prices
        mae = mae_loss_sum/len_bitcoin_prices    
        print(f'MSE Loss:{mse}')
        print(f'MAE Loss:{mae}')

    
    # if f is None:
    #     input_embedding='Linear'
    # else:
    #     input_embedding='Time2Vec'

    input_embedding='Linear' if f is None else 'Time2Vec'

    results = pd.read_csv('Achievements/Test Results.csv')
    new_row=pd.DataFrame(
        {
            'CryptoName': [crypto_name], 
            'h': [h], 
            'N': [N],
            'Input Empedding': [input_embedding],
            'MAE': [mae],
            'MSE': [mse]
            }
        )
    new_results= pd.concat((results, new_row))
    new_results.to_csv('Achievements/Test Results.csv', index=False)
    

if __name__=='__main__':
    seqlen_encoder=100
    batch_size=200
    h=8
    N=6
    f=torch.sin


    test_model('Bitcoin', seqlen_encoder, batch_size, h, N, f)
    test_model('Ethereum', seqlen_encoder, batch_size, h, N, f)
