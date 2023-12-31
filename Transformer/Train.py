from Transformer import Transformer
import Utils
# import sys
# if 'torch' not in sys.modules:
#     import torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchsummary
from torchmetrics.functional import r2_score
import pandas as pd
from tqdm import tqdm
import numpy as np

def train_model(
    crypto_name, d_model, f, h, N, d_FF, p,
    seqlen_encoder, seqlen_decoder, network_type, kernel_size,
    epoch_number, batch_size, loss_function, lr_config
    ):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device=torch.device('cuda:0')
        torch.cuda.manual_seed(46) 
        print('Using GPU')
    else:
        device=torch.device('cpu')
        torch.manual_seed(46)
        print('Using CPU')
        
    torch.set_default_device(device)

    transformer=Transformer(
        d_model, f, h, N, d_FF, p, 
        seqlen_encoder, seqlen_decoder, 
        network_type, kernel_size
        )
    # Initialzing all weights with Kaiming Uniform algorithm
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.kaiming_uniform_(p, nonlinearity='relu')

    # ===============================================
    # * Model Summary
    torchsummary.summary(
        transformer, 
        input_size=[(seqlen_encoder,), (seqlen_decoder,)], 
        batch_size=batch_size, device=device.type
        )

    trainable_parameters=0
    params={'Name': [], 'Shape': [], 'Requires Grad':[]}
    for name, param in transformer.named_parameters():
        params['Name'].append(name)
        params['Shape'].append(tuple(param.shape))
        params['Requires Grad'].append(param.requires_grad)

        this_param_weight_num=1
        for length in param.shape:
            this_param_weight_num*=length

        trainable_parameters+=this_param_weight_num

    params['Name'].append('Total Parameters')
    params['Shape'].append(trainable_parameters)
    params['Requires Grad'].append('-')
        
    params=pd.DataFrame(params)
    params.to_csv(f'Achievements/Model Parameters.csv', index=False)
    
    # ===============================================
    # * Reading Data:
    crypto_prices=pd.read_csv(f'Processed Prices/{crypto_name} Train.csv').astype('float32')
    crypto_prices=torch.tensor(np.array(crypto_prices))
    # Rescaling target data
    all_targets=crypto_prices[:, seqlen_encoder:].contiguous().squeeze()
    all_targets=Utils.rescale(all_targets, 'Std', crypto_name)
    crypto_prices_dataloader = DataLoader(
        crypto_prices, 
        batch_size,
        generator=torch.Generator(device)
    )
    
    # ===============================================
    # * Learning:
    optimizer=optim.Adam(
        params=transformer.parameters(),
        betas=(0.9,0.999),
        eps=1e-9,
        lr=0,
    )
    
    transformer.train()

    low=lr_config['low']
    up=lr_config['high']
    percentage_to_fall_down=lr_config['percentage_to_fall_down']
    percentage_to_rest=lr_config['percentage_to_rest']
    linearity=lr_config['linearity']

    # * Running Traing Epochs
    for epoch in range(1, epoch_number+1): # epoch=step_num
        all_predicted_outputs=torch.tensor([])
        
        lr=Utils.lr_function(
            low, up, epoch_number, 
            percentage_to_fall_down, percentage_to_rest, 
            epoch, linearity
            )
        optimizer.param_groups[0]['lr']=lr
        
        for batch in tqdm(crypto_prices_dataloader):
            inputs=batch[:, 0:seqlen_encoder]
            outputs=batch[:, seqlen_encoder-1: -1]
            targets=batch[:, seqlen_encoder:].squeeze().contiguous() # chizi hast ke bayad behesh beresim
            
            predicted_outputs=transformer.forward(inputs, outputs).squeeze().contiguous()
            all_predicted_outputs=torch.concat((all_predicted_outputs, predicted_outputs), dim=0)
            
            loss=loss_function(predicted_outputs, targets)
            optimizer.zero_grad() # x.grad=0
            loss.backward() # x.grad += dloss/dx
            optimizer.step() # x += -lr*x.grad

        all_predicted_outputs=Utils.rescale(all_predicted_outputs, 'Std', crypto_name)

        loss_all_predictions=round(loss_function(all_predicted_outputs, all_targets).item(), 5)
        r2_score_all_predictions=round(r2_score(all_predicted_outputs, all_targets).item(), 5)
        accuracy_all_predicton=round(Utils.accuracy(all_predicted_outputs, all_targets).item(), 5)

        print(f'Epoch: {epoch}, lr: {lr}')
        print(f'{loss_function.__name__.upper()} Loss: {loss_all_predictions}')
        print(f'R2 Score: {r2_score_all_predictions}')
        print(f'Accuracy: {accuracy_all_predicton}%')
        print('^^^^^^^================^^^^^^^\n')


    torch.save(transformer, f'Models/{crypto_name} Model.pth')
    # plot_prediction_train(all_targets, all_predicted_outputs, (4, 3), (3000,3020))
    
# ===================================================================================================
    
if __name__=='__main__':
    crypto_name='Bitcoin'
    # Model Parameters
    d_model=256
    f=torch.sin
    h=8
    N=2
    d_FF=512
    p=0.1
    seqlen_encoder=50 # in len_input hast
    seqlen_decoder=1 # len output
    network_type='CNN'
    kernel_size=1
    
    # Learning Parameters
    epoch_number=250
    batch_size=128
    
    loss_functions={
        'MSE': Utils.mse,
        'RMSE': Utils.rmse,
        'MAE': Utils.mae,
        'MAPE': Utils.mape,
        }
    loss_function=loss_functions['RMSE']

    lr_config={
        'low': 1e-6,
        'high': 1e-4,
        'percentage_to_fall_down': 0.05,
        'percentage_to_rest': 0.8,
        'linearity': 'sin'
        }

    train_model(crypto_name, d_model, f, h, N, d_FF, p,
                seqlen_encoder, seqlen_decoder, network_type, kernel_size,
                epoch_number, batch_size, loss_function, lr_config
                )