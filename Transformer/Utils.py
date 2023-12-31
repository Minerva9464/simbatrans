import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from datetime import datetime
from torchmetrics.functional import mean_absolute_percentage_error, r2_score

# ===================================================================================================

def lr_function(
    low, up, epoch_number, 
    percentage_to_fall_down, percentage_to_rest, 
    epoch, linearity
    ):
    if epoch < epoch_number*percentage_to_fall_down:
        return up
    elif epoch > epoch_number*percentage_to_rest:
        return low
    else:
        if linearity=='linear':
            m=(up-low)/(epoch_number*percentage_to_fall_down - epoch_number*percentage_to_rest)
            b=up - m*epoch_number*percentage_to_fall_down
            return m*epoch+b
        else:
            return 0.5*(1+1.1*np.cos(np.pi * epoch/epoch_number))*up+low
        # TODO: change coeffiecnts of above formula

# ===================================================================================================

def mse(pred, target): 
    return nn.MSELoss()(pred, target)
def rmse(pred, target): 
    return torch.sqrt(nn.MSELoss()(pred, target))
def mae(pred, target): 
    return nn.L1Loss()(pred, target)
def mape(pred, target): 
    return mean_absolute_percentage_error(pred, target)
def r2(pred, target): 
    return r2_score(pred, target)
def accuracy(pred, target):
    return 100*(1-mape(pred, target))

# ===================================================================================================

def scale(tensor, scale_type, crypto_name):
    if scale_type=='MinMax':
        result=pd.read_csv('Processed Prices/Scaler MinMax.csv')
        selected_crypto_min_max=result[result.CryptoName==crypto_name]
        min_crypto=selected_crypto_min_max.Min.iloc[0]
        max_crypto=selected_crypto_min_max.Max.iloc[0]

        return (tensor-min_crypto)/(max_crypto-min_crypto)
    
    else:
        result=pd.read_csv('Processed Prices/Scaler Std.csv')
        selected_crypto_mean_std=result[result.CryptoName==crypto_name]
        mean_crypto=selected_crypto_mean_std.Mean.iloc[0]
        std_crypto=selected_crypto_mean_std.Std.iloc[0]

        return (tensor-mean_crypto)/std_crypto

# ===================================================================================================

def rescale(data_to_rescale, scale_type, crypto_name):
    if scale_type=='MinMax':
        min_maxes=pd.read_csv('Processed Prices/Scaler MinMax.csv')
        selected_crypto_min_max=min_maxes[min_maxes.CryptoName==crypto_name]
        min_crypto=selected_crypto_min_max.Min.iloc[0]
        max_crypto=selected_crypto_min_max.Max.iloc[0]

        return data_to_rescale*(max_crypto-min_crypto)+min_crypto

    else:
        mean_stds=pd.read_csv('Processed Prices/Scaler Std.csv')
        selected_crypto_mean_std=mean_stds[mean_stds.CryptoName==crypto_name]
        mean_crypto=selected_crypto_mean_std.Mean.iloc[0]
        std_crypto=selected_crypto_mean_std.Std.iloc[0]

        return data_to_rescale*std_crypto+mean_crypto
    
# ===================================================================================================

def plot_prediction_train(
    all_targets: torch.tensor, all_predicted_outputs: torch.tensor, 
    plot_dimension: tuple, plot_range: tuple
    ):
    
    n_rows=plot_dimension[0]
    n_cols=plot_dimension[0]

    fig, axes=plt.subplots(n_rows, n_cols, figsize=(10,6))
    all_targets=all_targets.cpu()
    all_predicted_outputs=all_predicted_outputs.contiguous().detach().cpu() # detach: no requiers_grad
    
    selected_rows=random.sample(population=range(all_targets.shape[0]), k=n_rows*n_cols)
    row_to_plot=0
    fig.suptitle(f'{n_rows*n_cols} of Bitcoin Prices Sample')
    fig.tight_layout()
    for row in range(n_rows):
        for col in range(n_cols):
            output_row=selected_rows[row_to_plot]
            axes[row, col].plot(all_targets[output_row, :], color='royalblue')
            axes[row, col].plot(all_predicted_outputs[output_row, :], color='deeppink')
            axes[row, col].set_title(f'Row: {output_row}')
            row_to_plot+=1

    plt.legend(['trg', 'prd'], loc="upper left")
    plt.savefig(f'Achievements/Predictions Subplot.png', dpi=600)
    plt.show()

    # Total Data
    start_point=plot_range[0]
    end_point=plot_range[1]
    plt.figure(figsize=(10,6))
    plt.plot(all_targets[start_point:end_point, :].view(-1,1), color='royalblue', linewidth=3)
    plt.plot(all_predicted_outputs[start_point:end_point, :].view(-1,1), color='deeppink')

    plt.legend(['all_targets', 'all_predicted_outputs'], loc="upper left")
    plt.savefig(f'Achievements/Predictions Plot - Train.png', dpi=600)
    plt.show()

# ===================================================================================================

def plot_prediction_test(
    all_targets, all_predicted_outputs, 
    model_specs_suptitle, model_specs_file_name, 
    plot_range: tuple
    ):
    
    start_point=plot_range[0]
    end_point=plot_range[1]
    
    fig=plt.figure(figsize=(15,9))
    plt.plot(all_targets[start_point:end_point, :].view(-1,1), color='royalblue', linewidth=3)
    plt.plot(all_predicted_outputs[start_point:end_point, :].view(-1,1), color='deeppink')
    plt.legend(['target', 'prediction'], loc='upper left')
    plt.title('Prediction vs Target', fontsize=24, pad=10)
    plt.suptitle(model_specs_suptitle, fontsize=8)
    fig.tight_layout()
    plt.savefig(f'Achievements/Plots/Test Plot - {model_specs_file_name}.png', dpi=300)
    plt.subplots_adjust(top=0.89)
    # plt.show()
    
# ===================================================================================================

def save_results(
    crypto_name, d_model, h, N, d_FF, seqlen_encoder, seqlen_decoder, kernel_size,
    rmse, mse, mae, mape, r2, accuracy, note
    ):

    new_row=pd.DataFrame(
        {
            'CryptoName': [crypto_name], 
            'd_model': d_model,
            'h': [h], 
            'N': [N],
            'd_FF': [d_FF],
            'SequenceLenInputs': [seqlen_encoder],
            'SequenceLenOutputs': [seqlen_decoder],
            'KernelSize': [kernel_size],
            'RMSE': [rmse],
            'MSE': [mse],
            'MAE': [mae],
            'MAPE': [mape],
            'R2': [r2],
            'Accuracy': [accuracy],
            'Notes': [note],
            'DateTime': [datetime.now()],
            }
        )

    results = pd.read_csv('Achievements/Test Results.csv')
    new_results= pd.concat((results, new_row))
    new_results.to_csv('Achievements/Test Results.csv', index=False)