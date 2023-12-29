import Utils
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

def test_model(
    crypto_name, d_model, h, N, d_FF, seqlen_encoder, seqlen_decoder, kernel_size,
    batch_size
    ):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device=torch.device('cuda:0')
        print('Using GPU')
    else:
        device=torch.device('cpu')
        print('Using CPU')
    torch.set_default_device(device)

    crypto_prices=pd.read_csv(f'Processed Prices/{crypto_name} Test.csv').astype('float32')
    crypto_prices=torch.tensor(np.array(crypto_prices))

    crypto_prices_scaled=Utils.scale(crypto_prices, 'Std', crypto_name)

    crypto_prices_dataloader = DataLoader(
        crypto_prices_scaled, 
        batch_size,
        generator=torch.Generator(device)
        )

    all_targets=crypto_prices_scaled[:, seqlen_encoder:]
    all_predicted_outputs = torch.tensor([])

    learned_transformer=torch.load(f'Models/{crypto_name} Model.pth')
    
    with torch.no_grad():
        learned_transformer.eval()
        for batch in tqdm(crypto_prices_dataloader): # Har Batch
            inputs=batch[:, 0:seqlen_encoder]
            outputs=batch[:, seqlen_encoder-1] # * IMPORTANT: Avvalesh ye doonast faghat
            outputs=outputs.unsqueeze(1) # chon outputs hamishe yek sotoon dare => dim: 200 (Batch_size: 200). vali mikhaim beshe 200 x 1

            for prediction_step in range(seqlen_decoder):
                predicted_outputs=learned_transformer(inputs, outputs).squeeze(-1)
                one_new_predicted_price=predicted_outputs[:,-1].unsqueeze(-1)
                outputs=torch.concat((outputs, one_new_predicted_price), dim=1)

            all_predicted_outputs=torch.concat((all_predicted_outputs, predicted_outputs), dim=-2)

        # Rescale
        all_predicted_outputs=Utils.rescale(all_predicted_outputs, 'Std', crypto_name)
        all_targets=Utils.rescale(all_targets, 'Std', crypto_name)

        # Calculate Accuracy
        rmse=Utils.rmse(all_predicted_outputs, all_targets).item()
        mse=Utils.mse(all_predicted_outputs, all_targets).item()
        mae=Utils.mae(all_predicted_outputs, all_targets).item()
        mape=Utils.mape(all_predicted_outputs, all_targets).item()
        r2=Utils.r2(all_predicted_outputs, all_targets).item()
        accuracy=Utils.accuracy(all_predicted_outputs, all_targets).item()
        
        # Move to CPU to Save
        all_targets=all_targets.contiguous().cpu()
        all_predicted_outputs=all_predicted_outputs.contiguous().cpu()

        Utils.save_results(
            crypto_name, d_model, h, N, d_FF, 
            seqlen_encoder, seqlen_decoder, kernel_size, 
            rmse, mse, mae, mape, r2, accuracy
            )

        model_specs_file_name=(f'{crypto_name},{d_model},{h},{N},{d_FF},'
                                f'{seqlen_encoder},{seqlen_decoder},{kernel_size}')
        
        pd.DataFrame(all_predicted_outputs).to_csv(
            f'Achievements/Preds&Actuals/All Predictions - {model_specs_file_name}.csv',
            index=False
            )
        pd.DataFrame(all_targets).to_csv(
            f'Achievements/Preds&Actuals/All Targets - {model_specs_file_name}.csv', 
            index=False
            )

        model_specs_suptitle=(f'Crypto: {crypto_name} | d_model: {d_model} | h: {h} | N: {N}'
                                f' | d_FF: {d_FF} | Input Length: {seqlen_encoder} | '
                                f'Output Length: {seqlen_decoder} | Kernel Size: {kernel_size}')
        # All Data
        Utils.plot_prediction_test(
            all_targets, all_predicted_outputs, 
            model_specs_suptitle, model_specs_file_name, 
            (0, -1)
            )

        # # Good Part
        # Utils.plot_prediction_test(
        #     all_targets, all_predicted_outputs, 
        #     model_specs_suptitle, 'Zoomed - ' + model_specs_file_name, 
        #     (int(len_crypto_prices*0.77), int(len_crypto_prices*0.85))
        #     )
    
# ===================================================================================================
    
if __name__=='__main__':
    crypto_name='Bitcoin'
    d_model=256
    h=8
    N=2
    d_FF=512
    seqlen_encoder=50
    seqlen_decoder=1
    kernel_size=1

    batch_size=1000
    test_model(crypto_name, d_model, h, N, d_FF, seqlen_encoder, seqlen_decoder, kernel_size, batch_size)