from Preprocessing import data_wrangling
from Train import train_model
from Test import test_model
from Utils import rmse

from torch import sin

grid = [
    {
        'd_model': 256, 'h': 8, 'N': 2, 'd_FF': 512,
        'seqlen_encoder': 50, 'seqlen_decoder': 1, 'kernel_size': 1
        },
        # {
        # 'd_model': , 'h': , 'N': , 'd_FF': ,
        # 'seqlen_encoder': , 'seqlen_decoder': , 'kernel_size': 
        # }
    ]

test_range=(0.75, 0.85)
interval='2H'

crypto_name='Bitcoin'
f=sin
p=0.1
network_type='CNN'

epoch_number=2
batch_size=128
loss_function=rmse
lr_config={
            'low': 1e-4,
            'high': 1e-4,
            'percentage_to_fall_down': 0.02,
            'percentage_to_rest': 0.9,
            'linearity': 'linear'
            }

for grid_node in grid:
    d_model=grid_node['d_model']
    h=grid_node['h']
    N=grid_node['N']
    d_FF=grid_node['d_FF']
    seqlen_encoder=grid_node['seqlen_encoder']
    seqlen_decoder=grid_node['seqlen_decoder']
    kernel_size=grid_node['kernel_size']

    print('Train and Test Will Be Ready!')
    data_wrangling(crypto_name, seqlen_encoder, seqlen_decoder, test_range, interval)

    print('The Model Will Be Trained Now!')
    train_model(crypto_name, d_model, f, h, N, d_FF, p,
                seqlen_encoder, seqlen_decoder, network_type, kernel_size,
                epoch_number, batch_size, loss_function, lr_config
                )

    print('The Test Has Just Begun!')
    test_model(crypto_name, d_model, h, N, d_FF, seqlen_encoder, seqlen_decoder, 
                kernel_size, batch_size*4
                )

    print(f'\nGrid Node: {str(grid_node)[1:-1]} has been completed!\n')
    print('𓆩♡𓆪𓆩♡𓆪𓆩♡𓆪𓆩♡𓆪 ᰔᩚᰔᩚᰔᩚᰔᩚᰔᩚᰔᩚᰔᩚᰔᩚᰔᩚᰔᩚᩚᰔᩚᩚᰔᩚᩚᰔᩚᰔᩚᰔᩚᰔᩚᰔᩚᰔᩚᰔᩚ❦❦❦❦❦❦❦❦❦❦❦❦❦❦❦❦❦❦❦❦ꨄꨄꨄꨄꨄꨄꨄꨄꨄꨄꨄꨄꨄꨄꨄꨄꨄꨄ 𓆩♡𓆪𓆩♡𓆪𓆩♡𓆪𓆩♡𓆪\n')