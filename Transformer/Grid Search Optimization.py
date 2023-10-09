from Test import test_model
from Train import train_model
from torch import sin
from torch.nn import L1Loss
from tqdm import tqdm

grid = [
    {'h': 4, 'N': 3, 'f': sin, 'CryptoName': 'Bitcoin'},
    {'h': 4, 'N': 3, 'f': None, 'CryptoName': 'Bitcoin'},
    {'h': 4, 'N': 6, 'f': sin, 'CryptoName': 'Bitcoin'},
    {'h': 4, 'N': 6, 'f': None, 'CryptoName': 'Bitcoin'},
    {'h': 4, 'N': 12, 'f': sin, 'CryptoName': 'Bitcoin'},
    {'h': 4, 'N': 12, 'f': None, 'CryptoName': 'Bitcoin'},
    
    {'h': 8, 'N': 3, 'f': sin, 'CryptoName': 'Bitcoin'},
    {'h': 8, 'N': 3, 'f': None, 'CryptoName': 'Bitcoin'},
    {'h': 8, 'N': 6, 'f': sin, 'CryptoName': 'Bitcoin'},
    {'h': 8, 'N': 6, 'f': None, 'CryptoName': 'Bitcoin'},
    {'h': 8, 'N': 12, 'f': sin, 'CryptoName': 'Bitcoin'},
    {'h': 8, 'N': 12, 'f': None, 'CryptoName': 'Bitcoin'},

    {'h': 12, 'N': 3, 'f': sin, 'CryptoName': 'Bitcoin'},
    {'h': 12, 'N': 3, 'f': None, 'CryptoName': 'Bitcoin'},
    {'h': 12, 'N': 6, 'f': sin, 'CryptoName': 'Bitcoin'},
    {'h': 12, 'N': 6, 'f': None, 'CryptoName': 'Bitcoin'},
    {'h': 12, 'N': 12, 'f': sin, 'CryptoName': 'Bitcoin'},
    {'h': 12, 'N': 12, 'f': None, 'CryptoName': 'Bitcoin'},
]

d_model=512
d_FF=2048
seqlen_encoder=100
seqlen_decoder=20 
p=0.1

# Learning Parameters
epoch_number=1
batch_size_train=50
batch_size_test=200
learning_rate=1e-3
loss_function=L1Loss()

for grid_node in tqdm(grid):
    train_model(
        grid_node['CryptoName'], d_model, grid_node['h'], grid_node['N'], d_FF, 
        seqlen_encoder, seqlen_decoder, grid_node['f'], p, epoch_number, batch_size_train, 
        learning_rate, loss_function
        )
    
    test_model(
        grid_node['CryptoName'], seqlen_encoder, seqlen_decoder,
        batch_size_test, grid_node['h'], grid_node['N'], grid_node['f']
        )