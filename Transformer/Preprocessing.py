import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def data_wrangling(crypto_name: str, input_length, output_length, test_range: tuple, freq='2H'):
    crypto_prices=pd.read_csv(f'Raw Prices/{crypto_name} Prices.csv')

    closing_prices=crypto_prices.ClosingPrice

    start_time=crypto_prices.DateTime.iloc[0]
    end_time=crypto_prices.DateTime.iloc[-1]
    date_range=pd.date_range(start_time, end_time, freq=freq).to_list()

    len_prices=len(closing_prices)
    start_train_range=int(len_prices*test_range[0])
    end_train_range=int(len_prices*test_range[1])

    # * Train:
    prices_train=pd.concat(
        (closing_prices[:start_train_range],
        closing_prices[end_train_range:])
        )
    len_train=len(prices_train)

    row_length=input_length+output_length
    prices_train_table=np.zeros((len_train-row_length+1, row_length))
    len_train_table=prices_train_table.shape[0]
    for row in tqdm(range(len_train_table)):
        prices_train_table[row, :]=prices_train[row: row+row_length]

    prices_train_table_df=pd.DataFrame(scale(prices_train_table, scale_type='Std', crypto_name=crypto_name))

    # * Test
    prices_test=closing_prices[start_train_range:end_train_range]
    len_test=len(prices_test)
    prices_test_table=np.zeros(((len_test-input_length)//output_length, row_length))
    for row in tqdm(range(prices_test_table.shape[0])):
        prices_test_table[row, :]=prices_test[row*output_length: row*output_length+row_length]
    prices_test_table_df=pd.DataFrame(prices_test_table)

    # * Plotting
    plt.figure(figsize=(10,6))
    plt.plot(date_range[:len_prices], closing_prices, color='teal')
    plt.title(crypto_name, fontdict={'size':24})
    plt.xlabel('Year')
    plt.ylabel('Price (USD)')
    plt.savefig(f'Achievements/{crypto_name} All Time Prices.png', dpi=600)
    # Plot train
    plt.title(f'{crypto_name}, Train and Test', fontdict={'size':24})
    plt.plot(date_range[:start_train_range], closing_prices[:start_train_range], color='deeppink')
    plt.plot(date_range[end_train_range:len_prices], closing_prices[end_train_range:], color='deeppink')
    plt.savefig(f'Achievements/{crypto_name} All Time Prices - Train & Test.png', dpi=600)
    plt.close()
    # plt.show()

    prices_train_table_df.to_csv(f'Processed Prices/{crypto_name} Train.csv', index=False)
    prices_test_table_df.to_csv(f'Processed Prices/{crypto_name} Test.csv', index=False)
    print(f'{crypto_name} processing has been completed!')
    print('=====================================================================================')

def scale(array, scale_type, crypto_name):
    if scale_type=='MinMax':
        max_array=array.max()
        min_array=array.min()
        scaled_array= (array-min_array)/(max_array-min_array)
        
        min_maxes = pd.read_csv('Processed Prices/Scaler.csv')
        new_row=pd.DataFrame({'CryptoName': [crypto_name], 'Min': [min_array], 'Max': [max_array]})
        new_min_maxes= pd.concat((min_maxes, new_row)).drop_duplicates(subset='CryptoName')
        new_min_maxes.to_csv('Processed Prices/Scaler MinMax.csv', index=False)
        return scaled_array
    
    else:
        mean=array.mean()
        std=array.std()
        scaled_array= (array-mean)/std
        
        mean_stds = pd.read_csv('Processed Prices/Scaler Std.csv')
        new_row=pd.DataFrame({'CryptoName': [crypto_name], 'Mean': [mean], 'Std': [std]})
        new_mean_stds= pd.concat((mean_stds, new_row)).drop_duplicates(subset='CryptoName')
        new_mean_stds.to_csv('Processed Prices/Scaler Std.csv', index=False)
        return scaled_array

    
if __name__ =='__main__':
    input_length=50
    output_length=2
    interval='2H'
    test_range=(0.75, 0.85)
    data_wrangling('Bitcoin', input_length, output_length, test_range, freq=interval)
    # data_wrangling('Ethereum', input_length, output_length)
    # data_wrangling('Cardano', input_length, output_length)
    # data_wrangling('Binance Coin', input_length, output_length)
    # data_wrangling('Dogecoin', input_length, output_length)
    # data_wrangling('Paxgold', input_length, output_length)
    # data_wrangling('Ripple', input_length, output_length)
    # data_wrangling('Solana', input_length, output_length)
    # data_wrangling('Tron', input_length, output_length)






