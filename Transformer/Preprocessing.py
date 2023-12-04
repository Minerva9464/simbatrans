import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def data_wrangling(crypto_name: str, input_length, output_length, freq='2H'):
    crypto_prices=pd.read_csv(f'Raw Prices/{crypto_name} Prices.csv')

    opening_prices=crypto_prices.OpeningPrice
    closing_prices=crypto_prices.ClosingPrice

    start_time = crypto_prices.DateTime.iloc[0]
    end_time = crypto_prices.DateTime.iloc[-1]
    date_range=pd.date_range(start_time, end_time, freq=freq).to_list()

    avg_prices=(opening_prices+closing_prices)/2
    avg_prices=opening_prices
    len_avg=len(avg_prices)

    plt.figure(figsize=(10,6))
    plt.plot(date_range[:len_avg], avg_prices, color='teal')
    plt.title(crypto_name, fontdict={'size':24})
    plt.xlabel('Year')
    plt.ylabel('Price (USD)')
    plt.savefig(f'Achievements/{crypto_name} All Time Prices.png', dpi=600)
    plt.close()
    # plt.show()

    # Train:
    row_length=input_length+output_length
    price_table=np.zeros((len_avg-row_length+1, row_length))
    print('=======================================')

    for row in tqdm(range(price_table.shape[0])):
        price_table[row, :]=avg_prices[row: row+row_length]

    price_table_train, _ = train_test_split(price_table, test_size=0.0001, random_state=38)
    price_table_train_df=pd.DataFrame(scale(price_table_train, 'std', crypto_name))

    # Test
    price_table_test=np.zeros(((len_avg-input_length)//output_length, row_length))
    for row in tqdm(range(price_table_test.shape[0])):
        price_table_test[row, :]=avg_prices[row*output_length: row*output_length+row_length]
    prices_table_test_df=pd.DataFrame(price_table_test)
    
    price_table_train_df.to_csv(f'Processed Prices/{crypto_name} Train.csv', index=False)
    prices_table_test_df.to_csv(f'Processed Prices/{crypto_name} Test.csv', index=False)
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
    data_wrangling('Bitcoin', input_length, output_length, freq=interval)
    # data_wrangling('Ethereum', input_length, output_length)
    # data_wrangling('Cardano', input_length, output_length)
    # data_wrangling('Binance Coin', input_length, output_length)
    # data_wrangling('Dogecoin', input_length, output_length)
    # data_wrangling('Paxgold', input_length, output_length)
    # data_wrangling('Ripple', input_length, output_length)
    # data_wrangling('Solana', input_length, output_length)
    # data_wrangling('Tron', input_length, output_length)






