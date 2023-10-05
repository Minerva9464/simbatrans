import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def data_wrangling(crypto_name: str, input_length, output_length):
    crypto_prices=pd.read_csv(f'Raw Prices/{crypto_name} Prices.csv')
    print(crypto_prices.head(5))
    print('=======================================')
    # print(bitcoin_prices.columns)
    opening_prices=crypto_prices.OpeningPrice
    closing_prices=crypto_prices.ClosingPrice

    # print(opening_prices)
    # print(closing_prices)

    avg_prices=(opening_prices+closing_prices)/2
    # print(avg_prices)
    len_avg=len(avg_prices)
    plt.plot(avg_prices)
    plt.show()

    avg_prices=min_max_scaler(avg_prices, crypto_name)

    row_length=input_length+output_length

    price_table=np.zeros((len_avg-row_length+1, row_length))
    print(price_table.shape)
    print('=======================================')

    for row in tqdm(range(price_table.shape[0])):
        price_table[row, :]=avg_prices[row: row+row_length]

    # print(price_table)
    
    prices_table_df=pd.DataFrame(price_table)

    price_table_train, price_table_test = train_test_split(price_table, test_size=0.2, random_state=38)
    prices_table_train_df=pd.DataFrame(price_table_train)
    prices_table_test_df=pd.DataFrame(price_table_test)

    prices_table_df.to_csv(f'Processed Prices/{crypto_name}.csv', index=False)
    prices_table_train_df.to_csv(f'Processed Prices/{crypto_name} Train.csv', index=False)
    prices_table_test_df.to_csv(f'Processed Prices/{crypto_name} Test.csv', index=False)


def min_max_scaler(v, crypto_name):
    max_v=v.max()
    min_v=v.min()
    scaled_v= (v-min_v)/(max_v-min_v)
    scaled_v=scaled_v
    
    min_maxes = pd.read_csv('Processed Prices/Scaler.csv')
    new_row=pd.DataFrame({'CryptoName': [crypto_name], 'Min': [min_v], 'Max': [max_v]})
    new_min_maxes= pd.concat((min_maxes, new_row)).drop_duplicates(subset='CryptoName')
    new_min_maxes.to_csv('Processed Prices/Scaler.csv', index=False)
    return scaled_v


input_length=100
output_length=20
data_wrangling('Bitcoin', input_length, output_length)


