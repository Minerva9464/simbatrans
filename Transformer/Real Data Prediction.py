import requests
from GetPrices import get_prices
import pandas as pd

def rescale(crypto_name, crypto_prices_scaled, min, max):
    result=pd.read_csv('Processed Prices/Scaler.csv')
    selected_crypto_min_max=result[result.CryptoName==crypto_name]
    min_crypto=selected_crypto_min_max.Min.iloc[0]
    max_crypto=selected_crypto_min_max.Max.iloc[0]
    crypto_prices_rescaled= crypto_prices_scaled*(max_crypto-min_crypto)+min_crypto
    return crypto_prices_rescaled
