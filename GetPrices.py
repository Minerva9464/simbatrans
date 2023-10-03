import requests
from datetime import datetime, timedelta
import json
import pandas as pd

def get_prices(symbol: str, start_time: datetime, interval: str='1h') -> pd.DataFrame:
    url = 'https://api.binance.com/api/v3/klines'

    start_time=str(int(start_time.timestamp()*1000))

    parameters={
        'symbol' : symbol,
        'interval' : interval,
        'startTime' : start_time,
        'limit': 1000
    }

    response=requests.get(url, params=parameters)
    data=pd.DataFrame(json.loads(response.text))

    data.columns=['OpenTime', 'OpeningPrice', 'HighPrice', 'LowPrice', 'ClosingPrice', 'TradingVolume','CloseTime', 
                'QAV', 'NumTrades','taker_base_vol', 'taker_quote_vol', 'ignore']
    data['DateTime'] = [datetime.fromtimestamp(int(x)/1000) for x in data.CloseTime]

    data = data.drop(['ignore', 'taker_base_vol', 'taker_quote_vol', 'OpenTime'], axis=1)
    
    return data

def get_all_prices(symbol, start_time=datetime(2001, 9, 1), end_time=datetime.now()):
    prices_data = pd.DataFrame({})

    while (start_time.year!=end_time.year or start_time.month!=end_time.month):
        prices_data = pd.concat([prices_data, get_prices(symbol=symbol, start_time=start_time)])
        start_time = prices_data.DateTime.iloc[-1]+timedelta(hours=1)
        print(f'Next Start Time of {coin_name} is: {start_time}')
    
    return prices_data

#---------------------------------------
#---------------------------------------
if __name__ == '__main__':
    symbols = {'BTCUSDT': 'Bitcoin',
                'ETHUSDT': 'Ethereum',
                'XRPUSDT': 'Ripple',
                'BNBUSDT': 'Binance Coin',
                'ADAUSDT': 'Cardano',
                'DOGEUSDT': 'Dogecoin',
                'SOLUSDT': 'Solana',
                'TRXUSDT': 'Tron',
                'PAXGUSDT': 'Paxgold',
                }

    start_time = datetime(2001,1,1)
    end_time = datetime.now()
    for symbol, coin_name in zip(symbols.keys(), symbols.values()):
        prices_data = get_all_prices(symbol, start_time=start_time, end_time=end_time)
        
        prices_data.to_csv(f'Prices/{coin_name} Prices.csv', index=False)
        print('================================================')
        print(f'||         {coin_name} is Saved! Yeeessss!        ||')
        print('================================================')
