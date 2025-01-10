#%%
from functions import (call_driver,
                       send_focus_key,
                       get_text_and_links2,
                       converter_data)
import pandas as pd

#from datetime import datetime, timedelta
#import numpy as np

# instantiate the drive with predefined option
driver = call_driver()

# the words to focus on the serach in google:
# exempli:
# 'brazilian volatile currency rate dollar news'
# 'Exchange Rate Volatility Floating Brazil news'
# 'Brazilian currency volatility Lula brazil news'
# 'exchange rate volatility in Brazil news'
# 'brazilian central bank dollar rate news'
#
phrases = [
    "Volatile Brazilian currency exchange rate dollar news",
    "Exchange rate fluctuation in Brazil dollar news",
    "Brazilian real volatility dollar exchange news",
    "Fluctuating exchange rates in Brazil dollar news",
    "Brazil's currency instability exchange rate news",
    "Central Bank of Brazil exchange rate update news",
    "Brazilian real dollar volatility report news",
    "Brazil's exchange rate market volatility news",
    "Dollar exchange rate trends in Brazil news",
    "Brazilian real fluctuations central bank news",
    "Brazil's currency and dollar rate volatility news",
    "Exchange rate dynamics in Brazil's market news",
    "Brazilian economy exchange rate dollar trends news",
    "Real-dollar volatility in Brazil central bank news",
    "Brazil's monetary policy exchange rate news",
    "Floating exchange rate trends in Brazil news",
    "Volatile dollar exchange in Brazil's economy news",
    "Central Bank of Brazil dollar volatility analysis news",
    "Brazilian real depreciation exchange rate news",
    "Brazil's currency market volatility dollar news"
]


#focus_key = 'brazilian central bank dollar rate news'

# send the focus_key to drive search box
for focus_key in phrases:
    send_focus_key(driver, focus_key=focus_key)

    headlines, texts, links, dates = get_text_and_links2(driver, pg_num=100)

    df0 = pd.concat([headlines, texts, links, dates], axis=1)

    meses_pt_para_en = {
            "jan": "Jan", "fev": "Feb", "mar": "Mar", "abr": "Apr", 
            "mai": "May", "jun": "Jun", "jul": "Jul", "ago": "Aug", 
            "set": "Sep", "out": "Oct", "nov": "Nov", "dez": "Dec"
        }
    df0 = df0.dropna()

    df0['dates'] = df0['dates'].replace(meses_pt_para_en, regex=True)

    df0['dates_b'] = df0['dates'].apply(converter_data)

    df0['dates_b'] = [str(x).split(' ')[0] for x in df0['dates_b']]

    df0

    today = pd.Timestamp.today().date()
    df0.to_csv(f'./news_data/news_df_{today}_{focus_key}.csv')

    print(f'\033[92m\n--- now run "python create_features.py" to prepare the dataset \033[0m')


# %%
