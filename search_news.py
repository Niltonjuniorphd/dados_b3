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
    "Volatile exchange Lula says dollar",
    "Exchange fluctuation Lula says",
    "real volatility Lula says",
    "Brazilian real dollar Lula says ",
    "Brazil exchange rate Lula says ",
    "Dollar rate Lula says",    
]

phrases2 = [
    "brazilian volatile currency news",
]
#focus_key = 'brazilian central bank dollar rate news'

# send the focus_key to drive search box
for focus_key in phrases2:
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
    df0.to_csv(f'./lula_say/news_df_short_{today}_{focus_key}.csv', encoding='utf-8')

    print(f'\033[92m\n--- now run "python create_features.py" to prepare the dataset \033[0m')


# %%
