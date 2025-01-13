#%%
from gnews import GNews
import pandas as pd


# %%
gn = GNews()
gn.language='en',
gn.period = '5y'
#gn.start_date = (2020, 1, 1)
#gn.end_date = (2025, 1, 12)
gn.max_results = 150

# google_news.start_date = (2020, 1, 1) # Search from 1st Jan 2020

keys = [
    'lula says dollar',
    'lula says inflation',
    'lula says currence',
    'lula says economi',
    'lula says real',
    'lula says growth',
    'lula says budget',
    'lula says trade',
    'lula says tax',
    'lula says reform',
    'lula says market',
    'lula says export',
    'lula says policy',
    'lula says industry',
    'lula says investment'
]

news = pd.DataFrame()
for i, key in enumerate(keys):
    print(f'key {i+1}/{len(key)}: {key}')
    response2 = gn.get_news(key)
    df2 = pd.DataFrame(response2)
    news = pd.concat([news, df2], axis=0)
    print(f'{len(df2)} news found\n')


news['date'] = pd.to_datetime(news['published date']).dt.strftime('%Y-%m-%d')
news = news.reset_index(drop=True)
news.drop(columns=['title', 'publisher'])
news

# %%
news.to_csv('./news_dataset/Gnews_large.csv', index=False)





