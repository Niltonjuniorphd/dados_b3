#%%
import pandas as pd
from gnews import GNews


#%%
google_news = GNews()
response = google_news.get_news('Brazilian currence dollar')
print(response)

df = pd.DataFrame(response)

df['date'] = pd.to_datetime(df['published date']).dt.strftime('%Y-%m-%d')
df


# %%
google_news2 = GNews(
    language='en',
    start_date=(2020, 1, 1),
    end_date=(2025, 1, 12),
    max_results=150
)
# google_news.start_date = (2020, 1, 1) # Search from 1st Jan 2020

keys = [
    'Brazilian dollar',
    'brazilian inflation',
    'brazilian currence'
]
news = pd.DataFrame()
for key in keys:
    response2 = google_news2.get_news(key)
    df2 = pd.DataFrame(response2)
    news = pd.concat([news, df2], axis=0)


news['date'] = pd.to_datetime(news['published date']).dt.strftime('%Y-%m-%d')
news = news.reset_index(drop=True)
news

# %%
news.to_csv('./news_dataset/Gnews.csv', index=False)
# %%
