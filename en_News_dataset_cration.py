#%%
import pandas as pd


files = [
    'news_df_2024-12-30_brasilian_currence_rate_dollar_news.csv',
    'news_df_2024-12-30_Brazilian currency volatility Lula.csv',
    'news_df_2024-12-30_cotação do dolar notícias.csv',
    'news_df_2024-12-30_brazilian volatile currence rate dollar news.csv',
    'news_df_2024-12-30_Dollar rise inflation brazil news.csv',
    'news_df_2024-12-30_Exchange Rate Volatility Floating Brazil.csv',
    'news_df_2024-12-30_USD dollar brazil real exchange.csv'
]
df0 = pd.DataFrame()
for file in files:
    print(file)
    df = pd.read_csv(file, index_col=0, encoding='ISO-8859-1', sep=',')
    df0 = pd.concat([df0, df])

df0.to_csv('en_News.csv')

