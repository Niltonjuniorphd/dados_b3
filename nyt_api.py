#%%
import requests
import pandas as pd
import os
from tqdm import tqdm
import time

#%%
def nyt_summary2(key_word):
    NYT_KEY = os.environ["NYT_KEY"]
    all_articles = []

    for page in tqdm(range(0, 5), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'):
        url = f'https://api.nytimes.com/svc/search/v2/articlesearch.json?q={key_word}&page={page}&api-key={NYT_KEY}'
        response = requests.get(url) 
        time.sleep(5)

        if response.status_code == 200:  # Verifica se a requisição foi bem-sucedida
            data = response.json()
            articles = data.get("response", {}).get("docs", [])
            print(f'number in articles: {len(articles)}')
            for article in articles:
                all_articles.append({
                    "source": article.get("source", ""),
                    "headline": article.get("headline", {}).get("main", ""),
                    "snippet": article.get("snippet", ""),
                    "lead_paragraph": article.get("lead_paragraph", ""),
                    "pub_date": article.get("pub_date", ""),
                    "web_url": article.get("web_url", "")
                })
        
        else:
            print(f'Error: {response.status_code}') 

    # Cria um DataFrame a partir da lista de artigos
    df = pd.DataFrame(all_articles)
    return df


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

#%%
nyt_news_data = pd.DataFrame()
for focus_key in phrases:
    print(focus_key)
    df = nyt_summary2(focus_key)
    nyt_news_data = pd.concat([nyt_news_data, df], ignore_index=True, axis=0)


# %%
nyt_news_data.to_csv('nyt_news_data.csv')

#%%
NYT_KEY = os.environ["NYT_KEY"]
key_word = 'Brazilian real volatility dollar exchange news'
url = f'https://api.nytimes.com/svc/search/v2/articlesearch.json?q={key_word}&page={1}&api-key={NYT_KEY}'
response = requests.get(url)
response.status_code
data = response.json()
articles = data.get("response", {}).get("docs", [])
print(f'number in articles: {len(articles)}')

all_articles = []
for article in articles:
    all_articles.append({
        "source": article.get("source", ""),
        "headline": article.get("headline", {}).get("main", ""),
        "snippet": article.get("snippet", ""),
        "lead_paragraph": article.get("lead_paragraph", ""),
        "pub_date": article.get("pub_date", ""),
        "web_url": article.get("web_url", "")
    })
# %%
