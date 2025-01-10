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

    for page in range(0, 3):
        url = f'https://api.nytimes.com/svc/search/v2/articlesearch.json?q={key_word}&begin_date=20200101&page={page}&api-key={NYT_KEY}'
        
        time.sleep(10)
        response = requests.get(url) 
        time.sleep(1)

        if response.status_code == 200:  # Verifica se a requisição foi bem-sucedida
            data = response.json()
            articles = data.get("response", {}).get("docs", [])
            print(f'number of news in articles: {len(articles)} on page {page}')
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
    "Volatile Brazilian currency exchange rate dollar",
    "Exchange rate fluctuation in Brazil dollar",
    "Brazilian real volatility dollar exchange",
    "Fluctuating exchange rates in Brazil dollar",
    "Brazil's currency instability exchange rate",
    "Central Bank of Brazil exchange rate update",
    "Brazilian real dollar volatility report",
    "Brazil's exchange rate market volatility",
    "Dollar exchange rate trends in Brazil",
    "Brazilian real fluctuations central bank",
    "Brazil's currency and dollar rate volatility",
    "Exchange rate dynamics in Brazil's market",
    "Brazilian economy exchange rate dollar trends",
    "Real-dollar volatility in Brazil central bank",
    "Brazil's monetary policy exchange rate",
    "Floating exchange rate trends in Brazil",
    "Volatile dollar exchange in Brazil's economy",
    "Central Bank of Brazil dollar volatility analysis",
    "Brazilian real depreciation exchange rate",
    "Brazil's currency market volatility dollar"
]

#%%
nyt_data = pd.DataFrame()
for focus_key in phrases:
    print(focus_key)
    time.sleep(5)
    df = nyt_summary2(focus_key)
    nyt_data = pd.concat([nyt_data, df], ignore_index=True, axis=0)


 # %%
nyt_data.to_csv('nyt_data.csv')

#%%
NYT_KEY = os.environ["NYT_KEY"]
key_word = 'Brazilian economy exchange rate dollar trends'
url = f'https://api.nytimes.com/svc/search/v2/articlesearch.json?q={key_word}&begin_date=20200101&page={1}&api-key={NYT_KEY}'
response = requests.get(url)
print(response.status_code)
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

all_articles
# %%
