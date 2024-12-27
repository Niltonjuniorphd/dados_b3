#%%
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
import matplotlib.pyplot as plt

#%%
df0 = pd.read_csv('news_df_30_pgs_ok.csv', index_col=0)
df0

# %%

df = df0.copy()
# 1. Contagem de caracteres
df['char_count'] = df['headlines'].apply(len)

# 2. Contagem de palavras
df['word_count'] = df['headlines'].apply(lambda x: len(x.split()))

# 3. Contagem de palavras únicas
df['unique_word_count'] = df['headlines'].apply(lambda x: len(set(x.split())))

# 4. Comprimento médio das palavras
df['avg_word_length'] = df['headlines'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()))

# 5. Contagem de pontuações
df['punctuation_count'] = df['headlines'].apply(lambda x: sum(1 for char in x if char in "!?.,;:"))

# 6. Análise de sentimento usando TextBlob
df['polarity'] = df['headlines'].apply(lambda x: TextBlob(x).sentiment.polarity)  # Varia entre -1 e 1
df['subjectivity'] = df['headlines'].apply(lambda x: TextBlob(x).sentiment.subjectivity)  # Varia entre 0 e 1

download('vader_lexicon')
# Inicializar o SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Gerar análise de sentimento
df['sentiment_scores'] = df['headlines'].apply(lambda x: sia.polarity_scores(x))

# Separar as pontuações em colunas individuais
df['neg'] = df['sentiment_scores'].apply(lambda x: x['neg'])  # Negatividade
df['neu'] = df['sentiment_scores'].apply(lambda x: x['neu'])  # Neutralidade
df['pos'] = df['sentiment_scores'].apply(lambda x: x['pos'])  # Positividade
df['compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])  # Escore geral


df

'''
# 7. Vetorização de palavras (TF ou TF-IDF como categoria)
vectorizer = CountVectorizer()
word_counts = vectorizer.fit_transform(df['headlines'])
word_count_df = pd.DataFrame(word_counts.toarray(), columns=vectorizer.get_feature_names_out())
df = pd.concat([df, word_count_df], axis=1)

print(df)
'''

# %%
df_features = df.iloc[:,4:]
df_features = df_features[df_features['dates_b'] != 'NaT']
df_features = df_features.drop(columns=['sentiment_scores'])
df_features = df_features.sort_values(by='dates_b', ascending=False)
df_features

# %%

dolar = pd.read_csv('2024_ExchangeRateFile_20241227_1.csv', index_col=0, encoding='ISO-8859-1', sep=';')  
dolar

# %%

df_merged = pd.merge(df_features, dolar, how='left', left_on='dates_b', right_on='RptDt')
df_merged = df_merged[df_merged['EcncIndDesc'] == 'Indicadores gerais']
df_merged['PricVal'] = df_merged['PricVal'].str.replace(',', '.').astype(float)
df_merged = df_merged.drop(columns=['Asst',	'TckrSymb', 'EcncIndDesc'])
df_merged

# %%
plt.scatter(df_merged['polarity'], df_merged['PricVal'])

#%%
plt.scatter(df_merged['subjectivity'], df_merged['PricVal'])

# %%
plt.scatter(pd.to_datetime(df_merged['dates_b']), df_merged['PricVal'])
plt.plot(pd.to_datetime(df_merged['dates_b']), df_merged['PricVal'])
plt.xticks(rotation=90)

# %%
plt.scatter(df_merged['avg_word_length'], df_merged['PricVal'])

# %%
plt.scatter(df_merged['neg'], df_merged['PricVal'])

# %%
plt.scatter(df_merged['pos'], df_merged['PricVal'])

# %%
plt.scatter(df_merged['neu'], df_merged['PricVal'])

# %%
plt.scatter(df_merged['compound'], df_merged['PricVal'])

# %%
df_merged.to_csv('news_df_features.csv')
# %%
