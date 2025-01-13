#%%
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from functions import metrics_plot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
import string
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
download('vader_lexicon')
download('stopwords')
download('punkt_tab')
download('wordnet')
import numpy as np
from sklearn.cluster import KMeans


class CreateFeatures2(BaseEstimator, TransformerMixin):
    def __init__(self, text_columns, max_features=20):
        self.text_columns = text_columns
        self.sia = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.punctuation = set(string.punctuation)
        self.max_features = max_features
        self.tfidf_vectorizers = {col: TfidfVectorizer(max_features=self.max_features) for col in self.text_columns}

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        # Tokeniza e converte para minúsculas
        tokens = word_tokenize(text.lower())
        # Remove stopwords e pontuação
        tokens = [
            word for word in tokens if word not in self.stop_words and word not in self.punctuation]
        lemmatized_text = " ".join(self.lemmatizer.lemmatize(word) for word in tokens)  # Lematiza cada palavra
        lemmatized_text = lemmatized_text.strip().replace("'s", "")  # Remove espaços em branco extras
        return lemmatized_text

    def fit(self, X, y=None):
        for text_column in self.text_columns:
            cleaned_texts = X[text_column].apply(self.preprocess_text)
            self.tfidf_vectorizers[text_column].fit(cleaned_texts)
        return self

    def transform(self, X):
        df = X.copy()

        for text_column in self.text_columns:
            # Pré-processamento do texto (lemmatização e remoção de ruídos)
            df[f'cleaned_{text_column}'] = df[text_column] #.apply(self.preprocess_text)

            # Criação de features baseadas no texto
            df[f'char_count_{text_column}'] = df[f'cleaned_{text_column}'].apply(len)
            df[f'word_count_{text_column}'] = df[f'cleaned_{text_column}'].apply(lambda x: len(x.split()))
            df[f'unique_word_count_{text_column}'] = df[f'cleaned_{text_column}'].apply(lambda x: len(set(x.split())))
            df[f'avg_word_length_{text_column}'] = df[f'cleaned_{text_column}'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()) if x.split() else 0)
            df[f'polarity_{text_column}'] = df[f'cleaned_{text_column}'].apply(lambda x: TextBlob(x).sentiment.polarity)
            df[f'subjectivity_{text_column}'] = df[f'cleaned_{text_column}'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

            # Análise de sentimento com SentimentIntensityAnalyzer
            sentiment_scores = df[f'cleaned_{text_column}'].apply(lambda x: self.sia.polarity_scores(x))
            df[f'neg_{text_column}'] = sentiment_scores.apply(lambda x: x['neg'])
            df[f'neu_{text_column}'] = sentiment_scores.apply(lambda x: x['neu'])
            df[f'pos_{text_column}'] = sentiment_scores.apply(lambda x: x['pos'])
            df[f'compound_{text_column}'] = sentiment_scores.apply(lambda x: x['compound'])

            # TF-IDF Features
            tfidf_matrix = self.tfidf_vectorizers[text_column].transform(df[f'cleaned_{text_column}'])
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                columns=[f"tfidf_{text_column}_{i}" for i in range(self.max_features)]
            )
            tfidf_df.index = df.index  # Alinha os índices
            df = pd.concat([df, tfidf_df], axis=1)

        # Retorna apenas as colunas geradas, excluindo a coluna intermediária 'cleaned_*'
        feature_columns = [
            col for col in df.columns if col not in self.text_columns and not col.startswith("cleaned_")]
        return df[feature_columns].round(2)


# %%
dataset_path = './news_dataset/Gnews_large.csv'
print(f'loading news_dataset from {dataset_path}')
try:
    df = pd.read_csv(f'{dataset_path}').reset_index(drop=True)
    dollar = pd.read_csv('./dollar_exchange_brazil_data/brazilian_dolar_real_exchange_data_BC_01-09-2025.csv', encoding='utf-8', sep=',').reset_index(drop=True)
    print(f'\033[92m-----loading dataset and dollar files done...\033[0m')
except:
    print(f'fail loading files')

dollar['date_dollar'] = pd.to_datetime(dollar['dataHoraCotacao']).dt.strftime('%Y-%m-%d')
dollar = dollar.drop(columns=['cotacaoVenda', 'dataHoraCotacao'])

df['date'] = pd.to_datetime(df['published date']).dt.strftime('%Y-%m-%d')
df = df.reset_index(drop=True)
df.drop(columns=['title', 'publisher'])
df

df_merged = pd.merge(df, dollar, how='left', left_on='date', right_on='date_dollar')
df_merged['cotacaoCompra'] = df_merged['cotacaoCompra'].str.replace(',', '.').astype(float)
df_merged = df_merged.drop(columns=['date_dollar', 'date'])

print(f'\nsum na:\n{df_merged.isna().sum()}\n')
df_merged = df_merged.dropna()

print(f'\nsum duplicates: {sum(df_merged.duplicated())}\n')
df_merged = df_merged.drop_duplicates()

df_merged = df_merged.reset_index(drop=True)# %

# %%
X = df_merged[['description']]
y = df_merged['cotacaoCompra']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline = Pipeline([
    ('features', CreateFeatures2(text_columns=X_train.columns)),
    ('clf', RandomForestRegressor(random_state=42))

])

pipeline.fit(X_train, y_train)

# Predições com o melhor modelo
y_pred_train = pipeline.predict(X_train)
y_pred = pipeline.predict(X_test)

metrics_plot(y_test, y_pred, y_train, y_pred_train, pipeline=pipeline)


# %%
cf = CreateFeatures2(text_columns=X_train.columns)
df_features = cf.fit_transform(df_merged)

X = df_features.drop(['title', 'published date', 'url', 'publisher', 'cotacaoCompra'], axis=1)
y = df_features['cotacaoCompra']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)

metrics_plot(y_test.values, y_pred, y_train, y_pred_train, pipeline=model)

# %%
erro = ((y_pred - y_test)**2 )**0.5
x_erro = X_test[erro < 0.05]

y_preds = pd.Series(y_pred, index=y_test.index)
sns.scatterplot(x=y_test, y=y_pred)
sns.scatterplot(x=y_test.loc[x_erro.index], y=y_preds.loc[x_erro.index], alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.show()

df_cluster = df_features.drop(['title', 'published date', 'url', 'publisher', 'cotacaoCompra'], axis=1)

scaler = StandardScaler()
df_cluster = pd.DataFrame(scaler.fit_transform(df_cluster), columns=df_cluster.columns)
# 1. Calcular a distância euclidiana para cada linha em relação à origem
df_cluster['euclidean_distance'] = np.sqrt((df_cluster ** 2).sum(axis=1))

# 2. Agrupamento por clusterização KNN
# Defina o número de clusters desejados
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_cluster['cluster'] = kmeans.fit_predict(df_cluster.drop(columns=['euclidean_distance']))

# Resultado final
df_cluster

df_clusterb = pd.concat([df_features[['title', 'published date', 'url', 'publisher', 'cotacaoCompra']], df_cluster], axis=1)

sns.scatterplot(x=df_clusterb['cotacaoCompra'], y=df_clusterb['euclidean_distance'], hue=df_clusterb['cluster'], palette='viridis')
sns.scatterplot(x=df_clusterb['cotacaoCompra'].loc[x_erro.index], y=df_clusterb['euclidean_distance'].loc[x_erro.index], marker='+', color='red')
plt.show()

for column in df_clusterb.iloc[:, 5:10]:
    sns.scatterplot(x=df_clusterb['cotacaoCompra'], y=df_clusterb[column], hue=df_clusterb['cluster'], palette='viridis')
    sns.scatterplot(x=df_clusterb['cotacaoCompra'].loc[x_erro.index], y=df_clusterb[column].loc[x_erro.index], marker='+', color='red')
    plt.show()

# %%
sns.boxplot(x=df_clusterb['cluster'], y=df_clusterb['cotacaoCompra'])

# %%
df_merged_f = df_merged[df_clusterb['cluster'] == 1]

X = df_merged_f[['description']]
y = df_merged_f['cotacaoCompra']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline = Pipeline([
    ('features', CreateFeatures2(text_columns=X_train.columns)),
    ('clf', RandomForestRegressor(random_state=42))

])

pipeline.fit(X_train, y_train)

# Predições com o melhor modelo
y_pred_train = pipeline.predict(X_train)
y_pred = pipeline.predict(X_test)

metrics_plot(y_test, y_pred, y_train, y_pred_train, pipeline=pipeline)
# %%
