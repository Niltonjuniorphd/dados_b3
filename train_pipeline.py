# %%
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



#functions and classes

def merge_dollar(dataset, dollar):
    # Convert date column and format as required
    dollar['date'] = pd.to_datetime(dollar['dataHoraCotacao']).dt.strftime('%Y-%m-%d')
    dollar = dollar.drop(columns=['cotacaoVenda', 'dataHoraCotacao'])
    df_merged = None  # Initialize df_merged to avoid UnboundLocalError

    try:
        df_merged = pd.merge(dataset, dollar, how='left', left_on='dates_b', right_on='date')
        df_merged['cotacaoCompra'] = df_merged['cotacaoCompra'].str.replace(',', '.').astype(float)
        df_merged = df_merged.drop(columns=['dates_b', 'date'])
        
        print(f'sum na: {df_merged.isna().sum()}')
        df_merged = df_merged.dropna()
        
        print(f'sum duplicates: {sum(df_merged.duplicated())}')
        df_merged = df_merged.drop_duplicates()

        df_merged = df_merged.reset_index(drop=True)

    except Exception as e:
        print(f'Something went wrong during the merge: {e}')
        return None  # Return None to indicate failure

    if df_merged is not None:
        return df_merged
    else:
        return None


class CreateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, text_columns):
        self.text_columns = text_columns
        self.sia = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        # Nenhuma operação de ajuste necessária
        return self

    def transform(self, X):
        df = X.copy()

        for text_column in self.text_columns:
            # Criação de features baseadas no texto
            # df[f'char_count_{text_column}'] = df[text_column].apply(len)
            # df[f'word_count_{text_column}'] = df[text_column].apply(lambda x: len(x.split()))
            # df[f'unique_word_count_{text_column}'] = df[text_column].apply(lambda x: len(set(x.split())))
            # df[f'avg_word_length_{text_column}'] = df[text_column].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()))
            # df[f'punctuation_count_{text_column}'] = df[text_column].apply(lambda x: sum(1 for char in x if char in "!?.,;:"))
            df[f'polarity_{text_column}'] = df[text_column].apply(
                lambda x: TextBlob(x).sentiment.polarity)
            df[f'subjectivity_{text_column}'] = df[text_column].apply(
                lambda x: TextBlob(x).sentiment.subjectivity)

            # Análise de sentimento com SentimentIntensityAnalyzer
            sentiment_scores = df[text_column].apply(lambda x: self.sia.polarity_scores(x))
            df[f'neg_{text_column}'] = sentiment_scores.apply(lambda x: x['neg'])
            df[f'neu_{text_column}'] = sentiment_scores.apply(lambda x: x['neu'])
            df[f'pos_{text_column}'] = sentiment_scores.apply(lambda x: x['pos'])
            df[f'compound_{text_column}'] = sentiment_scores.apply(lambda x: x['compound'])

        return df.iloc[:, 4:]  # Retorna apenas as colunas geradas


class CreateFeatures2(BaseEstimator, TransformerMixin):
    def __init__(self, text_columns, max_features=50):
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
            df[f'cleaned_{text_column}'] = df[text_column].apply(
                self.preprocess_text)

            # Criação de features baseadas no texto
            #df[f'char_count_{text_column}'] = df[f'cleaned_{text_column}'].apply(len)
            #df[f'word_count_{text_column}'] = df[f'cleaned_{text_column}'].apply(lambda x: len(x.split()))
            #df[f'unique_word_count_{text_column}'] = df[f'cleaned_{text_column}'].apply(lambda x: len(set(x.split())))
            #df[f'avg_word_length_{text_column}'] = df[f'cleaned_{text_column}'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()) if x.split() else 0)
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


def train_test(df_merged):

    df_merged_f = df_merged[df_merged['cotacaoCompra'] >= 4.5].copy()

    X = df_merged_f.drop(columns=['cotacaoCompra'], axis=1)
    y = df_merged_f['cotacaoCompra']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    return X_train, X_test, y_train, y_test


def train_test_diff(df_merged):

    # df_merged_f = df_merged[df_merged['cotacaoCompra'] >= 4.5].copy()
    df_merged_f = df_merged
    df_merged_f.loc[:, 'diff'] = df_merged_f['cotacaoCompra'].diff()
    df_merged_f = df_merged_f.dropna()
    X = df_merged_f.drop(columns=['cotacaoCompra', 'diff'], axis=1)
    y = df_merged_f['diff']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


# %%
dataset_path = './news_dataset/News_dataset_lula_say_short_b.csv'
print(f'loading news_dataset from {dataset_path}')
try:
    df = pd.read_csv(f'{dataset_path}').reset_index(drop=True)
    dollar = pd.read_csv(
        './dollar_exchange_brazil_data/brazilian_dolar_real_exchange_data_BC_01-09-2025.csv', encoding='utf-8', sep=',').reset_index(drop=True)
    print(f'\033[92m-----loading dataset and dollar files done...\033[0m')
except:
    print(f'fail loading files')

# for nyt news dataset:
# df0 = df0.drop(columns=['Unnamed: 0', 'source', 'web_url'])
# df0['dates_b'] = pd.to_datetime(df0['pub_date']).dt.strftime('%Y-%m-%d')
# df0 = df0.drop(columns=['pub_date'])

# for google search dataset:
df0 = df.drop(columns=['links', 'dates']).drop_duplicates().reset_index(drop=True)

# %%
df_merged = merge_dollar(df0, dollar)
df_merged['cotacaoCompra'] = df_merged['cotacaoCompra'].round(2)
df_merged

X_train, X_test, y_train, y_test = train_test(df_merged)
# X_train, X_test, y_train, y_test = train_test_diff(df_merged)


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

cf = CreateFeatures2(text_columns=['texts', 'headlines'])
df_features = cf.fit_transform(df_merged)
df_features = df_features[df_features['cotacaoCompra'] >= 4.5].copy()

X = df_features.drop(columns=['cotacaoCompra'], axis=1)
y = df_features['cotacaoCompra']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
metrics_plot(y_test, y_pred, y_train, y_pred_train, pipeline=model)

# %%


# Geração das features com o transformador
cf = CreateFeatures2(text_columns=['texts', 'headlines'])
df_features = cf.fit_transform(df_merged)

# Filtrar os dados com base na condição
df_features = df_features[df_features['cotacaoCompra'] >= 4.5].copy()

# Separar variáveis preditoras e alvo
X = df_features.drop(columns=['cotacaoCompra'], axis=1)
y = df_features['cotacaoCompra']

# Dividir em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

# Aplicar StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinar o modelo
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred = model.predict(X_test_scaled)
y_pred_train = model.predict(X_train_scaled)

# Avaliar o modelo
metrics_plot(y_test, y_pred, y_train, y_pred_train, pipeline=model)


# %%
X_train, X_test, y_train, y_test = train_test(df_merged)
pipeline = Pipeline([
    ('features', CreateFeatures2(text_columns=X_train.columns)),
    ('clf', RandomForestRegressor(random_state=42))
])
param_grid = {
    'clf__n_estimators': [50, 100, 300],
    'clf__max_depth': [10, 20, 30, None],
    'clf__min_samples_split': [2, 3, 5],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__criterion': ['squared_error', 'poisson']
}

grid_search = GridSearchCV(estimator=pipeline,
                           param_grid=param_grid,
                           cv=3,
                           scoring='neg_root_mean_squared_error',
                           n_jobs=1,
                           verbose=2)

grid_search.fit(X_train, y_train)

print('\033[92m\n-----  GridSearchCV done...  -----\033[0m\n')
print(f"Best Param: {grid_search.best_params_}")
print(f"Best performance (RMSE): {-grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_

# Predições com o melhor modelo
y_pred = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

metrics_plot(y_test, y_pred, y_train, y_pred_train, pipeline=best_model)


# %%
outoftime = pd.read_csv(f'news_df_30_pgs_ok.csv', index_col=0)
outoftime = outoftime.drop(columns=['links', 'dates', 'dates_b'])
# %%
y_out = pipeline.predict(outoftime)
# %%
