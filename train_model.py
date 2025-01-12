#%%
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import time
from functions import metrics_plot


dataset_path = './news_dataset/News_dataset_lula_say.csv'
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

def train_test(df_merged):

    df_merged_f = df_merged[df_merged['cotacaoCompra'] >= 4.5].copy()

    X = df_merged_f.drop(columns=['cotacaoCompra'], axis=1)
    y = df_merged_f['cotacaoCompra']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    return X_train, X_test, y_train, y_test

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

df_merged = merge_dollar(df0, dollar)
df_merged['cotacaoCompra'] = df_merged['cotacaoCompra'].round(2)
df_merged

X_train, X_test, y_train, y_test = train_test(df_merged)


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


# %%
