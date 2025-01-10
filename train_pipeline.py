#%%
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV


#from datetime import datetime

#%%
dataset_path = './news_dataset/News_dataset_lula_say.csv'
print(f'loading news_dataset from {dataset_path}')
try:
    df0 = pd.read_csv(f'{dataset_path}')
    dollar = pd.read_csv('./dollar_exchange_brazil_data/brazilian_dolar_real_exchange_data_BC_01-09-2025.csv', encoding='utf-8', sep=',')
    print(f'\033[92m-----loading dataset and dollar files done...\033[0m')
except:
    print(f'fail loading files')

# for nyt news dataset:
#df0 = df0.drop(columns=['Unnamed: 0', 'source', 'web_url'])
#df0['dates_b'] = pd.to_datetime(df0['pub_date']).dt.strftime('%Y-%m-%d')
#df0 = df0.drop(columns=['pub_date'])

# for google search dataset:
df0 = df0.drop(columns=['links', 'dates'])


def merge_dollar(dataset, dollar):
    # Convert date column and format as required
    dollar['date'] = pd.to_datetime(dollar['dataHoraCotacao']).dt.strftime('%Y-%m-%d')
    dollar = dollar.drop(columns=['cotacaoVenda', 'dataHoraCotacao'])
    df_merged = None  # Initialize df_merged to avoid UnboundLocalError
    
    try:
        df_merged = pd.merge(dataset, dollar, how='left', left_on='dates_b', right_on='date')
        df_merged['cotacaoCompra'] = df_merged['cotacaoCompra'].str.replace(',', '.').astype(float)
        df_merged = df_merged.drop(columns=['dates_b', 'date'])
        df_merged = df_merged.dropna()
        df_merged = df_merged.drop_duplicates()

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
            df[f'char_count_{text_column}'] = df[text_column].apply(len)
            df[f'word_count_{text_column}'] = df[text_column].apply(lambda x: len(x.split()))
            df[f'unique_word_count_{text_column}'] = df[text_column].apply(lambda x: len(set(x.split())))
            df[f'avg_word_length_{text_column}'] = df[text_column].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()))
            #df[f'punctuation_count_{text_column}'] = df[text_column].apply(lambda x: sum(1 for char in x if char in "!?.,;:"))
            df[f'polarity_{text_column}'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity)
            df[f'subjectivity_{text_column}'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.subjectivity)

            # Análise de sentimento com SentimentIntensityAnalyzer
            sentiment_scores = df[text_column].apply(lambda x: self.sia.polarity_scores(x))
            df[f'neg_{text_column}'] = sentiment_scores.apply(lambda x: x['neg'])
            df[f'neu_{text_column}'] = sentiment_scores.apply(lambda x: x['neu'])
            df[f'pos_{text_column}'] = sentiment_scores.apply(lambda x: x['pos'])
            df[f'compound_{text_column}'] = sentiment_scores.apply(lambda x: x['compound'])

        return df.iloc[:, 4:]  # Retorna apenas as colunas geradas


def train_test(df_merged):

    #df_merged_f = df_merged[df_merged['cotacaoCompra'] >= 4.5].copy()

    X = df_merged.drop(columns=['cotacaoCompra'], axis=1)
    y = df_merged['cotacaoCompra']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test 


def train_test_diff(df_merged):

    #df_merged_f = df_merged[df_merged['cotacaoCompra'] >= 4.5].copy()
    df_merged_f = df_merged
    df_merged_f.loc[:,'diff'] = df_merged_f['cotacaoCompra'].diff()
    df_merged_f = df_merged_f.dropna()
    X = df_merged_f.drop(columns=['cotacaoCompra', 'diff'], axis=1)
    y = df_merged_f['diff']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test 


# %%
df_merged = merge_dollar(df0, dollar)
df_merged

X_train, X_test, y_train, y_test = train_test(df_merged)
#X_train, X_test, y_train, y_test = train_test_diff(df_merged)


pipeline = Pipeline([
    ('features', CreateFeatures(text_columns=X_train.columns)),
    ('clf', RandomForestRegressor(random_state=42))
])

pipeline.fit(X_train, y_train)

# Predições com o melhor modelo
y_pred = pipeline.predict(X_test)
y_pred_train = pipeline.predict(X_train)

RMSE = (sum((y_pred - y_test)**2)/len(y_test)**0.5)
RMSE_train = (sum((y_pred_train - y_train)**2)/len(y_train)**0.5)
MAE = mean_absolute_error(y_test, y_pred)
MAE_train = mean_absolute_error(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred)
r2_train = r2_score(y_train, y_pred_train)

# Exibindo os resultados
print(f"RMSE (test): {RMSE:.3f}")
print(f"RMSE (train): {RMSE_train:.3f}")
print(f"MAE (test): {MAE:.3f}")
print(f"MAE (train): {MAE_train:.3f}")
print(f"R² (test): {r2_test:.3f}")
print(f"R² (train): {r2_train:.3f}")
print('\nmetrics successfully calculated...\n')
print('plots will rise in a window. Close the window to see the next plot.')
# - Visualizações para análise de erros
# Resíduos
residuals = y_test - y_pred
residuals_train = y_train - y_pred_train

plt.figure(figsize=(14, 10))

plt.subplot(3, 2, 1)
sns.histplot(y_test, kde=True, color='blue', bins=30)
plt.title('Distribuição dos valores - Test')
plt.xlabel('valores')
plt.ylabel('Frequência')

plt.subplot(3, 2, 2)
sns.histplot(y_train, kde=True, color='blue', bins=30)
plt.title('Distribuição dos valores - Train')
plt.xlabel('valores')
plt.ylabel('')


plt.subplot(3, 2, 3)
sns.histplot(residuals, kde=True, color='blue', bins=30)
plt.title('Distribuição dos Resíduos - Test')
plt.xlabel('Resíduo')
plt.ylabel('Frequência')

plt.subplot(3, 2, 4)
sns.histplot(residuals_train, kde=True, color='red', bins=30)
plt.title('Distribuição dos Resíduos - Train')
plt.xlabel('Resíduo')
plt.ylabel('')

plt.subplot(3, 2, 5) 
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Previsões vs Real - Teste')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')

# Treinamento
plt.subplot(3, 2, 6) 
plt.scatter(y_train, y_pred_train, color='red', alpha=0.7)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='blue', linestyle='--')
plt.title('Previsões vs Real - Treinamento')
plt.xlabel('Valores Reais')
plt.ylabel('')

plt.tight_layout()
plt.show()

# - Análise de importância das variáveis (feature importance)
# Exibindo a importância das variáveis do modelo Random Forest
importances = pipeline.named_steps['clf'].feature_importances_
features = pipeline.named_steps['clf'].feature_names_in_

# Organizando as variáveis por importância
indices = importances.argsort()

# Plotando
plt.figure(figsize=(10, 6))
plt.barh(range(len(features)), importances[indices], align='center')
plt.yticks(range(len(features)), [features[i] for i in indices])
plt.title('Importância das Variáveis')
plt.xlabel('Importância')
plt.ylabel('Variáveis')
plt.show()

print('--- End of program ---')

#%%
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
                           n_jobs=-1,
                           verbose=2)

grid_search.fit(X_train, y_train)

print('\033[92m\n-----  GridSearchCV done...  -----\033[0m\n')
print(f"Best Param: {grid_search.best_params_}")
print(f"Best performance (RMSE): {-grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_

# Predições com o melhor modelo
y_pred = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

# - Avaliação do desempenho
# Cálculo do RMSE já foi feito, mas podemos também calcular o MAE (Erro Absoluto Médio)
MAE = mean_absolute_error(y_test, y_pred)
MAE_train = mean_absolute_error(y_train, y_pred_train)

# Cálculo do R² (coeficiente de determinação) para o modelo
r2_test = r2_score(y_test, y_pred)
r2_train = r2_score(y_train, y_pred_train)

# Exibindo os resultados
print(f"RMSE (test): {RMSE:.3f}")
print(f"RMSE (train): {RMSE_train:.3f}")
print(f"MAE (test): {MAE:.3f}")
print(f"MAE (train): {MAE_train:.3f}")
print(f"R² (test): {r2_test:.3f}")
print(f"R² (train): {r2_train:.3f}")
print('\nmetrics successfully calculated...\n')
print('plots will rise in a window. Close the window to see the next plot.')
# - Visualizações para análise de erros
# Resíduos
residuals = y_test - y_pred
residuals_train = y_train - y_pred_train

plt.figure(figsize=(14, 10))

plt.subplot(3, 2, 1)
sns.histplot(y_test, kde=True, color='blue', bins=30)
plt.title('Distribuição dos valores - Test')
plt.xlabel('valores')
plt.ylabel('Frequência')

plt.subplot(3, 2, 2)
sns.histplot(y_train, kde=True, color='blue', bins=30)
plt.title('Distribuição dos valores - Train')
plt.xlabel('valores')
plt.ylabel('')


plt.subplot(3, 2, 3)
sns.histplot(residuals, kde=True, color='blue', bins=30)
plt.title('Distribuição dos Resíduos - Test')
plt.xlabel('Resíduo')
plt.ylabel('Frequência')

plt.subplot(3, 2, 4)
sns.histplot(residuals_train, kde=True, color='red', bins=30)
plt.title('Distribuição dos Resíduos - Train')
plt.xlabel('Resíduo')
plt.ylabel('')

plt.subplot(3, 2, 5) 
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Previsões vs Real - Teste')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')

# Treinamento
plt.subplot(3, 2, 6) 
plt.scatter(y_train, y_pred_train, color='red', alpha=0.7)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='blue', linestyle='--')
plt.title('Previsões vs Real - Treinamento')
plt.xlabel('Valores Reais')
plt.ylabel('')

plt.tight_layout()
plt.show()

# - Análise de importância das variáveis (feature importance)
# Exibindo a importância das variáveis do modelo Random Forest
importances = pipeline.named_steps['clf'].feature_importances_
features = pipeline.named_steps['clf'].feature_names_in_

# Organizando as variáveis por importância
indices = importances.argsort()

# Plotando
plt.figure(figsize=(10, 6))
plt.barh(range(len(features)), importances[indices], align='center')
plt.yticks(range(len(features)), [features[i] for i in indices])
plt.title('Importância das Variáveis')
plt.xlabel('Importância')
plt.ylabel('Variáveis')
plt.show()

print('--- End of program ---')





# %%
outoftime = pd.read_csv(f'news_df_30_pgs_ok.csv', index_col=0)
outoftime = outoftime.drop(columns=['links', 'dates', 'dates_b'])
# %%
y_out = pipeline.predict(outoftime)
# %%


