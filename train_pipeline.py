#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
download('vader_lexicon')
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


#%%
dataset_path = 'en_News'


#%%
today = pd.Timestamp.today().date()
print(f'loading dataset news_df_features_{today}.csv')
try:
    #df0 = pd.read_csv(f'news_df_{today}.csv', index_col=0)
    df0 = pd.read_csv(f'{dataset_path}.csv', index_col=0)
    dollar = pd.read_csv('2024_ExchangeRateFile_20241230_1.csv', encoding='ISO-8859-1', sep=';')
except:
    print(f'file not found!')
print(f'\033[92m-----loading dataset news_df_{today} done...\033[0m')
print(f'\033[92m-----loading dataset news_df_features__{today} done...\033[0m\n')

#%%
def merge_dollar(df, dollar):
    
    #dollar = dollar[dollar['EcncIndDesc'] == 'Indicador Econômico']
    dollar.loc[:,'PricVal'] = dollar.loc[:,'PricVal'].str.replace(',', '.').astype(float)
    dollar = dollar.drop(columns=['Asst',	'TckrSymb', 'EcncIndDesc'])
    try:
        df_merged = pd.merge(df, dollar, how='left', left_on='dates_b', right_on='RptDt')
    except:
        print(f'some thing on merge went wrong...')

    return df_merged.dropna()

df_merged = merge_dollar(df0, dollar)

#%%
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
            df[f'punctuation_count_{text_column}'] = df[text_column].apply(lambda x: sum(1 for char in x if char in "!?.,;:"))
            df[f'polarity_{text_column}'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity)
            df[f'subjectivity_{text_column}'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.subjectivity)

            # Análise de sentimento com SentimentIntensityAnalyzer
            sentiment_scores = df[text_column].apply(lambda x: self.sia.polarity_scores(x))
            df[f'neg_{text_column}'] = sentiment_scores.apply(lambda x: x['neg'])
            df[f'neu_{text_column}'] = sentiment_scores.apply(lambda x: x['neu'])
            df[f'pos_{text_column}'] = sentiment_scores.apply(lambda x: x['pos'])
            df[f'compound_{text_column}'] = sentiment_scores.apply(lambda x: x['compound'])

            # Remover linhas com datas inválidas, se necessário
            #if 'dates_b' in df.columns:
            #    df = df[df['dates_b'] != 'NaT']

        return df.iloc[:, 4:]  # Retorna apenas as colunas geradas






# %%
X = df_merged.drop(columns=['links', 'dates', 'dates_b', 'PricVal', 'RptDt'], axis=1)

y = df_merged['PricVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


# %%
pipeline = Pipeline([
    ('feat', CreateFeatures(text_columns=['texts', 'headlines'])),
    ('clf', RandomForestRegressor(random_state=42))
])

pipeline.fit(X_train, y_train)

# %%
# Predições com o melhor modelo
y_pred = pipeline.predict(X_test)
y_pred_train = pipeline.predict(X_train)

RMSE = (sum((y_pred - y_test)**2)/len(y_test)**0.5)
print(f'RMSE_test: {RMSE}')

RMSE_train = (sum((y_pred_train - y_train)**2)/len(y_train)**0.5)
print(f'RMSE_train: {RMSE_train}')

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

# Plotando os resíduos
plt.figure(figsize=(14, 6))

# Resíduos no conjunto de teste
plt.subplot(1, 2, 1)
sns.histplot(residuals, kde=True, color='blue', bins=30)
plt.title('Distribuição dos Resíduos - Teste')
plt.xlabel('Resíduo')
plt.ylabel('Frequência')

# Resíduos no conjunto de treinamento
plt.subplot(1, 2, 2)
sns.histplot(residuals_train, kde=True, color='red', bins=30)
plt.title('Distribuição dos Resíduos - Treinamento')
plt.xlabel('Resíduo')
plt.ylabel('Frequência')

plt.tight_layout()
plt.show()

# - Gráficos de dispersão: valores previstos vs valores reais
# Teste
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Previsões vs Real - Teste')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.show()

# Treinamento
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_pred_train, color='red', alpha=0.7)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='blue', linestyle='--')
plt.title('Previsões vs Real - Treinamento')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.show()

#%%
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
