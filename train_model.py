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



#%%
df0 = pd.read_csv('news_df_features.csv', index_col=0)
df0

#%%
X = df0.drop(columns=['dates_b', 'PricVal'], axis=1)

y = df0['PricVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

pipeline = Pipeline([
    ('clf', RandomForestRegressor(random_state=42))
])

param_grid = {
    'clf__n_estimators': [50, 100, 300],
    'clf__max_depth': [10, 20, 30, None],
    'clf__min_samples_split': [2, 3, 5],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__criterion': ['squared_error', 'poisson']
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)

grid_search.fit(X_train, y_train)

print(f"Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor desempenho (RMSE): {-grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_

# Predições com o melhor modelo
y_pred = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

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
print(f"RMSE (Teste): {RMSE:.3f}")
print(f"RMSE (Treinamento): {RMSE_train:.3f}")
print(f"MAE (Teste): {MAE:.3f}")
print(f"MAE (Treinamento): {MAE_train:.3f}")
print(f"R² (Teste): {r2_test:.3f}")
print(f"R² (Treinamento): {r2_train:.3f}")

#%% - Visualizações para análise de erros
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

# - Análise de importância das variáveis (feature importance)
# Exibindo a importância das variáveis do modelo Random Forest
importances = best_model.named_steps['clf'].feature_importances_
features = X.columns

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

# %%
