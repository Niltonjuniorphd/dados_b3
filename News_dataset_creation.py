#%%
import pandas as pd
import os

# Caminho do diretório (substitua pelo caminho da pasta desejada)
folder_path = "./news_data"

# Obtendo a lista de arquivos no diretório
file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

df0 = pd.DataFrame()
for file in file_names:
    print(file)
    df = pd.read_csv(f'./news_data/{file}', index_col=0, encoding='ISO-8859-1', sep=',')
    df0 = pd.concat([df0, df])

df0.to_csv('./news_dataset/News_dataset.csv', index=False, encoding='utf-8')


# %%