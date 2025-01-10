#%%
import requests
from datetime import datetime, timedelta

initial_date = '01-01-2015'  # Substitua pela data inicial desejada (formato YYYY-MM-DD)

def get_dollar_data(initial_date):
    
    #data_final = '2025-01-08'    # Substitua pela data final desejada (formato YYYY-MM-DD)
    yesterday = str((datetime.today()- timedelta(days=1)).strftime('%m-%d-%Y')).split(' ')[0]
    today = str((datetime.today()).strftime('%m-%d-%Y')).split(' ')[0]

    url = f"https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/CotacaoDolarPeriodo(dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)?@dataInicial='{initial_date}'&@dataFinalCotacao='{yesterday}'&$top=10000&$format=text/csv&$select=cotacaoCompra,cotacaoVenda,dataHoraCotacao"
    try:
        # Fazendo a requisição
        response = requests.get(url)

        # Verificando se a requisição foi bem-sucedida
        if response.status_code == 200:
            # Salvando o conteúdo como arquivo CSV
            with open(f"./dollar_exchange_brazil_data/brazilian_dolar_real_exchange_data_BC_{today}.csv", "wb") as file:
                file.write(response.content)
            print(f"CSV file successful saved as brazilian_dolar_real_exchange_data_BC_{today}.csv")
        else:
            print(f"request fail with status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"request exception: {e}")

get_dollar_data(initial_date)

# %%
