### README.md


# Dados B3

This repository contains scripts and data analysis tools for working with news and daily exchange rate data for the USA dollar against the Brazilian currency Real.

## Problem domain:
The main objective of this project is to fetch and analyze news articles and correlate it with the Brazilian Real exchage rate data from the B3 (Brazilian Stock Exchange) website. 
- The **goals** is a model that can predict the impact of news on the exchange rate value.
- The data (***'texts'*** transformed in ***'numbers'***) will be used to train a model to predict the impact of news on the exchange rate Brasilian currencies.
- The focus of the news is on specific **key words** that are sent to the Google News search engine. The purpose of these key words is to strip unusable content and optimise the news spectrum.
- The standar key word in this projet is 'Lula diz que dolar' (Lula says the dollar), focusind the impact of Lula's words on the exchange rate.

```bash
- ... in the search_news.py file, the key words are defined as:
# the words to focus on the serach in google
focus_key = 'Lula diz que dolar'
```



## Instructions

### 1. Clone the Repository
To clone this repository to your local machine, use the following command:

```bash
git clone https://github.com/Niltonjuniorphd/dados_b3.git
cd dados_b3
```

### 2. Create a Virtual Environment
Create a virtual environment using `venv` to manage dependencies:
```bash
python -m venv .env
```

Activate the virtual environment:
(not required if using VSCode in Windows)
- On Linux/Mac:
  ```bash
  source venv/bin/activate
  ```
- On Windows:
  ```bash
  venv\Scripts\activate
  ```

### 3. Install Required Libraries
Install the required Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt --no-cache-dir

```

### 4. Runnig:
Execute the script `run.py` on the terminal for the steps to be carried out:
```bash
python run.py
```

## Notes
- Ensure you have Python 3.8 or higher installed on your system before proceeding.
- The scripts will creates CSV files on `root: dados_b3` directory.
- The external financial data file from the [B3's official](https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/consultas/boletim-diario/historico-de-taxas-de-cambio-resolucao-bcb-n-120/) website must be stored in the `root: dados_b3` directory before executions (see next section).
- Data collections was constructed in **portuguese language**.

<div align="center">
    <img src="images\b3_historical_image.png" alt="Exemplo de Imagem" width="500" height="300"/>
</div>

---

## How it Works
#### Please note that just executing the Script **'run.py'** will perform all the steps below.

### **Step 1 : Scrap and Persist Data**
The script **`search_news.py`** performs the following:
1. ***__ETL__***: The script **`search_news.py`** fetches news from **Google News** using Selenium. The **`focus_key`** variable defines the search keywords.
2. **Process**: Once the site responds, the script scrapes news articles, collecting **headlines**, **links** and **text**. It processes at least **30 pages**.
3. **Output**: Generates a CSV file named **`news_df_yyyy_mm_dd.csv`** to store the data.

- The image below displays the expected dataframe output:
<div align="center">
    <img src="images\text_data_collection.png" alt="df0" width="825" height="325"/>
</div>
---

### **Step 2: Create Features and Persist as CSV**
The script **`create_features.py`** performs the following:
1. **Sentiment Analysis**: Reads data from **`news_df_yyyy-mm-dd.csv`** created in the spep 1 and calculates sentiment scores using **TextBlob** and **VADER** tools. Articles are classified as **positive**, **negative**, or **neutral**.
2. **Metric Calculation**: Computes word counts, word frequency, and other metrics.
3. **Financial Data Integration**: Loads financial data from **`2024_ExchangeRateFile_20241227_1.csv`**  (downloaded from [B3's official site](https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/consultas/boletim-diario/historico-de-taxas-de-cambio-resolucao-bcb-n-120/)) and merges it with news sentiment data based on the **date** column. (Perhaps the file name should be changed to match the downloaded file)

```bash	
# Please pay attention in this detail:
# the file name in the create_features.py should be changed to match the downloaded file

dolar = pd.read_csv('2024_ExchangeRateFile_20241227_1.csv', index_col=0, encoding='ISO-8859-1', sep=';')
```
4. **Output**: Creates a merged DataFrame with features and saves it as **`news_df_features_yyyy-mm-dd.csv`**. 

- The image below displays the expected dataframe output:
<div align="center">
    <img src="images\df_merged.png" alt="df_merged" width="825" height="325"/>
</div>

***PricVal*** is the merged column daily exchange rate from the B3 website downloaded file.

---

### **Step 3: Train and Test the Model**
The script **`train_model.py`** performs the following:
1. **Data Preparation**: Reads data from **`news_df_features_yyyy-mm-dd.csv`** and splits it into **training** and **testing** sets.
2. **Model Training**: Uses **GridSearchCV** to optimize hyperparameters for a **Random Forest Regressor**. The model is trained on the training set.
3. **Evaluation**: Calculates and prints the **accuracy** on the testing set. Statistics and **plots** are displayed in sequential windows. (Close one plot window to view the next.)


#### -> Please be advised that certain logs will be displayed in the terminal:
<div align="center">
    <img src="images\train_log.png" alt="train_log" width="500" height="400"/>
</div>

--- 

#### -> metrics plots will rise in a window:

<div align="center">
    <img src="images\metrics.png" alt="metrics" width="800" height="400"/>
</div>



---




